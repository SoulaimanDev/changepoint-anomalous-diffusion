import unittest
import sys
from pathlib import Path
import torch
from harness import Observer,FirstNonfinite

class ObserverTests(unittest.TestCase):
    def test_original_core_backward_source_provenance(self):
        sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'work/runpod_audit/prepared_project'))
        from confirmatory.models.xlstm import MatrixLSTMCore
        core=MatrixLSTMCore(4,1);r={};o=Observer(core,r)
        with torch.no_grad():
            core.input_gate.weight.zero_();core.input_gate.bias.fill_(-100.)
            core.forget_gate.weight.zero_();core.forget_gate.bias.zero_()
        old=sys.gettrace();sys.settrace(o.trace_forward)
        try:y=core(torch.ones(1,1,4,requires_grad=True))
        finally:sys.settrace(old)
        self.assertTrue(torch.isfinite(y).all())
        o.phase='backward'
        with self.assertRaises(FirstNonfinite),o:y.sum().backward()
        self.assertEqual(r['first_nonfinite']['line'],145)
        self.assertEqual(r['first_nonfinite']['time_step_zero_based'],0)
        self.assertEqual(r['first_nonfinite']['autograd_node'],'ExpBackward0')

    def test_expected_mask_is_not_numerical_failure(self):
        r={};o=Observer(torch.nn.Identity(),r)
        mask=(torch.arange(99)>=19)&(torch.arange(99)<=79)
        x=torch.zeros(2,99,requires_grad=True)
        with o:
            z=x.masked_fill(~mask[None,:],float('-inf'))
            z=torch.log_softmax(z,dim=1)
            z=z.detach()
        self.assertGreater(r['intentional_mask_operations_ignored'],0)

    def test_nan_on_valid_bin_is_never_exempted(self):
        o=Observer(torch.nn.Identity(),{})
        x=torch.zeros(2,99);x[0,19]=float('nan')
        with self.assertRaises(FirstNonfinite),o:x.detach()

    def test_observer_stops_before_clamp(self):
        r={};o=Observer(torch.nn.Identity(),r)
        x=torch.tensor(-100.)
        with self.assertRaises(FirstNonfinite),o:torch.exp(-x).clamp(max=1e30)
        self.assertEqual(r['first_nonfinite']['operation'],'aten.exp.default')

    def test_observation_preserves_rng_forward_and_backward(self):
        def run(observe):
            torch.manual_seed(11)
            m=torch.nn.Sequential(torch.nn.Linear(3,4),torch.nn.Dropout(.1),torch.nn.Linear(4,1))
            x=torch.randn(8,3)
            if observe:
                with Observer(m,{}):
                    y=m(x);y.square().mean().backward()
            else:
                y=m(x);y.square().mean().backward()
            return y.detach(),[p.grad.clone() for p in m.parameters()],torch.get_rng_state()
        a,b=run(False),run(True)
        self.assertTrue(torch.equal(a[0],b[0]));self.assertTrue(torch.equal(a[2],b[2]))
        self.assertTrue(all(torch.equal(x,y) for x,y in zip(a[1],b[1])))

if __name__=='__main__':unittest.main()
