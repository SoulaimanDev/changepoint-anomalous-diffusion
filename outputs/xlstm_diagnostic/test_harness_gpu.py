"""CPU-only by default. Real CUDA test requires explicit opt-in; no training."""
import ast
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch
import harness_gpu as gpu


class DiagnosticTests(unittest.TestCase):
    def test_cuda_unavailable_fails_without_fallback(self):
        with patch.object(torch.cuda,'is_available',return_value=False):
            with self.assertRaisesRegex(RuntimeError,'no CPU fallback'):
                gpu.select_device('cuda')

    def test_cuda_selection_and_batch_transfer_requested(self):
        with patch.object(torch.cuda,'is_available',return_value=True):
            device=gpu.select_device('cuda')
        self.assertEqual(device.type,'cuda')
        batch={'x':np.zeros((2,99,1),np.float32),'has_changepoint':np.zeros(2,np.float32),'cp_dx':np.full(2,-1)}
        fake=Mock();fake.to.return_value=fake
        with patch.object(torch,'from_numpy',return_value=fake):
            gpu.prepare_batch(batch,np.array([0,1]),device)
        self.assertEqual(fake.to.call_count,3)
        self.assertTrue(all(c.args==(device,) for c in fake.to.call_args_list))

    def test_config_seed_and_order_have_no_device_branch(self):
        row={'trainable_parameters':173266,'configuration_slot':'cfg_01'}
        p={'batch_size':256,'learning_rate':0.0021799023913081548}
        loader=Mock(return_value=({}, {}, row));decode=Mock(return_value=p)
        for device in ['cpu','cuda']:
            result=gpu.frozen_identity(loader,decode)
            self.assertEqual(result[-1],11);self.assertEqual(result[-2],p)
            loader.assert_called_with('xlstm','cfg_01')
        src=Path(gpu.__file__).read_text()
        self.assertIn('torch.Generator().manual_seed(seed)',src)
        self.assertIn('seed_run(seed)',src)
        self.assertIn("build_model_from_hyperparameters('xlstm',p).to(device)",src)
        a=torch.randperm(200000,generator=torch.Generator().manual_seed(11))
        b=torch.randperm(200000,generator=torch.Generator().manual_seed(result[-1]))
        self.assertTrue(torch.equal(a,b))

    def test_protected_and_arbitrary_output_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);project=root/'project'
            for path in [project,root/'gpu_cfg01_seed11_001',
                root/'campaign_runpod_confirmatory_fresh/outputs/xlstm_diagnostic/gpu_cfg01_seed11_001',
                root/'outputs/xlstm_diagnostic/wrong_name']:
                with self.assertRaises(ValueError):gpu.validate_output(project,path)
            accepted=root/'outputs/xlstm_diagnostic/gpu_cfg01_seed11_001'
            self.assertEqual(gpu.validate_output(project,accepted),accepted.resolve())
            self.assertFalse(accepted.exists())

    def test_symlink_escape_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);target=root/'campaign_runpod_confirmatory_fresh';target.mkdir()
            link=root/'outputs/xlstm_diagnostic';link.parent.mkdir()
            try:link.symlink_to(target,target_is_directory=True)
            except OSError:self.skipTest('Symlinks unavailable')
            with self.assertRaises(ValueError):
                gpu.validate_output(root/'project',link/'gpu_cfg01_seed11_001')

    def test_train_only_guard_blocks_validation_test_and_write(self):
        root=Path(tempfile.gettempdir());train=root/'train_L100_dim1_with_without_dx.h5'
        opener=Mock();opened=[];guard=gpu.train_guard(opener,train,opened)
        for name in ['val_L100_dim1_with_without_dx.h5','validation_tuning.h5',
                     'validation_calibration.h5','test_L100_dim1_with_without_dx.h5']:
            with self.assertRaises(PermissionError):guard(root/name,'r')
        for mode in ['w','a','r+']:
            with self.assertRaises(PermissionError):guard(train,mode)
        opener.assert_not_called();guard(train,'r');opener.assert_called_once()
        self.assertEqual(opened,[str(train.resolve())])
        source=Path(gpu.__file__).read_text();tree=ast.parse(source)
        self.assertFalse(any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
            and n.func.attr in ('tuning','load_calibration') for n in ast.walk(tree)))
        self.assertNotIn('TuningData(',source)
        self.assertNotIn('np.load(',source)

    def test_observer_masks_and_stops_before_exp_clamp(self):
        report={};obs=gpu.Observer(torch.nn.Identity(),report)
        mask=(torch.arange(99)>=19)&(torch.arange(99)<=79)
        with obs:
            x=torch.zeros(2,99).masked_fill(~mask[None,:],float('-inf'))
            torch.log_softmax(x,dim=1).detach()
        with self.assertRaises(gpu.FirstNonfinite),obs:
            torch.exp(torch.tensor(100.)).clamp(max=1e30)
        self.assertEqual(report['first_nonfinite']['operation'],'aten.exp.default')

    def test_deadline_stops_before_next_operation(self):
        obs=gpu.Observer(torch.nn.Identity(),{});obs.deadline=0
        with self.assertRaises(gpu.DiagnosticTimeLimit),obs:torch.ones(1)

    @unittest.skipUnless(os.environ.get('RUN_CUDA_DIAGNOSTIC_TESTS')=='1',
                         'Explicit CUDA opt-in required; never run by default')
    def test_real_cuda_placement_and_observer(self):
        device=gpu.select_device('cuda')
        model=torch.nn.Linear(1,1).to(device)
        self.assertTrue(next(model.parameters()).is_cuda)
        batch={'x':np.zeros((2,99,1),np.float32),'has_changepoint':np.zeros(2,np.float32),'cp_dx':np.full(2,-1)}
        values=gpu.prepare_batch(batch,np.array([0,1]),device)
        self.assertTrue(all(t.is_cuda for t in values))
        with gpu.Observer(model,{}):out=model(values[0])
        self.assertTrue(out.is_cuda);self.assertEqual(out.dtype,torch.float32)
        mask=(torch.arange(99,device=device)>=19)&(torch.arange(99,device=device)<=79)
        with gpu.Observer(model,{}):
            torch.zeros(2,99,device=device).masked_fill(~mask[None,:],float('-inf')).detach()


if __name__=='__main__':unittest.main()
