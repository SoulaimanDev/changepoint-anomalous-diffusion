import unittest
import torch

class OriginalDefect(unittest.TestCase):
    def test_original_exp_then_clamp(self):
        m=torch.tensor(-100.,dtype=torch.float32,requires_grad=True)
        before=torch.exp(-m)
        after=before.clamp(max=1e30)
        self.assertTrue(torch.isinf(before).item())
        self.assertTrue(torch.isfinite(after).item())
        after.backward()
        self.assertTrue(torch.isnan(m.grad).item())

if __name__=='__main__':unittest.main()
