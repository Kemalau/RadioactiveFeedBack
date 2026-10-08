import unittest

try:
    import torch
    from radioactive_feedback.objective import group_advantages, grpo_loss
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'install the train extra to test the tensor objective')
class ObjectiveTests(unittest.TestCase):
    def test_advantages_sample_variance_and_constant_group(self):
        rewards = torch.tensor([[95.,85.], [90.,90.]])
        advantages = group_advantages(rewards)
        expected = torch.tensor([5.,-5.]) / (torch.tensor(50.) + 1e-6).sqrt()
        torch.testing.assert_close(advantages[0], expected)
        torch.testing.assert_close(advantages[1], torch.zeros(2))

    def test_clipping_kl_and_padding_do_not_change_loss(self):
        current = torch.tensor([[0.5,100.]], requires_grad=True)
        old = torch.zeros_like(current)
        reference = torch.zeros_like(current)
        mask = torch.tensor([[1,0]])
        loss, diagnostics = grpo_loss(current, old, reference, mask, torch.tensor([1.]), beta=0.04)
        expected = -1.2 + 0.04 * (torch.exp(torch.tensor(-0.5)) + 0.5 - 1)
        torch.testing.assert_close(loss, expected)
        loss.backward()
        self.assertEqual(current.grad[0,1].item(), 0)
        self.assertEqual(diagnostics['clip_fraction'], 1)


if __name__ == '__main__':
    unittest.main()
