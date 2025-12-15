import torch

from pytorch_layer_test_class import PytorchLayerTest


class TestSmoothL1Loss(PytorchLayerTest):

    def _prepare_input(self):
        return {
            "x": torch.randn(2, 3),
            "y": torch.randn(2, 3)
        }

    def create_model(self):
        class SmoothL1LossModel(torch.nn.Module):
            def forward(self, x, y):
                return torch.nn.functional.smooth_l1_loss(x, y)

        return SmoothL1LossModel(), None, "aten::smooth_l1_loss"

    def test_smooth_l1_loss(self, ie_device, precision):
        model, ref_net, kind = self.create_model()

        self._test(
            model=model,
            ref_net=ref_net,
            kind=kind,
            ie_device=ie_device,
            precision=precision,
            ir_version=None
        )
