## https://github.com/openvinotoolkit/openvino/issues/37215

import torch

from vllm.platforms import CpuArchEnum, current_platform
import openvino.frontend.pytorch.ov_custom_ops
from vllm.scalar_type import scalar_types

from vllm.model_executor.kernels.linear.mixed_precision.MPLinearKernel import MPLinearKernel, MPLinearLayerConfig


# This implementation is for the KleidiAI-accelerated w4a8int [Compressed-Tensor]
# quantization scheme on Arm CPUs:

# Used to replace the default Dynamic4bitLinearKernel with ct_gemm 
# ct_gemm will be replaced by matmul decompression subgraph by openvino

class OVDynamic4bitLinearKernel(MPLinearKernel):
    SUPPORTED_QUANT_TYPES = [scalar_types.int4]

    @classmethod
    def get_min_capability(cls) -> int:
        return 1

    @classmethod
    def can_implement(cls, c: MPLinearLayerConfig) -> tuple[bool, str | None]:
        if not current_platform.is_cpu():
            return False, "Only CPU is supported"
        if c.weight_type not in cls.SUPPORTED_QUANT_TYPES:
            return False, f"Unsupported quant type {c.weight_type}"
        if (
            current_platform.get_cpu_architecture() == CpuArchEnum.ARM
            and c.act_type
            not in [
                torch.float32,
                torch.float16,
            ]
        ):
            return (
                False,
                "OVDynamic4bitLinearKernel on Arm requires Float32 or"
                " Float16 activations",
            )
        if c.full_weight_shape[0] % c.group_size != 0:
            return (
                False,
                f"Group size ({c.group_size}) does not evenly divide"
                " the number of input features "
                f"({c.full_weight_shape[0]})",
            )
        if current_platform.get_cpu_architecture() == CpuArchEnum.ARM:
            try:
                # Attempt to retrieve the operation
                _ = torch.ops.ov_ext.ct_gemm
            except AttributeError:
                return (
                    False,
                    f"OpenVINO does not support"
                    " _dyn_quant_matmul_4bit. Install a newer version",
                )
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module):

        self.packed_weight = getattr(layer, self.w_q_name)
        
        self.scales = getattr(layer, self.w_s_name)
        self.packed_weight = self.packed_weight.add(8)
        uint8_packed = (self.packed_weight[::, 1::2] << 4 | self.packed_weight[::, ::2]).to(
            torch.uint8
        )
        self.packed_weight = uint8_packed
        return

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        
        # At the time of Openvino compilation ct_gemm will be replaced by Matmul subgraph
        out = torch.ops.ov_ext.ct_gemm(x, self.packed_weight, self.scales, int(32), True, None, layer.bias)
        return out

