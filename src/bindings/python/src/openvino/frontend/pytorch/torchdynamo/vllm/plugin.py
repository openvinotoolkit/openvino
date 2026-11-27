def register():
    """Entry point for `vllm.general_plugins`.
    """
    from vllm.model_executor.kernels.linear import register_linear_kernel
    from .kernels.ov_dynamic_4bit import OVDynamic4bitLinearKernel
    from vllm.platforms import PlatformEnum
    register_linear_kernel(OVDynamic4bitLinearKernel, PlatformEnum.CPU, "mp")
