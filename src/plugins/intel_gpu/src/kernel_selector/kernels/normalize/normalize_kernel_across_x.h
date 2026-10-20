#pragma once

#include "normalize_kernel_base.h"

namespace kernel_selector {
class NormalizeKernelAcrossX : public NormalizeKernelBase {
public:
    NormalizeKernelAcrossX() : NormalizeKernelBase("normalize_gpu_across_x") {}

    KernelsData GetKernelsData(const Params& params) const override;
    KernelsPriority GetKernelsPriority(const Params& params) const override;
    ParamsKey GetSupportedKey() const override;
};
}  // namespace kernel_selector