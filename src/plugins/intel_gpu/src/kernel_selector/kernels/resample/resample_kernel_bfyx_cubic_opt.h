// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "resample_kernel_base.h"

namespace kernel_selector {
class ResampleKernelBfyxCubicOpt : public ResampleKernelBase {
public:
    using Parent = ResampleKernelBase;
    ResampleKernelBfyxCubicOpt() : ResampleKernelBase("resample_bfyx_cubic_opt") {}
    virtual ~ResampleKernelBfyxCubicOpt() = default;

    KernelsData GetKernelsData(const Params& params) const override;
    KernelsPriority GetKernelsPriority(const Params& params) const override;
    ParamsKey GetSupportedKey() const override;

protected:
    bool Validate(const Params& p) const override;
    JitConstants GetJitConstants(const resample_params& params) const override;
    DispatchData SetDefault(const resample_params& arg) const override;
    std::vector<FusedOpType> GetSupportedFusedOps() const override {
        return { FusedOpType::QUANTIZE,
                 FusedOpType::ELTWISE,
                 FusedOpType::ACTIVATION };
    }

private:
    size_t GetOptimalBlockSize(const resample_params& params) const;
};
}  // namespace kernel_selector
