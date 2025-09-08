// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "resample_kernel_base.h"

namespace kernel_selector {
class ResampleKernelRef : public ResampleKernelBase {
public:
    using Parent = ResampleKernelBase;
    ResampleKernelRef() : ResampleKernelBase("resample_ref") {}
    virtual ~ResampleKernelRef() {}

    KernelsData GetKernelsData(const Params& params) const override;
    KernelsPriority GetKernelsPriority(const Params& params) const override;
    ParamsKey GetSupportedKey() const override;
    JitConstants GetJitConstants(const resample_params& params) const override;
    std::vector<FusedOpType> GetSupportedFusedOps() const override {
        return { FusedOpType::QUANTIZE,
                 FusedOpType::ELTWISE,
                 FusedOpType::ACTIVATION };
    }

protected:
    DispatchData SetDefault(const resample_params& arg) const override;
};
}  // namespace kernel_selector
