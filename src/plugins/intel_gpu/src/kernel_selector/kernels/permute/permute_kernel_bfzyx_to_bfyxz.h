// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "permute_kernel_base.h"

namespace kernel_selector {
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// PermuteKernel_bfzyx_to_bfyxz
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class PermuteKernel_bfzyx_to_bfyxz : public PermuteKernelBase {
public:
    using Parent = PermuteKernelBase;
    using Parent::Parent;
    PermuteKernel_bfzyx_to_bfyxz() : PermuteKernelBase("permute_bfzyx_to_bfyxz") {}
    virtual ~PermuteKernel_bfzyx_to_bfyxz() {}

    bool Validate(const Params& p) const override;
    KernelsPriority GetKernelsPriority(const Params& params) const override;
    ParamsKey GetSupportedKey() const override;
protected:
    JitConstants GetJitConstants(const permute_params& params, const CommonDispatchData& dispatchData) const override;
    CommonDispatchData SetDefault(const permute_params& params) const override;
    std::vector<FusedOpType> GetSupportedFusedOps() const override {
        return {
            FusedOpType::ACTIVATION,
            FusedOpType::QUANTIZE,
            FusedOpType::ELTWISE
        };
    }
};
}  // namespace kernel_selector
