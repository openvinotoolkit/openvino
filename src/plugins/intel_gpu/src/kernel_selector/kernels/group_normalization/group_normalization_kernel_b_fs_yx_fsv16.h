// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "group_normalization_kernel_base.h"

namespace kernel_selector {
class GroupNormalizationKernel_b_fs_yx_fsv16 : public GroupNormalizationKernelBase {
public:
    using Parent = GroupNormalizationKernelBase;

    GroupNormalizationKernel_b_fs_yx_fsv16() : GroupNormalizationKernelBase{"group_normalization_gpu_b_fs_yx_fsv16"} {}
    virtual ~GroupNormalizationKernel_b_fs_yx_fsv16() {}

    KernelsData GetKernelsData(const Params& params) const override;
    KernelsPriority GetKernelsPriority(const Params& params) const override;
    ParamsKey GetSupportedKey() const override;

protected:
    std::vector<FusedOpType> GetSupportedFusedOps() const override {
        return {
            FusedOpType::ACTIVATION,
            FusedOpType::QUANTIZE,
            FusedOpType::ELTWISE
        };
    }
    MultiDispatchData SetDefault(const group_normalization_params& params) const;
    JitConstants GetJitConstants(const group_normalization_params& params, GroupNormalizationKernelBase::DispatchData dispatchData) const;
    void GetUpdateDispatchDataFunc(KernelData& kd) const override;
    bool Validate(const Params& params) const override;
};

}  // namespace kernel_selector
