// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "group_normalization_kernel_base.h"

namespace kernel_selector {
class GroupNormalizationKernelRef : public GroupNormalizationKernelBase {
public:
    using Parent = GroupNormalizationKernelBase;
    enum KernelId {
        eCalcMeanKernel,
        eCalcStandardDeviationKernel,
        eNormalize,
        eKernelsNum
    };

    GroupNormalizationKernelRef() : GroupNormalizationKernelBase{"group_normalization_gpu_ref"} {}
    virtual ~GroupNormalizationKernelRef() {}

    KernelsData GetKernelsData(const Params& params) const override;
    KernelsPriority GetKernelsPriority(const Params& params) const override;
    ParamsKey GetSupportedKey() const override;
    std::vector<FusedOpType> GetSupportedFusedOps() const override {
        return {
            FusedOpType::ACTIVATION,
            FusedOpType::QUANTIZE,
            FusedOpType::ELTWISE
        };
    }

protected:
    DispatchData SetDefault(KernelId id, const group_normalization_params& params) const;
    JitConstants GetJitConstants(KernelId kernelId, const group_normalization_params& params) const;
    static void SetKernelArguments(const group_normalization_params& params,
                                   KernelId kernelId,
                                   cldnn::arguments_desc& arguments,
                                   std::vector<std::size_t>& internalBufferSizes);
};

}  // namespace kernel_selector
