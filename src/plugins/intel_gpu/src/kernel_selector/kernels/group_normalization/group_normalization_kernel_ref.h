// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once
#include "kernel_base_opencl.h"
#include "kernel_selector_params.h"

namespace kernel_selector {
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// GroupNormalizationParams
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct group_normalization_params : public base_params {
    group_normalization_params() : base_params(KernelType::GROUP_NORMALIZATION) {}

    std::int64_t num_groups{};
    double epsilon{};

    ParamsKey GetParamsKey() const override {
        return base_params::GetParamsKey();
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// group_normalization_optional_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct group_normalization_optional_params : optional_params {
    group_normalization_optional_params() : optional_params(KernelType::GROUP_NORMALIZATION) {}
};

class GroupNormalizationKernelRef : public KernelBaseOpenCL {
public:
    using DispatchData = CommonDispatchData;
    enum KernelId {
        eCalcMeanKernel,
        eCalcPow,
        eCalcStandardDeviationKernel,
        eNormalize,
        eKernelsNum
    };

    GroupNormalizationKernelRef() : KernelBaseOpenCL{"group_normalization_gpu_ref"} {}
    KernelsData GetKernelsData(const Params& params, const optional_params& options) const override;
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
