// Copyright (C) 2024 Intel Corporation
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

    std::int64_t num_groups = 1;
    double epsilon = 0.0f;

    ParamsKey GetParamsKey() const override {
        return base_params::GetParamsKey();
    }
};

class GroupNormalizationKernelBase : public KernelBaseOpenCL {
public:
    using KernelBaseOpenCL::KernelBaseOpenCL;
    virtual ~GroupNormalizationKernelBase() {}

    struct DispatchData : public CommonDispatchData {
        size_t itemsNum;
        size_t leftovers;
        size_t dataSetsCount;
        size_t dataSetSize;
        size_t maxSlmSize;

        DispatchData() : itemsNum(0), leftovers(0), dataSetsCount(0), dataSetSize(0), maxSlmSize(0) {}
    };

    struct MultiDispatchData {
        DispatchData stage_1;
        DispatchData stage_2;
        DispatchData stage_final;

        size_t item_groups;

        MultiDispatchData() : item_groups(0) {}
    };

protected:
    bool Validate(const Params&) const override;
    JitConstants GetJitConstants(const group_normalization_params& params) const;
    std::string GetKernelName(const group_normalization_params&) const { return kernelName; }
    Datatype GetActivationType(const group_normalization_params& params) const;
    Datatype GetAccumulatorType(const group_normalization_params& params) const;
};

}  // namespace kernel_selector
