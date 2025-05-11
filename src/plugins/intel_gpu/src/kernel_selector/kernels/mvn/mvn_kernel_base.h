// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_base_opencl.h"
#include "kernel_selector_params.h"
#include <string>

namespace kernel_selector {
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// mvn_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct mvn_params : public base_params {
    mvn_params() : base_params(KernelType::MVN) {}

    MVNMode mvnMode = MVNMode::WITHIN_CHANNELS;
    bool mvnNormalizeVariance = false;
    float epsilon = 0.0f;
    MVNEpsMode mvnEpsMode = MVNEpsMode::INSIDE_SQRT;

    ParamsKey GetParamsKey() const override {
        ParamsKey k = base_params::GetParamsKey();

        k.EnableMVNMode(mvnMode);

        if (mvnNormalizeVariance)
            k.EnableMVNNormalizeVariance();

        return k;
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// MVNKernelBase
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class MVNKernelBase : public KernelBaseOpenCL {
public:
    using KernelBaseOpenCL::KernelBaseOpenCL;
    virtual ~MVNKernelBase() {}

    struct DispatchData : public CommonDispatchData {
        size_t itemsNum;
        size_t leftovers;
        size_t dataSetsCount;
        size_t dataSetSize;
        size_t maxSlmSize;

        DispatchData() : itemsNum(0), leftovers(0), dataSetsCount(0), dataSetSize(0), maxSlmSize(0) {}
    };

protected:
    bool Validate(const Params&) const override;
    virtual JitConstants GetJitConstants(const mvn_params& params, DispatchData dispatchData) const;
    virtual DispatchData SetDefault(const mvn_params& params) const;
    virtual std::string GetKernelName(const mvn_params&) const { return kernelName; }
    KernelsData GetCommonKernelsData(const Params& params) const;
    Datatype GetActivationType(const mvn_params& params) const;
    void GetUpdateDispatchDataFunc(KernelData& kd) const override;
};
}  // namespace kernel_selector
