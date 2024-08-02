// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_base_opencl.h"

namespace kernel_selector {
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// rms_params
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct rms_params : public base_params {
    rms_params() : base_params(KernelType::RMS) {}
    float epsilon = 0.0f;
    int32_t ov_input_rank = -1;
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// RMSKernelBase
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class RMSKernelBase : public KernelBaseOpenCL {
public:
    using KernelBaseOpenCL::KernelBaseOpenCL;
    virtual ~RMSKernelBase() {}

    struct DispatchData : public CommonDispatchData {
        size_t dataSize;
        size_t dataCount;
        size_t maxSlmSize;
        size_t leftovers;
        size_t itemsNum;
        size_t subgroupBlockSize;

        DispatchData() : dataSize(0), dataCount(0), maxSlmSize(0), leftovers(0), itemsNum(0), subgroupBlockSize(0) {}
    };

protected:
    bool Validate(const Params&) const override;
    virtual JitConstants GetJitConstants(const rms_params& params, DispatchData dispatchData) const;
    virtual DispatchData SetDefault(const rms_params& params) const;
    KernelsData GetCommonKernelsData(const Params& params) const;
    Datatype GetAccumulatorType(const rms_params& params) const;
    void GetUpdateDispatchDataFunc(KernelData& kd) const override;
};
}  // namespace kernel_selector
