// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_base_opencl.h"
#include "kernel_selector_params.h"

namespace kernel_selector {
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// reduce_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct reduce_params : public base_params {
    reduce_params() : base_params(KernelType::REDUCE), reduceMode(ReduceMode::MAX), keepDims(0) {}

    ReduceMode reduceMode;
    std::vector<uint16_t> reduceAxes;
    int32_t keepDims;
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// ReduceKernelBase
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class ReduceKernelBase : public KernelBaseOpenCL {
public:
    using KernelBaseOpenCL::KernelBaseOpenCL;
    using DispatchData = CommonDispatchData;

    virtual ~ReduceKernelBase() {}

protected:
    bool Validate(const Params&) const override;
    virtual JitConstants GetJitConstants(const reduce_params& params) const;
    virtual CommonDispatchData SetDefault(const reduce_params& params) const = 0;
    Datatype GetAccumulatorType(const reduce_params& p) const;
    Datatype GetFinalAccumulatorType(const reduce_params& p) const;
    Datatype GetActivationType(const reduce_params& params) const;
    KernelsData GetCommonKernelsData(const Params& params) const;
    void GetUpdateDispatchDataFunc(KernelData& kd) const override;
};
}  // namespace kernel_selector
