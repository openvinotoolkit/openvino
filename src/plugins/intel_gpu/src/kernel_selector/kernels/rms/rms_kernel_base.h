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
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// rms_optional_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct rms_optional_params : optional_params {
    rms_optional_params() : optional_params(KernelType::RMS) {}
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
        size_t slmSize;
        size_t maxSlmSize;
        size_t leftovers;

        DispatchData() : dataSize(0), dataCount(0), slmSize(0), maxSlmSize(0), leftovers(0) {}
    };

protected:
    bool Validate(const Params&, const optional_params&) const override;
    virtual JitConstants GetJitConstants(const rms_params& params, DispatchData dispatchData) const;
    virtual DispatchData SetDefault(const rms_params& params) const;
    KernelsData GetCommonKernelsData(const Params& params, const optional_params&) const;
    Datatype GetAccumulatorType(const rms_params& params) const;
};
}  // namespace kernel_selector
