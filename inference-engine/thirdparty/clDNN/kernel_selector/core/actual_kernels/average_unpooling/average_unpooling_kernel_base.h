// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_base_opencl.h"
#include "kernel_selector_params.h"

namespace kernel_selector {
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// average_unpooling_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct average_unpooling_params : public base_params {
    average_unpooling_params() : base_params(KernelType::AVERAGE_UNPOOLING) {}

    uSize unpoolSize;
    uSize unpoolStride;
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// average_unpooling_optional_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct average_unpooling_optional_params : optional_params {
    average_unpooling_optional_params() : optional_params(KernelType::AVERAGE_UNPOOLING) {}
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// AverageUnpoolingKernelBase
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class AverageUnpoolingKernelBase : public KernelBaseOpenCL {
public:
    using KernelBaseOpenCL::KernelBaseOpenCL;
    virtual ~AverageUnpoolingKernelBase() {}

    struct DispatchData : public CommonDispatchData {
        bool needsBoundary = false;
    };

protected:
    bool Validate(const Params&, const optional_params&) const override;
    virtual JitConstants GetJitConstants(const average_unpooling_params& params) const;
    virtual DispatchData SetDefault(const average_unpooling_params& params) const;
    KernelsData GetCommonKernelsData(const Params& params, const optional_params&) const;
};
}  // namespace kernel_selector
