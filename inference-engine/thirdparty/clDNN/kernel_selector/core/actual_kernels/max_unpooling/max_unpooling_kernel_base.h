// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_base_opencl.h"
#include "kernel_selector_params.h"

namespace kernel_selector {
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// max_unpooling_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct max_unpooling_params : public base_params {
    max_unpooling_params() : base_params(KernelType::MAX_UNPOOLING) {}
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// max_unpooling_optional_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct max_unpooling_optional_params : optional_params {
    max_unpooling_optional_params() : optional_params(KernelType::MAX_UNPOOLING) {}
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// MaxUnpoolingKernelBase
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class MaxUnpoolingKernelBase : public KernelBaseOpenCL {
public:
    using KernelBaseOpenCL::KernelBaseOpenCL;
    virtual ~MaxUnpoolingKernelBase() {}

    struct DispatchData : public CommonDispatchData {
        bool needsBoundary = false;
    };

protected:
    bool Validate(const Params&, const optional_params&) const override;
    virtual JitConstants GetJitConstants(const max_unpooling_params& params) const;
    virtual DispatchData SetDefault(const max_unpooling_params& params) const;
    KernelsData GetCommonKernelsData(const Params& params, const optional_params&) const;
};
}  // namespace kernel_selector
