// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_base_opencl.h"
#include "quantize_kernel_params.h"

namespace kernel_selector {

class QuantizeKernelBase : public KernelBaseOpenCL {
public:
    using KernelBaseOpenCL::KernelBaseOpenCL;
    virtual ~QuantizeKernelBase() {}

    bool Validate(const Params& p) const override;
    KernelsData GetKernelsData(const Params& params) const override;

protected:
    virtual JitConstants GetJitConstants(const quantize_params& params, const CommonDispatchData& dispatchData) const;
    virtual CommonDispatchData SetDefault(const quantize_params& params) const = 0;
    void GetUpdateDispatchDataFunc(KernelData& kd) const override;
};
}  // namespace kernel_selector
