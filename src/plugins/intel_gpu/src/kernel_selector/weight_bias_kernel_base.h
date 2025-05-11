// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_base_opencl.h"
#include "weight_bias_params.h"

namespace kernel_selector {
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// WeightsBiasKernelBase
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class WeightBiasKernelBase : public KernelBaseOpenCL {
public:
    using KernelBaseOpenCL::KernelBaseOpenCL;
    virtual ~WeightBiasKernelBase() {}

protected:
    virtual JitConstants GetJitConstants(const weight_bias_params& params) const;
};
}  // namespace kernel_selector
