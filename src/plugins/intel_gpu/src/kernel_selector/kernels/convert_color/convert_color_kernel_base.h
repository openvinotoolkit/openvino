// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_base_opencl.h"
#include "kernel_selector_params.h"
#include <vector>

namespace kernel_selector {
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// convert_color_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct convert_color_params : public base_params {
    convert_color_params() : base_params(KernelType::CONVERT_COLOR) {}
    color_format input_color_format = color_format::BGR;
    color_format output_color_format = color_format::BGR;
    memory_type mem_type = memory_type::buffer;
};

struct convert_color_fuse_params : fuse_params {
    convert_color_fuse_params() : fuse_params(KernelType::CONVERT_COLOR) {}
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// ConvertColorKernelBase
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class ConvertColorKernelBase : public KernelBaseOpenCL {
public:
    using KernelBaseOpenCL::KernelBaseOpenCL;
    virtual ~ConvertColorKernelBase() {}

    struct DispatchData : public CommonDispatchData {};

protected:
    bool Validate(const Params&) const override;
    virtual JitConstants GetJitConstants(const convert_color_params& params) const;
    virtual CommonDispatchData SetDefault(const convert_color_params& params) const;
    KernelsData GetCommonKernelsData(const Params& params) const;
};
}  // namespace kernel_selector
