// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_base_opencl.h"
#include "kernel_selector_params.h"

namespace kernel_selector {
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// col2im_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct col2im_params : public base_params {
    col2im_params()
    : base_params(KernelType::COL2IM) {}
    // Required
    uSize output_size;
    uSize kernel_size;
    // Optional
    uSize stride;
    uSize dilation;
    uSize padding_begin;
    uSize padding_end;
};

struct col2im_fuse_params : fuse_params {
    col2im_fuse_params() : fuse_params(KernelType::COL2IM) {}
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Col2ImKernelBase
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class Col2ImKernelBase : public KernelBaseOpenCL {
public:
    using KernelBaseOpenCL::KernelBaseOpenCL;
    virtual ~Col2ImKernelBase() {}

    struct DispatchData : public CommonDispatchData {
    };

protected:
    bool Validate(const Params&) const override;
    virtual JitConstants GetJitConstants(const col2im_params& params) const;
    virtual CommonDispatchData SetDefault(const col2im_params& params) const = 0;
    KernelsData GetCommonKernelsData(const Params& params) const;
};
}  // namespace kernel_selector
