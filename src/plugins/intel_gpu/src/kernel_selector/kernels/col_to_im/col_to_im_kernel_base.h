// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_base_opencl.h"
#include "kernel_selector_params.h"

namespace kernel_selector {
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// col_to_im_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct col_to_im_params : public base_params {
    col_to_im_params()
    : base_params(KernelType::COL_TO_IM) {}
    uSize stride;
    uSize dilation;
    uSize padding_begin;
    uSize padding_end;
};

struct col_to_im_fuse_params : fuse_params {
    col_to_im_fuse_params() : fuse_params(KernelType::COL_TO_IM) {}
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// ColToImKernelBase
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class ColToImKernelBase : public KernelBaseOpenCL {
public:
    using KernelBaseOpenCL::KernelBaseOpenCL;
    virtual ~ColToImKernelBase() {}

    struct DispatchData : public CommonDispatchData {
    };

protected:
    bool Validate(const Params&) const override;
    virtual JitConstants GetJitConstants(const col_to_im_params& params) const;
    virtual CommonDispatchData SetDefault(const col_to_im_params& params) const = 0;
    KernelsData GetCommonKernelsData(const Params& params) const;
};
}  // namespace kernel_selector
