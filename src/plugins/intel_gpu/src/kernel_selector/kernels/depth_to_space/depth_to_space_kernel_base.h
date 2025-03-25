// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_base_opencl.h"
#include "kernel_selector_params.h"

namespace kernel_selector {
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// depth_to_space_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct depth_to_space_params : public base_params {
    depth_to_space_params()
    : base_params(KernelType::DEPTH_TO_SPACE)
    , block_size(0)
    , mode(DepthToSpaceMode::DEPTH_FIRST) {}
    size_t block_size;
    DepthToSpaceMode mode;
};

struct depth_to_space_fuse_params : fuse_params {
    depth_to_space_fuse_params() : fuse_params(KernelType::DEPTH_TO_SPACE) {}
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// DepthToSpaceKernelBase
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class DepthToSpaceKernelBase : public KernelBaseOpenCL {
public:
    using KernelBaseOpenCL::KernelBaseOpenCL;
    virtual ~DepthToSpaceKernelBase() {}

    struct DispatchData : public CommonDispatchData {
    };

protected:
    bool Validate(const Params&) const override;
    virtual JitConstants GetJitConstants(const depth_to_space_params& params) const;
    virtual CommonDispatchData SetDefault(const depth_to_space_params& params) const = 0;
    KernelsData GetCommonKernelsData(const Params& params) const;
};
}  // namespace kernel_selector
