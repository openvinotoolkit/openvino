// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_base_opencl.h"
#include "kernel_selector_params.h"
#include <vector>

namespace kernel_selector {
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// space_to_batch_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct space_to_batch_params : public base_params {
    space_to_batch_params() : base_params(KernelType::SPACE_TO_BATCH) {}
    DimTensor<uint32_t> block_shape;
    DimTensor<uint32_t> pads_begin;
    DimTensor<uint32_t> pads_end;

    base_params::ArgType block_type = base_params::ArgType::Input;
    base_params::ArgType begin_type = base_params::ArgType::Input;
    base_params::ArgType end_type = base_params::ArgType::Input;

    size_t block_dims = 0;
    size_t begin_dims = 0;
    size_t end_dims = 0;

    size_t block_input_index = 0;
    size_t begin_input_index = 0;
    size_t end_input_index = 0;
};

struct space_to_batch_fuse_params : fuse_params {
    space_to_batch_fuse_params() : fuse_params(KernelType::SPACE_TO_BATCH) {}
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// SpaceToBatchKernelBase
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class SpaceToBatchKernelBase : public KernelBaseOpenCL {
public:
    using KernelBaseOpenCL::KernelBaseOpenCL;
    virtual ~SpaceToBatchKernelBase() {}

    struct DispatchData : public CommonDispatchData {};

protected:
    bool Validate(const Params&) const override;
    virtual JitConstants GetJitConstants(const space_to_batch_params& params) const;
    virtual CommonDispatchData SetDefault(const space_to_batch_params& params) const;
    KernelsData GetCommonKernelsData(const Params& params) const;
};
}  // namespace kernel_selector
