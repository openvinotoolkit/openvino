// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_base_opencl.h"
#include "kernel_selector_params.h"
#include <vector>

namespace kernel_selector {
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// batch_to_space_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct batch_to_space_params : public base_params {
    batch_to_space_params() : base_params(KernelType::BATCH_TO_SPACE) {}
    DimTensor<uint32_t> block_shape;
    DimTensor<uint32_t> crops_begin;
    DimTensor<uint32_t> crops_end;
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// batch_to_space_optional_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct batch_to_space_optional_params : optional_params {
    batch_to_space_optional_params() : optional_params(KernelType::BATCH_TO_SPACE) {}
};

struct batch_to_space_fuse_params : fuse_params {
    batch_to_space_fuse_params() : fuse_params(KernelType::BATCH_TO_SPACE) {}
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// BatchToSpaceKernelBase
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class BatchToSpaceKernelBase : public KernelBaseOpenCL {
public:
    using KernelBaseOpenCL::KernelBaseOpenCL;
    virtual ~BatchToSpaceKernelBase() {}

    struct DispatchData : public CommonDispatchData {};

protected:
    bool Validate(const Params&, const optional_params&) const override;
    virtual JitConstants GetJitConstants(const batch_to_space_params& params) const;
    virtual CommonDispatchData SetDefault(const batch_to_space_params& params, const optional_params&) const;
    KernelsData GetCommonKernelsData(const Params& params, const optional_params&) const;
};
}  // namespace kernel_selector
