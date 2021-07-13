// Copyright (C) 2018-2021 Intel Corporation
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
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// space_to_batch_optional_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct space_to_batch_optional_params : optional_params {
    space_to_batch_optional_params() : optional_params(KernelType::SPACE_TO_BATCH) {}
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
    bool Validate(const Params&, const optional_params&) const override;
    virtual JitConstants GetJitConstants(const space_to_batch_params& params) const;
    virtual CommonDispatchData SetDefault(const space_to_batch_params& params, const optional_params&) const;
    KernelsData GetCommonKernelsData(const Params& params, const optional_params&) const;
};
}  // namespace kernel_selector
