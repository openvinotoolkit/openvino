/*
// Copyright (c) 2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

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

    virtual ParamsKey GetParamsKey() const { return base_params::GetParamsKey(); }
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
    virtual bool Validate(const Params&, const optional_params&) const;
    virtual JitConstants GetJitConstants(const space_to_batch_params& params) const;
    virtual CommonDispatchData SetDefault(const space_to_batch_params& params, const optional_params&) const;
    KernelsData GetCommonKernelsData(const Params& params, const optional_params&) const;
};
}  // namespace kernel_selector
