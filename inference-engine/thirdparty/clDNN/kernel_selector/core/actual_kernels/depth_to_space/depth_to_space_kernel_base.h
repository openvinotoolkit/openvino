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

#include "common_kernel_base.h"
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

    virtual ParamsKey GetParamsKey() const { return base_params::GetParamsKey(); }
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// depth_to_space_optional_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct depth_to_space_optional_params : optional_params {
    depth_to_space_optional_params() : optional_params(KernelType::DEPTH_TO_SPACE) {}
};

struct depth_to_space_fuse_params : fuse_params {
    depth_to_space_fuse_params() : fuse_params(KernelType::DEPTH_TO_SPACE) {}
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// DepthToSpaceKernelBase
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class DepthToSpaceKernelBase : public common_kernel_base {
public:
    using common_kernel_base::common_kernel_base;
    virtual ~DepthToSpaceKernelBase() {}

    struct DispatchData : public CommonDispatchData {
    };

protected:
    virtual bool Validate(const Params&, const optional_params&) const;
    virtual JitConstants GetJitConstants(const depth_to_space_params& params) const;
    virtual CommonDispatchData SetDefault(const depth_to_space_params& params) const;
    KernelsData GetCommonKernelsData(const Params& params, const optional_params&, float estimatedTime) const;
};
}  // namespace kernel_selector
