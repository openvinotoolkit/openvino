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

namespace kernel_selector {

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// space_to_depth_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct space_to_depth_params : public base_params {
    space_to_depth_params() : base_params(KernelType::SPACE_TO_DEPTH), depth_mode(SpaceToDepthMode::BLOCKS_FIRST), block_size(1) {}

    SpaceToDepthMode depth_mode;

    size_t block_size;

    virtual ParamsKey GetParamsKey() const { return base_params::GetParamsKey(); }
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// space_to_depth_optional_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct space_to_depth_optional_params : optional_params {
    space_to_depth_optional_params() : optional_params(KernelType::SPACE_TO_DEPTH) {}
};

class SpaceToDepthKernelRef : public common_kernel_base {
public:
    SpaceToDepthKernelRef() : common_kernel_base("space_to_depth_ref") {}
    virtual ~SpaceToDepthKernelRef() {}
    virtual JitConstants GetJitConstants(const space_to_depth_params& params) const;
    virtual CommonDispatchData SetDefault(const space_to_depth_params& params, const optional_params&) const;
    KernelsData GetKernelsData(const Params& params, const optional_params& options) const override;
    ParamsKey GetSupportedKey() const override;
};
}  // namespace kernel_selector
