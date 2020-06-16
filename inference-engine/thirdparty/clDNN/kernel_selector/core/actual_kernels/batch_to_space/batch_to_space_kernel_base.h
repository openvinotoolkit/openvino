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
#include <vector>

namespace kernel_selector {
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// batch_to_space_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct batch_to_space_params : public base_params {
    batch_to_space_params() : base_params(KernelType::BATCH_TO_SPACE) {}
    std::vector<std::vector<int32_t>> bts_params;

    virtual ParamsKey GetParamsKey() const { return base_params::GetParamsKey(); }
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
class BatchToSpaceKernelBase : public common_kernel_base {
public:
    using common_kernel_base::common_kernel_base;
    virtual ~BatchToSpaceKernelBase() {}

    struct DispatchData : public CommonDispatchData {};

protected:
    virtual bool Validate(const Params&, const optional_params&) const;
    virtual JitConstants GetJitConstants(const batch_to_space_params& params) const;
    virtual CommonDispatchData SetDefault(const batch_to_space_params& params, const optional_params&) const;
    KernelsData GetCommonKernelsData(const Params& params, const optional_params&, float estimatedTime) const;
};
}  // namespace kernel_selector
