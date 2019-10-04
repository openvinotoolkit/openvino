// Copyright (c) 2019 Intel Corporation
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

#pragma once

#include "common_kernel_base.h"
#include "kernel_selector_params.h"

namespace kernel_selector {
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// gather_tree_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct gather_tree_params : public base_params {
    gather_tree_params() : base_params(KernelType::GATHER_TREE) {}
};
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// gather_tree_optional_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct gather_tree_optional_params : optional_params {
    gather_tree_optional_params() : optional_params(KernelType::GATHER_TREE) {}
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// BorderKernelBase
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class GatherTreeKernelBase : public common_kernel_base {
public:
    using common_kernel_base::common_kernel_base;
    using DispatchData = CommonDispatchData;

    protected:
        JitConstants GetJitConstants(const gather_tree_params& params) const;
        DispatchData SetDefault(const gather_tree_params& params) const;
        KernelsData GetCommonKernelsData(const Params& params, const optional_params&, float estimated_time) const;
};
}  // namespace kernel_selector
