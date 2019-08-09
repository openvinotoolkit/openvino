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
// one_hot_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct one_hot_params : public base_params {
    one_hot_params() : base_params(KernelType::ONE_HOT) {}
    uint16_t one_hot_axis;
    int32_t one_hot_limit;
    float on_value;
    float off_value;
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// one_hot_optional_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct one_hot_optional_params : optional_params {
    one_hot_optional_params() : optional_params(KernelType::ONE_HOT) {}
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// OneHotKernelBase
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class OneHotKernelBase : public common_kernel_base {
public:
    using common_kernel_base::common_kernel_base;

    using DispatchData = CommonDispatchData;

protected:
    JitConstants GetJitConstants(const one_hot_params& params) const;
    static DispatchData SetDefault(const one_hot_params& params);
    KernelsData GetCommonKernelsData(const Params& params, const optional_params&, float estimated_time) const;
};
}  // namespace kernel_selector
