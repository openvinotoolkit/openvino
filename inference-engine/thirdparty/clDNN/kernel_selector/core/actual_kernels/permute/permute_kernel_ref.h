// Copyright (c) 2016-2019 Intel Corporation
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
#include <vector>

namespace kernel_selector {
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// permute_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct permute_params : public base_params {
    permute_params() : base_params(KernelType::PERMUTE) {}

    std::vector<uint16_t> order;

    virtual ParamsKey GetParamsKey() const { return base_params::GetParamsKey(); }
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// permute_optional_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct permute_optional_params : optional_params {
    permute_optional_params() : optional_params(KernelType::PERMUTE) {}
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// PermuteKernelRef
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class PermuteKernelRef : public common_kernel_base {
public:
    PermuteKernelRef() : common_kernel_base("permute_ref") {}
    virtual ~PermuteKernelRef() {}

    JitConstants GetJitConstants(const permute_params& params) const;
    KernelsData GetKernelsData(const Params& params, const optional_params& options) const override;
    ParamsKey GetSupportedKey() const override;
};
}  // namespace kernel_selector