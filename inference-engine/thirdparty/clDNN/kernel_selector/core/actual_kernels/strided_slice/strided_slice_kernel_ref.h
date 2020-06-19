// Copyright (c) 2019-2020 Intel Corporation
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
// strided_slice_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct strided_slice_params : public base_params {
    strided_slice_params() : base_params(KernelType::STRIDED_SLICE) {}

    std::vector<std::vector<int32_t>> striding_params;
    std::vector<uint8_t> begin_mask;
    std::vector<uint8_t> end_mask;
    std::vector<uint8_t> ellipsis_mask;
    std::vector<uint8_t> new_axis_mask;
    std::vector<uint8_t> shrink_axis_mask;

    virtual ParamsKey GetParamsKey() const { return base_params::GetParamsKey(); }
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// strided_slice_optional_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct strided_slice_optional_params : optional_params {
    strided_slice_optional_params() : optional_params(KernelType::STRIDED_SLICE) {}
};

class StridedSliceKernelRef : public common_kernel_base {
public:
    StridedSliceKernelRef() : common_kernel_base("strided_slice_ref") {}
    virtual ~StridedSliceKernelRef() {}
    virtual JitConstants GetJitConstants(const strided_slice_params& params) const;
    virtual CommonDispatchData SetDefault(const strided_slice_params& params, const optional_params&) const;
    KernelsData GetKernelsData(const Params& params, const optional_params& options) const override;
    ParamsKey GetSupportedKey() const override;

protected:
    bool Validate(const Params& p, const optional_params& o) const override;
};
}  // namespace kernel_selector
