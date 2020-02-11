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

namespace kernel_selector {
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// resample_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct resample_params : public base_params {
    resample_params() : base_params(KernelType::RESAMPLE) {}

    uint32_t pad_begin = 0;
    uint32_t pad_end = 0;
    uint32_t align_corners = 0;
    ResampleType resampleType = ResampleType::NEAREST_NEIGHBOR;

    virtual ParamsKey GetParamsKey() const {
        auto k = base_params::GetParamsKey();
        k.EnableReampleType(resampleType);
        return k;
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// resample_optional_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct resample_optional_params : optional_params {
    resample_optional_params() : optional_params(KernelType::RESAMPLE) {}
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// ResampleKernelBase
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class ResampleKernelBase : public common_kernel_base {
public:
    using DispatchData = CommonDispatchData;
    using common_kernel_base::common_kernel_base;

    virtual ~ResampleKernelBase() {}

protected:
    bool Validate(const Params& p, const optional_params& o) const override;
    virtual DispatchData SetDefault(const resample_params& arg) const;
    virtual JitConstants GetJitConstants(const resample_params& params) const;
    KernelsData GetCommonKernelsData(const Params& params, const optional_params& options) const;
    size_t GetFeatureBlockSize(const resample_params& params) const;
};
}  // namespace kernel_selector
