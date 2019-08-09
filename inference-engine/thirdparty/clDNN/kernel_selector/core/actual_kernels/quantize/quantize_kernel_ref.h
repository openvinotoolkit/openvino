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

namespace kernel_selector {
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// quantize_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct quantize_params : public base_params {
    quantize_params() : base_params(KernelType::QUANTIZE) {}

    int levels;
    bool packed_binary_output;

    virtual ParamsKey GetParamsKey() const { return base_params::GetParamsKey(); }
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// quantize_optional_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct quantize_optional_params : optional_params {
    quantize_optional_params() : optional_params(KernelType::QUANTIZE) {}
};

class QuantizeKernelRef : public common_kernel_base {
public:
    QuantizeKernelRef() : common_kernel_base("quantize_ref") {}
    virtual ~QuantizeKernelRef() {}

    virtual JitConstants GetJitConstants(const quantize_params& params) const;
    virtual CommonDispatchData SetDefault(const quantize_params& params, const optional_params&) const;
    KernelsData GetKernelsData(const Params& params, const optional_params& options) const override;
    ParamsKey GetSupportedKey() const override;
};
}  // namespace kernel_selector
