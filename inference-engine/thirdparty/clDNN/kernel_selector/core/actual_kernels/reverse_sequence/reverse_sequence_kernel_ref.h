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
// reverse_sequence_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct reverse_sequence_params : public base_params {
    reverse_sequence_params() : base_params(KernelType::REVERSE_SEQUENCE) {}

    int32_t seq_axis;
    int32_t batch_axis;

    virtual ParamsKey GetParamsKey() const { return base_params::GetParamsKey(); }
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// reverse_sequence_optional_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct reverse_sequence_optional_params : optional_params {
    reverse_sequence_optional_params() : optional_params(KernelType::REVERSE_SEQUENCE) {}
};

class ReverseSequenceKernelRef : public common_kernel_base {
public:
    ReverseSequenceKernelRef() : common_kernel_base("reverse_sequence_ref") {}
    virtual ~ReverseSequenceKernelRef() {}
    virtual JitConstants GetJitConstants(const reverse_sequence_params& params) const;
    virtual CommonDispatchData SetDefault(const reverse_sequence_params& params, const optional_params&) const;
    KernelsData GetKernelsData(const Params& params, const optional_params& options) const override;
    ParamsKey GetSupportedKey() const override;
};
}  // namespace kernel_selector
