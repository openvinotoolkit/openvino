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

#include "quantize_kernel_base.h"

namespace kernel_selector {

class QuantizeKernelScaleShift : public QuantizeKernelBase {
public:
    using Parent = QuantizeKernelBase;

    QuantizeKernelScaleShift() : QuantizeKernelBase("quantize_gpu_scale_shift_opt") {}
    virtual ~QuantizeKernelScaleShift() {}

    JitConstants GetJitConstants(const quantize_params& params, const CommonDispatchData& dispatchData) const override;
    CommonDispatchData SetDefault(const quantize_params& params, const optional_params&) const override;
    bool Validate(const Params& p, const optional_params& o) const override;
    ParamsKey GetSupportedKey() const override;
};
}  // namespace kernel_selector
