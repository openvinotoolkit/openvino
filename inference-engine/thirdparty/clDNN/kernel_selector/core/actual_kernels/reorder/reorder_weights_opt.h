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


#pragma once

#include "reorder_kernel_base.h"

namespace kernel_selector {
class ReorderWeightsOpt : public ReorderKernelBase {
public:
    ReorderWeightsOpt() : ReorderKernelBase("reorder_weights_opt") {}
    virtual ~ReorderWeightsOpt() {}

    KernelsData GetKernelsData(const Params& params, const optional_params& options) const override;
    ParamsKey GetSupportedKey() const override;
    DispatchData SetDefault(const reorder_weights_params& arg) const override;

protected:
    virtual bool Validate(const Params& params, const optional_params& options) const override;
    virtual JitConstants GetJitConstants(const reorder_weights_params& params) const override;
};
}  // namespace kernel_selector
