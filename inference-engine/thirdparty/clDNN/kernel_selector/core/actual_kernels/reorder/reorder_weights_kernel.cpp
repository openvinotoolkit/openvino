// Copyright (c) 2016 Intel Corporation
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


#include "kernel_selector_common.h"
#include "reorder_weights_kernel.h"
#include "kernel_selector_utils.h"

namespace kernel_selector {
ParamsKey ReorderWeightsKernel::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputWeightsType(WeightsType::INT8);
    k.EnableInputWeightsType(WeightsType::F16);
    k.EnableInputWeightsType(WeightsType::F32);
    k.EnableOutputWeightsType(WeightsType::INT8);
    k.EnableOutputWeightsType(WeightsType::F16);
    k.EnableOutputWeightsType(WeightsType::F32);
    k.EnableAllInputWeightsLayout();
    k.EnableAllOutputWeightsLayout();
    k.EnableDifferentTypes();
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    return k;
}

KernelsData ReorderWeightsKernel::GetKernelsData(const Params& params, const optional_params& options) const {
    const reorder_weights_params& orgParams = static_cast<const reorder_weights_params&>(params);
    return GetCommonKernelsData(orgParams, options, DONT_USE_IF_HAVE_SOMETHING_ELSE);
}
}  // namespace kernel_selector