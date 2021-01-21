// Copyright (c) 2016-2020 Intel Corporation
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


#include "reorder_weights_image_winograd_6x3_kernel.h"
#include "kernel_selector_utils.h"

namespace kernel_selector {
ParamsKey ReorderWeightsImageWinograd6x3Kernel::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputWeightsType(WeightsType::F16);
    k.EnableInputWeightsType(WeightsType::F32);
    k.EnableOutputWeightsType(WeightsType::F16);
    k.EnableOutputWeightsType(WeightsType::F32);
    k.EnableAllInputWeightsLayout();
    k.EnableOutputWeightsLayout(WeightsLayout::image_2d_weights_winograd_6x3_s1_fbxyb);
    k.EnableOutputWeightsLayout(WeightsLayout::image_2d_weights_winograd_6x3_s1_xfbyb);
    k.EnableWinogradReorder();
    k.EnableDifferentTypes();
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    return k;
}

ReorderWeightsImageWinograd6x3Kernel::DispatchData ReorderWeightsImageWinograd6x3Kernel::SetDefault(
    const reorder_weights_params& params) const {
    DispatchData dispatchData;

    const auto& input = params.input;

    dispatchData.gws[0] = 1;
    dispatchData.gws[1] = 3;
    dispatchData.gws[2] = static_cast<size_t>(input.IFM().v * input.OFM().v);

    dispatchData.lws[0] = 1;
    dispatchData.lws[1] = 1;
    dispatchData.lws[2] = 32;

    return dispatchData;
}

KernelsData ReorderWeightsImageWinograd6x3Kernel::GetKernelsData(const Params& params,
                                                                 const optional_params& options) const {
    const reorder_weights_params& orgParams = static_cast<const reorder_weights_params&>(params);
    return GetCommonKernelsData(orgParams, options);
}

KernelsPriority ReorderWeightsImageWinograd6x3Kernel::GetKernelsPriority(const Params& /*params*/, const optional_params& /*options*/) const {
    return FORCE_PRIORITY_4;
}
}  // namespace kernel_selector
