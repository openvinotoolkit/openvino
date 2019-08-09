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


#include "reorder_weights_binary_kernel.h"
#include "kernel_selector_utils.h"
#include <vector>

namespace kernel_selector {
ParamsKey ReorderWeightsBinaryKernel::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputWeightsType(WeightsType::BINARY);
    k.EnableOutputWeightsType(WeightsType::BINARY);
    k.EnableInputWeightsLayout(WeightsLayout::oiyx);
    k.EnableOutputWeightsLayout(WeightsLayout::os_is_yx_osv32_isv32p);
    k.EnableDifferentTypes();
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    return k;
}

ReorderWeightsBinaryKernel::DispatchData ReorderWeightsBinaryKernel::SetDefault(
    const reorder_weights_params& params) const {
    const auto& out = params.output;

    DispatchData kd;

    std::vector<size_t> global = {out.OFM().v, CeilDiv(out.IFM().v, 32), out.X().v * out.Y().v};
    auto local = GetOptimalLocalWorkGroupSizes(global);

    kd.gws0 = global[0];
    kd.gws1 = global[1];
    kd.gws2 = global[2];

    kd.lws0 = local[0];
    kd.lws1 = local[1];
    kd.lws2 = local[2];

    return kd;
}

KernelsData ReorderWeightsBinaryKernel::GetKernelsData(const Params& params, const optional_params& options) const {
    const reorder_weights_params& orgParams = static_cast<const reorder_weights_params&>(params);
    return GetCommonKernelsData(orgParams, options, FORCE_PRIORITY_4);
}
}  // namespace kernel_selector
