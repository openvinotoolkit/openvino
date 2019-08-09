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


#include "fully_connected_kernel_image_tutorial.h"
#include "kernel_selector_utils.h"
#include <vector>

namespace kernel_selector {
ParamsKey FullyConnected_image_tutorial::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableInputWeightsType(WeightsType::F16);
    k.EnableInputWeightsType(WeightsType::F32);
    k.EnableAllInputLayout();
    k.EnableInputLayout(DataLayout::bf);
    k.EnableOutputLayout(DataLayout::bf);
    k.EnableBiasPerOutput();
    k.EnableBiasPerFeature();
    k.EnableNonBiasTerm();
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    return k;
}

FullyConnected_image_tutorial::DispatchData FullyConnected_image_tutorial::SetDefault(
    const fully_connected_params& params,
    int) const {
    auto runInfo = Parent::SetDefault(params);

    std::vector<size_t> global = {params.output.Feature().v, params.output.Batch().v};
    std::vector<size_t> local = GetOptimalLocalWorkGroupSizes(global);

    runInfo.gws0 = global[0];
    runInfo.gws1 = global[1];
    runInfo.gws2 = 1;

    runInfo.lws0 = local[0];
    runInfo.lws1 = local[1];
    runInfo.lws2 = 1;

    runInfo.effiency = TUTORIAL_PRIORITY;

    return runInfo;
}

KernelsData FullyConnected_image_tutorial::GetKernelsData(const Params& params, const optional_params& options) const {
    KernelsData res = {};
    for (size_t i = 0; i < autoTuneOptions.size(); i++) {
        KernelsData kd = GetTunedKernelsDataByIndex(params,
                                                    options,
                                                    DataLayout::bfyx,
                                                    {WeightsLayout::image_2d_weights_c4_fyx_b},
                                                    DONT_USE_IF_HAVE_SOMETHING_ELSE,
                                                    static_cast<int>(i));
        if (!kd.empty()) {
            res.emplace_back(kd[0]);
        }
    }
    return res;
}
}  // namespace kernel_selector
