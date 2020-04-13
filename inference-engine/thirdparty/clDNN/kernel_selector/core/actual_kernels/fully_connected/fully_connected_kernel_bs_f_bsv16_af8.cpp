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


#include "fully_connected_kernel_bs_f_bsv16_af8.h"

namespace kernel_selector {
ParamsKey FullyConnected_bs_f_bsv16_af8::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableInputWeightsType(WeightsType::F16);
    k.EnableInputWeightsType(WeightsType::F32);
    k.EnableAllInputLayout();
    k.EnableOutputLayout(DataLayout::fb);
    k.EnableOutputLayout(DataLayout::bs_f_bsv16__af8);
    k.EnableBatching();
    k.EnableBiasPerFeature();
    k.EnableNonBiasTerm();
    k.EnableSubGroup();
    return k;
}

FullyConnected_bs_f_bsv16_af8::DispatchData FullyConnected_bs_f_bsv16_af8::SetDefault(const fully_connected_params& arg,
                                                                                      int) const {
    auto kd = FullyConnectedBlockKernelBase::SetDefault(arg);

    size_t groups_per_batches = GetLocalGroupsSize(arg);
    kd.gws0 = Align(arg.output.LogicalSize() / (GetBatchesPerWorkItem(arg) * groups_per_batches), 16);
    kd.gws1 = groups_per_batches;
    kd.lws0 = 16;
    kd.lws1 = 1;

    return kd;
}

static bool check_input_layout(const DataTensor& t) {
    bool b16_layout = false;
    b16_layout |= t.GetLayout() == DataLayout::bs_f_bsv16__af8;
    b16_layout |= DataTensor::Channelndex(t.GetLayout(), Tensor::DataChannelName::BATCH) == 0 && t.Batch().v == 16;
    return b16_layout;
}

bool FullyConnected_bs_f_bsv16_af8::Validate(const Params& p, const optional_params& o) const {
    if (!FullyConnectedBlockKernelBase::Validate(p, o)) {
        return false;
    }

    const auto& params = static_cast<const fully_connected_params&>(p);
    const auto& optParams = static_cast<const fully_connected_optional_params&>(o);

    if (!params.engineInfo.bSubGroupShortSupport && params.inputs[0].GetDType() == Datatype::F16) {
        return false;
    }

    const bool bProperBatch = params.inputs[0].Batch().v == 16;
    const bool bProperInput = check_input_layout(params.inputs[0]);
    const bool bSupportedLayout = optParams.allowInputReordering || bProperInput;

    if (!bProperBatch || !bSupportedLayout) {
        return false;
    }

    return true;
}

KernelsData FullyConnected_bs_f_bsv16_af8::GetKernelsData(const Params& params,
                                                          const optional_params& optParams) const {
    KernelsData res = {};
    for (size_t i = 0; i < autoTuneOptions.size(); i++) {
        KernelsData kd = GetTunedKernelsDataByIndex(params,
                                                    optParams,
                                                    DataLayout::bs_f_bsv16__af8,
                                                    WeightsLayout::os_i_osv16__ai8,
                                                    FORCE_PRIORITY_2,
                                                    static_cast<int>(i));
        if (!kd.empty()) {
            res.emplace_back(kd[0]);
        }
    }

    return res;
}
}  // namespace kernel_selector
