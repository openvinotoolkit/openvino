// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fully_connected_kernel_bs_f_bsv8_af8.h"

namespace kernel_selector {
ParamsKey FullyConnected_bs_f_bsv8_af8::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableInputWeightsType(WeightsType::F16);
    k.EnableInputWeightsType(WeightsType::F32);
    k.EnableAllInputLayout();
    k.EnableOutputLayout(DataLayout::fb);
    k.EnableOutputLayout(DataLayout::bs_f_bsv8__af8);
    k.EnableBatching();
    k.EnableBiasPerFeature();
    k.EnableNonBiasTerm();
    return k;
}

DeviceFeaturesKey FullyConnected_bs_f_bsv8_af8::get_required_device_features_key(const Params& params) const {
    auto k = get_common_subgroups_device_features_key(params);
    k.requires_subgroup_shuffle();

    return k;
}

FullyConnected_bs_f_bsv8_af8::DispatchData FullyConnected_bs_f_bsv8_af8::SetDefault(const fully_connected_params& arg,
                                                                                    int, int /*kernel_number*/) const {
    auto dispatchData = FullyConnectedBlockKernelBase::SetDefault(arg);

    size_t groups_per_batches = GetLocalGroupsSize(arg);
    dispatchData.gws[0] =
        Align(arg.outputs[0].LogicalSize() / (GetNeuronsPerWorkItem(arg) * GetBatchesPerWorkItem(arg) * groups_per_batches),
              8);
    dispatchData.gws[1] = groups_per_batches;
    dispatchData.lws[0] = 8;
    dispatchData.lws[1] = 1;

    return dispatchData;
}

static bool check_output_layout(const DataTensor& t) {
    bool b16_layout = false;
    b16_layout |= (t.GetLayout() == DataLayout::fb) && (t.Batch().v == 8);
    b16_layout |= (t.GetLayout() == DataLayout::bs_f_bsv8__af8);
    return b16_layout;
}

bool FullyConnected_bs_f_bsv8_af8::Validate(const Params& p) const {
    if (!FullyConnectedBlockKernelBase::Validate(p)) {
        return false;
    }

    if (!IsSIMDSizeSupported(p.engineInfo, 8))
        return false;

    const auto& params = static_cast<const fully_connected_params&>(p);

    if (!params.engineInfo.supports_intel_subgroups_short && params.inputs[0].GetDType() == Datatype::F16) {
        return false;
    }

    const bool bProperBatch = params.inputs[0].Batch().v >= 8 && params.inputs[0].Batch().v % 8 == 0;
    const bool bProperFeature = params.inputs[0].Feature().v >= 8 && params.inputs[0].Feature().v % 8 == 0;
    const bool bProperOutput = check_output_layout(params.outputs[0]);

    if (!bProperBatch || !bProperFeature || !bProperOutput) {
        return false;
    }

    if (!params.bias.empty()) {
        if (params.inputs[0].GetDType() != params.bias[0].GetDType()) {
            return false;
        }
    }

    return true;
}

KernelsData FullyConnected_bs_f_bsv8_af8::GetKernelsData(const Params& params) const {
    KernelsData res = {};
    for (size_t i = 0; i < autoTuneOptions.size(); i++) {
        KernelsData kd = GetTunedKernelsDataByIndex(params,
                                                    DataLayout::bs_f_bsv8__af8,
                                                    WeightsLayout::os_i_osv8__ai8,
                                                    static_cast<int>(i));
        if (!kd.empty()) {
            res.emplace_back(kd[0]);
        }
    }

    return res;
}

KernelsPriority FullyConnected_bs_f_bsv8_af8::GetKernelsPriority(const Params& /*params*/) const {
    return FORCE_PRIORITY_4;
}
}  // namespace kernel_selector
