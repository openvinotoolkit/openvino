// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fully_connected_kernel_fb_oi_b8_ref.h"

namespace kernel_selector {
ParamsKey FullyConnected_fb_oi_b8_ref::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableInputWeightsType(WeightsType::F16);
    k.EnableInputWeightsType(WeightsType::F32);
    k.EnableAllInputLayout();
    k.EnableOutputLayout(DataLayout::fb);
    k.EnableBatching();
    k.EnableBiasPerFeature();
    k.EnableNonBiasTerm();
    return k;
}

DeviceFeaturesKey FullyConnected_fb_oi_b8_ref::get_required_device_features_key(const Params& params) const {
    auto k = get_common_subgroups_device_features_key(params);
    k.requires_subgroup_shuffle();

    return k;
}

FullyConnected_fb_oi_b8_ref::DispatchData FullyConnected_fb_oi_b8_ref::SetDefault(const fully_connected_params& arg,
                                                                                  int, int /*kernel_number*/) const {
    auto dispatchData = FullyConnectedKernelBase::SetDefault(arg);

    const auto& output = arg.outputs[0];
    dispatchData.gws[0] = output.Batch().v;
    dispatchData.gws[1] = output.LogicalSize() / dispatchData.gws[0];
    dispatchData.lws[0] = 8;
    dispatchData.lws[1] = 1;

    return dispatchData;
}

bool FullyConnected_fb_oi_b8_ref::Validate(const Params& p) const {
    if (!FullyConnectedKernelBase::Validate(p)) {
        return false;
    }

    const auto& params = static_cast<const fully_connected_params&>(p);

    if (params.inputs[0].Batch().v != 8) {
        return false;
    }

    return true;
}

KernelsData FullyConnected_fb_oi_b8_ref::GetKernelsData(const Params& params) const {
    KernelsData res = {};
    for (size_t i = 0; i < autoTuneOptions.size(); i++) {
        KernelsData kd = GetTunedKernelsDataByIndex(params,
                                                    DataLayout::fb,
                                                    WeightsLayout::oi,
                                                    static_cast<int>(i));
        if (!kd.empty()) {
            res.emplace_back(kd[0]);
        }
    }
    return res;
}

KernelsPriority FullyConnected_fb_oi_b8_ref::GetKernelsPriority(const Params& /*params*/) const {
    return FORCE_PRIORITY_6;
}
}  // namespace kernel_selector
