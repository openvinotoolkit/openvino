// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fully_connected_kernel_fb_io_b8_f8.h"
#include <algorithm>

namespace kernel_selector {
ParamsKey FullyConnected_fb_io_b8_f8::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F32);
    k.EnableInputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableInputWeightsType(WeightsType::F32);
    k.EnableInputWeightsType(WeightsType::F16);
    k.EnableAllInputLayout();
    k.EnableOutputLayout(DataLayout::fb);
    k.EnableBatching();
    k.EnableBiasPerFeature();
    k.EnableNonBiasTerm();
    return k;
}

DeviceFeaturesKey FullyConnected_fb_io_b8_f8::get_required_device_features_key(const Params& params) const {
    DeviceFeaturesKey k;
    k.requires_subgroups();
    k.requires_subgroup_shuffle();

    return k;
}

size_t FullyConnected_fb_io_b8_f8::GetBatchesPerWorkItem(const fully_connected_params& params) const {
    auto batch_size = params.outputs[0].Batch().v;

    if (batch_size % 32 == 0)
        return std::min(batch_size, static_cast<size_t>(32U));

    if (batch_size % 16 == 0)
        return std::min(batch_size, static_cast<size_t>(16U));

    return std::min(batch_size, static_cast<size_t>(8U));
}

FullyConnected_fb_io_b8_f8::DispatchData FullyConnected_fb_io_b8_f8::SetDefault(const fully_connected_params& arg,
                                                                                int, int /*kernel_number*/) const {
    auto dispatchData = FullyConnectedBlockKernelBase::SetDefault(arg);

    const auto& output = arg.outputs[0];

    size_t groups_per_batches = GetLocalGroupsSize(arg);
    dispatchData.gws[0] =
        Align(output.LogicalSize() / (GetNeuronsPerWorkItem(arg) * GetBatchesPerWorkItem(arg) * groups_per_batches), 8);
    dispatchData.gws[1] = groups_per_batches;
    dispatchData.lws[0] = 8;
    dispatchData.lws[1] = 1;

    return dispatchData;
}

bool FullyConnected_fb_io_b8_f8::Validate(const Params& p) const {
    if (!FullyConnectedBlockKernelBase::Validate(p)) {
        return false;
    }

    if (!IsSIMDSizeSupported(p.engineInfo, 8))
        return false;

    const auto& params = static_cast<const fully_connected_params&>(p);

    const auto& output = params.outputs[0];
    const auto batches = output.Batch().v;
    const auto x_size = output.LogicalSize() / batches;

    const auto& input = params.inputs[0];
    const auto input_x_size = input.LogicalSize() / input.Batch().v;
    const bool proper_input_aligment = (input_x_size % 8) == 0;
    const bool proper_output_aligment =
        (output.LogicalSize() /
         (GetNeuronsPerWorkItem(params) * GetBatchesPerWorkItem(params) * GetLocalGroupsSize(params)) % 8) == 0;
    const bool bSupportedBatch = (batches % 8) == 0;
    const bool bSupportedFeature = (x_size % 8) == 0;

    if (!bSupportedBatch || !bSupportedFeature || !proper_input_aligment || !proper_output_aligment) {
        return false;
    }

    if (!params.bias.empty()) {
        if (params.inputs[0].GetDType() != params.bias[0].GetDType()) {
            return false;
        }
    }

    return true;
}

KernelsData FullyConnected_fb_io_b8_f8::GetKernelsData(const Params& params) const {
    assert(params.GetType() == KernelType::FULLY_CONNECTED);
    KernelsData res = {};

    for (size_t i = 0; i < autoTuneOptions.size(); i++) {
        KernelsData kd =
            GetTunedKernelsDataByIndex(params,  DataLayout::fb, WeightsLayout::io, static_cast<int>(i));
        if (!kd.empty()) {
            res.emplace_back(kd[0]);
        }
    }

    return res;
}

KernelsPriority FullyConnected_fb_io_b8_f8::GetKernelsPriority(const Params& params) const {
    const auto& p = static_cast<const fully_connected_params&>(params);

    return p.inputs[0].GetDType() == Datatype::F16 && p.outputs[0].Batch().v >= 16 ? FORCE_PRIORITY_3
                                                                               : FORCE_PRIORITY_5;
}
}  // namespace kernel_selector
