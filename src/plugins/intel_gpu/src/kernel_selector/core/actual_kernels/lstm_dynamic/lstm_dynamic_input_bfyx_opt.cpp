// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "lstm_dynamic/lstm_dynamic_input_bfyx_opt.h"
#include "kernel_selector_utils.h"

#include <vector>

namespace kernel_selector {

ParamsKey LSTM_DynamicInputKernelBfyxOpt::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableInputWeightsType(WeightsType::F32);
    k.EnableInputWeightsType(WeightsType::F16);
    k.EnableInputLayout(DataLayout::bfyx);
    k.EnableOutputLayout(DataLayout::bfyx);
    k.EnableDifferentTypes();
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    k.EnableLSTMGEMMBias();
    k.EnableNonBiasTerm();
    k.EnableBiasPerFeature();
    k.EnableBiasPerOutput();
    k.EnableSubGroup();
    k.EnableSubGroupShort();
    return k;
}

bool kernel_selector::LSTM_DynamicInputKernelBfyxOpt::Validate(const Params & p, const optional_params & o) const {
    if (!LSTM_DynamicInputKernelBase::Validate(p, o)) {
        return false;
    }

    const auto& params = static_cast<const lstm_dynamic_input_params&>(p);

    const auto& weights  = params.weights;
    const auto weights_x = weights.X().v;
    const auto weights_y = weights.Y().v;
    const auto& input = params.inputs[0];
    const auto& out   = params.outputs[0];

    bool input_X_div_by_8 = input.X().v % 8 == 0;
    bool weights_X_div_by_8 = weights_x % 8 == 0;
    bool weights_Y_div_by_8_x_simd_size = weights_y % (8 * simd_size) == 0;
    bool gws0_size = out.X().v / simd_size <= 512;  // ToDo remove condition and update .cl code for bigger gws0

    if (!input_X_div_by_8 ||
        !weights_X_div_by_8 ||
        !weights_Y_div_by_8_x_simd_size ||
        !gws0_size)
        return false;
    return true;
}

KernelsData LSTM_DynamicInputKernelBfyxOpt::GetKernelsData(const Params& params, const optional_params& options) const {
    if (!Validate(params, options)) {
        return {};
    }

    DispatchData dispatchData;

    KernelData kd = KernelData::Default<lstm_dynamic_input_params>(params);
    lstm_dynamic_input_params& dlstm_params = *static_cast<lstm_dynamic_input_params*>(kd.params.get());

    auto in_layout = dlstm_params.inputs[0].GetLayout();
    auto out_layout = dlstm_params.outputs[0].GetLayout();
    std::vector<std::vector<Tensor::DataChannelName>> dims_by_gws = {{ Tensor::DataChannelName::X },
                                                                     { Tensor::DataChannelName::Y, Tensor::DataChannelName::BATCH },
                                                                     { Tensor::DataChannelName::FEATURE }};

    const auto& out = dlstm_params.outputs[0];
    auto hidden_size = out.X().v;

    dispatchData.gws = { hidden_size / simd_size, out.Batch().v * out.Y().v, out.Feature().v };
    dispatchData.lws = GetOptimalLocalWorkGroupSizes(dispatchData.gws, params.engineInfo, in_layout, out_layout, dims_by_gws);

    bool succeed = UpdateWeightsParams(dlstm_params,
        options,
        WeightsLayout::dlstm_dir_io,
        kd.weightsReorderParams,
        GetSupportedKey());

    if (!succeed) {
        return {};
    }

    auto cldnn_jit = GetJitConstants(dlstm_params);
    auto entry_point = GetEntryPoint(kernelName, dlstm_params.layerID, params, options);
    auto jit = CreateJit(kernelName, cldnn_jit, entry_point);

    auto& kernel = kd.kernels[0];
    kernel.params.workGroups.global = dispatchData.gws;
    kernel.params.workGroups.local = dispatchData.lws;
    kernel.code.kernelString = GetKernelString(kernelName, jit, entry_point, params.engineInfo);
    SetKernelArguments(dlstm_params, kernel);

    return { kd };
}

KernelsPriority LSTM_DynamicInputKernelBfyxOpt::GetKernelsPriority(const Params& /*params*/, const optional_params& /*options*/) const {
    return FORCE_PRIORITY_5;
}
}  // namespace kernel_selector
