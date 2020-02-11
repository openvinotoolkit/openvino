/*
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
*/

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
    const auto& out   = params.output;

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

    DispatchData run_info;

    KernelData kd = KernelData::Default<lstm_dynamic_input_params>(params);
    lstm_dynamic_input_params& dlstm_params = *static_cast<lstm_dynamic_input_params*>(kd.params.get());

    const auto& out = dlstm_params.output;
    auto hidden_size = out.X().v;

    std::vector<size_t> global = { hidden_size / simd_size, out.Batch().v * out.Y().v, out.Feature().v };
    const auto& local = GetOptimalLocalWorkGroupSizes(global, params.engineInfo);

    run_info.gws0 = global[0];
    run_info.gws1 = global[1];
    run_info.gws2 = global[2];

    run_info.lws0 = local[0];
    run_info.lws1 = local[1];
    run_info.lws2 = local[2];

    run_info.fp16UnitUsed = dlstm_params.inputs[0].GetDType() == Datatype::F16;

    bool succeed = UpdateWeightsParams(dlstm_params,
        options,
        { WeightsLayout::dlstm_dir_io },
        kd.weightsReorderParams,
        GetSupportedKey());

    if (!succeed) {
        return {};
    }

    auto cldnn_jit = GetJitConstants(dlstm_params);
    auto entry_point = GetEntryPoint(kernelName, dlstm_params.layerID, options);
    auto jit = CreateJit(kernelName, cldnn_jit, entry_point);

    auto& kernel = kd.kernels[0];
    kernel.workGroups.global = { run_info.gws0, run_info.gws1, run_info.gws2 };
    kernel.workGroups.local = { run_info.lws0, run_info.lws1, run_info.lws2 };
    kernel.kernelString = GetKernelString(kernelName, jit, entry_point, params.engineInfo);
    SetKernelArguments(dlstm_params, kernel);

    kd.estimatedTime = FORCE_PRIORITY_5;
    return { kd };
}
}  // namespace kernel_selector
