/*
// Copyright (c) 2019-2020 Intel Corporation
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

#include "lstm_dynamic_input_kernel_base.h"
#include "kernel_selector_utils.h"
#include "common_tools.h"
#include <vector>

namespace kernel_selector {
JitConstants LSTM_DynamicInputKernelBase::GetJitConstants(const lstm_dynamic_input_params& params) const {
    JitConstants jit = MakeBaseParamsJitConstants(params);

    jit.AddConstants({MakeJitConstant("WEIGHTS", params.weights),
                      MakeJitConstant("DYN_LENGTH", params.inputs.at(1)),
                      MakeJitConstant("MAX_SEQUENCE_LENGTH", params.inputs.at(0).Feature().v)});

    // [2] Optionals
    if (!params.bias.empty()) {
        jit.AddConstants({MakeJitConstant("BIAS", params.bias[0]), MakeJitConstant("BIAS_TERM", true)});
    }

    return jit;
}

LSTM_DynamicInputKernelBase::DispatchData LSTM_DynamicInputKernelBase::SetDefault(
    const lstm_dynamic_input_params& params) {
    DispatchData dispatchData;
    const auto& out = params.output;

    // 4 * hidden, batch * dir, seq_len
    dispatchData.gws = { out.X().v, out.Batch().v * out.Y().v, out.Feature().v };
    dispatchData.lws = GetOptimalLocalWorkGroupSizes(dispatchData.gws, params.engineInfo);

    return dispatchData;
}

void kernel_selector::LSTM_DynamicInputKernelBase::SetKernelArguments(const lstm_dynamic_input_params& params, clKernelData& kernel) const {
    kernel.arguments.push_back({ ArgumentDescriptor::Types::INPUT, 0 });
    kernel.arguments.push_back({ ArgumentDescriptor::Types::INPUT, 1 });
    kernel.arguments.push_back({ ArgumentDescriptor::Types::OUTPUT, 0 });
    kernel.arguments.push_back({ ArgumentDescriptor::Types::WEIGHTS, 0 });
    if (!params.bias.empty()) {
        kernel.arguments.push_back({ ArgumentDescriptor::Types::BIAS, 0 });
    }
}

KernelsData LSTM_DynamicInputKernelBase::GetCommonKernelsData(const Params& params,
                                                              const optional_params& options) const {
    if (!Validate(params, options)) {
        return {};
    }

    const lstm_dynamic_input_params& orgParams = static_cast<const lstm_dynamic_input_params&>(params);

    auto dispatchData = SetDefault(orgParams);
    KernelData k_data = KernelData::Default<lstm_dynamic_input_params>(params, 1);

    auto cldnn_jit = GetJitConstants(orgParams);
    auto entry_point = GetEntryPoint(kernelName, orgParams.layerID, options);
    auto jit = CreateJit(kernelName, cldnn_jit, entry_point);

    auto& kernel = k_data.kernels[0];
    kernel.workGroups.global = dispatchData.gws;
    kernel.kernelString = GetKernelString(kernelName, jit, entry_point, params.engineInfo);
    SetKernelArguments(orgParams, kernel);

    return {k_data};
}
}  // namespace kernel_selector
