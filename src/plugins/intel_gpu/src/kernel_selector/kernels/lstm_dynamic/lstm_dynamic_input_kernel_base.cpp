// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

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
    auto in_layout = params.inputs[0].GetLayout();
    auto out_layout = params.outputs[0].GetLayout();
    std::vector<std::vector<Tensor::DataChannelName>> dims_by_gws = {{ Tensor::DataChannelName::X },
                                                                     { Tensor::DataChannelName::Y, Tensor::DataChannelName::BATCH },
                                                                     { Tensor::DataChannelName::FEATURE }};

    const auto& out = params.outputs[0];

    // 4 * hidden, batch * dir, seq_len
    dispatchData.gws = { out.X().v, out.Batch().v * out.Y().v, out.Feature().v };
    dispatchData.lws = GetOptimalLocalWorkGroupSizes(dispatchData.gws, params.engineInfo, in_layout, out_layout, dims_by_gws);

    return dispatchData;
}

void kernel_selector::LSTM_DynamicInputKernelBase::SetKernelArguments(const lstm_dynamic_input_params& params, clKernelData& kernel) const {
    kernel.params.arguments.push_back({ ArgumentDescriptor::Types::INPUT, 0 });
    kernel.params.arguments.push_back({ ArgumentDescriptor::Types::INPUT, 1 });
    kernel.params.arguments.push_back({ ArgumentDescriptor::Types::OUTPUT, 0 });
    kernel.params.arguments.push_back({ ArgumentDescriptor::Types::WEIGHTS, 0 });
    if (!params.bias.empty()) {
        kernel.params.arguments.push_back({ ArgumentDescriptor::Types::BIAS, 0 });
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
    auto entry_point = GetEntryPoint(kernelName, orgParams.layerID, params, options);
    auto jit = CreateJit(kernelName, cldnn_jit, entry_point);

    auto& kernel = k_data.kernels[0];
    kernel.params.workGroups.global = dispatchData.gws;
    kernel.code.kernelString = GetKernelString(kernelName, jit, entry_point, params.engineInfo);
    SetKernelArguments(orgParams, kernel);

    return {k_data};
}
}  // namespace kernel_selector
