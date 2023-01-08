// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "lstm_gemm_kernel_base.h"
#include "kernel_selector_utils.h"
#include "common_tools.h"

namespace kernel_selector {
JitConstants LSTMGemmKernelBase::GetJitConstants(const lstm_gemm_params& params) const {
    JitConstants jit = MakeBaseParamsJitConstants(params);
    const auto& weights = params.weights;
    const auto& recurrent = params.recurrent;
    const auto& hidden = params.hidden;
    const auto& bias = params.bias;
    if (params.hasBias) {
        jit.AddConstants({MakeJitConstant("BIAS", bias), MakeJitConstant("BIAS_TERM", true)});
    }
    if (params.hasHidden) {
        jit.AddConstants({MakeJitConstant("HIDDEN", hidden),
                          MakeJitConstant("HIDDEN_TERM", true),
                          MakeJitConstant("RECURRENT", recurrent),
                          MakeJitConstant("HIDDEN_DIRECTION", params.hidden_direction)});
    }
    jit.AddConstants({MakeJitConstant("WEIGHTS", weights)});
    jit.AddConstants({MakeJitConstant("DIRECTION", params.direction)});
    jit.AddConstants({MakeJitConstant("INPUT_DIRECTION", params.input_direction)});

    return jit;
}

KernelsData LSTMGemmKernelBase::GetCommonKernelsData(const Params& params, const optional_params& options) const {
    if (!Validate(params, options)) {
        return {};
    }

    const lstm_gemm_params& orgParams = static_cast<const lstm_gemm_params&>(params);

    KernelData kd = KernelData::Default<lstm_gemm_params>(params, orgParams.inputs.size());

    const auto& input = orgParams.inputs[0];

    auto newParams = orgParams;
    newParams.inputs.resize(1);
    newParams.inputs[0] = input;
    auto out = newParams.outputs[0];
    // TODO: reorder weights if needed
    auto& kernel = kd.kernels[0];
    auto cldnnJit = GetJitConstants(newParams);
    auto entryPoint = GetEntryPoint(kernelName, newParams.layerID, params, options);
    auto jit = CreateJit(kernelName, cldnnJit, entryPoint);

    kernel.params.workGroups.global = {out.X().v, out.Batch().v, 1};
    kernel.code.kernelString = GetKernelString(kernelName, jit, entryPoint, params.engineInfo);
    kernel.params.arguments.push_back({ArgumentDescriptor::Types::INPUT, 0});
    kernel.params.arguments.push_back({ArgumentDescriptor::Types::OUTPUT, 0});
    kernel.params.arguments.push_back({ArgumentDescriptor::Types::WEIGHTS, 0});
    if (orgParams.hasHidden) {
        kernel.params.arguments.push_back({ArgumentDescriptor::Types::HIDDEN, 0});
        kernel.params.arguments.push_back({ArgumentDescriptor::Types::RECURRENT, 0});
    }
    if (orgParams.hasBias) {
        kernel.params.arguments.push_back({ArgumentDescriptor::Types::BIAS, 0});
    }

    return {kd};
}
}  // namespace kernel_selector
