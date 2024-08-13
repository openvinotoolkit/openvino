// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "lstm_cell_kernel_base.h"
#include "kernel_selector_utils.h"
#include "common_tools.h"
#include <string>
#include <algorithm>

namespace kernel_selector {

JitConstants LSTMCellKernelBase::GetJitConstants(const lstm_cell_params& params) const {
    JitConstants jit = MakeBaseParamsJitConstants(params);
    if (params.input_forget) {
        jit.AddConstants({MakeJitConstant("INPUT_FORGET", true)});
    }
    jit.AddConstants({MakeJitConstant("DIRECTION", static_cast<int>(params.direction))});
    size_t size = params.inputs[1].Feature().v;
    jit.AddConstants({
        MakeJitConstant("GEMM_OFFSET_I", params.GetOffsetIndexI() * size),
        MakeJitConstant("GEMM_OFFSET_O", params.GetOffsetIndexO() * size),
        MakeJitConstant("GEMM_OFFSET_F", params.GetOffsetIndexF() * size),
        MakeJitConstant("GEMM_OFFSET_Z", params.GetOffsetIndexZ() * size),
    });
    jit.AddConstants({MakeJitConstant("BATCH_SIZE", params.inputs[1].Batch().v)});
    jit.AddConstants({MakeJitConstant("INPUT_SIZE", params.inputs[0].Feature().v)});
    auto hidden_size = static_cast<const int>(params.inputs[1].Feature().v);
    jit.AddConstants({MakeJitConstant("HIDDEN_SIZE", hidden_size)});
    auto out =  params.outputs[0];
    auto num_hidden_kernels = std::min({static_cast<int>(params.engineInfo.maxWorkGroupSize), static_cast<int>(out.Feature().v), 8});
    int num_hidden_to_do = hidden_size/num_hidden_kernels + (hidden_size % num_hidden_kernels  ? 1 : 0);
    jit.AddConstant({MakeJitConstant("NUM_HIDDEN_TO_DO", num_hidden_to_do)});
    auto ftype = params.inputs[0].GetDType();
    // if ReLU activation present, we have to reset accumulator type for the kernel to FP32
    // to avoid possible overflows on FP16, since ReLU doesn't limit upper border of its result
    for (size_t i = 0; i < params.activations.size(); i++) {
        if (params.activations[i].function == ActivationFunction::RELU) {
            ftype = Datatype::F32;
            break;
        }
    }
    jit.Merge(MakeTypeJitConstants(ftype, "ACCUMULATOR"));

    static const std::vector<std::string> asuffixes = {"_F", "_G", "_H", "_CLIP"};
    for (size_t i = 0; i < params.activations.size(); i++) {
        std::vector<base_activation_params> aparams = { params.activations[i] };
        jit.Merge(MakeActivationJitConstants(aparams, params.inputs[0].GetDType(), asuffixes[i]));
    }

    if (params.clip <= 0) {
        jit.AddConstants({
                MakeJitConstant("ACTIVATION_PARAMS_CLIP", ""),
                MakeJitConstant("ACTIVATION_CLIP(x, p)", "(x)"),
            });
    }

    return jit;
}

KernelsData LSTMCellKernelBase::GetCommonKernelsData(const Params& params) const {
    if (!Validate(params)) {
        return {};
    }

    const lstm_cell_params& orgParams = static_cast<const lstm_cell_params&>(params);

    KernelData kd = KernelData::Default<lstm_cell_params>(params, 1);

    auto out =  orgParams.outputs[0];

    auto& kernel = kd.kernels[0];
    kernel.params.arguments.push_back({ArgumentDescriptor::Types::INPUT, 0});
    kernel.params.arguments.push_back({ArgumentDescriptor::Types::INPUT, 1});
    kernel.params.arguments.push_back({ArgumentDescriptor::Types::INPUT, 2});
    kernel.params.arguments.push_back({ArgumentDescriptor::Types::INPUT, 3});
    kernel.params.arguments.push_back({ArgumentDescriptor::Types::INPUT, 4});
    kernel.params.arguments.push_back({ArgumentDescriptor::Types::INPUT, 5});
    kernel.params.arguments.push_back({ArgumentDescriptor::Types::OUTPUT, 0});
    kernel.params.arguments.push_back({ArgumentDescriptor::Types::OUTPUT, 1});
    auto cldnnJit = GetJitConstants(orgParams);
    auto entryPoint = GetEntryPoint(kernelName, orgParams.layerID, params);
    auto jit = CreateJit(kernelName, cldnnJit, entryPoint);
    auto num_hidden_kernels = static_cast<size_t>(std::min({params.engineInfo.maxWorkGroupSize, out.Feature().v, \
    static_cast<size_t>(8)}));
    kernel.params.workGroups.global = {num_hidden_kernels, out.Batch().v, 1};
    kernel.params.workGroups.local = {num_hidden_kernels, 1, 1};
    kernel.code.kernelString = GetKernelString(kernelName, jit, entryPoint, params.engineInfo);

    return {kd};
}
}  // namespace kernel_selector
