// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "lstm_kernel_base.h"
#include "kernel_selector_utils.h"
#include "common_tools.h"
#include <string>
#include <algorithm>

namespace kernel_selector {

JitConstants LSTMKernelBase::GetJitConstants(const lstm_params& params) const {
    JitConstants jit = MakeBaseParamsJitConstants(params);
    bool sequential = params.sequential;
    auto out =  params.outputs[0];
    if (params.input_forget) {
        jit.AddConstants({MakeJitConstant("INPUT_FORGET", true)});
    }
    jit.AddConstants({MakeJitConstant("VEC_SIZE", 4)});
    jit.AddConstants({MakeJitConstant("DIRECTION", static_cast<int>(params.direction))});
    const unsigned int gate_num = 4;
    jit.AddConstants({MakeJitConstant("GATE_NUM", gate_num)});
    if (sequential) {
        jit.AddConstants({MakeJitConstant("SEQUENCE", 1)});
    }
    int num_hidden_kernels;
    int hidden_size;
    if (sequential) {
        jit.AddConstants({MakeJitConstant("INPUT_SIZE", params.inputs[0].Y().v)});
        hidden_size = static_cast<int>(params.inputs[1].Y().v);
        num_hidden_kernels = std::min({static_cast<int>(params.engineInfo.maxWorkGroupSize), static_cast<int>(out.X().v)});
    } else {
        jit.AddConstants({MakeJitConstant("INPUT_SIZE", params.inputs[0].Feature().v)});
        hidden_size = static_cast<int>(params.inputs[1].Feature().v);
        num_hidden_kernels = std::min({static_cast<int>(params.engineInfo.maxWorkGroupSize), static_cast<int>(out.Feature().v)});
    }
    size_t size;
    if (sequential) {
        size = params.inputs[1].Y().v;
    } else {
        size = params.inputs[1].Feature().v;
    }
    jit.AddConstants({
        MakeJitConstant("GEMM_OFFSET_I", params.GetOffsetIndexI() * size),
        MakeJitConstant("GEMM_OFFSET_O", params.GetOffsetIndexO() * size),
        MakeJitConstant("GEMM_OFFSET_F", params.GetOffsetIndexF() * size),
        MakeJitConstant("GEMM_OFFSET_Z", params.GetOffsetIndexZ() * size),
    });
    jit.AddConstants({MakeJitConstant("BATCH_SIZE", params.inputs[1].Batch().v)});
    jit.AddConstants({MakeJitConstant("HIDDEN_SIZE", hidden_size)});
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

KernelsData LSTMKernelBase::GetCommonKernelsData(const Params& params) const {
    if (!Validate(params)) {
        return {};
    }

    const lstm_params& orgParams = static_cast<const lstm_params&>(params);
    bool sequential = orgParams.sequential;
    KernelData kd = KernelData::Default<lstm_params>(params, 1);

    auto out =  orgParams.outputs[0];

    auto& kernel = kd.kernels[0];
    kernel.params.arguments.push_back({ArgumentDescriptor::Types::INPUT, 0});
    kernel.params.arguments.push_back({ArgumentDescriptor::Types::INPUT, 1});
    kernel.params.arguments.push_back({ArgumentDescriptor::Types::INPUT, 2});
    kernel.params.arguments.push_back({ArgumentDescriptor::Types::INPUT, 3});
    kernel.params.arguments.push_back({ArgumentDescriptor::Types::INPUT, 4});
    kernel.params.arguments.push_back({ArgumentDescriptor::Types::INPUT, 5});
    if (sequential) {
        kernel.params.arguments.push_back({ArgumentDescriptor::Types::INPUT, 6});
    }
    kernel.params.arguments.push_back({ArgumentDescriptor::Types::OUTPUT, 0});
    kernel.params.arguments.push_back({ArgumentDescriptor::Types::OUTPUT, 1});
    if (sequential) {
        kernel.params.arguments.push_back({ArgumentDescriptor::Types::OUTPUT, 2});
    }
    auto cldnnJit = GetJitConstants(orgParams);
    auto entryPoint = GetEntryPoint(kernelName, orgParams.layerID, params);
    auto jit = CreateJit(kernelName, cldnnJit, entryPoint);
    size_t num_hidden_kernels;
    if (sequential) {
        num_hidden_kernels = static_cast<size_t>(std::min({params.engineInfo.maxWorkGroupSize, out.X().v}));
    } else {
        num_hidden_kernels = static_cast<size_t>(std::min({params.engineInfo.maxWorkGroupSize, out.Feature().v}));
    }
    kernel.params.workGroups.global = {num_hidden_kernels, out.Batch().v, 1};
    kernel.params.workGroups.local = {num_hidden_kernels, 1, 1};
    kernel.code.kernelString = GetKernelString(kernelName, jit, entryPoint, params.engineInfo);

    return {kd};
}
}  // namespace kernel_selector
