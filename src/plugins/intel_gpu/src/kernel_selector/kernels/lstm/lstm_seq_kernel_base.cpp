// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "lstm_seq_kernel_base.h"
#include "kernel_selector_utils.h"
#include "common_tools.h"
#include <string>

namespace kernel_selector {

JitConstants LSTMSeqKernelBase::GetJitConstants(const lstm_seq_params& params) const {
    JitConstants jit = MakeBaseParamsJitConstants(params);

    if (params.has_cell) {
        jit.AddConstants({MakeJitConstant("CELL_TERM", true),
                          MakeJitConstant("CELL_DIRECTION", params.cell_direction)});
    }
    if (params.input_forget) {
        jit.AddConstants({MakeJitConstant("INPUT_FORGET", true)});
    }
    jit.AddConstants({MakeJitConstant("DIRECTION", params.direction)});

    const auto& GEMMInput = params.inputs[0];
    size_t size = GEMMInput.X().v / 4;
    jit.AddConstants({
        MakeJitConstant("GEMM_OFFSET_I", params.GetOffsetIndexI() * size),
        MakeJitConstant("GEMM_OFFSET_O", params.GetOffsetIndexO() * size),
        MakeJitConstant("GEMM_OFFSET_F", params.GetOffsetIndexF() * size),
        MakeJitConstant("GEMM_OFFSET_Z", params.GetOffsetIndexZ() * size),
    });
    jit.AddConstants({MakeJitConstant("BATCH_SIZE", GEMMInput.Batch().v)});
    jit.AddConstants({MakeJitConstant("MAX_SEQ_LENGTH", GEMMInput.Feature().v)});
    jit.AddConstants({MakeJitConstant("INPUT_SIZE", GEMMInput.Y().v)});
    jit.AddConstants({MakeJitConstant("HIDDEN_SIZE", params.inputs[1].Y().v)});
    auto ftype = GetUnitType(params);
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
        jit.Merge(MakeActivationJitConstants(aparams, ftype, asuffixes[i]));
    }

    if (params.clip <= 0) {
        jit.AddConstants({
                MakeJitConstant("ACTIVATION_PARAMS_CLIP", ""),
                MakeJitConstant("ACTIVATION_CLIP(x, p)", "(x)"),
            });
    }

    return jit;
}

KernelsData LSTMSeqKernelBase::GetCommonKernelsData(const Params& params) const {
    if (!Validate(params)) {
        return {};
    }

    const lstm_seq_params& orgParams = static_cast<const lstm_seq_params&>(params);

    KernelData kd = KernelData::Default<lstm_seq_params>(params, 1);

    //const auto& input = orgParams.inputs[0];

    auto newParams = orgParams;
    auto out = newParams.outputs[0];

    auto& kernel = kd.kernels[0];
    auto cldnnJit = GetJitConstants(orgParams);
    auto entryPoint = GetEntryPoint(kernelName, newParams.layerID, params);
    auto jit = CreateJit(kernelName, cldnnJit, entryPoint);

    kernel.params.workGroups.global = {out.X().v, out.Batch().v, 1};
    kernel.params.workGroups.local = {out.X().v, 1, 1};
    kernel.code.kernelString = GetKernelString(kernelName, jit, entryPoint, params.engineInfo);
    kernel.params.arguments.push_back({ArgumentDescriptor::Types::INPUT, 0});
    kernel.params.arguments.push_back({ArgumentDescriptor::Types::INPUT, 1});
    kernel.params.arguments.push_back({ArgumentDescriptor::Types::INPUT, 2});
    kernel.params.arguments.push_back({ArgumentDescriptor::Types::INPUT, 3});
    kernel.params.arguments.push_back({ArgumentDescriptor::Types::INPUT, 4});
    kernel.params.arguments.push_back({ArgumentDescriptor::Types::INPUT, 5});
    kernel.params.arguments.push_back({ArgumentDescriptor::Types::INPUT, 6});
    kernel.params.arguments.push_back({ArgumentDescriptor::Types::OUTPUT, 0});
    kernel.params.arguments.push_back({ArgumentDescriptor::Types::OUTPUT, 1});
    kernel.params.arguments.push_back({ArgumentDescriptor::Types::OUTPUT, 2});

    return {kd};
}
}  // namespace kernel_selector
