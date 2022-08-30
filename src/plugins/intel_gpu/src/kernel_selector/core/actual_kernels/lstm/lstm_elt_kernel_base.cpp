// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "lstm_elt_kernel_base.h"
#include "kernel_selector_utils.h"
#include "common_tools.h"
#include <string>

namespace kernel_selector {

JitConstants LSTMEltKernelBase::GetJitConstants(const lstm_elt_params& params) const {
    JitConstants jit = MakeBaseParamsJitConstants(params);

    if (params.has_cell) {
        const auto& cell = params.cell;
        jit.AddConstants({MakeJitConstant("CELL_TERM", true),
                          MakeJitConstant("CELL", cell),
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

KernelsData LSTMEltKernelBase::GetCommonKernelsData(const Params& params, const optional_params& options) const {
    if (!Validate(params, options)) {
        return {};
    }

    const lstm_elt_params& orgParams = static_cast<const lstm_elt_params&>(params);

    KernelData kd = KernelData::Default<lstm_elt_params>(params, orgParams.inputs.size());

    const auto& input = orgParams.inputs[0];

    auto newParams = orgParams;
    newParams.inputs.resize(1);
    newParams.inputs[0] = input;
    auto out = newParams.outputs[0];

    auto& kernel = kd.kernels[0];
    auto cldnnJit = GetJitConstants(newParams);
    auto entryPoint = GetEntryPoint(kernelName, newParams.layerID, params, options);
    auto jit = CreateJit(kernelName, cldnnJit, entryPoint);

    kernel.params.workGroups.global = {out.X().v, out.Batch().v, 1};
    kernel.code.kernelString = GetKernelString(kernelName, jit, entryPoint, params.engineInfo);
    kernel.params.arguments.push_back({ArgumentDescriptor::Types::INPUT, 0});
    kernel.params.arguments.push_back({ArgumentDescriptor::Types::OUTPUT, 0});
    if (orgParams.has_cell) {
        kernel.params.arguments.push_back({ArgumentDescriptor::Types::CELL, 0});
    }

    return {kd};
}
}  // namespace kernel_selector
