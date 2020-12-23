/*
// Copyright (c) 2016-2020 Intel Corporation
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

    static const std::vector<std::string> asuffixes = {"_F","_G","_H","_CLIP"};
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

    float efficiency = FORCE_PRIORITY_1;
    const auto& input = orgParams.inputs[0];

    auto newParams = orgParams;
    newParams.inputs.resize(1);
    newParams.inputs[0] = input;
    auto out = newParams.output;

    auto& kernel = kd.kernels[0];
    auto cldnnJit = GetJitConstants(newParams);
    auto entryPoint = GetEntryPoint(kernelName, newParams.layerID, options);
    auto jit = CreateJit(kernelName, cldnnJit, entryPoint);

    kernel.workGroups.global = {out.X().v, out.Batch().v, 1};
    kernel.kernelString = GetKernelString(kernelName, jit, entryPoint, params.engineInfo);
    kernel.arguments.push_back({ArgumentDescriptor::Types::INPUT, 0});
    kernel.arguments.push_back({ArgumentDescriptor::Types::OUTPUT, 0});
    if (orgParams.has_cell) {
        kernel.arguments.push_back({ArgumentDescriptor::Types::CELL, 0});
    }

    kd.estimatedTime = efficiency;

    return {kd};
}
}  // namespace kernel_selector
