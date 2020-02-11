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

#include "lstm_dynamic_timeloop_kernel_base.h"
#include "kernel_selector_utils.h"
#include "common_tools.h"
#include <string>
#include <vector>

namespace kernel_selector {
JitConstants LSTM_DynamicTimeloopKernelBase::GetJitConstants(const lstm_dynamic_timeloop_params& params) const {
    JitConstants jit = MakeBaseParamsJitConstants(params);

    const auto& out = params.output;
    size_t hidden_size = out.X().v;

    // [1] Certainties
    jit.AddConstants({
        // IE default: fizo
        MakeJitConstant("GEMM_OFFSET_I", 1 * hidden_size),
        MakeJitConstant("GEMM_OFFSET_O", 3 * hidden_size),
        MakeJitConstant("GEMM_OFFSET_F", 0 * hidden_size),
        MakeJitConstant("GEMM_OFFSET_Z", 2 * hidden_size),
    });

    jit.AddConstants({MakeJitConstant("RECURRENT", params.recurrent),
                      MakeJitConstant("DYN_LENGTH", params.inputs.at(1)),
                      MakeJitConstant("HIDDEN_SIZE", hidden_size),
                      MakeJitConstant("MAX_SEQUENCE_LENGTH", params.inputs.at(0).Feature().v),
                      MakeJitConstant("ELEMENTS_TO_COUNT", hidden_size > 256 ? hidden_size / 256 : 1)});

    if (params.has_hidden) {
        const auto& hidden = params.hidden;
        jit.AddConstants({
            MakeJitConstant("INIT_HIDDEN_TERM", true),
            MakeJitConstant("INIT_HIDDEN", hidden),
        });
    }

    if (params.has_cell) {
        const auto& cell = params.cell;
        jit.AddConstants({
            MakeJitConstant("INIT_CELL_TERM", true),
            MakeJitConstant("INIT_CELL", cell),
        });
    }

    if (params.clip > 0) {
        std::string psclip = toCodeString(params.clip);
        std::string nsclip = toCodeString(-params.clip);
        jit.AddConstants(
            {MakeJitConstant("CLIP(x)",
                             "((x > " + psclip + ") ? " + psclip + ": (x < " + nsclip + ") ? " + nsclip + " : (x))")});
    } else {
        jit.AddConstants({MakeJitConstant("CLIP(x)", "(x)")});
    }
    if (params.input_forget) {
        jit.AddConstants({MakeJitConstant("INPUT_FORGET", true)});
    }

    if (params.has_last_hidden_output) {
        jit.AddConstants(
            {MakeJitConstant("LAST_HIDDEN", params.last_hidden_output), MakeJitConstant("LAST_HIDDEN_TERM", true)});
    }

    if (params.has_last_cell_output) {
        jit.AddConstants(
            {MakeJitConstant("LAST_CELL", params.last_cell_output), MakeJitConstant("LAST_CELL_TERM", true)});
    }

    return jit;
}

LSTM_DynamicTimeloopKernelBase::DispatchData LSTM_DynamicTimeloopKernelBase::SetDefault(
    const lstm_dynamic_timeloop_params& params) {
    DispatchData kd;
    const auto& out = params.output;
    kd.fp16UnitUsed = params.inputs[0].GetDType() == Datatype::F16;

    auto out_x_size = out.X().v;
    auto gws0 = out_x_size > 256 ? 256 : out_x_size;
    std::vector<size_t> global = {gws0, out.Batch().v, static_cast<size_t>(params.direction)};
    const auto& local = GetOptimalLocalWorkGroupSizes(global, params.engineInfo);

    kd.gws0 = global[0];
    kd.gws1 = global[1];
    kd.gws2 = global[2];

    kd.lws0 = local[0];
    kd.lws1 = local[1];
    kd.lws2 = local[2];

    return kd;
}

void kernel_selector::LSTM_DynamicTimeloopKernelBase::SetKernelArguments(const lstm_dynamic_timeloop_params& params, clKernelData& kernel) const {
    uint32_t input_idx = 0;
    kernel.arguments.push_back({ ArgumentDescriptor::Types::INPUT, input_idx++ });
    kernel.arguments.push_back({ ArgumentDescriptor::Types::INPUT, input_idx++ });
    kernel.arguments.push_back({ ArgumentDescriptor::Types::OUTPUT, 0 });
    kernel.arguments.push_back({ ArgumentDescriptor::Types::RECURRENT, 0 });
    if (params.has_hidden) {
        kernel.arguments.push_back({ ArgumentDescriptor::Types::HIDDEN, 0 });
    }
    if (params.has_cell) {
        kernel.arguments.push_back({ ArgumentDescriptor::Types::CELL, 0 });
    }
    if (params.has_last_hidden_output) {
        kernel.arguments.push_back({ ArgumentDescriptor::Types::INPUT, input_idx++ });
    }
    if (params.has_last_cell_output) {
        kernel.arguments.push_back({ ArgumentDescriptor::Types::INPUT, input_idx++ });
    }
}


KernelsData LSTM_DynamicTimeloopKernelBase::GetCommonKernelsData(const Params& params,
                                                                 const optional_params& options,
                                                                 float estimated_time) const {
    if (!Validate(params, options)) {
        return {};
    }

    const lstm_dynamic_timeloop_params& org_params = static_cast<const lstm_dynamic_timeloop_params&>(params);

    auto run_info = SetDefault(org_params);
    KernelData k_data = KernelData::Default<lstm_dynamic_timeloop_params>(params, 1);

    auto cldnn_jit = GetJitConstants(org_params);
    auto entry_point = GetEntryPoint(kernelName, org_params.layerID, options);
    auto jit = CreateJit(kernelName, cldnn_jit, entry_point);

    auto& kernel = k_data.kernels[0];
    kernel.workGroups.global = {run_info.gws0, run_info.gws1, run_info.gws2};
    kernel.workGroups.local  = {run_info.lws0, run_info.lws1, run_info.lws2};
    kernel.kernelString = GetKernelString(kernelName, jit, entry_point, params.engineInfo);
    SetKernelArguments(org_params, kernel);
    k_data.estimatedTime = estimated_time;
    return {k_data};
}
}  // namespace kernel_selector
