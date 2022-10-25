// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "lstm_dynamic_timeloop_kernel_base.h"
#include "kernel_selector_utils.h"
#include "common_tools.h"
#include <string>
#include <vector>

namespace kernel_selector {
JitConstants LSTM_DynamicTimeloopKernelBase::GetJitConstants(const lstm_dynamic_timeloop_params& params) const {
    JitConstants jit = MakeBaseParamsJitConstants(params);

    const auto& out = params.outputs[0];
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
    DispatchData dispatchData;
    const auto& out = params.outputs[0];

    auto out_x_size = out.X().v;
    auto gws0 = out_x_size > 256 ? 256 : out_x_size;
    dispatchData.gws = { gws0, out.Batch().v, static_cast<size_t>(params.direction) };
    dispatchData.lws = GetOptimalLocalWorkGroupSizes(dispatchData.gws, params.engineInfo);

    return dispatchData;
}

void kernel_selector::LSTM_DynamicTimeloopKernelBase::SetKernelArguments(const lstm_dynamic_timeloop_params& params, clKernelData& kernel) const {
    uint32_t input_idx = 0;
    kernel.params.arguments.push_back({ ArgumentDescriptor::Types::INPUT, input_idx++ });
    kernel.params.arguments.push_back({ ArgumentDescriptor::Types::INPUT, input_idx++ });
    kernel.params.arguments.push_back({ ArgumentDescriptor::Types::OUTPUT, 0 });
    kernel.params.arguments.push_back({ ArgumentDescriptor::Types::RECURRENT, 0 });
    if (params.has_hidden) {
        kernel.params.arguments.push_back({ ArgumentDescriptor::Types::HIDDEN, 0 });
    }
    if (params.has_cell) {
        kernel.params.arguments.push_back({ ArgumentDescriptor::Types::CELL, 0 });
    }
    if (params.has_last_hidden_output) {
        kernel.params.arguments.push_back({ ArgumentDescriptor::Types::INPUT, input_idx++ });
    }
    if (params.has_last_cell_output) {
        kernel.params.arguments.push_back({ ArgumentDescriptor::Types::INPUT, input_idx++ });
    }
}


KernelsData LSTM_DynamicTimeloopKernelBase::GetCommonKernelsData(const Params& params,
                                                                 const optional_params& options) const {
    if (!Validate(params, options)) {
        return {};
    }

    const lstm_dynamic_timeloop_params& org_params = static_cast<const lstm_dynamic_timeloop_params&>(params);

    auto dispatchData = SetDefault(org_params);
    KernelData k_data = KernelData::Default<lstm_dynamic_timeloop_params>(params, 1);

    auto cldnn_jit = GetJitConstants(org_params);
    auto entry_point = GetEntryPoint(kernelName, org_params.layerID, params, options);
    auto jit = CreateJit(kernelName, cldnn_jit, entry_point);

    auto& kernel = k_data.kernels[0];
    kernel.params.workGroups.global = dispatchData.gws;
    kernel.params.workGroups.local  = dispatchData.lws;
    kernel.code.kernelString = GetKernelString(kernelName, jit, entry_point, params.engineInfo);
    SetKernelArguments(org_params, kernel);
    return {k_data};
}
}  // namespace kernel_selector
