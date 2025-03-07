// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "lstm_seq_cm.h"
#include "kernel_selector_utils.h"
#include "common_tools.h"

namespace kernel_selector {

ParamsKey LSTMSeqKernel_CM::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::INT32);
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableDifferentTypes();
    k.EnableInputLayout(DataLayout::bfyx);
    k.EnableAllOutputLayout();
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    return k;
}

KernelsData LSTMSeqKernel_CM::GetKernelsData(const Params& params) const {
    if (!Validate(params)) {
        return {};
    }
    auto shape = GetShape(params);

    KernelData kd = KernelData::Default<lstm_params>(params, 2);
    auto options = std::string(" -Qxcm_jit_option=-DPASTokenReduction "
              " -mllvm --vc-disable-indvars-opt=true "
              " /Qxcm_jit_option=-enableBCR /Qxcm_doubleGRF "
              " -DXETLA_CODE_BASE=__CM__ ");
    auto shape_key = std::string {"is"} + std::to_string(shape.input_size);
    auto gemm_entry_point = std::string {"xetla_lstm_gemm_"} + shape_key;
    auto loop_entry_point = std::string {"xetla_lstm_loop_"} + shape_key;

    auto constants = JitConstants{};
    constants.AddConstant({MakeJitConstant("INPUT_SIZE", std::to_string(shape.input_size))});

    constants.AddConstant({MakeJitConstant("KERNEL_NAME", gemm_entry_point)});
    auto gemm_jit = CreateJit(constants);

    constants.RemoveConstant("KERNEL_NAME");
    constants.AddConstant({MakeJitConstant("KERNEL_NAME", loop_entry_point)});
    auto loop_jit = CreateJit(constants);


    // Request temporary buffers
    kd.internalBufferDataType = Datatype::F32;
    auto temp_buffer_size = shape.num_dir * shape.seq_len * shape.batch_size * shape.hidden_size * shape.num_gates * sizeof(float);
    kd.internalBuffers.push_back(temp_buffer_size);

    auto& gemm_part = kd.kernels[0];
    gemm_part.params.arguments.push_back({ArgumentDescriptor::Types::INPUT, 0});
    gemm_part.params.arguments.push_back({ArgumentDescriptor::Types::INPUT, 3});
    gemm_part.params.arguments.push_back({ArgumentDescriptor::Types::INPUT, 5});
    gemm_part.params.arguments.push_back({ArgumentDescriptor::Types::INPUT, 6});
    gemm_part.params.arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 0});

    // Calc gws, lws
    {
        size_t matrix_m_ih = shape.seq_len;
        size_t matrix_n_ih = shape.hidden_size * shape.num_gates;

        size_t wg_m_ih = 40;
        size_t wg_n_ih = 256;

        size_t sg_m_ih = 24;
        size_t sg_n_ih = 32;

        size_t local_kslicing_ih = 1;
        size_t subgroup_range_m = (wg_m_ih + sg_m_ih - 1) / sg_m_ih;
        size_t subgroup_range_n = (wg_n_ih + sg_n_ih - 1) / sg_n_ih;

        size_t group_range_m = (matrix_m_ih + wg_m_ih - 1) / wg_m_ih;
        size_t group_range_n = (matrix_n_ih + wg_n_ih - 1) / wg_n_ih;

        gemm_part.params.workGroups.local = {subgroup_range_n, subgroup_range_m, local_kslicing_ih};
        gemm_part.params.workGroups.global = {group_range_n*subgroup_range_n, group_range_m*subgroup_range_m, shape.num_dir*local_kslicing_ih};
    }
    gemm_part.code.kernelString = GetKernelString("xetla_lstm_gemm", gemm_jit, gemm_entry_point);
    gemm_part.code.kernelString->options += options;
    gemm_part.code.kernelString->batch_compilation = true;

    auto& loop_part = kd.kernels[1];
    loop_part.params.arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 0});
    loop_part.params.arguments.push_back({ArgumentDescriptor::Types::INPUT, 1});
    loop_part.params.arguments.push_back({ArgumentDescriptor::Types::INPUT, 2});
    loop_part.params.arguments.push_back({ArgumentDescriptor::Types::INPUT, 4});
    loop_part.params.arguments.push_back({ArgumentDescriptor::Types::INPUT, 6});
    loop_part.params.arguments.push_back({ArgumentDescriptor::Types::OUTPUT, 0});
    loop_part.params.arguments.push_back({ArgumentDescriptor::Types::OUTPUT, 1});
    loop_part.params.arguments.push_back({ArgumentDescriptor::Types::OUTPUT, 2});

    // Calc gws, lws
    {
        size_t wg_m_hh = 1;
        size_t wg_n_hh = shape.hidden_size * shape.num_gates;

        size_t sg_m_hh = 1;
        size_t sg_n_hh = 16;

        size_t matrix_m_hh = 1;
        size_t matrix_n_hh = shape.hidden_size * shape.num_gates;

        size_t group_range_m = (matrix_m_hh + wg_m_hh - 1) / wg_m_hh;
        size_t group_range_n = (matrix_n_hh + wg_n_hh - 1) / wg_n_hh;
        size_t subgroup_range_m = (wg_m_hh + sg_m_hh - 1) / sg_m_hh;
        size_t subgroup_range_n = (wg_n_hh + sg_n_hh - 1) / sg_n_hh;

        loop_part.params.workGroups.local = {subgroup_range_n, subgroup_range_m, 1};
        loop_part.params.workGroups.global = {group_range_n*subgroup_range_n, group_range_m*subgroup_range_m, shape.num_dir};
    }
    loop_part.code.kernelString = GetKernelString("xetla_lstm_loop", loop_jit, loop_entry_point);
    loop_part.code.kernelString->options += options;
    loop_part.code.kernelString->batch_compilation = true;

    return {kd};
}

KernelsPriority LSTMSeqKernel_CM::GetKernelsPriority(const Params& /*params*/) const {
    return FORCE_PRIORITY_1;
}

bool LSTMSeqKernel_CM::Validate(const Params& p) const {
    const lstm_params& orgParams = static_cast<const lstm_params&>(p);
    const auto expected_inputs = 7;

    const auto &seq_lengths = orgParams.inputs[expected_inputs-1];
    if (seq_lengths.GetLayout() != DataLayout::bfyx || seq_lengths.GetDType() != kernel_selector::Datatype::INT32) {
        return false;
    }
    for (int i = 0; i < expected_inputs - 1; i++) {
        const auto &dtensor = orgParams.inputs[i];
        if (dtensor.GetLayout() != DataLayout::bfyx || dtensor.GetDType() != kernel_selector::Datatype::F16) {
            return false;
        }
    }

    if (orgParams.activations.size() != 3 ||
        orgParams.activations[0].function != ActivationFunction::LOGISTIC ||
        orgParams.activations[1].function != ActivationFunction::HYPERBOLIC_TAN ||
        orgParams.activations[2].function != ActivationFunction::HYPERBOLIC_TAN) {
            return false;
    }

    auto shape = GetShape(p);
    if (shape.hidden_size != 128 || shape.batch_size != 1 || shape.num_dir != 2 ||
        (shape.input_size != 64 && shape.input_size != 256)) {
        return false;
    }

    return true;
}
LSTMSeqKernel_CM::lstm_shape LSTMSeqKernel_CM::GetShape(const Params& params) const {
    const lstm_params& orgParams = static_cast<const lstm_params&>(params);
    const auto &x = orgParams.inputs[0];
    const auto &ini_hidden = orgParams.inputs[1];
    const auto &out = orgParams.outputs[0];

    lstm_shape shape;
    shape.num_gates = 4; // always 4
    shape.batch_size = x.Batch().v;
    shape.hidden_size = ini_hidden.Y().v;
    shape.input_size = x.Y().v;
    shape.seq_len = out.Y().v;
    shape.num_dir = ini_hidden.Feature().v;

    return shape;
}

}  // namespace kernel_selector
