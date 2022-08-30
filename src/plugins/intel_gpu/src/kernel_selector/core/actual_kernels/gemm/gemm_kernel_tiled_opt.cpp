// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gemm_kernel_tiled_opt.h"
#include <iostream>

namespace kernel_selector {
ParamsKey GemmKernelTiledOpt::GetSupportedKey() const {
    ParamsKey k;

    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::INT8);
    k.EnableOutputDataType(Datatype::UINT8);
    k.EnableInputLayout(DataLayout::bfyx);
    k.EnableOutputLayout(DataLayout::bfyx);
    k.EnableInputLayout(DataLayout::bfzyx);
    k.EnableOutputLayout(DataLayout::bfzyx);
    k.EnableInputLayout(DataLayout::bfwzyx);
    k.EnableOutputLayout(DataLayout::bfwzyx);

    k.EnableBatching();
    k.EnableDifferentTypes();

    return k;
}

GemmKernelBase::DispatchData GemmKernelTiledOpt::SetDefault(const gemm_params& params) const {
    const auto& output = params.outputs[0];

    DispatchData dispatchData;
    GemmTuningData td = SetTuningParams(params);

    auto total_batches = output.LogicalSize() / (output.X().v * output.Y().v);
    std::vector<size_t> global = { output.X().v, output.Y().v, total_batches };

    dispatchData.gws[0] = Align(global[0], td.tile_n_size) / (td.tile_n_size / td.simd_size);
    dispatchData.gws[1] = Align(global[1], td.tile_m_size) / td.tile_m_size;
    dispatchData.gws[2] = global[2];

    dispatchData.lws[0] = td.simd_size;
    dispatchData.lws[1] = 1;
    dispatchData.lws[2] = 1;

    return dispatchData;
}

GemmKernelTiledOpt::GemmTuningData GemmKernelTiledOpt::SetTuningParams(const gemm_params& params) const {
    const auto& output = params.outputs[0];

    GemmKernelTiledOpt::GemmTuningData tuning_data;

    auto m_size = output.Y().v;
    auto n_size = output.X().v;
    auto k_size = params.transpose_input0 ? params.inputs[0].Y().v : params.inputs[0].X().v;

    auto total_batches = output.LogicalSize() / (output.X().v * output.Y().v);
    tuning_data.simd_size = 8;

    tuning_data.tile_n_size = tuning_data.simd_size;
    while (tuning_data.tile_n_size < 64 && n_size / (tuning_data.tile_n_size * 2) >= 1) {
        tuning_data.tile_n_size *= 2;
    }

    // tuning_data.tile_k_size must be the same as simd_size when k % tile_k != 0
    tuning_data.tile_k_size = tuning_data.simd_size;
    tuning_data.tile_m_size = tuning_data.simd_size;

    bool leftovers = m_size % tuning_data.tile_m_size || k_size % tuning_data.tile_k_size || n_size % tuning_data.tile_n_size;

    if (leftovers || total_batches > 1 || params.transpose_input0 || params.transpose_input1) {
        tuning_data.simd_size = 16;
        tuning_data.tile_n_size = tuning_data.simd_size;
        tuning_data.tile_k_size = tuning_data.simd_size;
        tuning_data.tile_m_size = tuning_data.simd_size;
    }

    return tuning_data;
}

JitConstants GemmKernelTiledOpt::GetJitConstants(const gemm_params& params) const {
    JitConstants jit = Parent::GetJitConstants(params);

    const auto& output = params.outputs[0];
    GemmTuningData tuning_data = SetTuningParams(params);

    auto m_size = output.Y().v;
    auto n_size = output.X().v;
    auto k_size = params.transpose_input0 ? params.inputs[0].Y().v : params.inputs[0].X().v;
    auto leftover_m = m_size % tuning_data.tile_m_size;
    auto leftover_n = n_size % tuning_data.tile_n_size;
    auto leftover_k = k_size % tuning_data.tile_k_size;
    auto b_vec_size = tuning_data.tile_n_size / tuning_data.simd_size;

    jit.Merge(MakeTypeJitConstants(params.inputs[0].GetDType(), "ACCUMULATOR"));

    jit.AddConstants({
        MakeJitConstant("M", m_size),
        MakeJitConstant("K", k_size),
        MakeJitConstant("N", n_size),
        MakeJitConstant("SIMD_WIDTH", tuning_data.simd_size),
        MakeJitConstant("TILE_M", tuning_data.tile_m_size),
        MakeJitConstant("TILE_K", tuning_data.tile_k_size),
        MakeJitConstant("TILE_N", tuning_data.tile_n_size),
        MakeJitConstant("K_FULL_ITERATIONS", k_size / tuning_data.tile_k_size),
        MakeJitConstant("TILE_M_NOT_DIVISIBLE", leftover_m != 0),
        MakeJitConstant("TILE_K_NOT_DIVISIBLE", leftover_k != 0),
        MakeJitConstant("TILE_N_NOT_DIVISIBLE", leftover_n != 0),
        MakeJitConstant("TILE_M_LEFTOVER", leftover_m),
        MakeJitConstant("TILE_K_LEFTOVER", leftover_k),
        MakeJitConstant("TILE_N_LEFTOVER", leftover_n),
    });

    if (tuning_data.tile_k_size > tuning_data.simd_size) {
        jit.AddConstants({
            MakeJitConstant("A_VEC_SIZE", tuning_data.tile_k_size / tuning_data.simd_size),
            MakeJitConstant("A_FLOATN", std::string("CAT(INPUT0_TYPE, ") + toCodeString(tuning_data.tile_k_size / tuning_data.simd_size) + ")"),
        });
    } else {
        jit.AddConstants({
            MakeJitConstant("A_VEC_SIZE", 1),
            MakeJitConstant("A_FLOATN", std::string("INPUT0_TYPE")),
        });
    }

    if (tuning_data.tile_n_size > tuning_data.simd_size) {
        jit.AddConstants({
            MakeJitConstant("B_VEC_SIZE", b_vec_size),
            MakeJitConstant("B_FLOATN", std::string("CAT(INPUT1_TYPE, ") + toCodeString(b_vec_size) + ")"),
            MakeJitConstant("OUTPUT_TYPE_VEC", std::string("CAT(OUTPUT_TYPE, ") + toCodeString(b_vec_size) + ")"),
            MakeJitConstant("ACCUMULATOR_TYPE_VEC", std::string("CAT(ACCUMULATOR_TYPE, ") + toCodeString(b_vec_size) + ")"),
        });
    } else {
        b_vec_size = 1;
        jit.AddConstants({
            MakeJitConstant("B_VEC_SIZE", b_vec_size),
            MakeJitConstant("B_FLOATN", std::string("INPUT1_TYPE")),
            MakeJitConstant("OUTPUT_TYPE_VEC", std::string("OUTPUT_TYPE")),
            MakeJitConstant("ACCUMULATOR_TYPE_VEC", std::string("ACCUMULATOR_TYPE")),
        });
    }

    if (!params.fused_ops.empty()) {
        auto input_dt = GetActivationType(params);
        FusedOpsConfiguration conf_vec = { "_VEC", {"b", "f", "(y + write_id)", "x"},
                                           "dequantized",
                                           input_dt,
                                           b_vec_size,
                                           LoadType::LT_ALIGNED_READ,
                                           BoundaryCheck::ENABLED,
                                           IndexType::TENSOR_COORD,
                                           Tensor::DataChannelName::Y };
        FusedOpsConfiguration conf_scalar = { "_SCALAR", {"b", "f", "(y + write_id)", "x"},
                                               "dequantized",
                                               input_dt,
                                               1,
                                               LoadType::LT_UNALIGNED,
                                               BoundaryCheck::ENABLED,
                                               IndexType::TENSOR_COORD,
                                               Tensor::DataChannelName::Y };
        jit.Merge(MakeFusedOpsJitConstants(params, { conf_vec, conf_scalar }));
    }

    return jit;
}

KernelsData GemmKernelTiledOpt::GetKernelsData(const Params& params, const optional_params& options) const {
    return GetCommonKernelsData(params, options);
}

KernelsPriority GemmKernelTiledOpt::GetKernelsPriority(const Params& params, const optional_params& /*options*/) const {
    const auto& gmm_params = static_cast<const gemm_params&>(params);

    return gmm_params.transpose_input0 || gmm_params.transpose_input1 ? FORCE_PRIORITY_6 : FORCE_PRIORITY_3;
}

bool GemmKernelTiledOpt::Validate(const Params& params, const optional_params& options) const {
    if (!Parent::Validate(params, options))
        return false;

    const auto& gmm_params = static_cast<const gemm_params&>(params);
    bool gemm_leftovers = gmm_params.inputs[0].X().v % 16 || gmm_params.inputs[0].Y().v % 16 ||
                          gmm_params.inputs[1].X().v % 16 || gmm_params.inputs[1].Y().v % 16;
    if ((gmm_params.transpose_input0 || gmm_params.transpose_input1) && gemm_leftovers)
        return false;

    for (size_t i = 1; i < gmm_params.inputs.size(); i++)
        if (gmm_params.inputs[0].GetDType() != gmm_params.inputs[i].GetDType())
            return false;

    return true;
}
}  // namespace kernel_selector
