/*
// Copyright (c) 2018-2020 Intel Corporation
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

#include "gemm_kernel_tiled_opt.h"
#include <iostream>

namespace kernel_selector {
ParamsKey GemmKernelTiledOpt::GetSupportedKey() const {
    ParamsKey k;

    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableInputLayout(DataLayout::bfyx);
    k.EnableOutputLayout(DataLayout::bfyx);
    k.EnableInputLayout(DataLayout::bfzyx);
    k.EnableOutputLayout(DataLayout::bfzyx);
    k.EnableInputLayout(DataLayout::bfwzyx);
    k.EnableOutputLayout(DataLayout::bfwzyx);

    k.EnableBatching();

    return k;
}

GemmKernelBase::DispatchData GemmKernelTiledOpt::SetDefault(const gemm_params& params) const {
    const auto& output = params.output;

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
    const auto& output = params.output;

    auto m_size = output.Y().v;
    auto n_size = output.X().v;
    auto k_size = params.transpose_input0 ? params.inputs[0].Y().v : params.inputs[0].X().v;

    auto total_batches = output.LogicalSize() / (output.X().v * output.Y().v);
    tuning_data.simd_size = 8;

    if (n_size >= 8) {
        tuning_data.tile_n_size = tuning_data.simd_size;

        while (tuning_data.tile_n_size < 64 && n_size / (tuning_data.tile_n_size * 2) >= 1) {
            tuning_data.tile_n_size *= 2;
        }
    }

    // tuning_data.tile_k_size must be the same as simd_size when k % tile_k != 0
    tuning_data.tile_k_size = tuning_data.simd_size;
    tuning_data.tile_m_size = 8;

    bool leftovers = m_size % tuning_data.tile_m_size || k_size % tuning_data.tile_k_size || n_size % tuning_data.tile_n_size;

    if (leftovers || total_batches > 1 || params.transpose_input0 || params.transpose_input1) {
        tuning_data.simd_size = 16;
        tuning_data.tile_n_size = tuning_data.simd_size;
        tuning_data.tile_k_size = tuning_data.simd_size;
        tuning_data.tile_m_size = 16;
    }

    return tuning_data;
}

JitConstants GemmKernelTiledOpt::GetJitConstants(const gemm_params& params) const {
    JitConstants jit = Parent::GetJitConstants(params);

    const auto& output = params.output;

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
            MakeJitConstant("A_FLOATN", std::string("UNIT_TYPE") + std::to_string(tuning_data.tile_k_size / tuning_data.simd_size)),
        });
    }
    else {
        jit.AddConstants({
            MakeJitConstant("A_VEC_SIZE", 1),
            MakeJitConstant("A_FLOATN", std::string("UNIT_TYPE")),
        });
    }

    if (tuning_data.tile_n_size > tuning_data.simd_size) {
        jit.AddConstants({
            MakeJitConstant("B_VEC_SIZE", b_vec_size),
            MakeJitConstant("B_FLOATN", std::string("UNIT_TYPE") + std::to_string(b_vec_size)),
        });
    }
    else {
        b_vec_size = 1;
        jit.AddConstants({
            MakeJitConstant("B_VEC_SIZE", 1),
            MakeJitConstant("B_FLOATN", std::string("UNIT_TYPE")),
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
                                               LoadType::LT_ALIGNED_READ,
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

    return true;
}
}  // namespace kernel_selector
