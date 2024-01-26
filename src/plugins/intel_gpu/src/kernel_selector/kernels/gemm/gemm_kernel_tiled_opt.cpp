// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gemm_kernel_tiled_opt.h"
#include "kernel_selector_utils.h"
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

    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    k.EnableDifferentTypes();
    k.EnableDynamicShapesSupport();
    k.EnableIndirectGemm();

    return k;
}

DeviceFeaturesKey GemmKernelTiledOpt::get_required_device_features_key(const Params& params, const optional_params& options) const {
    auto k = get_common_subgroups_device_features_key(params, options);
    k.requires_subgroup_shuffle();

    return k;
}

GemmKernelBase::DispatchData GemmKernelTiledOpt::SetDefault(const gemm_params& params) const {
    const auto& output = params.outputs[0];

    DispatchData dispatchData;
    if (!params.has_dynamic_tensors()) {
        GemmTuningData td = SetTuningParams(params);

        auto total_batches = output.LogicalSize() /
                            (GetOuputSize(params.output_order, output, 'X') * GetOuputSize(params.output_order, output, 'Y'));
        std::vector<size_t> global = { GetOuputSize(params.output_order, output, 'X'), GetOuputSize(params.output_order, output, 'Y'),
                                       total_batches };

        dispatchData.gws[0] = Align(global[0], td.tile_n_size) / (td.tile_n_size / td.simd_size);
        dispatchData.gws[1] = Align(global[1], td.tile_m_size) / td.tile_m_size;
        dispatchData.gws[2] = global[2];

        dispatchData.lws[0] = td.simd_size;
        dispatchData.lws[1] = 1;
        dispatchData.lws[2] = 1;
    }
    return dispatchData;
}

GemmKernelTiledOpt::GemmTuningData GemmKernelTiledOpt::SetTuningParams(const gemm_params& params) const {
    const auto& output = params.outputs[0];

    GemmKernelTiledOpt::GemmTuningData tuning_data;

    if (!params.is_shape_agnostic) {
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

        if (leftovers || total_batches > 1 || params.transpose_input0 || params.transpose_input1 || !IsSIMDSizeSupported(params.engineInfo, 8)) {
            tuning_data.simd_size = 16;
            tuning_data.tile_n_size = tuning_data.simd_size;
            tuning_data.tile_k_size = tuning_data.simd_size;
            tuning_data.tile_m_size = tuning_data.simd_size;
        }
    } else {
        // In shape agnostic kernel case, the vector size of FusedOpsConfiguration cannot be specified at build time,
        // so the tile sizes must be the same as simd_size
        tuning_data.simd_size = 16;
        tuning_data.tile_n_size = tuning_data.simd_size;
        tuning_data.tile_k_size = tuning_data.simd_size;
        tuning_data.tile_m_size = tuning_data.simd_size;
    }

    return tuning_data;
}

JitConstants GemmKernelTiledOpt::GetJitConstants(const gemm_params& params) const {
    JitConstants jit = Parent::GetJitConstants(params);

    GemmTuningData tuning_data = SetTuningParams(params);
    auto b_vec_size = tuning_data.tile_n_size / tuning_data.simd_size;

    jit.Merge(MakeTypeJitConstants(params.inputs[0].GetDType(), "ACCUMULATOR"));
    if (params.has_dynamic_tensors()) {
        DimensionAccessHelper dims0(params.inputs[0]);
        DimensionAccessHelper dims1(params.inputs[1]);
        DimensionAccessHelper dims0_padded(params.inputs[0], true);
        DimensionAccessHelper dims1_padded(params.inputs[1], true);
        // Note: Actually currently this kernel is not being selected if it is shape agnostic impl && transposed inputs
        // Because we cannot get the original rank
        auto input0_dims = ConvTo8dims(params.input0_order);
        auto input1_dims = ConvTo8dims(params.input1_order);
        auto m_size = dims0.dims_sizes[input0_dims[6]];
        auto n_size = dims1.dims_sizes[input1_dims[7]];
        auto n_padded_size = "(" + dims1_padded.dims_sizes[input1_dims[7]] + ")";
        auto k_size = dims0.dims_sizes[input0_dims[7]];
        auto k_padded_size_in0 = "(" + dims0_padded.dims_sizes[input0_dims[7]] + ")";
        const std::string leftover_m = "(" + m_size + "%" + std::to_string(tuning_data.tile_m_size) + ")";
        const std::string leftover_n = "(" + n_size + "%" + std::to_string(tuning_data.tile_n_size) + ")";
        const std::string leftover_k = "(" + k_size + "%" + std::to_string(tuning_data.tile_k_size) + ")";
        const std::string not_divisible_m = "(" + leftover_m + "!=0)";
        const std::string not_divisible_n = "(" + leftover_n + "!=0)";
        const std::string not_divisible_k = "(" + leftover_k + "!=0)";
        const std::string full_iteration_k = "(" + k_size + "/" + std::to_string(tuning_data.tile_k_size) + ")";

        jit.AddConstants({
            MakeJitConstant("M", m_size),
            MakeJitConstant("K", k_size),
            MakeJitConstant("N", n_size),
            MakeJitConstant("K_PADDED_IN0", k_padded_size_in0),
            MakeJitConstant("N_PADDED", n_padded_size),
            MakeJitConstant("SIMD_WIDTH", tuning_data.simd_size),
            MakeJitConstant("TILE_M", tuning_data.tile_m_size),
            MakeJitConstant("TILE_K", tuning_data.tile_k_size),
            MakeJitConstant("TILE_N", tuning_data.tile_n_size),
            MakeJitConstant("K_FULL_ITERATIONS", full_iteration_k),
            MakeJitConstant("TILE_M_NOT_DIVISIBLE", not_divisible_m),
            MakeJitConstant("TILE_K_NOT_DIVISIBLE", not_divisible_k),
            MakeJitConstant("TILE_N_NOT_DIVISIBLE", not_divisible_n),
            MakeJitConstant("TILE_M_LEFTOVER", leftover_m),
            MakeJitConstant("TILE_K_LEFTOVER", leftover_k),
            MakeJitConstant("TILE_N_LEFTOVER", leftover_n),
            MakeJitConstant("TR_B", GetTransposedDims(params.output_order, true).at(0)),
            MakeJitConstant("TR_F", GetTransposedDims(params.output_order, true).at(1)),
            MakeJitConstant("TR_W", GetTransposedDims(params.output_order, true).at(4)),
            MakeJitConstant("TR_Z", GetTransposedDims(params.output_order, true).at(5)),
            MakeJitConstant("TR_Y", GetTransposedDims(params.output_order, true).at(6)),
            MakeJitConstant("TR_X", GetTransposedDims(params.output_order, true).at(7)),
        });

        bool has_dynamic_k_padding = params.transpose_input0 ? params.inputs[0].Y().pad.is_dynamic
                                                             : params.inputs[0].X().pad.is_dynamic;
        bool has_dynamic_n_padding = params.transpose_input1 ? params.inputs[1].Y().pad.is_dynamic
                                                             : params.inputs[1].X().pad.is_dynamic;
        if (has_dynamic_k_padding)
            jit.AddConstant(MakeJitConstant("HAS_DYNAMIC_K_PADDING", 1));
        if (has_dynamic_n_padding)
            jit.AddConstant(MakeJitConstant("HAS_DYNAMIC_N_PADDING", 1));
    } else {
        auto get_transposed_dim_size = [](const kernel_selector::DataTensor &data_tensor,
                                          const std::vector<int64_t>& dims_order, const std::string dim) {
            int64_t target_dim_idx;
            const size_t rank = data_tensor.GetDims().size();
            if (dims_order.size() > 1 && dim.compare("Y") == 0) {
                target_dim_idx = dims_order.at(dims_order.size() - 2);
            } else if (dims_order.size() > 0 && dim.compare("X") == 0) {
                target_dim_idx = dims_order.back();
            } else if (dims_order.size() == 0 && dim.compare("Y") == 0) {
                target_dim_idx = rank - 2;
            } else if (dims_order.size() == 0 && dim.compare("X") == 0) {
                target_dim_idx = rank - 1;
            } else {
                OPENVINO_THROW("Unsupported dimension: ", dim);
            }

            size_t loc = static_cast<size_t>(target_dim_idx);
            if (dims_order.size() > 0) {
                loc += (dims_order.size() < rank) ? (rank - dims_order.size()) : 0;
            }

            if (loc == 0) {
                return data_tensor.Batch().v;
            } else if (loc == 1) {
                return data_tensor.Feature().v;
            } else if (loc == (rank - 1) && rank >= 3) {
                return data_tensor.X().v;
            } else if (loc == (rank - 2) && rank >= 4) {
                return data_tensor.Y().v;
            } else if (loc == (rank - 3) && rank >= 5) {
                return data_tensor.Z().v;
            } else if (loc == (rank - 4) && rank >= 6) {
                return data_tensor.W().v;
            }
            OPENVINO_THROW("Target dimension is not found.");
        };

        auto m_size = get_transposed_dim_size(params.inputs[0], params.input0_order, "Y");
        auto n_size = get_transposed_dim_size(params.inputs[1], params.input1_order, "X");
        auto k_size = get_transposed_dim_size(params.inputs[0], params.input0_order, "X");
        auto leftover_m = m_size % tuning_data.tile_m_size;
        auto leftover_n = n_size % tuning_data.tile_n_size;
        auto leftover_k = k_size % tuning_data.tile_k_size;

        jit.AddConstants({
            MakeJitConstant("M", m_size),
            MakeJitConstant("K", k_size),
            MakeJitConstant("N", n_size),
            MakeJitConstant("K_PADDED_IN0", k_size),
            MakeJitConstant("N_PADDED", n_size),
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
            MakeJitConstant("TR_B", GetTransposedDims(params.output_order, true).at(0)),
            MakeJitConstant("TR_F", GetTransposedDims(params.output_order, true).at(1)),
            MakeJitConstant("TR_W", GetTransposedDims(params.output_order, true).at(4)),
            MakeJitConstant("TR_Z", GetTransposedDims(params.output_order, true).at(5)),
            MakeJitConstant("TR_Y", GetTransposedDims(params.output_order, true).at(6)),
            MakeJitConstant("TR_X", GetTransposedDims(params.output_order, true).at(7)),
        });

        if (params.inputs[0].LogicalSize() != params.inputs[0].PhysicalSize())
            jit.AddConstant(MakeJitConstant("INPUT0_HAS_PADDING", 1));
        if (params.inputs[1].LogicalSize() != params.inputs[1].PhysicalSize())
            jit.AddConstant(MakeJitConstant("INPUT1_HAS_PADDING", 1));
    }

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

    if (gmm_params.outputs[0].PitchesDifferFromLogicalDims())
        return false;

    for (size_t input_idx = 0; input_idx < gmm_params.inputs.size(); ++input_idx) {
        auto& input = gmm_params.inputs[input_idx];
        if (!Tensor::SimpleLayout(input.GetLayout())) {
            return false;
        }
        // Supports outer padding as first element offset and dynamic padding for Batch, Feature, X, Y dimensions for first and second inputs
        // in case of shape agnostic kernel
        bool proper_pad_f = input.Feature().pad.is_dynamic ? false : input.Feature().pad.Total() == 0;
        bool proper_pad_x = input.X().pad.is_dynamic ? false : input.X().pad.Total() == 0;
        bool proper_pad_y = input.Y().pad.is_dynamic ? false : input.Y().pad.Total() == 0;
        if (gmm_params.is_shape_agnostic && input_idx < 2) {
            proper_pad_f |= input.Feature().pad.is_dynamic;
            proper_pad_x |= input.X().pad.is_dynamic;
            proper_pad_y |= input.Y().pad.is_dynamic;
        }

        if (!proper_pad_x || !proper_pad_y || input.Z().pad.Total() != 0 || !proper_pad_f)
            return false;
    }

    if (gmm_params.has_dynamic_inputs() && !gmm_params.is_shape_agnostic)
        return false;

    for (size_t i = 1; i < gmm_params.inputs.size(); i++)
        if (gmm_params.inputs[0].GetDType() != gmm_params.inputs[i].GetDType())
            return false;

    return true;
}
}  // namespace kernel_selector
