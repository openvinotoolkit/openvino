// Copyright (C) 2018-2024 Intel Corporation
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
    k.EnableInputDataType(Datatype::INT32);
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

DeviceFeaturesKey GemmKernelTiledOpt::get_required_device_features_key(const Params& params) const {
    auto k = get_common_subgroups_device_features_key(params);
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
        GPU_DEBUG_TRACE_DETAIL << "Draft for global work item size: [" << global[0] << ", " << global[1] << ", " << global[2] << "], " << std::endl;

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
        // Increasing tile_n_size has performance improvement when m_size and n_size are not shallow and n_size is aligned at 32.
        // TODO: Support TILE_K_LEFTOVER true case at static shape
        if (m_size >= 128 && n_size >= 128 && (n_size % 32 == 0) && tuning_data.simd_size == 16 &&
            (k_size % tuning_data.tile_k_size == 0) && params.fused_ops.empty())
            tuning_data.tile_n_size = 32;

        GPU_DEBUG_LOG << params.layerID << ": m_size: " << m_size << ", n_size: " << n_size << ", k_size: " << k_size << std::endl;
    } else {
        // In shape agnostic kernel case, the vector size of FusedOpsConfiguration cannot be specified at build time,
        // so the tile sizes must be the same as simd_size
        tuning_data.simd_size = 16;
        tuning_data.tile_k_size = tuning_data.simd_size;
        tuning_data.tile_m_size = tuning_data.simd_size;
        bool output_ndim_transposed = (params.output_order.size() > 0 && (params.output_order.back() != (static_cast<int>(params.output_order.size()) - 1)));
        if ((params.transpose_input0 == 0 /*X_LAST*/) && (params.transpose_input1 == 0 /*X_LAST*/ || params.transpose_input1 == 1 /*Y_LAST*/)
            && (!params.indirect_input0 && !params.inputs[0].has_dynamic_pad() && params.indirect_axis != 1)
            && (!output_ndim_transposed || params.fused_ops.empty())
            && !params.engineInfo.supports_immad) {
            // - Not supports transposed input0 / transposed input1 for OTHER mode yet
            // - If output X dim (= N) is transposed, cannot read eltwise as aligned data
            tuning_data.tile_n_size = 32;
        } else {
            tuning_data.tile_n_size = 16;
        }
    }

    GPU_DEBUG_LOG << params.layerID << ": tile_m_size: " << tuning_data.tile_m_size
                    << ", tile_n_size: " << tuning_data.tile_n_size
                    << ", tile_k_size: " << tuning_data.tile_k_size
                    << ", simd_size: " << tuning_data.simd_size << std::endl;

    return tuning_data;
}

JitConstants GemmKernelTiledOpt::GetJitConstants(const gemm_params& params) const {
    JitConstants jit = Parent::GetJitConstants(params);

    GemmTuningData tuning_data = SetTuningParams(params);
    auto b_vec_size = tuning_data.tile_n_size / tuning_data.simd_size;

    jit.Merge(MakeTypeJitConstants(params.inputs[0].GetDType(), "ACCUMULATOR"));
    if (params.has_dynamic_tensors()) {
        DimensionAccessHelperJit dims0(params.inputs[0]);
        DimensionAccessHelperJit dims1(params.inputs[1]);
        DimensionAccessHelperJit dims0_padded(params.inputs[0], true);
        DimensionAccessHelperJit dims1_padded(params.inputs[1], true);
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
        std::string n_aligned_4byte = "0";
        std::string k_aligned_4byte = "0";
        if (BytesPerElement(params.inputs[0].GetDType()) == 4 || BytesPerElement(params.inputs[0].GetDType()) == 8) {
            n_aligned_4byte = "1";
            k_aligned_4byte = "1";
        } else {
            auto bytes_per_element = std::to_string(BytesPerElement(params.inputs[0].GetDType()));
            if (n_size.find("shape_info") == std::string::npos) {
                n_aligned_4byte = "(" + n_size + "*" + bytes_per_element + " % 4 == 0)";
            }
            if (k_size.find("shape_info") == std::string::npos) {
                k_aligned_4byte = "(" + k_size + "*" + bytes_per_element + " % 4 == 0)";
            }
        }

        jit.AddConstants({
            MakeJitConstant("M", m_size),
            MakeJitConstant("K", k_size),
            MakeJitConstant("N", n_size),
            MakeJitConstant("K_PADDED_IN0", k_padded_size_in0),
            MakeJitConstant("N_PADDED", n_padded_size),
            MakeJitConstant("K_IS_ALIGNED_4BYTE", k_aligned_4byte),
            MakeJitConstant("N_IS_ALIGNED_4BYTE", n_aligned_4byte),
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

        bool transpose_output = (params.output_order.size() > 0 && (params.output_order.back() != (static_cast<int>(params.output_order.size()) - 1)));
        if (transpose_output)
            jit.AddConstant(MakeJitConstant("TRANSPOSE_OUTPUT", 2 /* set as TRANSPOSE_OTHER */));
        else
            jit.AddConstant(MakeJitConstant("TRANSPOSE_OUTPUT", 0 /* set as TRANSPOSE_X_LAST */));

        if (dims0_padded.has_dynamic_pad)
            jit.AddConstant(MakeJitConstant("INPUT0_HAS_DYNAMIC_PADDING", 1));
        if (dims1_padded.has_dynamic_pad)
            jit.AddConstant(MakeJitConstant("INPUT1_HAS_DYNAMIC_PADDING", 1));
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
        auto n_aligned_4byte = (n_size * BytesPerElement(params.inputs[0].GetDType())) % 4 == 0;
        auto k_aligned_4byte = (k_size * BytesPerElement(params.inputs[0].GetDType())) % 4 == 0;

        jit.AddConstants({
            MakeJitConstant("M", m_size),
            MakeJitConstant("K", k_size),
            MakeJitConstant("N", n_size),
            MakeJitConstant("K_PADDED_IN0", k_size),
            MakeJitConstant("N_PADDED", n_size),
            MakeJitConstant("K_IS_ALIGNED_4BYTE", k_aligned_4byte),
            MakeJitConstant("N_IS_ALIGNED_4BYTE", n_aligned_4byte),
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
        auto vec_load_type = LoadType::LT_ALIGNED_READ;
        for (auto op : params.fused_ops) {
            if (op.GetType() == FusedOpType::ELTWISE) {
                auto vec_axis_dim = op.tensors[0].X().v;
                // If vector axis of the eltwise input data is to be broadcasted we cannot use aligned load
                if ((vec_axis_dim == 1 && op.tensors[0].LogicalSize() != 1) && (params.inputs[1].X().v != vec_axis_dim)) {
                    vec_load_type = LoadType::LT_UNALIGNED;
                }
            }
        }
        FusedOpsConfiguration conf_vec = { "_VEC", {"b", "f", "(y + write_id)", "x"},
                                           "dequantized",
                                           input_dt,
                                           b_vec_size,
                                           vec_load_type,
                                           BoundaryCheck::ENABLED,
                                           IndexType::TENSOR_COORD,
                                           Tensor::DataChannelName::X };

        FusedOpsConfiguration conf_scalar = { "_SCALAR", {"b", "f", "(y + write_id)", "x"},
                                               "dequantized",
                                               input_dt,
                                               1,
                                               LoadType::LT_UNALIGNED,
                                               BoundaryCheck::ENABLED,
                                               IndexType::TENSOR_COORD,
                                               Tensor::DataChannelName::X };
        jit.Merge(MakeFusedOpsJitConstants(params, { conf_vec, conf_scalar }));
    }

    return jit;
}

KernelsData GemmKernelTiledOpt::GetKernelsData(const Params& params) const {
    if (!Validate(params)) {
        return KernelsData();
    }

    const auto& prim_params = static_cast<const gemm_params&>(params);
    size_t num_kernels = params.is_shape_agnostic ? 4 : 1;
    auto dispatchData = SetDefault(prim_params);
    KernelData k_data = KernelData::Default<gemm_params>(params, num_kernels);
    GetUpdateDispatchDataFunc(k_data);
    auto cldnn_jit = GetJitConstants(prim_params);
    for (size_t i = 0; i < num_kernels; i++) {
        if (params.is_shape_agnostic) {
            cldnn_jit.RemoveConstant("TILE_K_NOT_DIVISIBLE");
            cldnn_jit.RemoveConstant("TILE_N_NOT_DIVISIBLE");
            if (i == 0) {
                cldnn_jit.AddConstant(MakeJitConstant("TILE_K_NOT_DIVISIBLE", "0"));
                cldnn_jit.AddConstant(MakeJitConstant("TILE_N_NOT_DIVISIBLE", "0"));
            } else if (i == 1) {
                cldnn_jit.AddConstant(MakeJitConstant("TILE_K_NOT_DIVISIBLE", "0"));
                cldnn_jit.AddConstant(MakeJitConstant("TILE_N_NOT_DIVISIBLE", "1"));
            } else if (i == 2) {
                cldnn_jit.AddConstant(MakeJitConstant("TILE_K_NOT_DIVISIBLE", "1"));
                cldnn_jit.AddConstant(MakeJitConstant("TILE_N_NOT_DIVISIBLE", "0"));
            } else if (i == 3) {
                cldnn_jit.AddConstant(MakeJitConstant("TILE_K_NOT_DIVISIBLE", "1"));
                cldnn_jit.AddConstant(MakeJitConstant("TILE_N_NOT_DIVISIBLE", "1"));
            }
        }
        auto entry_point = GetEntryPoint(kernelName, prim_params.layerID, params, i);
        auto jit = CreateJit(kernelName, cldnn_jit, entry_point);

        auto& kernel = k_data.kernels[i];
        FillCLKernelData(kernel,
                        dispatchData,
                        params.engineInfo,
                        kernelName,
                        jit,
                        entry_point,
                        EXE_MODE_DEFAULT,
                        false,
                        false,
                        (uint32_t)prim_params.inputs.size(),
                        GetFusedPrimitiveInputsCount(params),
                        1,
                        prim_params.is_shape_agnostic);
    }

    return {k_data};
}

KernelsPriority GemmKernelTiledOpt::GetKernelsPriority(const Params& params) const {
    const auto& gmm_params = static_cast<const gemm_params&>(params);

    return gmm_params.transpose_input0 || gmm_params.transpose_input1 ? FORCE_PRIORITY_6 : FORCE_PRIORITY_3;
}

bool GemmKernelTiledOpt::Validate(const Params& params) const {
    if (!Parent::Validate(params))
        return false;

    const auto& gmm_params = static_cast<const gemm_params&>(params);

    if (gmm_params.outputs[0].PitchesDifferFromLogicalDims())
        return false;

    size_t num_inputs = (gmm_params.indirect_input0 || gmm_params.indirect_input1) ? gmm_params.inputs.size() - 1 : gmm_params.inputs.size();
    for (size_t input_idx = 0; input_idx < num_inputs; ++input_idx) {
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

    for (size_t i = 1; i < num_inputs; i++)
        if (gmm_params.inputs[0].GetDType() != gmm_params.inputs[i].GetDType())
            return false;

    return true;
}

void GemmKernelTiledOpt::GetUpdateDispatchDataFunc(KernelData& kd) const {
    if (kd.kernels.size() == 1) {
        Parent::GetUpdateDispatchDataFunc(kd);
    } else {
        kd.update_dispatch_data_func = [this](const Params& params, KernelData& kd) {
            const auto& prim_params = static_cast<const gemm_params&>(params);

            auto getTensorValue = [](const DataTensor& t, const int64_t dim_idx) -> size_t {
                switch (dim_idx) {
                    case 1:
                        return t.Feature().v;
                    case 2:
                        return t.U().v;
                    case 3:
                        return t.V().v;
                    case 4:
                        return t.W().v;
                    case 5:
                        return t.Z().v;
                    case 6:
                        return t.Y().v;
                    case 7:
                        return t.X().v;
                    default:
                        return t.Batch().v;
                }
            };

            GemmTuningData tuning_data = SetTuningParams(prim_params);
            auto input0_dims = ConvTo8dims(prim_params.input0_order);
            auto input1_dims = ConvTo8dims(prim_params.input1_order);
            auto k_size = getTensorValue(prim_params.inputs[0], input0_dims[7]);
            auto n_size = getTensorValue(prim_params.inputs[1], input1_dims[7]);
            bool not_divisible_k = ((k_size % tuning_data.tile_k_size) != 0);
            bool not_divisible_n = ((n_size % tuning_data.tile_n_size) != 0);
            size_t execute_kernel_idx = 0;
            if (not_divisible_k == false && not_divisible_n == false) {
                execute_kernel_idx = 0;
            } else if (not_divisible_k == false && not_divisible_n == true) {
                execute_kernel_idx = 1;
            } else if (not_divisible_k == true && not_divisible_n == false) {
                execute_kernel_idx = 2;
            } else if (not_divisible_k == true && not_divisible_n == true) {
                execute_kernel_idx = 3;
            }

            auto dispatchData = SetDefault(prim_params);
            for (size_t i = 0; i < kd.kernels.size(); i++) {
                kd.kernels[i].params.workGroups.global = dispatchData.gws;
                kd.kernels[i].params.workGroups.local = dispatchData.lws;
                if (execute_kernel_idx == i) {
                    kd.kernels[i].skip_execution = KernelData::SkipKernelExecution(prim_params);
                } else {
                    kd.kernels[i].skip_execution = true;
                }
            }
        };
    }
}
}  // namespace kernel_selector
