// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "reduce_kernel_b_fs_yx_fsv16.h"
#include "kernel_selector_utils.h"
#include <vector>
#include <string>
#include "common_tools.h"

namespace kernel_selector {

static const size_t SIMD = 16;
static const size_t XY_OPT_F_LIMITS = 96;
static const size_t AXIS_F = 1;
static const size_t AXIS_Y = 2;
static const size_t AXIS_X = 3;
using NDims = std::vector<kernel_selector::Tensor::Dim>;

static size_t calc_read_offset(const reduce_params& params) {
    auto read_offset = 1;
    if (BytesPerElement(params.inputs[0].GetDType()) == 4)
        read_offset = 4;
    else if (BytesPerElement(params.inputs[0].GetDType()) == 2)
        read_offset = 8;
    else if (BytesPerElement(params.inputs[0].GetDType()) == 1)
        read_offset = 16;
    return read_offset;
}

static NDims get_input_dims(const reduce_params& params) {
    auto input = params.inputs[0];
    auto in_dims = input.GetDims();
    std::reverse(in_dims.begin(), in_dims.end());
    return in_dims;
}

static NDims calc_in_dims(const reduce_params& params) {
    auto input = params.inputs[0];
    auto in_dims = input.GetDims();
    auto reduce_axes = params.reduceAxes;

    std::vector<size_t> ordered_axes = {0, 1, 3, 2};
    std::reverse(in_dims.begin(), in_dims.end());
    for (size_t a = 0; a < params.reduceAxes.size(); a++) {
        in_dims[ordered_axes[params.reduceAxes[a]]].v = 1;
    }

    return in_dims;
}

static bool is_xy_opt_supported(const ReduceMode& mode) {
    switch (mode) {
        case ReduceMode::MAX:
        case ReduceMode::MIN:
        case ReduceMode::MEAN:
        case ReduceMode::SUM:
        case ReduceMode::AND:
        case ReduceMode::OR:
        case ReduceMode::L1:
        case ReduceMode::LOG_SUM_EXP:
            return true;
        // prod, sum_squre, L2 and log_sum doesn't work with reduce(x,y) optimization.
        case ReduceMode::PROD:
        case ReduceMode::SUM_SQUARE:
        case ReduceMode::L2:
        case ReduceMode::LOG_SUM:
        default:
            return false;
    }
}

static bool can_opt_reduce_xy(const reduce_params& params) {
    auto axes = params.reduceAxes;
    auto input_dims = get_input_dims(params);
    return is_xy_opt_supported(params.reduceMode) && axes.size() == 2 &&
        std::find(axes.begin(), axes.end(), AXIS_Y) != std::end(axes) &&
        std::find(axes.begin(), axes.end(), AXIS_X) != std::end(axes) &&
        input_dims[1].v <= XY_OPT_F_LIMITS;
}

static bool reducing_unaligned_f_axis(const reduce_params& params) {
    if (count(params.reduceAxes.begin(), params.reduceAxes.end(), AXIS_F) > 0) {
        if (params.inputs[0].Feature().v % 16 != 0)
            return true;
    }

    return false;
}

ParamsKey ReduceKernel_b_fs_yx_fsv16::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableInputDataType(Datatype::INT32);
    k.EnableInputDataType(Datatype::INT8);
    k.EnableInputDataType(Datatype::UINT8);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::INT32);
    k.EnableOutputDataType(Datatype::INT8);
    k.EnableOutputDataType(Datatype::UINT8);
    k.EnableInputLayout(DataLayout::b_fs_yx_fsv16);
    k.EnableOutputLayout(DataLayout::b_fs_yx_fsv16);
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    k.EnableDifferentTypes();
    return k;
}

DeviceFeaturesKey ReduceKernel_b_fs_yx_fsv16::get_required_device_features_key(const Params& params) const {
    auto k = get_common_subgroups_device_features_key(params);
    k.requires_subgroup_reduce();
    k.requires_subgroup_shuffle();
    k.requires_subgroup_shuffle_relative();

    return k;
}

CommonDispatchData ReduceKernel_b_fs_yx_fsv16::SetDefault(const reduce_params& params) const {
    CommonDispatchData dispatchData;

    auto in_dims = calc_in_dims(params);

    if (can_opt_reduce_xy(params)) {
        auto input_dims = get_input_dims(params);
        dispatchData.gws = { 16,
                            std::min(CeilDiv(input_dims[2].v, SIMD), SIMD),
                            CeilDiv(in_dims[1].v, SIMD) * in_dims[0].v };                 // F, B
        dispatchData.lws = { 16, dispatchData.gws[1], 1 };
    } else {
        dispatchData.gws = { 16,
                         CeilDiv(in_dims[3].v, calc_read_offset(params)) * in_dims[2].v,  // X, Y
                         CeilDiv(in_dims[1].v, SIMD) * in_dims[0].v };                    // F, B
        dispatchData.lws = { SIMD, 1, 1 };
    }

    return dispatchData;
}

JitConstants ReduceKernel_b_fs_yx_fsv16::GetJitConstants(const reduce_params& params) const {
    auto jit = ReduceKernelBase::GetJitConstants(params);
    auto in_dims = calc_in_dims(params);
    auto read_offset = calc_read_offset(params);

    // Optimization of reduce(x,y) when feature depth is shallow.
    // In this case, tile the input tensor and create partial result to generate more work items
    if (can_opt_reduce_xy(params)) {
        auto input_dims = get_input_dims(params);
        auto num_block_y = std::min(CeilDiv(input_dims[2].v, SIMD), SIMD);
        jit.AddConstant(MakeJitConstant("IS_REDUCE_XY", 1));
        jit.AddConstant(MakeJitConstant("BLOCK_Y_NUM", num_block_y));
        jit.AddConstant(MakeJitConstant("BLOCK_Y_SIZE", CeilDiv(input_dims[2].v, num_block_y)));
    } else {
        jit.AddConstant(MakeJitConstant("IS_REDUCE_XY", 0));
    }

    // Universal output sizes for keep dims = true/false cases
    jit.AddConstant(MakeJitConstant("COMMON_OUTPUT_SIZE_X", in_dims[3].v));
    jit.AddConstant(MakeJitConstant("COMMON_OUTPUT_SIZE_Y", in_dims[2].v));
    jit.AddConstant(MakeJitConstant("COMMON_OUTPUT_FEATURE_NUM", in_dims[1].v));
    jit.AddConstant(MakeJitConstant("COMMON_OUTPUT_BATCH_NUM", in_dims[0].v));
    jit.AddConstant(MakeJitConstant("READ_OFFSET", read_offset));
    jit.AddConstant(MakeJitConstant("BLOCK_READ(ptr,offset)", "DT_INPUT_BLOCK_READ" + toCodeString(read_offset) + "(ptr,offset)"));
    jit.Merge(MakeTypeJitConstants(GetActivationType(params), "ACTIVATION"));
    jit.Merge(MakeTypeJitConstants(GetAccumulatorType(params), "ACCUMULATOR"));
    jit.Merge(MakeTypeJitConstants(GetFinalAccumulatorType(params), "FINAL_ACCUMULATOR"));

    if (!params.fused_ops.empty()) {
        auto input_dt = GetActivationType(params);
        std::vector<std::string> idx_order = {"b", "f", "y", "x"};
        std::string var_name = "reduce_result";

        bool cant_handle_vec16 = read_offset > 8 ? true : false;
        size_t vec_size = cant_handle_vec16 ? 8 : read_offset;

        FusedOpsConfiguration conf_scalar = {"_SCALAR",
                                             idx_order,
                                             var_name,
                                             input_dt,
                                             1,
                                             LoadType::LT_ALIGNED_READ,
                                             BoundaryCheck::DISABLED,
                                             IndexType::TENSOR_COORD,
                                             Tensor::DataChannelName::X};

        if (cant_handle_vec16) {
            FusedOpsConfiguration conf_vector_1 = {"_VECTOR_1",
                                                   idx_order,
                                                   var_name+".lo",
                                                   input_dt,
                                                   vec_size,
                                                   LoadType::LT_ALIGNED_READ,
                                                   BoundaryCheck::DISABLED,
                                                   IndexType::TENSOR_COORD,
                                                   Tensor::DataChannelName::X};

            std::vector<std::string> idx_order_vec_2 = {"b", "f", "y", "x + 8"};
            FusedOpsConfiguration conf_vector_2 = {"_VECTOR_2",
                                                   idx_order_vec_2,
                                                   var_name+".hi",
                                                   input_dt,
                                                   vec_size,
                                                   LoadType::LT_ALIGNED_READ,
                                                   BoundaryCheck::DISABLED,
                                                   IndexType::TENSOR_COORD,
                                                   Tensor::DataChannelName::X};

            jit.AddConstant(MakeJitConstant("FUSED_OPS_VECTOR", "{FUSED_OPS_VECTOR_1;final_result.lo=FUSED_OPS_RESULT_VECTOR_1;}"
                                                                "{FUSED_OPS_VECTOR_2;final_result.hi=FUSED_OPS_RESULT_VECTOR_2;}"));
            jit.AddConstant(MakeJitConstant("FUSED_OPS_RESULT_VECTOR", "final_result"));
            jit.Merge(MakeFusedOpsJitConstants(params, {conf_scalar, conf_vector_1, conf_vector_2}));
        } else {
            FusedOpsConfiguration conf_vector = {"_VECTOR",
                                                 idx_order,
                                                 var_name,
                                                 input_dt,
                                                 vec_size,
                                                 LoadType::LT_ALIGNED_READ,
                                                 BoundaryCheck::DISABLED,
                                                 IndexType::TENSOR_COORD,
                                                 Tensor::DataChannelName::X};

            jit.Merge(MakeFusedOpsJitConstants(params, {conf_vector, conf_scalar}));
        }
    }

    // Some reduction modes are affected by 0 value (e.g. min, max, prod ...)
    bool zero_invariant_mode = params.reduceMode == ReduceMode::L1 || params.reduceMode == ReduceMode::L2 ||
                               params.reduceMode == ReduceMode::LOG_SUM || params.reduceMode == ReduceMode::LOG_SUM_EXP ||
                               params.reduceMode == ReduceMode::MEAN || params.reduceMode == ReduceMode::OR ||
                               params.reduceMode == ReduceMode::SUM || params.reduceMode == ReduceMode::SUM_SQUARE;

    if (zero_invariant_mode && reducing_unaligned_f_axis(params)) {
        jit.AddConstant(MakeJitConstant("ZERO_INVARIANT_REDUCTION", 1));
    }

    return jit;
}

KernelsData ReduceKernel_b_fs_yx_fsv16::GetKernelsData(const Params& params) const {
    KernelsData kds = GetCommonKernelsData(params);
    const reduce_params& orgParams = static_cast<const reduce_params&>(params);

    // To get perf gain of reduction of un-aligned f axis,
    // Reduce kernel uses 0 value out of range in inner block by disabling re-use memory
    if (orgParams.inputs[0].Feature().v % 16 != 0) {
        kds[0].can_reuse_memory = false;
    }

    return kds;
}

KernelsPriority ReduceKernel_b_fs_yx_fsv16::GetKernelsPriority(const Params& /*params*/) const {
    return FORCE_PRIORITY_6;
}
}  // namespace kernel_selector
