// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "eltwise_kernel_blocked_opt.h"
#include "common_tools.h"
#include "kernel_selector_utils.h"
#include <algorithm>
#include <string>
#include <vector>

namespace kernel_selector {
static inline bool InputHasFeatureBroadcast(const eltwise_params& params, const size_t op_num, const size_t input_idx);
static inline bool IsBroadcastingPossibleInput(const DataTensor& input, const DataTensor& output);
static inline int SelectVecSizeFromFormat(const DataTensor&);
static inline int GetInnerFeatureBlockSize(const DataTensor&);
static inline int GetInnerBatchBlockSize(const DataTensor&);
static inline size_t CalculateTotalWorkItemCount(const eltwise_params& params);


ParamsKey EltwiseKernel_blocked_opt::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableInputDataType(Datatype::INT8);
    k.EnableInputDataType(Datatype::UINT8);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::INT8);
    k.EnableOutputDataType(Datatype::UINT8);
    k.EnableInputLayout(DataLayout::b_fs_yx_fsv4);
    k.EnableOutputLayout(DataLayout::b_fs_yx_fsv4);
    k.EnableInputLayout(DataLayout::b_fs_yx_fsv16);
    k.EnableOutputLayout(DataLayout::b_fs_yx_fsv16);
    k.EnableInputLayout(DataLayout::b_fs_zyx_fsv16);
    k.EnableOutputLayout(DataLayout::b_fs_zyx_fsv16);
    k.EnableInputLayout(DataLayout::b_fs_yx_fsv32);
    k.EnableOutputLayout(DataLayout::b_fs_yx_fsv32);
    k.EnableInputLayout(DataLayout::b_fs_zyx_fsv32);
    k.EnableOutputLayout(DataLayout::b_fs_zyx_fsv32);
    k.EnableInputLayout(DataLayout::bs_fs_yx_bsv32_fsv32);
    k.EnableOutputLayout(DataLayout::bs_fs_yx_bsv32_fsv32);
    k.EnableInputLayout(DataLayout::bs_fs_yx_bsv16_fsv32);
    k.EnableOutputLayout(DataLayout::bs_fs_yx_bsv16_fsv32);
    k.EnableInputLayout(DataLayout::bs_fs_yx_bsv32_fsv16);
    k.EnableOutputLayout(DataLayout::bs_fs_yx_bsv32_fsv16);
    k.EnableInputLayout(DataLayout::bs_fs_yx_bsv16_fsv16);
    k.EnableOutputLayout(DataLayout::bs_fs_yx_bsv16_fsv16);
    k.EnableInputLayout(DataLayout::bs_fs_zyx_bsv32_fsv32);
    k.EnableOutputLayout(DataLayout::bs_fs_zyx_bsv32_fsv32);
    k.EnableInputLayout(DataLayout::bs_fs_zyx_bsv16_fsv32);
    k.EnableOutputLayout(DataLayout::bs_fs_zyx_bsv16_fsv32);
    k.EnableInputLayout(DataLayout::bs_fs_zyx_bsv32_fsv16);
    k.EnableOutputLayout(DataLayout::bs_fs_zyx_bsv32_fsv16);
    k.EnableInputLayout(DataLayout::bs_fs_zyx_bsv16_fsv16);
    k.EnableOutputLayout(DataLayout::bs_fs_zyx_bsv16_fsv16);
    k.EnableDifferentTypes();
    k.EnableBatching();
    k.EnableTensorPitches();
    k.EnableTensorOffset();
    k.EnableEltwiseBroadcast();
    return k;
}

KernelsData EltwiseKernel_blocked_opt::GetKernelsData(const Params& params) const {
    if (!Validate(params)) {
        return {};
    }

    KernelData kd = KernelData::Default<eltwise_params>(params);
    eltwise_params& newParams = *static_cast<eltwise_params*>(kd.params.get());

    auto entry_point = GetEntryPoint(kernelName, newParams.layerID, params);
    auto cldnn_jit = GetJitConstants(newParams);
    auto jit = CreateJit(kernelName, cldnn_jit, entry_point);

    DispatchData dispatchData = SetDefault(newParams);

    auto& kernel = kd.kernels[0];

    kernel.params.workGroups.global = dispatchData.gws;
    kernel.params.workGroups.local = dispatchData.lws;

    kernel.code.kernelString = GetKernelString(kernelName, jit, entry_point, params.engineInfo, EXE_MODE_DEFAULT);
    kernel.params.arguments = GetArgsDesc((uint32_t)newParams.inputs.size(),
                                   false,
                                   false,
                                   GetFusedPrimitiveInputsCount(params));

    return {kd};
}

KernelsPriority EltwiseKernel_blocked_opt::GetKernelsPriority(const Params& /*params*/) const {
    return FORCE_PRIORITY_1;
}

// Protected
bool EltwiseKernel_blocked_opt::Validate(const Params& params) const {
    if (!EltwiseKernelBase::Validate(params)) {
        return false;
    }

    const auto& ewParams = static_cast<const eltwise_params&>(params);
    if (IsUnsupportedModeForVecCode(ewParams))
        return false;

    for (size_t i = 0; i < ewParams.inputs.size(); i++) {
        if ((SelectVecSizeFromFormat(ewParams.inputs[i]) == 1) &&
            !IsBroadcastingPossibleInput(ewParams.inputs[i], ewParams.outputs[0])) {
            return false;
        }
    }

    const auto vec_size = SelectVecSizeFromFormat(ewParams.outputs[0]);
    const auto input0 = ewParams.inputs[0];
    const auto& output = ewParams.outputs[0];
    // Check that padding before features doesn't mis-align the blocks
    if (input0.Feature().pad.before % vec_size != 0 || output.Feature().pad.before % vec_size != 0)
        return false;

    auto compareTensors = [](const DataTensor& input0, const DataTensor& input1) -> bool {
        // Check all parameters except DataType
        auto& input0_dims = input0.GetDims();
        auto& input1_dims = input1.GetDims();
        bool same = input0.GetLayout() == input1.GetLayout() &&
                    input0.GetPaddedVal() == input1.GetPaddedVal() &&
                    input0.GetViewOffset() == input1.GetViewOffset() &&
                    input0_dims.size() == input1_dims.size();
        for (size_t i = 0; i < input0_dims.size(); i++) {
            same &= input0_dims[i].v == input1_dims[i].v &&
                    input0_dims[i].pad.before == input1_dims[i].pad.before &&
                    input0_dims[i].pad.after == input1_dims[i].pad.after &&
                    input0_dims[i].pitch == input1_dims[i].pitch;
        }

        return same;
    };

    for (size_t i = 1; i < ewParams.inputs.size(); i++) {
        if (ewParams.inputs[i].LogicalSize() == input0.LogicalSize() && !(compareTensors(ewParams.inputs[i], input0)))
            return false;
        if (ewParams.inputs[i].Feature().pad.before % vec_size != 0) {
            return false;
        }
    }

    return true;
}

JitConstants EltwiseKernel_blocked_opt::MakeLoadJitConstants(const eltwise_params& params, bool /*use_vload*/) const {
    const auto vec_size = SelectVecSizeFromFormat(params.outputs[0]);
    JitConstants jit = {};
    std::string vload_decls;

    auto Padded = [](const DataTensor& tensor) -> bool {
        bool is_padded = false;
        auto& tensor_dims = tensor.GetDims();
        for (size_t i = 0; i < tensor_dims.size(); i++) {
            is_padded |= tensor_dims[i].pad.Total() != 0;
        }
        return is_padded;
    };

    jit.AddConstant(MakeJitConstant("PADDED_OUTPUT", Padded(params.outputs[0])));

    for (size_t op_num = 0; op_num < params.operations.size(); op_num++) {
        const std::string op_num_str = toCodeString(op_num);
        const auto &ew = params.operations[op_num];
        // Every input selects a proper indexing rule either a raw global_id or a formatted GET_INDEX
        // A formatted GET_INDEX is calculated from its global id.
        // One operation for a eltwise could take both indexing methods like the following :
        // float8 tmp_broadcast0 = (float8)(input0[get_b_fs_yx_fsv_index_safe( b, (f_block * 8), y, x, ..., 32)]);;
        // float8 tmp_a0_1 = convert_float8(vload8(global_id, input1));;
        // const float8 tmp0 = (float8)tmp_broadcast0 + (float8)tmp_a0_1; res = tmp0;
        for (size_t input_idx = 0; input_idx < ew.inputs.size(); input_idx++) {
            const auto &input = ew.inputs[input_idx];
            const std::string name = "INPUT_" + op_num_str + "_" + toCodeString(input_idx);
            // Broadcasting
            bool feature_broadcasting = (params.inputs[input_idx].Feature().v == 1 && params.outputs[0].Feature().v != 1);
            bool spatial_broadcasting = (params.inputs[input_idx].LogicalSize() == params.outputs[0].Feature().v &&
                                        params.inputs[input_idx].LogicalSize() == params.inputs[input_idx].Feature().v &&
                                        GetInnerBatchBlockSize(params.inputs[input_idx]) == 1 && !Padded(params.inputs[input_idx]));
            bool full_tensor = (params.inputs[input_idx].LogicalSize() == params.outputs[0].LogicalSize() && !Padded(params.inputs[input_idx]));

            // Based on dimension, get a string of indexing for formmatted GET_INDEX
            std::string default_indexing_str;
            if (DataTensor::ChannelsCount(params.inputs[input_idx].GetLayout()) == 4)
                default_indexing_str = "b, (f_block * " + toCodeString(vec_size) +"), y, x";
            else if (DataTensor::ChannelsCount(params.inputs[input_idx].GetLayout()) == 5)
                default_indexing_str = "b, (f_block * " + toCodeString(vec_size) +"), z, y, x";
            else
                OPENVINO_THROW("MakeLoadJit : Unexpected dimension for eltwise optimized kernel.");

            // Generate Jit
            switch (input.mode) {
                case EltwiseInputMode::SCALAR:
                    jit.AddConstant(MakeJitConstant(name, input.scalar));
                    break;
                case EltwiseInputMode::INPUT_BUFFER:
                {
                    // Strings for readability
                    const std::string idx_order = "INPUT" + toCodeString(input.index) + "_IDX_ORDER";
                    const std::string temp_vec_type = "MAKE_VECTOR_TYPE(ACCUMULATOR_TYPE, " + toCodeString(vec_size) + ")";
                    const std::string temp_vec_var = " tmp_a" + toCodeString(op_num) + "_" + toCodeString(input_idx);
                    const std::string input_i = "input" + toCodeString(input.index);
                    const std::string vload_n = "vload" + toCodeString(vec_size);
                    jit.AddConstant(MakeJitConstant(idx_order, default_indexing_str));

                    // Select indexing method based on broadcasting type and tensor size
                    if (params.inputs[input.index].LogicalSize() == 1) {
                        // Sample : half8 tmp_a0_1 = (half8)(input1[0]);
                        const std::string vload_name = "DO_VLOAD" + toCodeString(op_num) + "_" + toCodeString(input_idx);
                        const std::string vload_value = "\\\n\t " + temp_vec_type + temp_vec_var + " = " +
                                                        "(" + temp_vec_type + ")(" + input_i + "[0])";

                        jit.AddConstant(MakeJitConstant(vload_name, vload_value));
                        jit.AddConstant(MakeJitConstant(name, "tmp_a" + toCodeString(op_num) + "_" + toCodeString(input_idx)));
                    } else if (feature_broadcasting) {
                        // Load as scalar and broadcast to vector variable
                        // Sample : half8 tmp_broadcast0 = (half8)(input0[get_b_fs_yx_fsv_index_safe( b, (f_block * 8), y, x, ..., 32)]);;
                        const std::string broadcast_name = "DO_FEATURE_BROADCAST" + toCodeString(op_num) + "_" + toCodeString(input_idx);
                        std::string broadcast_value = "\\\n\t " + temp_vec_type + " tmp_broadcast" + toCodeString(op_num) + " = " +
                                                        "(" + temp_vec_type + ")(" + input_i +
                                                        "[GET_INDEX(INPUT, " + toCodeString(input.index) + ", " + idx_order + ")]);";

                        jit.AddConstant(MakeJitConstant(broadcast_name, broadcast_value));
                        jit.AddConstant(MakeJitConstant(name, "tmp_broadcast" + toCodeString(op_num)));
                    } else if (spatial_broadcasting) {
                        // Load as vector. No need to use GET_INDEX: use f_block for raw indexing
                        // Sample : half8 tmp_a0_1 = convert_half8(vload8(f_block, input1));;
                        const std::string vload_name = "DO_VLOAD" + toCodeString(op_num) + "_" + toCodeString(input_idx);
                        const std::string vload_value = "\\\n\t " + temp_vec_type + temp_vec_var + " = " +
                                                        "TO_TYPE(" + temp_vec_type + ", " + vload_n + "(f_block, " + input_i + "));";

                        jit.AddConstant(MakeJitConstant(vload_name, vload_value));
                        jit.AddConstant(MakeJitConstant(name, "tmp_a" + toCodeString(op_num) + "_" + toCodeString(input_idx)));
                    } else if (full_tensor) {
                        // Load as vector. Use raw global id to reduce overhead of formatted indexing
                        // Sample : half8 tmp_a0_0 = convert_half8(vload8(global_id, input0));;
                        const std::string vload_name = "DO_VLOAD" + toCodeString(op_num) + "_" + toCodeString(input_idx);
                        const std::string vload_value = "\\\n\t " + temp_vec_type + temp_vec_var + " = " +
                                                        "TO_TYPE(" + temp_vec_type + ", " + vload_n + "(global_id, " + input_i + "));";

                        jit.AddConstant(MakeJitConstant(vload_name, vload_value));
                        jit.AddConstant(MakeJitConstant(name, "tmp_a" + toCodeString(op_num) + "_" + toCodeString(input_idx)));
                    } else {
                        // A default vector load using formatted GET_INDEX
                        // Sample : half8 tmp_a0_0 = convert_half8(vload8(0, &input0[get_b_fs_yx_fsv_index( b, (f_block * 8), y, x, ..., 32)]));;
                        const std::string vload_name = "DO_VLOAD" + toCodeString(op_num) + "_" + toCodeString(input_idx);
                        const std::string vload_value = "\\\n\t " + temp_vec_type + temp_vec_var + " = " +
                                                        "TO_TYPE(" + temp_vec_type + ", " + vload_n + "(0, &input" + toCodeString(input.index) +
                                                        "[GET_INDEX(INPUT," + toCodeString(input.index) + ", " + idx_order + ")]));";

                        jit.AddConstant(MakeJitConstant(vload_name, vload_value));
                        jit.AddConstant(MakeJitConstant(name, "tmp_a" + toCodeString(op_num) + "_" + toCodeString(input_idx)));
                    }
                }  // EltwiseInputMode::INPUT_BUFFER
                    break;
                case EltwiseInputMode::OUTPUT_BUFFER:
                    jit.AddConstant(MakeJitConstant(name, "output[off]"));
                    break;
                case EltwiseInputMode::UNORDERED_ACCESS_INPUT_BUFFER:
                    jit.AddConstant(MakeJitConstant(
                            name,
                            "input" + toCodeString(input.index) + "[(size_t)tmp" + toCodeString(input.tmpIndex) + "]"));
                    break;
                case EltwiseInputMode::INTERMEDIATE_RESULTS_INDEX:
                    jit.AddConstant(MakeJitConstant(name, "tmp" + toCodeString(input.tmpIndex)));
                    break;
                default:
                    break;
            }
        }
    }

    return jit;
}

JitConstants EltwiseKernel_blocked_opt::GetJitConstants(const eltwise_params& params) const {
    JitConstants jit = MakeBaseParamsJitConstants(params);
    const auto vec_size = SelectVecSizeFromFormat(params.outputs[0]);
    const auto inner_feature_blk_size = GetInnerFeatureBlockSize(params.outputs[0]);
    const auto inner_batch_blk_size = GetInnerBatchBlockSize(params.outputs[0]);

    jit.Merge(MakeTypeJitConstants(GetAccumulatorType(params), "ACCUMULATOR"));
    jit.AddConstant(MakeJitConstant("BLOCK_SIZE", vec_size));
    // Define inner blocks for batch & feature axes
    jit.AddConstant(MakeJitConstant("F_BLOCK_COUNT", inner_feature_blk_size / vec_size));
    jit.AddConstant(MakeJitConstant("INNER_BATCH_SIZE", inner_batch_blk_size));
    jit.AddConstant(MakeJitConstant("INNER_BLOCKS_COUNT", (inner_feature_blk_size / vec_size) * inner_batch_blk_size));
    // Define spatial size to calculate batch and feature from a raw global id of each work group
    jit.AddConstant(MakeJitConstant("OUTPUT_SIZE_XY", params.outputs[0].X().v * params.outputs[0].Y().v));
    // To calculate batch, define outer block size of feature axis (divided by the inner feature-block size)
    jit.AddConstant(MakeJitConstant("OUT_F_BLOCK", CeilDiv(params.outputs[0].Feature().v, inner_feature_blk_size)));

    bool use_vload = false;
    jit.Merge(MakeInputDeclsJitConstants(params, use_vload));
    jit.Merge(MakeLoadJitConstants(params, use_vload));
    jit.Merge(GetOperationsJitConstants(params, use_vload, vec_size));

    std::string do_eltwise;
    auto& operations = params.operations;
    for (size_t op_num = 0; op_num < operations.size(); op_num++) {
        const auto &ew = operations[op_num];
        for (size_t input_idx = 0; input_idx < ew.inputs.size(); input_idx++) {
            const auto &input = ew.inputs[input_idx];
            if (input.mode != EltwiseInputMode::INPUT_BUFFER && input.mode != EltwiseInputMode::SCALAR)
                continue;

            if (InputHasFeatureBroadcast(params, op_num, input_idx)) {
                do_eltwise += "\\\n\tDO_FEATURE_BROADCAST" + toCodeString(op_num) + "_" + toCodeString(input_idx) + ";";
            } else {
                do_eltwise += "\\\n\tDO_VLOAD" + toCodeString(op_num) + "_" + toCodeString(input_idx) + ";";
            }
        }
        do_eltwise += "\\\n\tOPERATION" + toCodeString(op_num) + ";";
    }

    do_eltwise += "\\\n\tres = tmp" + toCodeString(operations.size() - 1) + ";";

    jit.AddConstant(MakeJitConstant("DO_ELTWISE", do_eltwise));

    if (params.layoutBased || params.int8_quantization || params.broadcast) {
        jit.Merge(GetTensorFriendlyWorkGroupsJit(params.outputs[0]));
    }

    if (!params.stride.empty()) {
        jit.AddConstant(MakeJitConstant("INPUT_STRIDED", 1));
    }

    jit.Merge(MakeActivationJitConstants(params.activations, params.outputs[0].GetDType(), "_TYPED"));

    if (params.outputs[0].Feature().v % vec_size != 0)
        jit.AddConstant(MakeJitConstant("LEFTOVERS", params.outputs[0].Feature().v % vec_size));

    // Fused post_ops
    if (!params.fused_ops.empty()) {
        kernel_selector::Datatype input_dt = GetAccumulatorType(params);
        std::vector<std::string> idx_order;
        if (DataTensor::ChannelsCount(params.outputs[0].GetLayout()) == 4) {
            idx_order = {"b", "f_block * " + toCodeString(vec_size), "y", "x"};
        } else if (DataTensor::ChannelsCount(params.outputs[0].GetLayout()) == 5) {
            idx_order = {"b", "f_block * " + toCodeString(vec_size), "z", "y", "x"};
        }

        FusedOpsConfiguration conf = {"", idx_order, "res", input_dt, (size_t)vec_size};

        conf.vec_axis = Tensor::DataChannelName::FEATURE;
        jit.Merge(MakeFusedOpsJitConstants(params, {conf}));
    }

    jit.AddConstant(MakeJitConstant("QUANTIZATION_TERM", params.int8_quantization));
    jit.AddConstant(MakeJitConstant("VEC_SIZE", vec_size));
    jit.AddConstant(MakeJitConstant("VSTORE_N", "vstore" + toCodeString(vec_size)));

    if (params.broadcast) {
        bool need_idx_safe = true;
        for (size_t i = 0; i < params.inputs.size(); i++) {
            if (params.inputs[i].LogicalSize() == 1) {
                need_idx_safe = false;
                break;
            }
        }

        if (need_idx_safe)
            jit.AddConstant(MakeJitConstant("ELTWISE_BROADCAST", params.broadcast));
    }

    return jit;
}

EltwiseKernelBase::DispatchData EltwiseKernel_blocked_opt::SetDefault(const eltwise_params& params) const {
    DispatchData dispatchData;

    // Instead of using multiple channel(e.g. {{FEATURE}, {SPATIAL}, {BATCH}}), this kernel uses 1 channel which contains all logical size.
    // so that each global id can be an index of each work group.
    // It also makes an index for fomatted GET_INDEX macro if needed(e.g. feature broadcasting, fusing).
    KernelData kd = KernelData::Default<eltwise_params>(params);
    dispatchData.gws = {std::max(CalculateTotalWorkItemCount(params) / SelectVecSizeFromFormat(params.outputs[0]), (size_t)1), 1, 1};
    dispatchData.lws = GetOptimalLocalWorkGroupSizes(dispatchData.gws, params.engineInfo);

    return dispatchData;
}

// Local
static inline size_t CalculateTotalWorkItemCount(const eltwise_params& params) {
    auto feature = Align(params.outputs[0].Feature().v, GetInnerFeatureBlockSize(params.outputs[0]));
    auto batch = Align(params.outputs[0].Batch().v, GetInnerBatchBlockSize(params.outputs[0]));
    size_t spatial = 0;
    if (DataTensor::ChannelsCount(params.outputs[0].GetLayout()) == 5)
        spatial = params.outputs[0].X().v * params.outputs[0].Y().v * params.outputs[0].Z().v;
    else
        spatial = params.outputs[0].X().v * params.outputs[0].Y().v;

    return (feature * batch * spatial);
}

static inline int SelectVecSizeFromFormat(const DataTensor& tensor) {
    // No feature inner block : not acceptable for calculation of ordered index
    auto layout = tensor.GetLayout();
    switch (layout) {
    case DataLayout::b_fs_yx_fsv4:
        return 4;
    case DataLayout::b_fs_yx_fsv16:
    case DataLayout::b_fs_zyx_fsv16:
    case DataLayout::b_fs_yx_fsv32:
    case DataLayout::b_fs_zyx_fsv32:
    case DataLayout::bs_fs_yx_bsv32_fsv32:
    case DataLayout::bs_fs_yx_bsv32_fsv16:
    case DataLayout::bs_fs_yx_bsv16_fsv32:
    case DataLayout::bs_fs_yx_bsv16_fsv16:
    case DataLayout::bs_fs_zyx_bsv32_fsv32:
    case DataLayout::bs_fs_zyx_bsv32_fsv16:
    case DataLayout::bs_fs_zyx_bsv16_fsv32:
    case DataLayout::bs_fs_zyx_bsv16_fsv16:
        return 8;
    default:
        return 1;
    }
}

static inline int GetInnerBatchBlockSize(const DataTensor& tensor) {
    auto layout = tensor.GetLayout();
    switch (layout) {
    case DataLayout::b_fs_yx_fsv4:
    case DataLayout::b_fs_yx_fsv16:
    case DataLayout::b_fs_zyx_fsv16:
    case DataLayout::b_fs_yx_fsv32:
    case DataLayout::b_fs_zyx_fsv32:
        return 1;
    case DataLayout::bs_fs_yx_bsv16_fsv32:
    case DataLayout::bs_fs_yx_bsv16_fsv16:
    case DataLayout::bs_fs_zyx_bsv16_fsv32:
    case DataLayout::bs_fs_zyx_bsv16_fsv16:
        return 16;
    case DataLayout::bs_fs_yx_bsv32_fsv32:
    case DataLayout::bs_fs_yx_bsv32_fsv16:
    case DataLayout::bs_fs_zyx_bsv32_fsv32:
    case DataLayout::bs_fs_zyx_bsv32_fsv16:
        return 32;
    default:
        OPENVINO_THROW("GetInnerBatchBlockSize : Unexpected format for eltwise_blocked_optimized kernel.");
    }

    return 1;
}

static inline int GetInnerFeatureBlockSize(const DataTensor& tensor) {
    auto layout = tensor.GetLayout();
    switch (layout) {
    case DataLayout::b_fs_yx_fsv4:
        return 4;
    case DataLayout::b_fs_yx_fsv16:
    case DataLayout::b_fs_zyx_fsv16:
    case DataLayout::bs_fs_yx_bsv32_fsv16:
    case DataLayout::bs_fs_yx_bsv16_fsv16:
    case DataLayout::bs_fs_zyx_bsv32_fsv16:
    case DataLayout::bs_fs_zyx_bsv16_fsv16:
        return 16;
    case DataLayout::b_fs_yx_fsv32:
    case DataLayout::b_fs_zyx_fsv32:
    case DataLayout::bs_fs_yx_bsv32_fsv32:
    case DataLayout::bs_fs_yx_bsv16_fsv32:
    case DataLayout::bs_fs_zyx_bsv32_fsv32:
    case DataLayout::bs_fs_zyx_bsv16_fsv32:
        return 32;
    default:
        OPENVINO_THROW("GetInnerFeatureBlockSize : Unexpected format for eltwise_blocked_optimized kernel.");
    }

    return 1;
}

static inline bool IsBroadcastingPossibleInput(const DataTensor& input, const DataTensor& output) {
    if ((input.LogicalSize() == 1) ||
        (input.LogicalSize() == output.Feature().v && input.Feature().v == output.Feature().v)) {
            return true;
        }
    return false;
}

static inline bool InputHasFeatureBroadcast(const eltwise_params& params, const size_t op_num, const size_t input_idx) {
    const auto &ew = params.operations[op_num];

    const auto &input = ew.inputs[input_idx];
    if (input.mode == EltwiseInputMode::INPUT_BUFFER) {
        if (params.inputs[input_idx].LogicalSize() != 1
            && params.inputs[input_idx].Feature().v == 1
            && params.outputs[0].Feature().v != 1) {
                return true;
            }
    }

    return false;
}
}  // namespace kernel_selector
