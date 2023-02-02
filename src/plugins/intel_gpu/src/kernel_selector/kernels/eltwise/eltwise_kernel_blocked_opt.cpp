// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "eltwise_kernel_blocked_opt.h"
#include "kernel_selector_utils.h"
#include <algorithm>
#include <string>
#include <vector>

namespace kernel_selector {
static inline bool InputHasFeatureBroadcast(const eltwise_params& params, const size_t op_num, const size_t input_idx);
static inline bool IsBroadcastingPossibleInput(const DataTensor& input, const DataTensor& output);
static inline int GetFeatureBlockSizeFromFormat(const eltwise_params& params, size_t index);

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

KernelsData EltwiseKernel_blocked_opt::GetKernelsData(const Params& params, const optional_params& options) const {
    if (!Validate(params, options)) {
        return {};
    }

    KernelData kd = KernelData::Default<eltwise_params>(params);
    eltwise_params& newParams = *static_cast<eltwise_params*>(kd.params.get());

    auto entry_point = GetEntryPoint(kernelName, newParams.layerID, params, options);
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

KernelsPriority EltwiseKernel_blocked_opt::GetKernelsPriority(const Params& /*params*/, const optional_params& /*options*/) const {
    return FORCE_PRIORITY_1;
}

// Protected
bool EltwiseKernel_blocked_opt::Validate(const Params& params, const optional_params& o) const {
    if (!EltwiseKernelBase::Validate(params, o)) {
        return false;
    }

    const auto& ewParams = static_cast<const eltwise_params&>(params);
    if (IsUnsupportedModeForVecCode(ewParams))
        return false;

    for (size_t i = 0; i < ewParams.inputs.size(); i++) {
        if ((GetFeatureBlockSizeFromFormat(ewParams, i) == 1) &&
            !IsBroadcastingPossibleInput(ewParams.inputs[i], ewParams.outputs[0])) {
            return false;
        }
    }

    const auto vec_size = GetFeatureBlockSizeFromFormat(ewParams, 0);
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
    const auto vec_size = GetFeatureBlockSizeFromFormat(params, 0);
    JitConstants jit = {};
    std::string vload_decls;


    // Make load jit constants
    for (size_t op_num = 0; op_num < params.operations.size(); op_num++) {
        const std::string op_num_str = toCodeString(op_num);
        const auto &ew = params.operations[op_num];
        for (size_t input_idx = 0; input_idx < ew.inputs.size(); input_idx++) {
            const auto &input = ew.inputs[input_idx];
            const std::string name = "INPUT_" + op_num_str + "_" + toCodeString(input_idx);

            // Get a string for a default index based on dimension
            std::string default_indexing_str;
            if (DataTensor::ChannelsCount(params.inputs[input_idx].GetLayout()) == 4)
                default_indexing_str = "b, (f_block * " + toCodeString(vec_size) +"), y, x";
            else if (DataTensor::ChannelsCount(params.inputs[input_idx].GetLayout()) == 5)
                default_indexing_str = "b, (f_block * " + toCodeString(vec_size) +"), z, y, x";
            else
                IE_ASSERT("MakeLoadJit : Unexpected dimension for eltwise optimized kernel.");

            switch (input.mode) {
                case EltwiseInputMode::SCALAR:
                    jit.AddConstant(MakeJitConstant(name, input.scalar));
                    break;
                case EltwiseInputMode::INPUT_BUFFER:
                {
                    const std::string idx_order = "INPUT" + toCodeString(input.index) + "_IDX_ORDER";
                    jit.AddConstant(MakeJitConstant(idx_order, default_indexing_str));

                    if (params.inputs[input.index].LogicalSize() == 1) {
                        const std::string vload_name = "DO_VLOAD" + toCodeString(op_num) + "_" + toCodeString(input_idx);
                        const std::string vload_value = "\\\n\tMAKE_VECTOR_TYPE(ACCUMULATOR_TYPE, " + toCodeString(vec_size) + ") " +
                                                        "tmp_a" + toCodeString(op_num) + "_" + toCodeString(input_idx) + " = " +
                                                        "(MAKE_VECTOR_TYPE(ACCUMULATOR_TYPE, " + toCodeString(vec_size) + "))" +
                                                        "(input" + toCodeString(input.index) + "[0])";
                        jit.AddConstant(MakeJitConstant(vload_name, vload_value));
                        jit.AddConstant(MakeJitConstant(name, "tmp_a" + toCodeString(op_num) + "_" + toCodeString(input_idx)));
                    } else {
                        bool feature_broadcasting = (params.inputs[input_idx].Feature().v == 1 && params.outputs[0].Feature().v != 1);

                        if (feature_broadcasting) {
                            const std::string broadcast_name = "DO_FEATURE_BROADCAST" + toCodeString(op_num) + "_" + toCodeString(input_idx);
                            std::string broadcast_value = "\\\n\tMAKE_VECTOR_TYPE(ACCUMULATOR_TYPE, " + toCodeString(vec_size) + ") tmp_b" +
                                                          toCodeString(op_num) + " = (MAKE_VECTOR_TYPE(ACCUMULATOR_TYPE, " + toCodeString(vec_size) + "))" +
                                                          "(input" + toCodeString(input.index) + "[GET_INDEX(INPUT, " + toCodeString(input.index) +
                                                          ", " + idx_order + ")]);";

                            jit.AddConstant(MakeJitConstant(broadcast_name, broadcast_value));
                            jit.AddConstant(MakeJitConstant(name, "tmp_b" + toCodeString(op_num)));
                        } else {
                            const std::string vload_name = "DO_VLOAD" + toCodeString(op_num) + "_" + toCodeString(input_idx);
                            const std::string vload_value = "\\\n\tMAKE_VECTOR_TYPE(ACCUMULATOR_TYPE, " + toCodeString(vec_size) + ")" +
                                                            " tmp_a" + toCodeString(op_num) + "_" + toCodeString(input_idx) +
                                                            " = TO_TYPE(MAKE_VECTOR_TYPE(ACCUMULATOR_TYPE, " + toCodeString(vec_size) + "), vload" +
                                                            toCodeString(vec_size) + "(0, &input" + toCodeString(input.index) +
                                                            "[GET_INDEX(INPUT," + toCodeString(input.index) + ", " + idx_order + ")]));";

                            jit.AddConstant(MakeJitConstant(vload_name, vload_value));
                            jit.AddConstant(MakeJitConstant(name, "tmp_a" + toCodeString(op_num) + "_" + toCodeString(input_idx)));
                        }
                    }
                    break;
                }
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
    const auto vec_size = GetFeatureBlockSizeFromFormat(params, 0);

    jit.Merge(MakeTypeJitConstants(GetAccumulatorType(params), "ACCUMULATOR"));
    jit.AddConstant(MakeJitConstant("BLOCK_SIZE", vec_size));
    jit.AddConstant(MakeJitConstant("XY_BLOCK", params.outputs[0].X().v * params.outputs[0].Y().v));

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

    // Fused_ops
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
    auto in_layout = params.inputs[0].GetLayout();
    auto out_layout = params.outputs[0].GetLayout();
    std::vector<std::vector<Tensor::DataChannelName>> dims_by_gws = {{Tensor::DataChannelName::FEATURE},
                                                                     {Tensor::DataChannelName::X, Tensor::DataChannelName::Y},
                                                                     {Tensor::DataChannelName::BATCH}};
    // Global workgroup size 0: feature, 1: spatial, 2: batch
    dispatchData.gws[0] = CeilDiv(params.outputs[0].Feature().v, GetFeatureBlockSizeFromFormat(params, 0));
    dispatchData.gws[2] = params.outputs[0].Batch().v;
    if (DataTensor::ChannelsCount(params.outputs[0].GetLayout()) == 5)
        dispatchData.gws[1] = params.outputs[0].X().v * params.outputs[0].Y().v * params.outputs[0].Z().v;
    else if (DataTensor::ChannelsCount(params.outputs[0].GetLayout()) == 4)
        dispatchData.gws[1] = params.outputs[0].X().v * params.outputs[0].Y().v;
    else
        IE_ASSERT("Unexpected dimension for eltwise_blocked_opt kernel.");

    // Calculate local workgroup size
    dispatchData.lws = GetOptimalLocalWorkGroupSizes(dispatchData.gws, params.engineInfo, in_layout, out_layout, dims_by_gws);
    if (out_layout == DataLayout::b_fs_yx_fsv4) {
        dispatchData.lws[0] = 1;
        dispatchData.lws[2] = 1;
    }

    return dispatchData;
}

// Local
static inline int GetFeatureBlockSizeFromFormat(const eltwise_params& arg, size_t index) {
    auto in_layout = arg.inputs[index].GetLayout();
    switch (in_layout) {
    case DataLayout::b_fs_yx_fsv4:
        return 4;
    case DataLayout::b_fs_yx_fsv16:
    case DataLayout::b_fs_yx_fsv32:
    case DataLayout::b_fs_zyx_fsv16:
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
