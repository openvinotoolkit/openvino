// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "eltwise_kernel_b_fs_yx_fsv16.h"
#include "kernel_selector_utils.h"
#include <string>
#include <algorithm>

namespace kernel_selector {

static inline bool IsBroadcastingPossibleInput(const DataTensor& input, const DataTensor& output) {
    if ((input.LogicalSize() == 1) ||
        (input.LogicalSize() == output.Feature().v && input.Feature().v == output.Feature().v)) {
            return true;
        }
    return false;
}

ParamsKey EltwiseKernel_b_fs_yx_fsv16::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableInputDataType(Datatype::INT8);
    k.EnableInputDataType(Datatype::UINT8);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::INT8);
    k.EnableOutputDataType(Datatype::UINT8);
    k.EnableInputLayout(DataLayout::b_fs_yx_fsv16);
    k.EnableInputLayout(DataLayout::b_fs_zyx_fsv16);
    k.EnableOutputLayout(DataLayout::b_fs_yx_fsv16);
    k.EnableOutputLayout(DataLayout::b_fs_zyx_fsv16);
    k.EnableDifferentTypes();
    k.EnableBatching();
    k.EnableTensorPitches();
    k.EnableTensorOffset();
    k.EnableEltwiseBroadcast();
    k.EnableSubGroup();
    k.EnableSubGroupShort();
    return k;
}

static inline size_t GetBlockSize(const eltwise_params& params) {
    // Set blocksize 1 when broadcasting X dim
    for (size_t i = 0; i < params.inputs.size(); i++) {
        if ((params.inputs[i].X().v == 1) && !IsBroadcastingPossibleInput(params.inputs[i], params.output)) {
            return 1;
        }
    }

    size_t optimal_bs_values[] = {8, 4, 2, 1};

    for (auto bs : optimal_bs_values) {
        if ((params.output.X().v) % bs == 0) {
            return bs;
        }
    }

    return 1;
}

static inline bool OpHasFeatureBroadcast(const eltwise_params& params, const size_t op_num) {
    const auto &ew = params.operations[op_num];

    for (size_t input_idx = 0; input_idx < ew.inputs.size(); input_idx++) {
        const auto &input = ew.inputs[input_idx];
        if (input.mode == EltwiseInputMode::INPUT_BUFFER) {
            if (params.inputs[input_idx].LogicalSize() != 1 &&
                params.inputs[input_idx].Feature().v == 1 &&
                params.output.Feature().v != 1) {
                    return true;
                }
        }
    }

    return false;
}

JitConstants EltwiseKernel_b_fs_yx_fsv16::MakeLoadJitConstants(const eltwise_params& params, bool /*useVload8*/) const {
    JitConstants jit = {};
    std::string vload_decls;
    for (size_t op_num = 0; op_num < params.operations.size(); op_num++) {
        const std::string op_num_str = std::to_string(op_num);
        const auto &ew = params.operations[op_num];
        for (size_t input_idx = 0; input_idx < ew.inputs.size(); input_idx++) {
            const auto &input = ew.inputs[input_idx];
            const std::string name = "INPUT_" + op_num_str + "_" + std::to_string(input_idx);

            switch (input.mode) {
                case EltwiseInputMode::SCALAR:
                    jit.AddConstant(MakeJitConstant(name, input.scalar));
                    break;
                case EltwiseInputMode::INPUT_BUFFER:
                {
                    if (params.inputs[input.index].LogicalSize() == params.output.Feature().v &&
                        params.inputs[input.index].LogicalSize() == params.inputs[input.index].Feature().v) {
                        std::string block_read_str = "BLOCK_READN(INPUT" + std::to_string(input.index) + "_TYPE, " +
                                                     "1, " +
                                                     "input" + std::to_string(input.index) +
                                                     ", INPUT" + std::to_string(input.index);
                        if (DataTensor::ChannelsCount(params.inputs[input_idx].GetLayout()) == 4) {
                            jit.AddConstant(MakeJitConstant(name, block_read_str + "_GET_INDEX(b, f_block*16, y, x))"));
                        } else {
                            jit.AddConstant(MakeJitConstant(name, block_read_str + "_GET_INDEX(b, f_block*16, z, y, x))"));
                        }
                    } else if (params.inputs[input.index].LogicalSize() == 1) {
                        jit.AddConstant(MakeJitConstant(name,
                                                        "input" + std::to_string(input.index) +
                                                        "[0]"));
                    } else {
                        const std::string idx_order = "INPUT" + std::to_string(input.index) + "_IDX_ORDER";
                        if (DataTensor::ChannelsCount(params.inputs[input_idx].GetLayout()) == 4) {
                            jit.AddConstant(MakeJitConstant(idx_order, "b, f_block*16, y, x"));
                        } else {
                            jit.AddConstant(MakeJitConstant(idx_order, "b, f_block*16, z, y, x"));
                        }
                        bool feature_broadcasting = (params.inputs[input_idx].Feature().v == 1 && params.output.Feature().v != 1);

                        const std::string block_read_str = "TO_TYPE(MAKE_VECTOR_TYPE(ACCUMULATOR_TYPE, BLOCK_SIZE), BLOCK_READN(INPUT" +
                                                                std::to_string(input.index) + "_TYPE, BLOCK_SIZE, " +
                                                                "input" + std::to_string(input.index) + ", " +
                                                                "GET_INDEX(INPUT, " + std::to_string(input.index) + ", " + idx_order + ")))";
                        if (feature_broadcasting) {
                            const std::string broadcast_name = "DO_FEATURE_BROADCAST" + std::to_string(op_num);
                            std::string sub_group_broadcast;
                            if (GetBlockSize(params) == 1) {
                                sub_group_broadcast = "\\\n\ttmp_b" + std::to_string(op_num) +
                                                    " = sub_group_broadcast(tmp_b" + std::to_string(op_num) + ", 0);";
                            } else {
                                sub_group_broadcast = "\\\n\tunroll_for (uint i = 0; i < BLOCK_SIZE; ++i) tmp_b" + std::to_string(op_num) +
                                                    "[i] = sub_group_broadcast(tmp_b" + std::to_string(op_num) + "[i], 0);";
                            }

                            std::string broadcast_value = "\\\n\tMAKE_VECTOR_TYPE(ACCUMULATOR_TYPE, BLOCK_SIZE) tmp_b" + std::to_string(op_num) +
                                                        " = " + block_read_str + ";" + sub_group_broadcast;

                            jit.AddConstant(MakeJitConstant(broadcast_name, broadcast_value));
                            jit.AddConstant(MakeJitConstant(name, "tmp_b" + std::to_string(op_num)));
                        } else {
                            jit.AddConstant(MakeJitConstant(name, block_read_str));
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
                            "input" + std::to_string(input.index) + "[(size_t)tmp" + std::to_string(input.tmpIndex) + "]"));
                    break;
                case EltwiseInputMode::INTERMEDIATE_RESULTS_INDEX:
                    jit.AddConstant(MakeJitConstant(name, "tmp" + std::to_string(input.tmpIndex)));
                    break;
                default:
                    break;
            }
        }
    }

    return jit;
}

JitConstants EltwiseKernel_b_fs_yx_fsv16::GetJitConstants(const eltwise_params& params) const {
    JitConstants jit = MakeBaseParamsJitConstants(params);
    bool useVload8 = false;

    auto blockSize = GetBlockSize(params);
    jit.Merge(MakeTypeJitConstants(GetAccumulatorType(params), "ACCUMULATOR"));
    jit.AddConstant(MakeJitConstant("BLOCK_SIZE", blockSize));
    jit.AddConstant(MakeJitConstant("BLOCKS_COUNT", CeilDiv(params.output.X().v, blockSize)));

    jit.Merge(MakeInputDeclsJitConstants(params, useVload8));
    jit.Merge(MakeLoadJitConstants(params, useVload8));
    jit.Merge(GetOperationsJitConstants(params, useVload8, blockSize));

    std::string do_eltwise;
    auto& operations = params.operations;
    for (size_t op_num = 0; op_num < operations.size(); op_num++) {
        if (OpHasFeatureBroadcast(params, op_num)) {
            do_eltwise += "\\\n\tDO_FEATURE_BROADCAST" + std::to_string(op_num) + ";";
        }
        do_eltwise += "\\\n\tOPERATION" + std::to_string(op_num) + ";";
    }

    do_eltwise += "\\\n\tres = tmp" + std::to_string(operations.size() - 1) + ";";

    jit.AddConstant(MakeJitConstant("DO_ELTWISE", do_eltwise));

    if (params.layoutBased || params.int8_quantization || params.broadcast) {
        jit.Merge(GetTensorFriendlyWorkGroupsJit(params.output));
    }

    if (!params.stride.empty()) {
        jit.AddConstant(MakeJitConstant("INPUT_STRIDED", 1));
    }

    jit.Merge(MakeActivationJitConstants(params.activations, params.output.GetDType(), "_TYPED"));

    if (params.output.Feature().v % 16 != 0)
        jit.AddConstant(MakeJitConstant("LEFTOVERS", params.output.Feature().v % 16));

    if (!params.fused_ops.empty()) {
        kernel_selector::Datatype input_dt = GetAccumulatorType(params);

        std::vector<std::string> idx_order;
        if (DataTensor::ChannelsCount(params.output.GetLayout()) == 4) {
            idx_order = {"b", "f_block*16", "y", "x"};
        } else if (DataTensor::ChannelsCount(params.output.GetLayout()) == 5) {
            idx_order = {"b", "f_block*16", "z", "y", "x"};
        }

        FusedOpsConfiguration conf = {"", idx_order, "res", input_dt, blockSize};
        conf.load_type = FusedOpsConfiguration::LoadType::LT_ALIGNED_READ;
        conf.vec_axis = Tensor::DataChannelName::X;

        jit.Merge(MakeFusedOpsJitConstants(params, {conf}));
    }

    if (params.broadcast) {
        bool need_idx_safe = true;
        for (size_t i = 0; i < params.inputs.size(); i++) {
            if (IsBroadcastingPossibleInput(params.inputs[i], params.output)) {
                    need_idx_safe = false;
                    break;
            }
        }
        if (need_idx_safe)
            jit.AddConstant(MakeJitConstant("ELTWISE_BROADCAST", params.broadcast));
    }

    return jit;
}

bool EltwiseKernel_b_fs_yx_fsv16::Validate(const Params& p, const optional_params& o) const {
    if (!EltwiseKernelBase::Validate(p, o)) {
        return false;
    }

    const auto& params = static_cast<const eltwise_params&>(p);

    const auto count = params.output.PhysicalSize();

    if (count % 8 != 0)
        return false;

    if (IsUnsupportedModeForVecCode(params))
        return false;

    for (size_t i = 0; i < params.inputs.size(); i++) {
        if ((params.inputs[i].GetLayout() != DataLayout::b_fs_yx_fsv16) &&
            (params.inputs[i].GetLayout() != DataLayout::b_fs_zyx_fsv16) &&
            !IsBroadcastingPossibleInput(params.inputs[i], params.output)) {
            return false;
        }
    }

    auto input0 = params.inputs[0];

    // Check that padding before features doesn't miss-align the blocks
    auto feature_block_size = 16;
    if (input0.Feature().pad.before % feature_block_size != 0 || params.output.Feature().pad.before % feature_block_size != 0) {
        return false;
    }

    auto compareTensors = [](const DataTensor& input0, const DataTensor& input1) -> bool {
        // Check all parameters except DataType
        auto& input0_dims = input0.GetDims();
        auto& input1_dims = input1.GetDims();
        bool same = input0.GetLayout() == input1.GetLayout() &&
                    input0.GetPaddedVal() == input1.GetPaddedVal() &&
                    input0.GetViewOffset() == input1.GetViewOffset() &&
                    input0_dims.size() == input1_dims.size();
        if (same) {
            for (size_t i = 0; i < input0_dims.size(); i++) {
                same &= input0_dims[i].v == input1_dims[i].v &&
                        input0_dims[i].pad.before == input1_dims[i].pad.before &&
                        input0_dims[i].pad.after == input1_dims[i].pad.after &&
                        input0_dims[i].pitch == input1_dims[i].pitch;
            }
        }
        return same;
    };

    for (size_t i = 1; i < params.inputs.size(); i++) {
        if (params.inputs[i].LogicalSize() == input0.LogicalSize() && !(compareTensors(params.inputs[i], input0)))
            return false;
        if (params.inputs[i].Feature().pad.before % feature_block_size != 0) {
            return false;
        }
    }

    return true;
}

EltwiseKernelBase::DispatchData EltwiseKernel_b_fs_yx_fsv16::SetDefault(const eltwise_params& params) const {
    DispatchData dispatchData;

    dispatchData.gws[0] = Align(params.output.Feature().v, 16);
    dispatchData.gws[1] = CeilDiv(params.output.X().v, GetBlockSize(params)) * params.output.Y().v * params.output.Z().v;
    dispatchData.gws[2] = params.output.Batch().v;

    dispatchData.lws[0] = 16;
    dispatchData.lws[1] = 16;
    while (dispatchData.lws[1] > 1) {
        if (dispatchData.gws[1] % dispatchData.lws[1] == 0)
            break;
        dispatchData.lws[1]--;
    }
    dispatchData.lws[2] = 1;

    return dispatchData;
}

KernelsPriority EltwiseKernel_b_fs_yx_fsv16::GetKernelsPriority(const Params& /*params*/, const optional_params& /*options*/) const {
    return FORCE_PRIORITY_1;
}

KernelsData EltwiseKernel_b_fs_yx_fsv16::GetKernelsData(const Params& params, const optional_params& options) const {
    if (!Validate(params, options)) {
        return {};
    }

    KernelData kd = KernelData::Default<eltwise_params>(params);
    eltwise_params& newParams = *static_cast<eltwise_params*>(kd.params.get());

    auto entry_point = GetEntryPoint(kernelName, newParams.layerID, options);
    auto cldnn_jit = GetJitConstants(newParams);
    auto jit = CreateJit(kernelName, cldnn_jit, entry_point);

    DispatchData dispatchData = SetDefault(newParams);

    auto& kernel = kd.kernels[0];

    kernel.workGroups.global = dispatchData.gws;
    kernel.workGroups.local = dispatchData.lws;

    kernel.kernelString = GetKernelString(kernelName, jit, entry_point, params.engineInfo, DEFAULT);
    kernel.arguments = GetArgsDesc((uint32_t)newParams.inputs.size(),
                                   false,
                                   false,
                                   GetFusedPrimitiveInputsCount(params));

    return {kd};
}
}  // namespace kernel_selector
