// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "eltwise_kernel_b_fs_yx_fsv4.h"
#include "kernel_selector_utils.h"
#include <algorithm>
#include <string>
#include <vector>

namespace kernel_selector {
static inline bool InputHasFeatureBroadcast(const eltwise_params& params, const size_t op_num, const size_t input_idx);

ParamsKey EltwiseKernel_b_fs_yx_fsv4::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableInputDataType(Datatype::INT8);
    k.EnableInputDataType(Datatype::UINT8);
    k.EnableOutputDataType(Datatype::INT8);
    k.EnableOutputDataType(Datatype::UINT8);
    k.EnableInputLayout(DataLayout::b_fs_yx_fsv4);
    k.EnableOutputLayout(DataLayout::b_fs_yx_fsv4);
    k.EnableDifferentTypes();
    k.EnableBatching();
    k.EnableTensorPitches();
    k.EnableTensorOffset();
    k.EnableEltwiseBroadcast();
    return k;
}

KernelsData EltwiseKernel_b_fs_yx_fsv4::GetKernelsData(const Params& params, const optional_params& options) const {
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

KernelsPriority EltwiseKernel_b_fs_yx_fsv4::GetKernelsPriority(const Params& /*params*/, const optional_params& /*options*/) const {
    return FORCE_PRIORITY_1;
}

// Protected
bool EltwiseKernel_b_fs_yx_fsv4::Validate(const Params& params, const optional_params& o) const {
    if (!EltwiseKernelBase::Validate(params, o)) {
        return false;
    }

    const auto& ewParams = static_cast<const eltwise_params&>(params);

    const auto& output = ewParams.output;
    const auto count = output.PhysicalSize();

    if (count % vec_size != 0)
        return false;

    for (size_t i = 0; i < ewParams.inputs.size(); i++) {
        if ((ewParams.inputs[i].GetLayout() != DataLayout::b_fs_yx_fsv4) &&
            (ewParams.inputs[i].LogicalSize() != 1)) {
            return false;
        }
    }

    auto input0 = ewParams.inputs[0];

    // Check that padding before features doesn't miss-align the blocks
    if (input0.Feature().pad.before % vec_size != 0 || output.Feature().pad.before % vec_size != 0) {
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

    for (size_t i = 1; i < ewParams.inputs.size(); i++) {
        if (ewParams.inputs[i].LogicalSize() == input0.LogicalSize() && !(compareTensors(ewParams.inputs[i], input0)))
            return false;
        if (ewParams.inputs[i].Feature().pad.before % vec_size != 0) {
            return false;
        }
    }

    return true;
}

JitConstants EltwiseKernel_b_fs_yx_fsv4::MakeLoadJitConstants(const eltwise_params& params, bool /*useVload8*/) const {
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
                    const std::string idx_order = "INPUT" + std::to_string(input.index) + "_IDX_ORDER";
                    jit.AddConstant(MakeJitConstant(idx_order, "b, f_block*4, y, x"));

                    if (params.inputs[input.index].LogicalSize() == 1) {
                        const std::string vload_name = "DO_VLOAD" + std::to_string(op_num) + "_" + std::to_string(input_idx);
                        const std::string vload_value = "\\\n\tMAKE_VECTOR_TYPE(ACCUMULATOR_TYPE, 4) tmp_a" + std::to_string(op_num) +
                                                        "_" + std::to_string(input_idx) + " = " "(MAKE_VECTOR_TYPE(ACCUMULATOR_TYPE, 4))" +
                                                        "(input" + std::to_string(input.index) + "[0])";
                        jit.AddConstant(MakeJitConstant(vload_name, vload_value));
                        jit.AddConstant(MakeJitConstant(name, "tmp_a" + std::to_string(op_num) + "_" + std::to_string(input_idx)));
                    } else {
                        bool feature_broadcasting = (params.inputs[input_idx].Feature().v == 1 && params.output.Feature().v != 1);

                        if (feature_broadcasting) {
                            const std::string broadcast_name = "DO_FEATURE_BROADCAST" + std::to_string(op_num) + "_" + std::to_string(input_idx);
                            std::string broadcast_value = "\\\n\tMAKE_VECTOR_TYPE(ACCUMULATOR_TYPE, 4) tmp_b" + std::to_string(op_num) +
                                                        " = " "(MAKE_VECTOR_TYPE(ACCUMULATOR_TYPE, 4))"+"(input" + std::to_string(input.index) +
                                                        "[GET_INDEX(INPUT, " + std::to_string(input.index) + ", " + idx_order + ")]);";

                            jit.AddConstant(MakeJitConstant(broadcast_name, broadcast_value));
                            jit.AddConstant(MakeJitConstant(name, "tmp_b" + std::to_string(op_num)));
                        } else {
                            const std::string vload_name = "DO_VLOAD" + std::to_string(op_num) + "_" + std::to_string(input_idx);
                            const std::string vload_value = "\\\n\tMAKE_VECTOR_TYPE(ACCUMULATOR_TYPE, 4) tmp_a" + std::to_string(op_num) +
                                                            "_" + std::to_string(input_idx) + " = TO_TYPE(MAKE_VECTOR_TYPE(ACCUMULATOR_TYPE, " +
                                                            std::to_string(vec_size) + "), vload4(0, &input" + std::to_string(input.index) +
                                                            "[GET_INDEX(INPUT," + std::to_string(input.index) + ", " + idx_order + ")]));";

                            jit.AddConstant(MakeJitConstant(vload_name, vload_value));
                            jit.AddConstant(MakeJitConstant(name, "tmp_a" + std::to_string(op_num) + "_" + std::to_string(input_idx)));
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

JitConstants EltwiseKernel_b_fs_yx_fsv4::GetJitConstants(const eltwise_params& params) const {
    JitConstants jit = MakeBaseParamsJitConstants(params);
    bool useVload8 = false;

    auto blockSize = vec_size;
    jit.Merge(MakeTypeJitConstants(GetAccumulatorType(params), "ACCUMULATOR"));
    jit.AddConstant(MakeJitConstant("BLOCK_SIZE", blockSize));
    jit.AddConstant(MakeJitConstant("BLOCKS_COUNT", CeilDiv(params.output.X().v, blockSize)));

    jit.Merge(MakeInputDeclsJitConstants(params, useVload8));
    jit.Merge(MakeLoadJitConstants(params, useVload8));
    jit.Merge(GetOperationsJitConstants(params, useVload8, vec_size));

    std::string do_eltwise;
    auto& operations = params.operations;
    for (size_t op_num = 0; op_num < operations.size(); op_num++) {
        const auto &ew = operations[op_num];
        for (size_t input_idx = 0; input_idx < ew.inputs.size(); input_idx++) {
            const auto &input = ew.inputs[input_idx];
            if (input.mode != EltwiseInputMode::INPUT_BUFFER && input.mode != EltwiseInputMode::SCALAR)
                continue;

            if (InputHasFeatureBroadcast(params, op_num, input_idx)) {
                do_eltwise += "\\\n\tDO_FEATURE_BROADCAST" + std::to_string(op_num) + "_" + std::to_string(input_idx) + ";";
            } else {
                do_eltwise += "\\\n\tDO_VLOAD" + std::to_string(op_num) + "_" + std::to_string(input_idx) + ";";
            }
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

    if (params.output.Feature().v % 4 != 0)
        jit.AddConstant(MakeJitConstant("LEFTOVERS", params.output.Feature().v % 4));

    if (!params.fused_ops.empty()) {
        kernel_selector::Datatype input_dt = GetAccumulatorType(params);
        std::vector<std::string> idx_order = {"b", "f_block*4", "y", "x"};
        FusedOpsConfiguration conf = {"", idx_order, "res", input_dt, (size_t)vec_size};
        conf.vec_axis = Tensor::DataChannelName::FEATURE;
        jit.Merge(MakeFusedOpsJitConstants(params, {conf}));
    }

    jit.AddConstant(MakeJitConstant("QUANTIZATION_TERM", params.int8_quantization));
    jit.AddConstant(MakeJitConstant("VEC_SIZE", vec_size));

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

EltwiseKernelBase::DispatchData EltwiseKernel_b_fs_yx_fsv4::SetDefault(const eltwise_params& params) const {
    DispatchData dispatchData;

    dispatchData.gws[0] = params.output.X().v * params.output.Y().v;
    dispatchData.gws[1] = CeilDiv(params.output.Feature().v, 4);
    dispatchData.gws[2] = params.output.Batch().v;

    dispatchData.lws = GetOptimalLocalWorkGroupSizes(dispatchData.gws, params.engineInfo);
    dispatchData.lws[1] = 1;
    dispatchData.lws[2] = 1;

    return dispatchData;
}

// Local
static inline bool InputHasFeatureBroadcast(const eltwise_params& params, const size_t op_num, const size_t input_idx) {
    const auto &ew = params.operations[op_num];

    const auto &input = ew.inputs[input_idx];
    if (input.mode == EltwiseInputMode::INPUT_BUFFER) {
        if (params.inputs[input_idx].LogicalSize() != 1
            && params.inputs[input_idx].Feature().v == 1
            && params.output.Feature().v != 1) {
                return true;
            }
    }

    return false;
}
}  // namespace kernel_selector
