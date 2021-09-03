// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <iostream>
#include "binary_convolution_kernel_1x1_b_fs_yx_fsv16.h"
#include <string>
#include <core/actual_kernels/activation/activation_kernel_base.h>
#include <core/actual_kernels/eltwise/eltwise_kernel_base.h>

namespace kernel_selector {

static const int sub_group_size = 16;
static const int ic_pack_size = 32;
static const int xy_block_size = 16;

ParamsKey BinaryConvolutionKernel1x1_b_fs_yx_fsv16::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::BINARY);
    k.EnableInputWeightsType(WeightsType::BINARY);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::UINT8);
    k.EnableOutputDataType(Datatype::INT8);
    k.EnableOutputDataType(Datatype::INT32);
    k.EnableInputLayout(DataLayout::b_fs_yx_32fp);
    k.EnableOutputLayout(DataLayout::b_fs_yx_fsv16);
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableNonBiasTerm();
    k.EnableBatching();
    k.EnableDifferentTypes();
    return k;
}

BinaryConvolutionKernelBase::DispatchData BinaryConvolutionKernel1x1_b_fs_yx_fsv16::SetDefault(
    const binary_convolution_params& params,
    int) const {
    DispatchData dispatchData = BinaryConvolutionKernelBase::SetDefault(params);

    const auto& out = params.output;

    auto x = out.X().v;
    auto y = out.Y().v;
    auto f = out.Feature().v;
    auto b = out.Batch().v;

    dispatchData.gws[0] = Align(x * y, sub_group_size);
    dispatchData.gws[1] = CeilDiv(f, sub_group_size);  // 1 WI calcs 16 OC
    dispatchData.gws[2] = b;

    dispatchData.lws = { static_cast<size_t>(sub_group_size), 1, 1 };

    return dispatchData;
}

KernelsPriority BinaryConvolutionKernel1x1_b_fs_yx_fsv16::GetKernelsPriority(const Params& /*params*/, const optional_params& /*options*/) const {
    return FORCE_PRIORITY_1;
}

bool BinaryConvolutionKernel1x1_b_fs_yx_fsv16::Validate(const Params& p, const optional_params& o) const {
    if (!BinaryConvolutionKernelBase::Validate(p, o) || !CovolutionBinaryCheckInput(p, o))
        return false;

    const auto& params = static_cast<const binary_convolution_params&>(p);

    const auto& input = params.inputs[0];
    const auto& output = params.output;

    const bool bOutputSizes = output.X().v != input.X().v || output.Y().v != input.Y().v;
    const bool bFilterSize = params.filterSize.x != 1 || params.filterSize.y != 1;
    const bool bStride = params.stride.x != 1 || params.stride.y != 1;
    const bool bGroups = params.split > 1 || params.groups > 1 || params.depthwise_separable_opt;

    if (bOutputSizes || bFilterSize || bStride || bGroups)
        return false;

    return true;
}

JitConstants BinaryConvolutionKernel1x1_b_fs_yx_fsv16::GetJitConstants(const binary_convolution_params& params,
                                                                       const DispatchData& dispatchData) const {
    auto jit = Parent::GetJitConstants(params, dispatchData);

    jit.AddConstant(MakeJitConstant("SUB_GROUP_SIZE", sub_group_size));
    jit.AddConstant(MakeJitConstant("INPUT0_FEATURE_NUM_PACKED", CeilDiv(params.inputs[0].Feature().v, ic_pack_size)));
    jit.AddConstant(MakeJitConstant("OUTPUT_FEATURE_NUM_PACKED", CeilDiv(params.output.Feature().v, ic_pack_size)));
    jit.AddConstant(MakeJitConstant("PADDED_INPUT", params.inputs[0].X().pad.Total() != 0));
    jit.AddConstant(MakeJitConstant("PADDED_OUTPUT", params.output.X().pad.Total() != 0));
    jit.AddConstant(MakeJitConstant("XY_BLOCK_SIZE", xy_block_size));
    if (params.inputs[0].Feature().v % ic_pack_size) {
        jit.AddConstant(MakeJitConstant("LEFTOVERS_IC", params.inputs[0].Feature().v % ic_pack_size));
        jit.AddConstant(MakeJitConstant("FILTER_MASK",
                                        (0xFFFFFFFF >> (ic_pack_size - params.inputs[0].Feature().v % ic_pack_size))));
    }

    if (params.output.Feature().v % 32 != 0) {
        jit.AddConstant(MakeJitConstant("LEFTOVERS_OC", true));
    }

    if (params.output.GetDType() == Datatype::BINARY) {
        jit.AddConstant(MakeJitConstant("BINARY_PACKED_OUTPUT", 1));
    }

    return jit;
}

JitConstants BinaryConvolutionKernel1x1_b_fs_yx_fsv16::GetFusedPrimitivesJitConstants(const binary_convolution_params& params,
                                                                                      const DispatchData& /*kd*/) const {
    JitConstants jit = {};

    auto input_dt = GetUnitType(params);
    FusedOpsConfiguration conf = {"", {"b", "(f_block*16 + oc)", "y", "x"}, "res", input_dt, 1 };
    jit.Merge(MakeFusedOpsDeclsJitConstants(params, {conf}));

    size_t op_id = 0;
    std::string input_decls = "";
    std::string eltwise_fused_ops = "";
    std::string prepare_data = "";
    for (auto& fused_dep : params.fused_ops) {
        auto fused_dep_codegen = FusedOpsCodeGenerator(fused_dep);

        auto get_aligned_load = [&](std::string ptr, std::string byte_offset) -> std::string {
            if (fused_dep.tensors[0].GetDType() == Datatype::F32)
                return "(intel_sub_group_block_read((const __global uint*)(" + ptr + ") + (" + byte_offset + ")))";
            else
                return "(intel_sub_group_block_read_us((const __global ushort*)(" + ptr + ") + (" + byte_offset +
                       ")))";
        };

        auto get_shuffle = [&](std::string var, std::string lid) -> std::string {
            return "(intel_sub_group_shuffle(" + var + ", " + lid + "))";
        };

        std::string data_type = fused_dep_codegen.GetInputTypeName(0, 1);
        std::string vec_data_type = fused_dep_codegen.GetInputTypeName(0, 1);
        std::string sc = "sc" + std::to_string(op_id);
        std::string e_add = "e_add" + std::to_string(op_id);
        std::string e_mul = "e_mul" + std::to_string(op_id);

        switch (fused_dep.GetType()) {
            case KernelType::SCALE: {
                std::string cast_type = (fused_dep.tensors[0].GetDType() == Datatype::F32) ? "as_float" : "as_half";
                if (fused_dep.tensors.size() == 1) {
                    std::string var_name = fused_dep_codegen.GetInputVarName(0);
                    prepare_data += "\\\n\t" + vec_data_type + " " + var_name + " = " + cast_type +
                                    get_aligned_load(fused_dep_codegen.GetInputPtrName(0), "f_block*OC_BLOCK_SIZE") + ";";
                    eltwise_fused_ops += "\\\n\t" + data_type + " " + sc + " = " + get_shuffle(var_name, "oc") + ";";
                    eltwise_fused_ops += "\\\n\tres = res*" + var_name + ";";
                } else {
                    std::string var0_name = fused_dep_codegen.GetInputVarName(0);
                    std::string var1_name = fused_dep_codegen.GetInputVarName(1);
                    prepare_data += "\\\n\t" + vec_data_type + " " + var0_name + " = " + cast_type +
                                    get_aligned_load(fused_dep_codegen.GetInputPtrName(0), "f_block*OC_BLOCK_SIZE") + ";";
                    prepare_data += "\\\n\t" + vec_data_type + " " + var1_name + " = " + cast_type +
                                    get_aligned_load(fused_dep_codegen.GetInputPtrName(1), "f_block*OC_BLOCK_SIZE") + ";";
                    eltwise_fused_ops += "\\\n\tres = res*" + var0_name + " + " + var1_name + ";";
                }

                break;
            }

            case KernelType::ACTIVATION: {
                auto p = fused_dep.GetOpParams<activation_fuse_params>();
                base_activation_params activation = p->param;
                if (activation.function != ActivationFunction::NONE) {
                    auto suffix = "_FUSED_OP" + std::to_string(op_id);

                    jit.Merge(MakeActivationJitConstants(activation, fused_dep.output_tensor.GetDType(), suffix));
                    eltwise_fused_ops += "\\\n\tres = ACTIVATION" + suffix + "((OUTPUT_TYPE)res, ACTIVATION_PARAMS" + suffix + ");";
                }

                break;
            }

            case KernelType::ELTWISE: {
                std::string cast_type = (fused_dep.tensors[0].GetDType() == Datatype::F32) ? "as_float" : "as_half";
                std::string var_name = fused_dep_codegen.GetInputVarName(0);
                prepare_data += "\\\n\t" + vec_data_type + " " + var_name + " = " + cast_type +
                                get_aligned_load(fused_dep_codegen.GetInputPtrName(0), "f_block*OC_BLOCK_SIZE") + ";";

                auto eltwise_p = std::dynamic_pointer_cast<eltwise_fuse_params>(fused_dep.op_params);

                if (eltwise_p->mode == EltwiseMode::ADD) {
                    eltwise_fused_ops += "\\\n\t" + data_type + " " + e_add + " = " + get_shuffle(var_name, "oc") + ";";
                    eltwise_fused_ops += "\\\n\tres = res+" + var_name + ";";
                } else {
                    eltwise_fused_ops += "\\\n\t" + data_type + " " + e_mul + " = " + get_shuffle(var_name, "oc") + ";";
                    eltwise_fused_ops += "\\\n\tres = res*" + var_name + ";";
                }

                break;
            }

            default:
                throw std::invalid_argument("Invalid fused op in binary_convolution_1x1_fsv16 kernel: " + params.layerID);
        }

        op_id++;
    }
    jit.AddConstant(MakeJitConstant("DO_ELTWISE_FUSED_OPS", eltwise_fused_ops));
    jit.AddConstant(MakeJitConstant("FUSED_OPS_PREPARE_DATA", prepare_data));
    jit.AddConstant(MakeJitConstant("CUSTOM_FUSED_OPS", true));

    return jit;
}

KernelsData BinaryConvolutionKernel1x1_b_fs_yx_fsv16::GetKernelsData(const Params& params, const optional_params& options) const {
    return GetTunedKernelsDataByIndex(params, options);
}
}  // namespace kernel_selector
