// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <iostream>
#include "binary_convolution_kernel_generic.h"
#include <string>
#include <core/actual_kernels/activation/activation_kernel_base.h>
#include <core/actual_kernels/eltwise/eltwise_kernel_base.h>

namespace kernel_selector {

static const int sub_group_size = 16;
static const int ic_pack_size = 32;
static const int x_block_size = 16;

ParamsKey BinaryConvolutionKernelGeneric::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::BINARY);
    k.EnableInputWeightsType(WeightsType::BINARY);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::INT32);
    k.EnableOutputDataType(Datatype::BINARY);
    k.EnableInputLayout(DataLayout::b_fs_yx_32fp);
    k.EnableOutputLayout(DataLayout::bfyx);
    k.EnableOutputLayout(DataLayout::b_fs_yx_32fp);
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableNonBiasTerm();
    k.EnableBatching();
    k.EnableDifferentTypes();
    return k;
}

BinaryConvolutionKernelBase::DispatchData BinaryConvolutionKernelGeneric::SetDefault(const binary_convolution_params& params,
                                                                                     int) const {
    DispatchData dispatchData = BinaryConvolutionKernelBase::SetDefault(params);

    const auto& out = params.output;

    auto x = out.X().v;
    auto y = out.Y().v;
    auto f = out.Feature().v;
    auto b = out.Batch().v;

    dispatchData.gws[0] = Align(x, sub_group_size) * y;
    dispatchData.gws[1] = CeilDiv(f, 2 * sub_group_size);  // 1 WI calc 2 OC x 16 X
    dispatchData.gws[2] = b;

    dispatchData.lws[0] = sub_group_size;
    dispatchData.lws[1] = 1;
    dispatchData.lws[2] = 1;

    return dispatchData;
}

KernelsPriority BinaryConvolutionKernelGeneric::GetKernelsPriority(const Params& /*params*/, const optional_params& /*options*/) const {
    return FORCE_PRIORITY_2;
}

bool BinaryConvolutionKernelGeneric::Validate(const Params& p, const optional_params& o) const {
    if (!BinaryConvolutionKernelBase::Validate(p, o) || !CovolutionBinaryCheckInput(p, o))
        return false;

    const auto& params = static_cast<const binary_convolution_params&>(p);

    if (params.split > 1 || params.groups > 1 || params.depthwise_separable_opt)
        return false;

    return true;
}

JitConstants BinaryConvolutionKernelGeneric::GetJitConstants(const binary_convolution_params& params,
                                                             const DispatchData& dispatchData) const {
    auto jit = Parent::GetJitConstants(params, dispatchData);

    auto input = params.inputs[0];
    auto output = params.output;
    size_t input_line_size = params.stride.x * (x_block_size - 1) + params.weights.X().v;

    int pad_physical_val = params.pad_value == -1.0f ? 0x00000000 : 0xFFFFFFFF;
    jit.AddConstant(MakeJitConstant("SUB_GROUP_SIZE", sub_group_size));
    jit.AddConstant(MakeJitConstant("INPUT0_FEATURE_NUM_PACKED", CeilDiv(params.inputs[0].Feature().v, ic_pack_size)));
    jit.AddConstant(MakeJitConstant("OUTPUT_FEATURE_NUM_PACKED", CeilDiv(params.output.Feature().v, ic_pack_size)));
    jit.AddConstant(MakeJitConstant("PAD_VALUE", pad_physical_val));
    jit.AddConstant(MakeJitConstant("OUTPUT_X_BLOCK_SIZE", x_block_size));
    jit.AddConstant(MakeJitConstant("INPUT_ELEMENTS_PER_WI", CeilDiv(input_line_size, sub_group_size)));
    jit.AddConstant(MakeJitConstant("X_BLOCKS", CeilDiv(output.X().v, x_block_size)));
    jit.AddConstant(MakeJitConstant("EXCLUDE_PAD", params.pad_value == 0.0f));
    if (params.inputs[0].Feature().v % ic_pack_size) {
        jit.AddConstant(MakeJitConstant("LEFTOVERS_IC", params.inputs[0].Feature().v % ic_pack_size));
        jit.AddConstant(MakeJitConstant("FILTER_MASK",
                                        (0xFFFFFFFF >> (ic_pack_size - params.inputs[0].Feature().v % ic_pack_size))));
    }

    if (params.output.GetDType() == Datatype::BINARY) {
        jit.AddConstant(MakeJitConstant("BINARY_PACKED_OUTPUT", 1));
    }

    return jit;
}

JitConstants BinaryConvolutionKernelGeneric::GetFusedPrimitivesJitConstants(const binary_convolution_params& params,
                                                                            const DispatchData& /*kd*/) const {
    JitConstants jit = {};

    auto input_dt = GetUnitType(params);
    FusedOpsConfiguration conf = {"", {"b", "(f_block*16 + i)", "y", "x"}, "res", input_dt, 1 };
    jit.Merge(MakeFusedOpsDeclsJitConstants(params, {conf}));

    size_t op_id = 0;
    std::string eltwise_fused_ops = "";
    std::string channel_pack_fused_ops = "";
    std::string prepare_data = "";
    for (auto& fused_dep : params.fused_ops) {
        auto fused_dep_codegen = FusedOpsCodeGenerator(fused_dep);
        auto get_aligned_load2 = [&](std::string ptr, std::string byte_offset) -> std::string {
            if (fused_dep.tensors[0].GetDType() == Datatype::F32)
                return "(intel_sub_group_block_read2((const __global uint*)(" + ptr + ") + (" + byte_offset + ")))";
            else
                return "(intel_sub_group_block_read_us2((const __global ushort*)(" + ptr + ") + (" + byte_offset +
                       ")))";
        };
        std::string data_type = fused_dep_codegen.GetInputTypeName(0, 1);
        std::string vec_data_type = fused_dep_codegen.GetInputTypeName(0, 2);
        std::string sc = "sc" + std::to_string(op_id);
        std::string sh = "sh" + std::to_string(op_id);
        std::string e_add = "e_add" + std::to_string(op_id);
        std::string e_mul = "e_mul" + std::to_string(op_id);

        switch (fused_dep.GetType()) {
            case KernelType::SCALE: {
                std::string cast_type = (fused_dep.tensors[0].GetDType() == Datatype::F32) ? "as_float2" : "as_half2";
                if (fused_dep.tensors.size() == 1) {
                    std::string var_name = fused_dep_codegen.GetInputVarName(0);
                    prepare_data += vec_data_type + " " + var_name + " = " + cast_type +
                                    get_aligned_load2(fused_dep_codegen.GetInputPtrName(0), "f_block*OC_BLOCK_SIZE") + ";";
                    eltwise_fused_ops += data_type + " " + sc + " = (i < 16) ? " + var_name + ".s0" + " : " + var_name + ".s1;";
                    eltwise_fused_ops += "res = res*" + sc +";";
                } else {
                    std::string var0_name = fused_dep_codegen.GetInputVarName(0);
                    std::string var1_name = fused_dep_codegen.GetInputVarName(1);
                    prepare_data += vec_data_type + " " + var0_name + " = " + cast_type +
                                    get_aligned_load2(fused_dep_codegen.GetInputPtrName(0), "f_block*OC_BLOCK_SIZE") + ";";
                    prepare_data += vec_data_type + " " + var1_name + " = " + cast_type +
                                    get_aligned_load2(fused_dep_codegen.GetInputPtrName(1), "f_block*OC_BLOCK_SIZE") + ";";
                    eltwise_fused_ops +=
                        data_type + " " + sc + " = (i < 16) ? " + var0_name + ".s0" + " : " + var0_name + ".s1;";
                    eltwise_fused_ops +=
                        data_type + " " + sh + " = (i < 16) ? " + var1_name + ".s0" + " : " + var1_name + ".s1;";
                    eltwise_fused_ops += "res = res*" + sc + " + " + sh + ";";
                }

                break;
            }

            case KernelType::QUANTIZE: {
                std::string var_name_in = fused_dep_codegen.GetInputVarName(0);
                std::string var_name_out = fused_dep_codegen.GetInputVarName(3);
                std::string cast_type_vec = (fused_dep.tensors[0].GetDType() == Datatype::F32) ? "as_float2" : "as_half2";
                std::string cast_type = (fused_dep.tensors[0].GetDType() == Datatype::F32) ? "as_float" : "as_half";

                if (fused_dep.tensors[0].Feature().v == params.output.Feature().v) {
                    prepare_data += vec_data_type + " " + var_name_in + " = " + cast_type_vec +
                                    get_aligned_load2(fused_dep_codegen.GetInputPtrName(0), "f_block*OC_BLOCK_SIZE") + ";";
                } else {
                    prepare_data += data_type + " " + var_name_in + " = " + cast_type +
                                 + "(" + fused_dep_codegen.GetInputPtrName(0) + "[0]);";
                }

                if (fused_dep.tensors[2].Feature().v == params.output.Feature().v) {
                    prepare_data += vec_data_type + " " + var_name_out + " = " + cast_type_vec +
                                    get_aligned_load2(fused_dep_codegen.GetInputPtrName(3), "f_block*OC_BLOCK_SIZE") + ";";
                } else {
                    prepare_data += data_type + " " + var_name_out + " = " + cast_type +
                                    "(" + fused_dep_codegen.GetInputPtrName(3)+"[0]);";
                }

                std::string var_in_s0 = fused_dep.tensors[0].Feature().v == params.output.Feature().v ? var_name_in + ".s0" : var_name_in;
                std::string var_in_s1 = fused_dep.tensors[0].Feature().v == params.output.Feature().v ? var_name_in + ".s1" : var_name_in;

                std::string var_out_s0 = fused_dep.tensors[3].Feature().v == params.output.Feature().v ? var_name_out + ".s0" : var_name_out;
                std::string var_out_s1 = fused_dep.tensors[3].Feature().v == params.output.Feature().v ? var_name_out + ".s1" : var_name_out;

                channel_pack_fused_ops += "\\\n\tfor (int i = 0; i < 16; i++) {";
                channel_pack_fused_ops += "\\\n\tint ch0, ch1;";
                if (fused_dep.tensors[2].Feature().v == params.output.Feature().v) {
                    channel_pack_fused_ops += "\\\n\tif ("+ var_out_s0 + " == UNIT_VAL_ONE) ";
                    channel_pack_fused_ops += "\\\n\t\tch0 = dst[0*SUB_GROUP_SIZE + i] > " + var_in_s0 + " ? (1 << lid) : 0;";
                    channel_pack_fused_ops += "\\\n\telse ";
                    channel_pack_fused_ops += "\\\n\t\tch0 = dst[0*SUB_GROUP_SIZE + i] <= " + var_in_s0 + " ? (1 << lid) : 0;";
                    channel_pack_fused_ops += "\\\n\tif ("+ var_out_s1 + " == UNIT_VAL_ONE) ";
                    channel_pack_fused_ops += "\\\n\t\tch1 = dst[1*SUB_GROUP_SIZE + i] > " + var_in_s1 + " ? "
                                                         "(1 << (SUB_GROUP_SIZE + lid)) : 0;";
                    channel_pack_fused_ops += "\\\n\telse ";
                    channel_pack_fused_ops += "\\\n\t\tch1 = dst[1*SUB_GROUP_SIZE + i] <= " + var_in_s1 + " ? "
                                                         "(1 << (SUB_GROUP_SIZE + lid)) : 0;";
                } else {
                    channel_pack_fused_ops += "\\\n\tif ("+ var_out_s0 + " == UNIT_VAL_ONE) {";
                    channel_pack_fused_ops += "\\\n\t\tch0 = dst[0*SUB_GROUP_SIZE + i] > " + var_in_s0 + " ? (1 << lid) : 0;";
                    channel_pack_fused_ops += "\\\n\t\tch1 = dst[1*SUB_GROUP_SIZE + i] > " + var_in_s1 + " ? "
                                                         "(1 << (SUB_GROUP_SIZE + lid)) : 0;";
                    channel_pack_fused_ops += "\\\n\t} else {";
                    channel_pack_fused_ops += "\\\n\t\tch0 = dst[0*SUB_GROUP_SIZE + i] <= " + var_in_s0 + " ? (1 << lid) : 0;";
                    channel_pack_fused_ops += "\\\n\t\tch1 = dst[1*SUB_GROUP_SIZE + i] <= " + var_in_s1 + " ? "
                                                                                                       "(1 << (SUB_GROUP_SIZE + lid)) : 0;";
                    channel_pack_fused_ops += "\\\n\t}";
                }
                channel_pack_fused_ops += "\\\n\tint packed = ch0 + ch1;";
                channel_pack_fused_ops += "\\\n\tpacked_out[i] = sub_group_reduce_add(packed);";
                channel_pack_fused_ops += "\\\n\t}";

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
                std::string cast_type = (fused_dep.tensors[0].GetDType() == Datatype::F32) ? "as_float2" : "as_half2";
                std::string var_name = fused_dep_codegen.GetInputVarName(0);
                prepare_data += vec_data_type + " " + var_name + " = " + cast_type +
                                get_aligned_load2(fused_dep_codegen.GetInputPtrName(0), "f_block*OC_BLOCK_SIZE") + ";";

                auto eltwise_p = std::dynamic_pointer_cast<eltwise_fuse_params>(fused_dep.op_params);

                if (eltwise_p->mode == EltwiseMode::ADD) {
                    eltwise_fused_ops += data_type + " " + e_add + " = (i < 16) ? " + var_name + ".s0" + " : " + var_name + ".s1;";
                    eltwise_fused_ops += "res = res+" + e_add +";";
                } else {
                    eltwise_fused_ops += data_type + " " + e_mul + " = (i < 16) ? " + var_name + ".s0" + " : " + var_name + ".s1;";
                    eltwise_fused_ops += "res = res*" + e_mul +";";
                }

                break;
            }

            default:
                throw std::invalid_argument("Invalid fused op in binary_convolution_generic kernel: " + params.layerID);
        }

        op_id++;
    }
    jit.AddConstant(MakeJitConstant("DO_ELTWISE_FUSED_OPS", eltwise_fused_ops));
    jit.AddConstant(MakeJitConstant("DO_CHANNEL_PACK_OPS", channel_pack_fused_ops));
    jit.AddConstant(MakeJitConstant("FUSED_OPS_PREPARE_DATA", prepare_data));
    jit.AddConstant(MakeJitConstant("CUSTOM_FUSED_OPS", true));

    return jit;
}

KernelsData BinaryConvolutionKernelGeneric::GetKernelsData(const Params& params, const optional_params& options) const {
    return GetTunedKernelsDataByIndex(params, options);
}
}  // namespace kernel_selector
