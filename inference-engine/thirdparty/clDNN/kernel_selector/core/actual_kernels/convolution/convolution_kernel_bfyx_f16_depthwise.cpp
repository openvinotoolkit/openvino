// Copyright (c) 2018-2019 Intel Corporation
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


#include <iostream>
#include "convolution_kernel_bfyx_f16_depthwise.h"
#include "kernel_selector_utils.h"
#include <string>

namespace kernel_selector {
static const size_t sub_group_size = 16;
static const size_t feature_block_size = 16;

ParamsKey ConvolutionKernel_bfyx_f16_depthwise::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputWeightsType(WeightsType::F16);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableInputWeightsType(WeightsType::F32);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableInputLayout(DataLayout::bfyx_f16);
    k.EnableOutputLayout(DataLayout::bfyx_f16);
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBiasPerFeature();
    k.EnableNonBiasTerm();
    k.EnableBatching();
    k.EnableGroupedConvolution();
    k.EnableSubGroup();
    k.EnableSubGroupShort();
    k.EnableDepthwiseSeparableOpt();
    return k;
}

bool ConvolutionKernel_bfyx_f16_depthwise::Validate(const Params& p, const optional_params&) const {
    const convolution_params& cp = static_cast<const convolution_params&>(p);
    if (!cp.depthwise_separable_opt || (cp.inputs[0].Feature().v != cp.split && cp.inputs[0].Feature().v != cp.groups))
        return false;

    if (cp.filterSize.x != 3 || cp.filterSize.y != 3 || cp.inputs[0].Batch().v != 1)
        return false;

    if (cp.stride.x != 1 && cp.stride.x != 2)
        return false;

    return true;
}

ConvolutionKernelBase::DispatchData ConvolutionKernel_bfyx_f16_depthwise::SetDefault(const convolution_params& params,
                                                                                     int) const {
    DispatchData runInfo = Parent::SetDefault(params);
    const auto& out = params.output;

    runInfo.gws0 = CeilDiv(out.X().v, 8) * out.Y().v;
    runInfo.gws1 = Align(out.Feature().v, feature_block_size);
    runInfo.gws2 = out.Batch().v;
    runInfo.lws0 = 1;
    runInfo.lws1 = sub_group_size;
    runInfo.lws2 = 1;

    if (out.Batch().v == 1)
        runInfo.effiency = FORCE_PRIORITY_1;
    else
        runInfo.effiency = FORCE_PRIORITY_7;

    return runInfo;
}

JitConstants ConvolutionKernel_bfyx_f16_depthwise::GetJitConstants(const convolution_params& params,
                                                                   const DispatchData& kd) const {
    auto jit = ConvolutionKernelBase::GetJitConstants(params, kd);

    jit.AddConstant(MakeJitConstant("SUB_GROUP_SIZE", sub_group_size));
    jit.AddConstant(MakeJitConstant("X_BLOCKS", CeilDiv(params.output.X().v, 8)));
    jit.AddConstant(MakeJitConstant("IC_BLOCK", feature_block_size));
    if (params.output.Feature().v % feature_block_size != 0) {
        jit.AddConstant(MakeJitConstant("OUTPUT_LEFTOVERS", 1));
    }

    return jit;
}

KernelsData ConvolutionKernel_bfyx_f16_depthwise::GetKernelsData(const Params& params,
                                                                 const optional_params& options) const {
    return GetCommonKernelsData(params, options);
}

JitConstants ConvolutionKernel_bfyx_f16_depthwise::GetFusedPrimitivesJitConstants(const convolution_params& params,
                                                                                  const DispatchData& /*kd*/) const {
    JitConstants jit = {};

    size_t op_id = 0;
    std::string input_decls = "";
    std::string load_decls_vec = "";
    std::string load_decls = "";
    std::string eltwise_fused_ops_vec = "";
    std::string eltwise_fused_ops = "";

    auto make_jit_vector_type = [](std::string tensor_name, size_t vec_size) -> std::string {
        if (vec_size == 0 || vec_size > 8)
            throw std::invalid_argument("Invalid vector size in jit definitions");
        if (vec_size > 1)
            return "MAKE_VECTOR_TYPE(" + tensor_name + "_TYPE," + std::to_string(vec_size) + ")";
        else
            return tensor_name + "_TYPE";
    };

    auto make_jit_load = [](std::string tensor_name, std::string ptr_name, size_t vec_size) -> std::string {
        if (vec_size == 0 || vec_size > 8)
            throw std::invalid_argument("Invalid vector size in jit definitions");

        std::string index_func_call_vec = tensor_name + "_GET_INDEX(b, f_block*16, y, x)";
        std::string index_func_call = tensor_name + "_GET_INDEX(b, f_block*16, y, (x+i))";
        if (vec_size > 1)
            return " UNIT_BLOCK_READ" + std::to_string(vec_size) + "(" + ptr_name + ", " + index_func_call_vec + ")";
        else
            return " UNIT_BLOCK_READ(" + ptr_name + ", " + index_func_call + ")";
    };

    for (auto& fused_dep : params.fused_ops) {
        std::string op_type = "";
        switch (fused_dep.type) {
            case convolution_params::fused_operation_desc::Type::ELTWISE: {
                op_type = "eltwise";
                eltwise_fused_ops_vec += "dst = (dst + " + op_type + "_data);";
                eltwise_fused_ops += "dst[i] = (dst[i] + " + op_type + "_data);";
                break;
            }
            default:
                throw std::invalid_argument("Invalid fused op in convolution kernel: " + params.layerID);
        }

        for (size_t op_input_id = 0; op_input_id < fused_dep.tensors.size(); op_input_id++) {
            std::string name = "FUSED_OP_" + std::to_string(op_id) + "_INPUT" + std::to_string(op_input_id);
            std::string ptr_name = op_type + "_input" + std::to_string(op_input_id);

            std::string var_name = op_type + "_data";
            jit.AddConstant(MakeJitConstant(name, fused_dep.tensors[op_input_id]));
            input_decls += "const __global " + toCLType(fused_dep.tensors[op_input_id].GetDType()) +
                           "* " + ptr_name + ",";
            load_decls_vec += make_jit_vector_type(name, 8) + " " + var_name + " = " +
                              make_jit_load(name, ptr_name, 8) + ";";
            load_decls += make_jit_vector_type(name, 1) + " " + var_name + " = " +
                          make_jit_load(name, ptr_name, 1) + ";";
        }

        if (fused_dep.activation.function != ActivationFunction::NONE) {
            std::string temp_op_type = op_type;
            for (auto& ch : temp_op_type)
                ch = static_cast<char>(std::toupper(static_cast<unsigned char>(ch)));
            std::string suffix = "_" + temp_op_type;

            jit.Merge(MakeActivationJitConstants(fused_dep.activation, suffix));
            eltwise_fused_ops_vec += "dst = ACTIVATION"+suffix+"(dst, ACTIVATION_PARAMS"+suffix+");";
            eltwise_fused_ops += "dst[i] = ACTIVATION"+suffix+"(dst[i], ACTIVATION_PARAMS"+suffix+");";
        }
        op_id++;
    }
    jit.AddConstant(MakeJitConstant("FUSED_OPS_DECLS", input_decls));
    jit.AddConstant(MakeJitConstant("FUSED_OPS_LOAD_DATA_VEC", load_decls_vec));
    jit.AddConstant(MakeJitConstant("FUSED_OPS_LOAD_DATA", load_decls));
    jit.AddConstant(MakeJitConstant("DO_ELTWISE_FUSED_OPS_VEC", eltwise_fused_ops_vec));
    jit.AddConstant(MakeJitConstant("DO_ELTWISE_FUSED_OPS", eltwise_fused_ops));

    return jit;
}
}  // namespace kernel_selector
