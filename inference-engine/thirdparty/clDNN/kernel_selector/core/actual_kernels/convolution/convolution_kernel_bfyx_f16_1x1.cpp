// Copyright (c) 2016-2019 Intel Corporation
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
#include "convolution_kernel_bfyx_f16_1x1.h"
#include "kernel_selector_utils.h"
#include <string>

namespace kernel_selector {
static const size_t sub_group_size = 16;
static const size_t feature_block_size = 16;

ConvolutionKernel_bfyx_f16_1x1::ConvolutionKernel_bfyx_f16_1x1() : ConvolutionKernelBase("convolution_gpu_bfyx_f16_1x1") {
    std::vector<size_t> outputBlockWidths = {2, 4, 8};
    std::vector<std::string> executionModes = ConvolutionKernelBase::autoTuneOptions;

    for (auto w : outputBlockWidths) {
        for (auto exeMode : executionModes) {
            autoTuneOptions.emplace_back(AutoTuneOption{w, exeMode});
        }
    }
}

ConvolutionKernel_bfyx_f16_1x1::AutoTuneOption ConvolutionKernel_bfyx_f16_1x1::GetAutoTuneOptions(const Params& params,
                                                                                          int /*autoTuneIndex*/) const {
    const convolution_params& cp = static_cast<const convolution_params&>(params);

    if (cp.output.X().v*cp.output.Y().v > 4)
        return {8, DEFAULT};
    else
        return {2, DEFAULT};
}


ParamsKey ConvolutionKernel_bfyx_f16_1x1::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableInputWeightsType(WeightsType::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableInputWeightsType(WeightsType::F32);
    k.EnableInputLayout(DataLayout::bfyx_f16);
    k.EnableOutputLayout(DataLayout::bfyx_f16);
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBiasPerFeature();
    k.EnableNonBiasTerm();
    k.EnableBatching();
    k.EnableSubGroup();
    k.EnableSubGroupShort();
    return k;
}

ConvolutionKernelBase::DispatchData ConvolutionKernel_bfyx_f16_1x1::SetDefault(const convolution_params& params,
                                                                               int autoTuneIndex) const {
    DispatchData kd = ConvolutionKernelBase::SetDefault(params);

    auto autoTune = GetAutoTuneOptions(params, autoTuneIndex);
    kd.cldnnStyle.blockWidth = autoTune.blockWidth;

    const auto& out = params.output;
    auto x = out.X().v;
    auto y = out.Y().v;
    auto f = out.Feature().v;
    auto b = out.Batch().v;

    kd.gws0 = CeilDiv(x * y, autoTune.blockWidth);
    kd.gws1 = Align(f, feature_block_size);
    kd.gws2 = b;

    kd.lws0 = 1;
    kd.lws1 = sub_group_size;
    kd.lws2 = 1;

    if (b == 1)
        kd.effiency = FORCE_PRIORITY_1;
    else
        kd.effiency = FORCE_PRIORITY_7;

    return kd;
}

bool ConvolutionKernel_bfyx_f16_1x1::Validate(const Params& p, const optional_params& o) const {
    if (!ConvolutionKernelBase::Validate(p, o)) {
        return false;
    }

    const auto& params = static_cast<const convolution_params&>(p);

    const auto& input = params.inputs[0];
    const auto& output = params.output;

    const bool bOutputSizes =
        output.X().v != input.X().v || output.Y().v != input.Y().v || output.Feature().v % 16 != 0;
    const bool bFilterSize = params.filterSize.x != 1 || params.filterSize.y != 1;
    const bool bStride = params.stride.x != 1 || params.stride.y != 1;

    if  (bOutputSizes || bFilterSize || bStride) {
        return false;
    }

    return true;
}

JitConstants ConvolutionKernel_bfyx_f16_1x1::GetJitConstants(const convolution_params& params,
                                                             const DispatchData& runInfo) const {
    auto jit = Parent::GetJitConstants(params, runInfo);

    auto blockWidth = runInfo.cldnnStyle.blockWidth;

    jit.AddConstant(MakeJitConstant("SUB_GROUP_SIZE", sub_group_size));
    jit.AddConstant(MakeJitConstant("PADDED_INPUT", params.inputs[0].X().pad.Total() != 0));
    jit.AddConstant(MakeJitConstant("PADDED_OUTPUT", params.output.X().pad.Total() != 0));

    jit.AddConstant(MakeJitConstant("X_BLOCK_SIZE", blockWidth));
    jit.AddConstant(MakeJitConstant("X_BLOCKS", CeilDiv(params.output.X().v, blockWidth)));
    jit.AddConstant(MakeJitConstant("IC_BLOCKS", CeilDiv(params.inputs[0].Feature().v, feature_block_size)));
    if (params.output.Feature().v % feature_block_size != 0) {
        jit.AddConstant(MakeJitConstant("OUTPUT_LEFTOVERS", 1));
    }
    if (params.inputs[0].Feature().v % feature_block_size != 0) {
        jit.AddConstant(MakeJitConstant("INPUT_LEFTOVERS", 1));
    }

    return jit;
}

JitConstants ConvolutionKernel_bfyx_f16_1x1::GetFusedPrimitivesJitConstants(const convolution_params& params,
                                                                            const DispatchData& kd) const {
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
        std::string index_func_call = tensor_name + "_GET_INDEX(b, f_block*16, yi, xi)";
        if (vec_size > 1)
            return " UNIT_BLOCK_READ" + std::to_string(vec_size) + "(" + ptr_name + ", " + index_func_call_vec + ")";
        else
            return " UNIT_BLOCK_READ(" + ptr_name + ", " + index_func_call + ")";
    };

    for (auto& fused_dep : params.fused_ops) {
        std::string op_type = "";
        switch (fused_dep.type) {
            case convolution_params::fused_operation_desc::Type::ELTWISE: {
                op_type = "eltwise" + std::to_string(op_id);
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
            load_decls_vec += make_jit_vector_type(name, kd.cldnnStyle.blockWidth) + " " + var_name + " = " +
                              make_jit_load(name, ptr_name, kd.cldnnStyle.blockWidth) + ";";
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

KernelsData ConvolutionKernel_bfyx_f16_1x1::GetKernelsData(const Params& params, const optional_params& options) const {
    return GetCommonKernelsData(params, options, DEFAULT, -1);
}
}  // namespace kernel_selector
