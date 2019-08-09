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


#include "convolution_kernel_bfyx_to_bfyx_f16.h"
#include "kernel_selector_utils.h"
#include <vector>
#include <algorithm>

namespace kernel_selector {

static const size_t sub_group_size = 16;
static const size_t feature_block_size = 16;

ConvolutionKernel_bfyx_to_bfyx_f16::ConvolutionKernel_bfyx_to_bfyx_f16()
    : ConvolutionKernelBase("convolution_gpu_bfyx_to_bfyx_f16") {
    std::vector<size_t> outputBlockWidths = {2, 4, 8};
    std::vector<std::string> executionModes = ConvolutionKernelBase::autoTuneOptions;

    for (auto w : outputBlockWidths) {
        for (auto exeMode : executionModes) {
            autoTuneOptions.emplace_back(AutoTuneOption{w, exeMode});
        }
    }
}

ConvolutionKernel_bfyx_to_bfyx_f16::AutoTuneOption ConvolutionKernel_bfyx_to_bfyx_f16::GetAutoTuneOptions(
    const Params& /* arg*/,
    int autoTuneIndex) const {
    if (autoTuneIndex >= 0 && autoTuneIndex < static_cast<int>(autoTuneOptions.size()))
        return autoTuneOptions[autoTuneIndex];

    return {8, AGE_BASED};
}

ParamsKey ConvolutionKernel_bfyx_to_bfyx_f16::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableInputWeightsType(WeightsType::F16);
    k.EnableInputWeightsType(WeightsType::F32);
    k.EnableInputLayout(DataLayout::bfyx);
    k.EnableOutputLayout(DataLayout::bfyx_f16);
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    // TODO Add dilation support to kernel
    // k.EnableDilation();
    k.EnableBiasPerFeature();
    // TODO Add bias per output support to kernel
    // k.EnableBiasPerOutput();
    k.EnableNonBiasTerm();
    k.EnableSplitSupport();
    k.EnableDepthwiseSeparableOpt();
    k.EnableBatching();
    k.EnableSubGroup();
    k.EnableSubGroupShort();
    return k;
}

ConvolutionKernelBase::DispatchData ConvolutionKernel_bfyx_to_bfyx_f16::SetDefault(const convolution_params& params,
                                                                                   int autoTuneIndex) const {
    DispatchData kd = ConvolutionKernelBase::SetDefault(params);

    const auto& out = params.output;

    auto autoTune = GetAutoTuneOptions(params, autoTuneIndex);
    kd.cldnnStyle.blockWidth = autoTune.blockWidth;

    auto x = out.X().v;
    auto y = out.Y().v;
    auto f = out.Feature().v;
    auto b = out.Batch().v;

    kd.gws0 = CeilDiv(x, autoTune.blockWidth) * y;
    kd.gws1 = Align(f, sub_group_size);
    kd.gws2 = b;

    kd.lws0 = 1;
    kd.lws1 = sub_group_size;
    kd.lws2 = 1;

    if (b == 1)
        kd.effiency = FORCE_PRIORITY_2;
    else
        kd.effiency = FORCE_PRIORITY_7;

    return kd;
}

bool ConvolutionKernel_bfyx_to_bfyx_f16::Validate(const Params& p, const optional_params& o) const {
    if (!ConvolutionKernelBase::Validate(p, o)) {
        return false;
    }

    const auto& params = static_cast<const convolution_params&>(p);

    const auto& input = params.inputs[0];
    const auto& output = params.output;

    // TODO Add support for different input features number in kernel
    if (input.Feature().v != 3) {
        return false;
    }

    // Check that padding before features doesn't miss-align the blocks
    if (input.Feature().pad.before % feature_block_size != 0 || output.Feature().pad.before % feature_block_size != 0) {
        return false;
    }

    return true;
}

JitConstants ConvolutionKernel_bfyx_to_bfyx_f16::GetJitConstants(const convolution_params& params,
                                                                 const DispatchData& runInfo) const {
    auto input = params.inputs[0];
    auto output = params.output;
    auto jit = Parent::GetJitConstants(params, runInfo);

    auto blockWidth = runInfo.cldnnStyle.blockWidth;

    size_t input_line_size = std::min(params.stride.x * (blockWidth - 1) + (params.weights.X().v - 1)*params.dilation.x + 1,
                                      input.X().v + input.X().pad.Total());
    size_t input_block_size = CeilDiv(input_line_size * params.filterSize.y, sub_group_size);

    jit.AddConstant(MakeJitConstant("OUTPUT_X_BLOCK_SIZE", blockWidth));

    jit.AddConstant(MakeJitConstant("INPUT_LINE_SIZE", input_line_size));
    jit.AddConstant(MakeJitConstant("INPUT_BLOCK_SIZE", input_block_size));

    jit.AddConstant(MakeJitConstant("SUB_GROUP_SIZE", sub_group_size));
    jit.AddConstant(MakeJitConstant("X_BLOCKS", CeilDiv(output.X().v, blockWidth)));

    if (params.output.Feature().v % feature_block_size != 0) {
        jit.AddConstant(MakeJitConstant("OUTPUT_LEFTOVERS", 1));
    }

    return jit;
}

KernelsData ConvolutionKernel_bfyx_to_bfyx_f16::GetTunedKernelsDataByIndex(const Params& params,
                                                                           const optional_params& options,
                                                                           const int autoTuneIndex) const {
    auto tuneOptions = GetAutoTuneOptions(params, autoTuneIndex);
    return GetCommonKernelsData(params, options, tuneOptions.exeMode, autoTuneIndex);
}

KernelsData ConvolutionKernel_bfyx_to_bfyx_f16::GetKernelsData(const Params& params,
                                                               const optional_params& options) const {
    return GetTunedKernelsDataByIndex(params, options);
}

JitConstants ConvolutionKernel_bfyx_to_bfyx_f16::GetFusedPrimitivesJitConstants(const convolution_params& params,
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
        std::string index_func_call = tensor_name + "_GET_INDEX(b, f_block*16, y, x+i)";
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
                throw std::invalid_argument("Invalid fused op in binary_convolution kernel: " + params.layerID);
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

KernelsData ConvolutionKernel_bfyx_to_bfyx_f16::GetKernelsDataForAutoTune(const Params& params,
                                                                          const optional_params& options) const {
    if (!Validate(params, options)) {
        return {};
    }

    KernelsData res = {};

    for (size_t i = 0; i < autoTuneOptions.size(); i++) {
        KernelsData kd = GetTunedKernelsDataByIndex(params, options, static_cast<int>(i));
        if (!kd.empty()) {
            res.emplace_back(kd[0]);
        }
    }

    return res;
}
}  // namespace kernel_selector
