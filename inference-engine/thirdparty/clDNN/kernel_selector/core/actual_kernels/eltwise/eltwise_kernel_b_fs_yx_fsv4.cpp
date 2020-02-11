/*
// Copyright (c) 2019 Intel Corporation
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
*/

#include "eltwise_kernel_b_fs_yx_fsv4.h"
#include "kernel_selector_utils.h"
#include <string>
#include <vector>

namespace kernel_selector {

ParamsKey EltwiseKernel_b_fs_yx_fsv4::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::INT8);
    k.EnableInputDataType(Datatype::UINT8);
    k.EnableOutputDataType(Datatype::INT8);
    k.EnableOutputDataType(Datatype::UINT8);
    k.EnableInputLayout(DataLayout::b_fs_yx_fsv4);
    k.EnableOutputLayout(DataLayout::b_fs_yx_fsv4);
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    k.EnableInt8Quantization();
    k.EnableEltwiseStride();
    return k;
}

EltwiseKernelBase::DispatchData EltwiseKernel_b_fs_yx_fsv4::SetDefault(const eltwise_params& params) const {
    DispatchData kd;

    // Because of very specific requirements for data, we may linearize the data,
    // i.e. use only one dimension, e.g. 'X'.

    // GWS:
    // we process 4*4 (4 int8 bytes per on block_read4 reading) features per workitem
    kd.gws0 = params.output.X().v * params.output.Y().v * params.output.Batch().v * params.output.Feature().v / (4 * 4);
    kd.gws1 = 1;
    kd.gws2 = 1;
    // LWS:
    kd.lws0 = 8;
    kd.lws1 = 1;
    kd.lws2 = 1;

    kd.effiency = FORCE_PRIORITY_1;
    return kd;
}

bool EltwiseKernel_b_fs_yx_fsv4::Validate(const Params& params, const optional_params& options) const {
    // Requirents to use 'eltwise_b_fs_yx_fsv4' kernel are below:
    // 1. No stride
    // 2. All dimensions for all inputs are the same
    // 3. No padding
    // So, it can be linearized

    if (!Parent::Validate(params, options)) {
        return false;
    }

    KernelData kd = KernelData::Default<eltwise_params>(params);
    eltwise_params& newParams = *static_cast<eltwise_params*>(kd.params.get());

    // 1. No stride
    if (!newParams.stride.empty()) {
        return false;
    }

    for (size_t i = 0; i < newParams.inputs.size() - 1; i++) {
        // 2. All dimensions for all inputs are the same
        if (!(newParams.inputs[i] == newParams.inputs[i + 1])) {
            return false;
        }
    }

    const auto& in = newParams.inputs[0];
    for (size_t i = 0; i < in.Dimentions(); i++) {
        // 3. No padding
        if ((in.GetDims()[i].pad.before != 0) || (in.GetDims()[i].pad.after != 0)) {
            return false;
        }
    }

    return true;
}

JitConstants EltwiseKernel_b_fs_yx_fsv4::GetJitConstants(const eltwise_params& params) const {
    JitConstants jit = MakeBaseParamsJitConstants(params);

    if (params.inputs[0].GetDType() == Datatype::UINT8) {
        // Special handler for unsigned types
        jit.AddConstants({MakeJitConstant("ELTW_UNSIGNED", 1)});
    }

    ///////////////
    jit.AddConstants({
        MakeJitConstant("ELTWISE_LAYOUT_BASED", params.layoutBased),
        MakeJitConstant("QUANTIZATION_TERM", params.int8_quantization),
    });

    if (params.int8_quantization) {
        if (params.output_calibration) {
            jit.AddConstant(MakeJitConstant("CALIBRATION_TERM", params.output_calibration));
            jit.AddConstant(MakeJitConstant("O_QF", params.output_calibration_factors[0]));

        } else {
            jit.AddConstants({MakeJitConstant("O_QF", params.output_quantization_factor)});
        }
    }

    std::string inputs_decls;
    auto& updateInputs = params.updateInputIds;

    for (size_t i = 0; i < params.inputs.size(); i++) {
        // const should be added only to inputs which will not be updated
        std::string const_str = "const";
        for (size_t update_input_idx = 0; update_input_idx < updateInputs.size(); update_input_idx++) {
            if (updateInputs[update_input_idx].inputId == i) {
                const_str = "";
                break;
            }
        }

        inputs_decls +=
            const_str + " __global " + toCLType(params.inputs[i].GetDType()) + "* input" + std::to_string(i) + ", ";
    }

    jit.AddConstant(MakeJitConstant("INPUTS_DECLS", inputs_decls));
    jit.AddConstant(MakeJitConstant("ELTWISE_NO_PITCH_SAME_DIMS", CheckInputsOutputNoPitchSameDims(params)));

    std::string do_eltwise;

    auto& operations = params.operations;
    auto& coefficients = params.coefficients;

    for (size_t op_num = 0; op_num < operations.size(); op_num++) {
        const std::string op_num_str = std::to_string(op_num);
        const auto& ew = operations[op_num];

        for (size_t input_idx = 0; input_idx < ew.inputs.size(); input_idx++) {
            const auto& input = ew.inputs[input_idx];
            const std::string name = "INPUT_" + op_num_str + "_" + std::to_string(input_idx);
            switch (input.mode) {
                case EltwiseInputMode::SCALAR:
                    jit.AddConstant(MakeJitConstant(name, input.scalar));
                    break;
                case EltwiseInputMode::INPUT_BUFFER:
                    jit.AddConstant(MakeJitConstant(name,
                                                    "GET_INPUT(input" + std::to_string(input.index) + ", INPUT" +
                                                        std::to_string(input.index) + ")"));
                    break;
                case EltwiseInputMode::OUTPUT_BUFFER:
                    jit.AddConstant(MakeJitConstant(name, "output[GET_INDEX(OUTPUT, )]"));
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
        std::string input0_str, input1_str, cast_type, op;

        cast_type = "(int16)";
        op = "const int16 tmp" + op_num_str + " = ";

        input0_str = cast_type + "INPUT_" + op_num_str + "_0";
        input1_str = cast_type + "INPUT_" + op_num_str + "_1";

        if (ew.mode == EltwiseMode::ADD) {
            std::vector<std::string> coeff_strings(ew.inputs.size(), "");
            for (size_t input_idx = 0; input_idx < ew.inputs.size(); input_idx++) {
                const auto& input = ew.inputs[input_idx];
                if (input.mode == EltwiseInputMode::INPUT_BUFFER && input.index < coefficients.size()) {
                    const float c = coefficients[input.index];
                    if (c != 1.0f)
                        coeff_strings[input_idx] = cast_type + "(" + std::to_string(c) + ")*";
                }
            }

            input0_str = coeff_strings[0] + input0_str;
            input1_str = coeff_strings[1] + input1_str;
        }

        switch (ew.mode) {
            case EltwiseMode::ADD:
                op += input0_str + " + " + input1_str;
                break;
            case EltwiseMode::SUB:
                op += input0_str + " - " + input1_str;
                break;
            case EltwiseMode::MUL:
                op += input0_str + " * " + input1_str;
                break;
            case EltwiseMode::DIV:
                op += input0_str + " / " + input1_str;
                break;
            case EltwiseMode::MODULU:
            case EltwiseMode::MIN:
            case EltwiseMode::MAX: {
                auto mode = (ew.mode == EltwiseMode::MODULU ? "mod" : (ew.mode == EltwiseMode::MIN ? "min" : "max"));
                auto input_0_type = params.inputs[0].GetDType();
                auto input_1_type = params.inputs[1].GetDType();

                // input_0 == int
                if (input_0_type == kernel_selector::Datatype::INT8 ||
                    input_0_type == kernel_selector::Datatype::UINT8) {
                    // input_0 == int && input_1 == int
                    if (input_1_type == kernel_selector::Datatype::INT8 ||
                        input_1_type == kernel_selector::Datatype::UINT8) {
                        if (ew.mode == EltwiseMode::MODULU)
                            op += input0_str + " % " + input1_str;
                        else
                            op += cast_type + mode + "(" + input0_str + ", " + input1_str + ")";
                    // input_0 == int && input_1 != int
                    } else {
                        op += cast_type + "f" + mode + "(convert_float(" + input0_str + "), " + input1_str + ")";
                    }
                // input_0 != int && input_1 == int
                } else if (input_1_type == kernel_selector::Datatype::INT8 ||
                         input_1_type == kernel_selector::Datatype::UINT8) {
                    op += cast_type + "f" + mode + "(" + input0_str + ", convert_float(" + input1_str + "))";
                // input_0 != int && input_1 != int
                } else {
                    op += cast_type + "f" + mode + "(" + input0_str + ", " + input1_str + ")";
                }
            } break;
            case EltwiseMode::POW:
                op += cast_type + "pow(" + input0_str + ", " + input1_str + ")";
                break;
            case EltwiseMode::SQRT:
                op += cast_type + "sqrt(" + input0_str + ")";
                break;
            case EltwiseMode::RSQRT:
                op += cast_type + "1/sqrt(" + input0_str + ")";
                break;
            case EltwiseMode::ASSIGN:
                op += input0_str;
                break;
            default:
                break;
        }

        std::string opname = "OPERATION" + op_num_str;
        jit.AddConstant(MakeJitConstant(opname, op));
        do_eltwise += "\\\n\t" + opname + ";";
    }

    for (size_t update_input_idx = 0; update_input_idx < updateInputs.size(); update_input_idx++)
        do_eltwise += "\\\n\tinput" + std::to_string(updateInputs[update_input_idx].inputId) + "[GET_INDEX(INPUT, " +
                      std::to_string(updateInputs[update_input_idx].inputId) + ")] = tmp" +
                      std::to_string(updateInputs[update_input_idx].tmpId) + ";";

    do_eltwise += "\\\n\tres = tmp" + std::to_string(operations.size() - 1) + ";";

    jit.AddConstant(MakeJitConstant("DO_ELTWISE", do_eltwise));

    if (params.layoutBased || params.int8_quantization) {
        jit.Merge(GetTensorFriendlyWorkGroupsJit(params.inputs[0]));
    }

    if (!params.stride.empty()) {
        jit.AddConstant(MakeJitConstant("INPUT_STRIDED", 1));
    }

    ///////////////
    return jit;
}

KernelsData EltwiseKernel_b_fs_yx_fsv4::GetKernelsData(const Params& params, const optional_params& options) const {
    return GetCommonKernelsData(params, options);
}
}  // namespace kernel_selector
