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


#include "eltwise_kernel_fs_bs_yx_bsv4_fsv32.h"
#include "kernel_selector_utils.h"
#include <string>
#include <vector>

namespace kernel_selector {

ParamsKey EltwiseKernel_fs_bs_yx_bsv4_fsv32::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::INT8);
    k.EnableOutputDataType(Datatype::INT8);
    k.EnableInputLayout(DataLayout::fs_bs_yx_bsv4_fsv32);
    k.EnableOutputLayout(DataLayout::fs_bs_yx_bsv4_fsv32);
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    k.EnableInt8Quantization();
    k.EnableOutputCalibration();
    k.EnableEltwiseStride();
    return k;
}

EltwiseKernelBase::DispatchData EltwiseKernel_fs_bs_yx_bsv4_fsv32::SetDefault(const eltwise_params& params) const {
    DispatchData kd;

    kd.gws0 = params.output.X().v;
    kd.gws1 = params.output.Y().v;
    // we process 4 batches and 4 features per workitem
    kd.gws2 = (params.output.Batch().v / 4) * (params.output.Feature().v / 4);
    kd.lws0 = 1;
    kd.lws1 = 1;
    kd.lws2 = 8;

    kd.effiency = FORCE_PRIORITY_3;
    return kd;
}

JitConstants EltwiseKernel_fs_bs_yx_bsv4_fsv32::GetJitConstants(const eltwise_params& params) const {
    JitConstants jit = MakeBaseParamsJitConstants(params);

    const size_t in_x_pitch = 32 * 4;
    const size_t in_y_pitch = 32 * 4 * params.inputs[0].X().LogicalDimPadded();
    const size_t in_b_block_pitch = in_y_pitch * params.inputs[0].Y().LogicalDimPadded();
    const size_t in_f_block_pitch = in_b_block_pitch * ((params.inputs[0].Batch().v + 3) / 4);
    const size_t in_offset =
        in_x_pitch * params.inputs[0].X().pad.before + in_y_pitch * params.inputs[0].Y().pad.before;

    jit.AddConstant(MakeJitConstant("IN_X_PITCH", in_x_pitch));
    jit.AddConstant(MakeJitConstant("IN_Y_PITCH", in_y_pitch));
    jit.AddConstant(MakeJitConstant("IN_B_BLOCK_PITCH", in_b_block_pitch));
    jit.AddConstant(MakeJitConstant("IN_F_BLOCK_PITCH", in_f_block_pitch));
    jit.AddConstant(MakeJitConstant("IN_OFFSET", in_offset));

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

        if (!params.stride.empty()) {
            jit.AddConstant(MakeJitConstant("INPUT" + std::to_string(i) + "_STRIDE_X", params.stride[i].x));
            jit.AddConstant(MakeJitConstant("INPUT" + std::to_string(i) + "_STRIDE_Y", params.stride[i].y));
        }
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

        if (params.int8_quantization) {
            cast_type = "(int16)";
            op = "const int16 tmp" + op_num_str + " = ";
        } else {
            cast_type = "(UNIT_TYPE)";
            op = "const UNIT_TYPE tmp" + op_num_str + " = ";
        }

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
                    input_0_type == kernel_selector::Datatype::INT32 ||
                    input_0_type == kernel_selector::Datatype::INT64) {
                    // input_0 == int && input_1 == int
                    if (input_1_type == kernel_selector::Datatype::INT8 ||
                        input_1_type == kernel_selector::Datatype::INT32 ||
                        input_1_type == kernel_selector::Datatype::INT64) {
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
                         input_1_type == kernel_selector::Datatype::INT32 ||
                         input_1_type == kernel_selector::Datatype::INT64) {
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
            case EltwiseMode::SQUARED_DIFF:
                op += cast_type + "((" + input0_str + " - " + input1_str +
                      ")"
                      " * (" +
                      input0_str + " - " + input1_str + "))";
                break;
            case EltwiseMode::EQ:
                op += cast_type + "(" + input0_str + " == " + input1_str + ")";
                break;
            case EltwiseMode::NE:
                op += cast_type + "(" + input0_str + " != " + input1_str + ")";
                break;
            case EltwiseMode::LT:
                op += cast_type + "(" + input0_str + " < " + input1_str + ")";
                break;
            case EltwiseMode::LE:
                op += cast_type + "(" + input0_str + " <= " + input1_str + ")";
                break;
            case EltwiseMode::GT:
                op += cast_type + "(" + input0_str + " > " + input1_str + ")";
                break;
            case EltwiseMode::GE:
                op += cast_type + "(" + input0_str + " >= " + input1_str + ")";
                break;
            case EltwiseMode::LOGIC_AND:
                op += cast_type + "(" + input0_str + " && " + input1_str + ")";
                break;
            case EltwiseMode::LOGIC_OR:
                op += cast_type + "(" + input0_str + " || " + input1_str + ")";
                break;
            case EltwiseMode::LOGIC_XOR:
                op += cast_type + "(!" + input0_str + " != !" + input1_str + ")";
                break;
            case EltwiseMode::FLOOR_MOD:
                op += cast_type + "(" + input0_str + " - " + input0_str + " / " + input1_str + " * " + input1_str + ")";
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

KernelsData EltwiseKernel_fs_bs_yx_bsv4_fsv32::GetKernelsData(const Params& params,
                                                              const optional_params& options) const {
    return GetCommonKernelsData(params, options);
}
}  // namespace kernel_selector
