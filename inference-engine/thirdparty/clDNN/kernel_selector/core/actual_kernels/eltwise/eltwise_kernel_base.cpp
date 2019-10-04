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

#include "eltwise_kernel_base.h"
#include "kernel_selector_utils.h"
#include <string>
#include <vector>

namespace kernel_selector {
static uint32_t GetNumberOfInputs(EltwiseMode m) {
    switch (m) {
        case EltwiseMode::ADD:
        case EltwiseMode::SUB:
        case EltwiseMode::MUL:
        case EltwiseMode::DIV:
        case EltwiseMode::MIN:
        case EltwiseMode::MAX:
        case EltwiseMode::POW:
        case EltwiseMode::MODULU:
        case EltwiseMode::EQ:
        case EltwiseMode::NE:
        case EltwiseMode::LT:
        case EltwiseMode::LE:
        case EltwiseMode::GT:
        case EltwiseMode::GE:
        case EltwiseMode::LOGIC_AND:
        case EltwiseMode::LOGIC_OR:
        case EltwiseMode::LOGIC_XOR:
        case EltwiseMode::SQUARED_DIFF:
        case EltwiseMode::FLOOR_MOD:
            return 2;
        case EltwiseMode::SQRT:
        case EltwiseMode::RSQRT:
        case EltwiseMode::ASSIGN:
            return 1;
        default:
            return 0;
    }
}

ParamsKey eltwise_params::GetParamsKey() const {
    ParamsKey k = base_params::GetParamsKey();
    if (int8_quantization) {
        k.EnableInt8Quantization();
    }

    if (output_calibration) {
        k.EnableOutputCalibration();
    }

    if (inputs_calibration) {
        k.EnableEltwiseInputsCalibration();
    }

    if (!stride.empty()) {
        k.EnableEltwiseStride();
    }

    if (broadcast) {
        k.EnableEltwiseBroadcast();
    }

    return k;
}

bool EltwiseKernelBase::Validate(const Params& p, const optional_params& o) const {
    if (p.GetType() != KernelType::ELTWISE || o.GetType() != KernelType::ELTWISE) {
        return false;
    }

    const eltwise_params& params = static_cast<const eltwise_params&>(p);

    if (params.inputs.size() == 0) {
        return false;
    }

    auto& operations = params.operations;

    if (operations.size() == 0) {
        return false;
    }

    for (size_t op_num = 0; op_num < operations.size(); op_num++) {
        const auto& ew = operations[op_num];

        if (ew.inputs.size() != GetNumberOfInputs(ew.mode)) {
            return false;
        }

        for (size_t input_idx = 0; input_idx < ew.inputs.size(); input_idx++) {
            const auto& input = ew.inputs[input_idx];
            if (input.mode == EltwiseInputMode::INPUT_BUFFER && input.index >= params.inputs.size()) {
                return false;
            }
        }
    }

    return true;
}

JitConstants EltwiseKernelBase::GetJitConstantsCommon(const eltwise_params& params, bool useVload8) const {
    JitConstants jit = MakeBaseParamsJitConstants(params);

    auto GetIdxOrderVecForLayout = [&](DataLayout l, bool layoutBased, uSize stride) -> std::vector<std::string> {
        // TODO: Generalize this method
        std::vector<std::string> bfyx_idx_order = {};
        if (layoutBased) {
            bfyx_idx_order = { "d4", "d3", "d2", "d1" };
        } else {
            if (l == DataLayout::yxfb) {
                bfyx_idx_order = { "d1", "d2", "d4", "d3" };
            } else if (l == DataLayout::fyxb) {
                bfyx_idx_order = { "d1", "d4", "d3", "d2" };
            } else {
                bfyx_idx_order = { "d4", "d3", "d2", "d1" };
            }
        }

        if (!params.stride.empty()) {
            bfyx_idx_order[2] = "(" + bfyx_idx_order[2] + "*" + std::to_string(stride.y) + ")";
            bfyx_idx_order[3] = "(" + bfyx_idx_order[3] + "*" + std::to_string(stride.x) + ")";
        }

        return bfyx_idx_order;
    };

    auto GetIdxOrderStringForLayout = [&](DataLayout l, bool layoutBased, uSize stride) -> std::string {
        std::vector<std::string> bfyx_idx_order = GetIdxOrderVecForLayout(l, layoutBased, stride);

        return bfyx_idx_order[0] + "," +
               bfyx_idx_order[1] + "," +
               bfyx_idx_order[2] + "," +
               bfyx_idx_order[3];
    };

    jit.AddConstants({
        MakeJitConstant("ELTWISE_LAYOUT_BASED", params.layoutBased),
        MakeJitConstant("QUANTIZATION_TERM", params.int8_quantization),
        MakeJitConstant("ELTWISE_BROADCAST", params.broadcast),
    });

    if (params.int8_quantization) {
        if (params.output_calibration) {
            jit.AddConstant(MakeJitConstant("CALIBRATION_TERM", params.output_calibration));
            jit.AddConstant(MakeJitConstant("O_QF", params.output_calibration_factors[0]));

        } else {
            jit.AddConstants({MakeJitConstant("O_QF", params.output_quantization_factor)});
        }
    }

    std::string inputs_decls, vload_decls;
    auto& updateInputs = params.updateInputIds;

    std::string out_idx_order = "OUTPUT_IDX_ORDER";
    uSize out_stride = {1, 1, 1};
    if (useVload8) {
        jit.AddConstant(MakeJitConstant(out_idx_order, "d1"));
    } else {
        if (CheckInputsOutputNoPitchSameDims(params) &&
            !(params.layoutBased || params.int8_quantization || params.broadcast)) {
            jit.AddConstant(MakeJitConstant(out_idx_order, "d1"));
        } else {
            size_t out_c = DataTensor::ChannelsCount(params.output.GetLayout());
            if (out_c <= 4) {
                jit.AddConstant(MakeJitConstant(out_idx_order, GetIdxOrderStringForLayout(params.output.GetLayout(),
                                                                                          params.layoutBased || params.broadcast,
                                                                                          out_stride)));
            } else if (out_c == 5) {
                jit.AddConstant(MakeJitConstant(out_idx_order, "d5,d4,d3,d2,d1"));
            } else {
                assert(0);
            }
        }
    }
    if (!params.stride.empty()) {
        jit.AddConstant(MakeJitConstant("OUTPUT_STRIDE_X", out_stride.x));
        jit.AddConstant(MakeJitConstant("OUTPUT_STRIDE_Y", out_stride.y));
        jit.AddConstant(MakeJitConstant("OUTPUT_STRIDE_Z", out_stride.z));
    }

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
            jit.AddConstant(MakeJitConstant("INPUT" + std::to_string(i) + "_STRIDE_Z", params.stride[i].z));
        }
        std::string idx_order = "INPUT" + std::to_string(i) + "_IDX_ORDER";
        if (useVload8) {
            vload_decls += "\\\n\tconst " + toCLType(params.inputs[i].GetDType()) + "8 in" + std::to_string(i);
            if (params.inputs[i].PhysicalSize() == 1)  // Scalar case
                vload_decls += " = (" + toCLType(params.inputs[i].GetDType()) + "8)(input" + std::to_string(i) + "[0]";
            else  // Buffer case
                vload_decls += " = vload8(global_id, input" + std::to_string(i);
            vload_decls += ");";
            jit.AddConstant(MakeJitConstant(idx_order, "d1"));
        } else {
            if (CheckInputsOutputNoPitchSameDims(params) &&
                !(params.layoutBased || params.int8_quantization || params.broadcast)) {
                jit.AddConstant(MakeJitConstant(idx_order, "d1"));
            } else {
                size_t in_c = DataTensor::ChannelsCount(params.inputs[i].GetLayout());
                size_t out_c = DataTensor::ChannelsCount(params.output.GetLayout());
                auto in_stride = params.stride.empty() ? out_stride : params.stride[i];
                if (out_c <= 4 && in_c <= 4) {
                    jit.AddConstant(MakeJitConstant(idx_order, GetIdxOrderStringForLayout(params.inputs[i].GetLayout(),
                                                                                          params.layoutBased || params.broadcast,
                                                                                          in_stride)));
                } else if (out_c == 5) {
                    if (in_c < 5) {
                        // Skip Z coord for 4d tensors
                        jit.AddConstant(MakeJitConstant(idx_order, "d5,d4,d2,d1"));
                    } else if (in_c == 5) {
                        jit.AddConstant(MakeJitConstant(idx_order, "d5,d4,d3,d2,d1"));
                    }
                } else if (out_c <= 4 && in_c == 5) {
                    // quite strange case, but can happen due to reorders fusing
                    // it means that z coord is equal to 1, so z offset will be always equal to 0
                    jit.AddConstant(MakeJitConstant(idx_order, "d4,d3,0,d2,d1"));
                } else {
                    assert(0);
                }
            }
        }
    }

    jit.AddConstant(MakeJitConstant("INPUTS_DECLS", inputs_decls));
    jit.AddConstant(MakeJitConstant("ELTWISE_NO_PITCH_SAME_DIMS", CheckInputsOutputNoPitchSameDims(params)));

    if (useVload8)
        jit.AddConstant(MakeJitConstant("VLOAD_DECLS", vload_decls));

    std::string do_eltwise;

    auto& operations = params.operations;
    auto& coefficients = params.coefficients;

    for (size_t op_num = 0; op_num < operations.size(); op_num++) {
        const std::string op_num_str = std::to_string(op_num);
        const auto& ew = operations[op_num];

        for (size_t input_idx = 0; input_idx < ew.inputs.size(); input_idx++) {
            const auto& input = ew.inputs[input_idx];
            const std::string name = "INPUT_" + op_num_str + "_" + std::to_string(input_idx);
            std::string idx_order = "INPUT" + std::to_string(input.index) + "_IDX_ORDER";

            switch (input.mode) {
                case EltwiseInputMode::SCALAR:
                    jit.AddConstant(MakeJitConstant(name, input.scalar));
                    break;
                case EltwiseInputMode::INPUT_BUFFER:
                    if (useVload8)
                        jit.AddConstant(MakeJitConstant(name, "in" + std::to_string(input.index)));
                    else
                        jit.AddConstant(MakeJitConstant(name,
                                                        "input" + std::to_string(input.index) +
                                                        "[GET_INDEX(INPUT, " + std::to_string(input.index) +
                                                        "," + idx_order +")]"));
                    break;
                case EltwiseInputMode::OUTPUT_BUFFER:
                    jit.AddConstant(MakeJitConstant(name, "output[GET_INDEX(OUTPUT,,"+ out_idx_order +")]"));
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

        std::string input0_str, input1_str, cast_type, output_cast, op;

        if (useVload8) {
            cast_type = "(MAKE_VECTOR_TYPE(UNIT_TYPE, 8))";
            op = "const MAKE_VECTOR_TYPE(UNIT_TYPE, 8) tmp" + op_num_str + " = ";
        } else if (params.int8_quantization) {
            cast_type = "(int)";
            op = "const int tmp" + op_num_str + " = ";
        } else {
            cast_type = "(UNIT_TYPE)";
            op = "const UNIT_TYPE tmp" + op_num_str + " = ";
        }

        if (params.output.GetDType() == Datatype::INT8 && !params.int8_quantization) {
            output_cast = "(char)";
            cast_type = "(" + toCLType(params.inputs[op_num].GetDType()) + ")";
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
                    } else {
                    // input_0 == int && input_1 != int
                        op += cast_type + "f" + mode + "(convert_float(" + input0_str + "), " + input1_str + ")";
                    }
                } else if (input_1_type == kernel_selector::Datatype::INT8 ||
                         input_1_type == kernel_selector::Datatype::INT32 ||
                         input_1_type == kernel_selector::Datatype::INT64) {
                    // input_0 != int && input_1 == int
                    op += cast_type + "f" + mode + "(" + input0_str + ", convert_float(" + input1_str + "))";
                } else {
                    // input_0 != int && input_1 != int
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
                op += output_cast + "(" + input0_str + " == " + input1_str + ")";
                break;
            case EltwiseMode::NE:
                op += output_cast + "(" + input0_str + " != " + input1_str + ")";
                break;
            case EltwiseMode::LT:
                op += output_cast + "(" + input0_str + " < " + input1_str + ")";
                break;
            case EltwiseMode::LE:
                op += output_cast + "(" + input0_str + " <= " + input1_str + ")";
                break;
            case EltwiseMode::GT:
                op += output_cast + "(" + input0_str + " > " + input1_str + ")";
                break;
            case EltwiseMode::GE:
                op += output_cast + "(" + input0_str + " >= " + input1_str + ")";
                break;
            case EltwiseMode::LOGIC_AND:
                op += output_cast + "(" + input0_str + " && " + input1_str + ")";
                break;
            case EltwiseMode::LOGIC_OR:
                op += output_cast + "(" + input0_str + " || " + input1_str + ")";
                break;
            case EltwiseMode::LOGIC_XOR:
                op += output_cast + "(!" + input0_str + " != !" + input1_str + ")";
                break;
            case EltwiseMode::FLOOR_MOD:
                op += output_cast + "(" + input0_str + " - " + input0_str + " / " + input1_str + " * " + input1_str + ")";
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
                      std::to_string(updateInputs[update_input_idx].inputId) + ", " +
                      "INPUT"+std::to_string(updateInputs[update_input_idx].inputId) + "_IDX_ORDER)] = tmp" +
                      std::to_string(updateInputs[update_input_idx].tmpId) + ";";

    do_eltwise += "\\\n\tres = tmp" + std::to_string(operations.size() - 1) + ";";

    jit.AddConstant(MakeJitConstant("DO_ELTWISE", do_eltwise));

    if (params.layoutBased || params.int8_quantization || params.broadcast) {
        jit.Merge(GetTensorFriendlyWorkGroupsJit(params.output));
    }

    if (!params.stride.empty()) {
        jit.AddConstant(MakeJitConstant("INPUT_STRIDED", 1));
    }

    return jit;
}

JitConstants EltwiseKernelBase::GetJitConstants(const eltwise_params& params) const {
    return GetJitConstantsCommon(params, false);
}

EltwiseKernelBase::DispatchData EltwiseKernelBase::SetDefault(const eltwise_params& params) const {
    DispatchData kd;

    if (params.layoutBased || params.int8_quantization || params.broadcast) {
        auto global = GetTensorFriendlyWorkGroups(params.output);
        kd.gws0 = global[0];
        kd.gws1 = global[1];
        kd.gws2 = global[2];
    } else if (CheckInputsOutputNoPitchSameDims(params)) {
        kd.gws0 = params.output.LogicalSize();
        kd.gws1 = 1;
        kd.gws2 = 1;
    } else {
        const auto& out = params.output;

        std::vector<size_t> gws;
        for (const auto& o : out.GetDims()) {
            gws.push_back(o.v);
        }

        size_t n_dims;
        if ((out.GetLayout() == DataLayout::bfzyx)  || (out.GetLayout() == DataLayout::bfzyx_f16))
            n_dims = 5;
        else
            n_dims = 4;

        for (size_t i = gws.size(); i < n_dims; i++) {
            gws.push_back(1U);
        }

        kd.gws0 = gws[0];
        if (n_dims == 5) {
            kd.gws1 = gws[1] * gws[2];  // y*z
            kd.gws2 = gws[3] * gws[4];
        } else {
            kd.gws1 = gws[1];
            kd.gws2 = gws[2] * gws[3];
        }
    }

    auto local = GetOptimalLocalWorkGroupSizes({kd.gws0, kd.gws1, kd.gws2});

    if (params.output.GetLayout() == DataLayout::bfyx_f16 && params.output.Feature().v % 16 == 0 &&
        kd.gws1 % 16 == 0) {
        kd.lws0 = 1;
        kd.lws1 = 16;
        kd.lws2 = 1;
    } else {
        kd.lws0 = local[0];
        kd.lws1 = local[1];
        kd.lws2 = local[2];
    }

    return kd;
}

KernelsData EltwiseKernelBase::GetCommonKernelsData(const Params& params, const optional_params& options) const {
    if (!Validate(params, options)) {
        return {};
    }

    KernelData kd = KernelData::Default<eltwise_params>(params);
    eltwise_params& newParams = *static_cast<eltwise_params*>(kd.params.get());

    auto entry_point = GetEntryPoint(kernelName, newParams.layerID, options);
    auto cldnn_jit = GetJitConstants(newParams);
    std::string jit = CreateJit(kernelName, cldnn_jit, entry_point);

    DispatchData runInfo = SetDefault(newParams);

    auto& kernel = kd.kernels[0];

    kernel.workGroups.global = {runInfo.gws0, runInfo.gws1, runInfo.gws2};
    kernel.workGroups.local = {runInfo.lws0, runInfo.lws1, runInfo.lws2};

    kernel.kernelString = GetKernelString(kernelName, jit, entry_point, params.engineInfo, DEFAULT);
    kernel.arguments = GetArgsDesc((uint32_t)newParams.inputs.size(),
                                   false,
                                   false,
                                   newParams.int8_quantization,
                                   newParams.output_calibration);

    kd.estimatedTime = DONT_USE_IF_HAVE_SOMETHING_ELSE;

    return {kd};
}
}  // namespace kernel_selector
