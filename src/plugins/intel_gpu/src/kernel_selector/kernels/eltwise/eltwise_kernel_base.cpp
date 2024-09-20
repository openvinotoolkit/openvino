// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "eltwise_kernel_base.h"
#include "kernel_selector_utils.h"
#include <string>
#include <vector>

namespace kernel_selector {
namespace {
std::vector<size_t> GetLimitedOptimalLocalWorkGroupSizes(std::vector<size_t> gws, const EngineInfo& info, std::vector<size_t> limited_size_lws) {
    const size_t lws_max = info.maxWorkGroupSize;
    const size_t optimal_lws_values[] = {256, 227, 224, 192, 160, 128, 96, 64, 32, 16, 8, 7, 6, 5, 4, 3, 2, 1};
    size_t total_lws = 1;
    std::vector<size_t> lws;
    for (size_t i = 0; i < gws.size(); ++i) {
        auto rest_lws = lws_max / total_lws;
        size_t lws_idx = 0;
        while (rest_lws < optimal_lws_values[lws_idx] || optimal_lws_values[lws_idx] > limited_size_lws[i]) lws_idx++;

        while (gws[i] % optimal_lws_values[lws_idx]) lws_idx++;

        lws.push_back(optimal_lws_values[lws_idx]);
        total_lws *= optimal_lws_values[lws_idx];
    }

    return lws;
}

uint32_t GetNumberOfInputs(EltwiseMode m) {
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
        case EltwiseMode::RIGHT_SHIFT:
        case EltwiseMode::LEFT_SHIFT:
        case EltwiseMode::BITWISE_AND:
        case EltwiseMode::BITWISE_OR:
        case EltwiseMode::BITWISE_XOR:
            return 2;
        case EltwiseMode::SQRT:
        case EltwiseMode::RSQRT:
        case EltwiseMode::ASSIGN:
        case EltwiseMode::IS_FINITE:
        case EltwiseMode::IS_INF:
        case EltwiseMode::IS_NAN:
            return 1;
        default:
            return 0;
    }
}
}  // namespace

ParamsKey eltwise_params::GetParamsKey() const {
    ParamsKey k = base_params::GetParamsKey();

    if (!stride.empty()) {
        k.EnableEltwiseStride();
    }

    if (broadcast) {
        k.EnableEltwiseBroadcast();
    }

    return k;
}

static bool IsBitwiseMode(EltwiseMode mode) {
    return mode == EltwiseMode::BITWISE_AND || mode == EltwiseMode::LEFT_SHIFT || mode == EltwiseMode::RIGHT_SHIFT ||
           mode == EltwiseMode::BITWISE_OR || mode == EltwiseMode::BITWISE_XOR;
}

Datatype EltwiseKernelBase::GetAccumulatorType(const eltwise_params &params) const {
    // NOTE: Workaround for not promoting shift operations. Not sure what should happen
    // if shift op is just one operation of other elementwise operations. My guess is that is should be promoted as
    // well, but in reality more robust solution will be needed or (better) - assumption that types are not promoted. So
    // probably this is a temporary solution.
    if (IsBitwiseMode(params.operations[0].mode)) {
        return params.inputs[0].GetDType();
    }

    if (params.int8_quantization)
        return Datatype::INT32;

    Datatype types[] = { Datatype::F32, Datatype::F16, Datatype::INT64, Datatype::INT32, Datatype::UINT32};

    for (Datatype type : types)
        for (auto& in : params.inputs)
            if (in.GetDType() == type)
                return type;

    return Datatype::F32;
}

bool EltwiseKernelBase::Validate(const Params& p) const {
    if (p.GetType() != KernelType::ELTWISE) {
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

    const eltwise_params& orgParams = static_cast<const eltwise_params&>(p);
    for (auto& fused_op : orgParams.fused_ops) {
        if (!IsFusedPrimitiveSupported(fused_op))
            return false;
    }

    return true;
}

bool EltwiseKernelBase::IsUnsupportedModeForVecCode(const eltwise_params& params) const {
    // These modes are supposed to produce BOOL output type
    // but this kernel uses vector data types, and these operation will produce 0xFFFF / 0x0000 instead of 0 / 1 values
    // The value might be then converted to fp16/fp32 and used for some arithmetic, which will lead to invalid results, thus reject these modes
    // to fallback on ref kernel with scalar types.
    // TODO: Consider updating optimized kernels to produce 0/1 output for vector code if such operation is a bottleneck in some model
    const std::vector<EltwiseMode> unsupported_modes = {
        EltwiseMode::EQ,
        EltwiseMode::NE,
        EltwiseMode::LT,
        EltwiseMode::LE,
        EltwiseMode::GT,
        EltwiseMode::GE,
        EltwiseMode::LOGIC_AND,
        EltwiseMode::LOGIC_OR,
        EltwiseMode::LOGIC_XOR,
        EltwiseMode::FLOOR_MOD,
        EltwiseMode::IS_FINITE,
        EltwiseMode::IS_INF,
        EltwiseMode::IS_NAN,
    };

    for (size_t op_num = 0; op_num <  params.operations.size(); op_num++) {
        const auto& ew =  params.operations[op_num];
        if (std::find(unsupported_modes.begin(), unsupported_modes.end(), ew.mode) != unsupported_modes.end())
            return true;
    }

    return false;
}

JitConstants EltwiseKernelBase::GetOperationsJitConstants(const eltwise_params& params, bool useVload8, size_t blockSize) const {
    JitConstants jit = {};
    for (size_t op_num = 0; op_num < params.operations.size(); op_num++) {
        const std::string op_num_str = toCodeString(op_num);
        const auto& ew = params.operations[op_num];

        std::string op, cast_type;
        std::string input0_str = cast_type + "INPUT_" + op_num_str + "_0";
        std::string input1_str = cast_type + "INPUT_" + op_num_str + "_1";
        auto& coefficients = params.coefficients;

        if (useVload8) {
            cast_type = "(MAKE_VECTOR_TYPE(ACCUMULATOR_TYPE, 8))";
            op = "const MAKE_VECTOR_TYPE(ACCUMULATOR_TYPE, 8) tmp" + op_num_str + " = ";
        } else if (blockSize > 1) {
            cast_type = "(MAKE_VECTOR_TYPE(ACCUMULATOR_TYPE, BLOCK_SIZE))";
            op = "const MAKE_VECTOR_TYPE(ACCUMULATOR_TYPE, BLOCK_SIZE) tmp" + op_num_str + " = ";
        } else {
            cast_type = "(ACCUMULATOR_TYPE)";
            op = "const ACCUMULATOR_TYPE tmp" + op_num_str + " = ";
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
                        coeff_strings[input_idx] = cast_type + "(" + toCodeString(c) + ")*";
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
                op += "(" + input0_str + " == " + input1_str + ")";
                break;
            case EltwiseMode::NE:
                op += "(" + input0_str + " != " + input1_str + ")";
                break;
            case EltwiseMode::LT:
                op += "(" + input0_str + " < " + input1_str + ")";
                break;
            case EltwiseMode::LE:
                op += "(" + input0_str + " <= " + input1_str + ")";
                break;
            case EltwiseMode::GT:
                op += "(" + input0_str + " > " + input1_str + ")";
                break;
            case EltwiseMode::GE:
                op += "(" + input0_str + " >= " + input1_str + ")";
                break;
            case EltwiseMode::LOGIC_AND:
                op += "(" + input0_str + " && " + input1_str + ")";
                break;
            case EltwiseMode::LOGIC_OR:
                op += "(" + input0_str + " || " + input1_str + ")";
                break;
            case EltwiseMode::LOGIC_XOR:
                op += "(!" + input0_str + " != !" + input1_str + ")";
                break;
            case EltwiseMode::FLOOR_MOD: {
                auto input_1_type = params.inputs[1].GetDType();
                if (input_1_type == kernel_selector::Datatype::F16 || input_1_type == kernel_selector::Datatype::F32) {
                    op += "(" + input0_str + " - floor(" + input0_str + " / " + input1_str + ") * " + input1_str + ")";
                } else {
                    op += "(" + input0_str + " - floor(" + input0_str + " / convert_float(" + input1_str + ")) * " + input1_str + ")";
                }
                break;
            }
            case EltwiseMode::ASSIGN:
                op += input0_str;
                break;
            case EltwiseMode::IS_FINITE:
                op += "(isfinite(" + input0_str + "))";
                break;
            case EltwiseMode::IS_INF:
                op += "(isinf(" + input0_str + ") && (" + toCodeString(coefficients.at(0)) + " && signbit(" +
                      input0_str + ") || " + toCodeString(coefficients.at(1)) + " && !signbit(" + input0_str + ")))";
                break;
            case EltwiseMode::IS_NAN:
                op += "(isnan(" + input0_str + "))";
                break;
            case EltwiseMode::RIGHT_SHIFT:
                op += "(" + input0_str + " >> " + input1_str + ")";
                break;
            case EltwiseMode::LEFT_SHIFT:
                op += "(" + input0_str + " << " + input1_str + ")";
                break;
            case EltwiseMode::BITWISE_AND:
                op += "(" + input0_str + " & " + input1_str + ")";
                break;
            case EltwiseMode::BITWISE_OR:
                op += "(" + input0_str + " | " + input1_str + ")";
                break;
            case EltwiseMode::BITWISE_XOR:
                op += "(" + input0_str + " ^ " + input1_str + ")";
                break;
            default:
                break;
        }

        jit.AddConstant(MakeJitConstant("OPERATION" + op_num_str, op));
    }

    return jit;
}

JitConstants EltwiseKernelBase::MakeLoadJitConstants(const eltwise_params& params,
                                                     bool useVload8) const {
    JitConstants jit = {};
    std::string vload_decls;

    for (size_t op_num = 0; op_num < params.operations.size(); op_num++) {
        const std::string op_num_str = toCodeString(op_num);
        const auto &ew = params.operations[op_num];
        bool is_dynamic_crop_kernel = params.is_shape_agnostic && params.operations[op_num].mode == EltwiseMode::ASSIGN;
        if (is_dynamic_crop_kernel)
            jit.AddConstant(MakeJitConstant("IS_DYNAMIC_CROP", 1));
        for (size_t input_idx = 0; input_idx < ew.inputs.size(); input_idx++) {
            const auto &input = ew.inputs[input_idx];
            const std::string name = "INPUT_" + op_num_str + "_" + toCodeString(input_idx);
            std::string idx_order = "INPUT" + toCodeString(input.index) + "_IDX_ORDER";

            switch (input.mode) {
                case EltwiseInputMode::SCALAR:
                    jit.AddConstant(MakeJitConstant(name, input.scalar));
                    break;
                case EltwiseInputMode::INPUT_BUFFER:
                    if (useVload8)
                        jit.AddConstant(MakeJitConstant(name, "in" + toCodeString(input.index)));
                    else
                        jit.AddConstant(MakeJitConstant(name,
                                                        "input" + toCodeString(input.index) +
                                                        "[GET_INDEX(INPUT, " + toCodeString(input.index) +
                                                        "," + idx_order + ") " + (is_dynamic_crop_kernel ? "+ runtime_offset]" : "]")));
                    break;
                case EltwiseInputMode::OUTPUT_BUFFER:
                    jit.AddConstant(MakeJitConstant(name, "output[GET_INDEX(OUTPUT,,OUTPUT_IDX_ORDER)]"));
                    break;
                case EltwiseInputMode::UNORDERED_ACCESS_INPUT_BUFFER:
                    jit.AddConstant(MakeJitConstant(
                            name,
                            "input" + toCodeString(input.index) + "[(size_t)tmp" + toCodeString(input.tmpIndex) + "]"));
                    break;
                case EltwiseInputMode::INTERMEDIATE_RESULTS_INDEX:
                    jit.AddConstant(MakeJitConstant(name, "tmp" + toCodeString(input.tmpIndex)));
                    break;
                default:
                    break;
            }
        }
    }

    if (useVload8) {
        for (size_t i = 0; i < params.inputs.size(); i++) {
            vload_decls += "\\\n\tconst " + toCLType(params.inputs[i].GetDType()) + "8 in" + toCodeString(i);
            if (params.inputs[i].PhysicalSize() == 1)  // Scalar case
                vload_decls += " = (" + toCLType(params.inputs[i].GetDType()) + "8)(input" + toCodeString(i) + "[0]";
            else  // Buffer case
                vload_decls += " = vload8(global_id, input" + toCodeString(i);
            vload_decls += ");";
        }
        jit.AddConstant(MakeJitConstant("VLOAD_DECLS", vload_decls));
    }
    return jit;
}

JitConstants EltwiseKernelBase::MakeInputDeclsJitConstants(const eltwise_params& params,
                                                           bool /*useVload8*/) const {
    JitConstants jit = {};
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
        inputs_decls += const_str + " __global " + toCLType(params.inputs[i].GetDType()) + "* input" + toCodeString(i) + ", ";
    }
    jit.AddConstant(MakeJitConstant("INPUTS_DECLS", inputs_decls));
    return jit;
}

JitConstants EltwiseKernelBase::MakeIndexJitConstants(const eltwise_params& params,
                                                      bool useVload8) const {
    JitConstants jit = {};
    auto& updateInputs = params.updateInputIds;

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
            } else if (l == DataLayout::byxf) {
                bfyx_idx_order = { "d4", "d1", "d3", "d2" };
            } else if (l == DataLayout::fs_b_yx_fsv32) {
                bfyx_idx_order = { "d3", "d4", "d2", "d1" };
            } else {
                bfyx_idx_order = { "d4", "d3", "d2", "d1" };
            }
        }

        if (!params.stride.empty()) {
            bfyx_idx_order[2] = "(" + bfyx_idx_order[2] + "*" + toCodeString(stride.y) + ")";
            bfyx_idx_order[3] = "(" + bfyx_idx_order[3] + "*" + toCodeString(stride.x) + ")";
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

    std::string out_idx_order = "OUTPUT_IDX_ORDER";
    if (useVload8) {
        jit.AddConstant(MakeJitConstant(out_idx_order, "d1"));
    } else {
        if (CheckInputsOutputNoPitchSameDims(params) &&
            !(params.layoutBased || params.int8_quantization || params.broadcast)) {
            jit.AddConstant(MakeJitConstant(out_idx_order, "d1"));
        } else {
            size_t out_c = DataTensor::ChannelsCount(params.outputs[0].GetLayout());
            if (out_c <= 4) {
                jit.AddConstant(MakeJitConstant(out_idx_order, GetIdxOrderStringForLayout(params.outputs[0].GetLayout(),
                                                                                          params.layoutBased || params.broadcast,
                                                                                          {1, 1, 1})));
            } else {
                std::string idx_order;
                for (size_t i = 0; i < out_c; i++) {
                    idx_order += "d" + std::to_string(out_c - i) + ((i == (out_c - 1)) ? "" : ",");
                }
                jit.AddConstant(MakeJitConstant(out_idx_order, idx_order));
            }
        }
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

        if (!params.stride.empty()) {
            jit.AddConstant(MakeJitConstant("INPUT" + toCodeString(i) + "_STRIDE_X", params.stride[i].x));
            jit.AddConstant(MakeJitConstant("INPUT" + toCodeString(i) + "_STRIDE_Y", params.stride[i].y));
            jit.AddConstant(MakeJitConstant("INPUT" + toCodeString(i) + "_STRIDE_Z", params.stride[i].z));
        }

        std::string idx_order = "INPUT" + toCodeString(i) + "_IDX_ORDER";
        if (useVload8) {
            jit.AddConstant(MakeJitConstant(idx_order, "d1"));
        } else {
            if (CheckInputsOutputNoPitchSameDims(params) &&
                !(params.layoutBased || params.int8_quantization || params.broadcast)) {
                jit.AddConstant(MakeJitConstant(idx_order, "d1"));
            } else {
                size_t in_c = DataTensor::ChannelsCount(params.inputs[i].GetLayout());
                size_t out_c = DataTensor::ChannelsCount(params.outputs[0].GetLayout());
                auto in_stride = params.stride.empty() ? uSize{1, 1, 1} : params.stride[i];
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
                } else if (out_c == 6) {
                    if (in_c < 5) {
                        jit.AddConstant(MakeJitConstant(idx_order, "d6,d5,d2,d1"));
                    } else if (in_c == 5) {
                        jit.AddConstant(MakeJitConstant(idx_order, "d6,d5,d3,d2,d1"));
                    } else {
                        jit.AddConstant(MakeJitConstant(idx_order, "d6,d5,d4,d3,d2,d1"));
                    }
                } else if (out_c == 7) {
                    if (in_c < 5) {
                        jit.AddConstant(MakeJitConstant(idx_order, "d7,d6,d2,d1"));
                    } else if (in_c == 5) {
                        jit.AddConstant(MakeJitConstant(idx_order, "d7,d6,d3,d2,d1"));
                    } else if (in_c == 6) {
                        jit.AddConstant(MakeJitConstant(idx_order, "d7,d6,d4,d3,d2,d1"));
                    } else {
                        jit.AddConstant(MakeJitConstant(idx_order, "d7,d6,d5,d4,d3,d2,d1"));
                    }
                } else if (out_c == 8) {
                    if (in_c < 5) {
                        jit.AddConstant(MakeJitConstant(idx_order, "d8,d7,d2,d1"));
                    } else if (in_c == 5) {
                        jit.AddConstant(MakeJitConstant(idx_order, "d8,d7,d3,d2,d1"));
                    } else if (in_c == 6) {
                        jit.AddConstant(MakeJitConstant(idx_order, "d8,d7,d4,d3,d2,d1"));
                    } else if (in_c == 7) {
                        jit.AddConstant(MakeJitConstant(idx_order, "d8,d7,d5,d4,d3,d2,d1"));
                    } else {
                        jit.AddConstant(MakeJitConstant(idx_order, "d8,d7,d6,d5,d4,d3,d2,d1"));
                    }
                } else {
                    assert(0);
                }
            }
        }
    }

    return jit;
}

JitConstants EltwiseKernelBase::GetJitConstantsCommon(const eltwise_params& params, bool useVload8) const {
    JitConstants jit = MakeBaseParamsJitConstants(params);

    jit.AddConstants({
        MakeJitConstant("ELTWISE_LAYOUT_BASED", params.layoutBased),
        MakeJitConstant("QUANTIZATION_TERM", params.int8_quantization),
        MakeJitConstant("ELTWISE_BROADCAST", params.broadcast),
    });

    jit.Merge(MakeTypeJitConstants(GetAccumulatorType(params), "ACCUMULATOR"));
    jit.AddConstant(MakeJitConstant("ELTWISE_NO_PITCH_SAME_DIMS", CheckInputsOutputNoPitchSameDims(params)));

    jit.Merge(MakeInputDeclsJitConstants(params, useVload8));
    jit.Merge(MakeIndexJitConstants(params, useVload8));
    jit.Merge(MakeLoadJitConstants(params, useVload8));
    jit.Merge(GetOperationsJitConstants(params, useVload8));

    std::string do_eltwise;
    auto& operations = params.operations;
    for (size_t op_num = 0; op_num < operations.size(); op_num++) {
        do_eltwise += "\\\n\tOPERATION" + toCodeString(op_num) + ";";
    }

    auto& updateInputs = params.updateInputIds;
    for (size_t update_input_idx = 0; update_input_idx < updateInputs.size(); update_input_idx++)
        do_eltwise += "\\\n\tinput" + toCodeString(updateInputs[update_input_idx].inputId) + "[GET_INDEX(INPUT, " +
                      toCodeString(updateInputs[update_input_idx].inputId) + ", " +
                      "INPUT"+toCodeString(updateInputs[update_input_idx].inputId) + "_IDX_ORDER)] = tmp" +
                      toCodeString(updateInputs[update_input_idx].tmpId) + ";";

    do_eltwise += "\\\n\tres = tmp" + toCodeString(operations.size() - 1) + ";";

    jit.AddConstant(MakeJitConstant("DO_ELTWISE", do_eltwise));

    if (params.layoutBased || params.int8_quantization || params.broadcast) {
        jit.Merge(GetTensorFriendlyWorkGroupsJit(params.outputs[0]));
    }

    if (!params.stride.empty()) {
        jit.AddConstant(MakeJitConstant("INPUT_STRIDED", 1));
    }

    jit.Merge(MakeActivationJitConstants(params.activations, GetAccumulatorType(params), "_TYPED"));

    return jit;
}

JitConstants EltwiseKernelBase::GetJitConstants(const eltwise_params& params) const {
    return GetJitConstantsCommon(params, false);
}

EltwiseKernelBase::DispatchData EltwiseKernelBase::SetDefault(const eltwise_params& params) const {
    DispatchData dispatchData;

    if (params.layoutBased || params.int8_quantization || params.broadcast) {
        dispatchData.gws = GetTensorFriendlyWorkGroups(params.outputs[0]);
    } else if (CheckInputsOutputNoPitchSameDims(params)) {
        dispatchData.gws[0] = params.outputs[0].LogicalSize();
        dispatchData.gws[1] = 1;
        dispatchData.gws[2] = 1;
    } else {
        const auto& out = params.outputs[0];

        std::vector<size_t> gws;
        for (const auto& o : out.GetDims()) {
            gws.push_back(o.v);
        }

        size_t n_dims = DataTensor::ChannelsCount(out.GetLayout());
        for (size_t i = gws.size(); i < n_dims; i++) {
            gws.push_back(1U);
        }

        dispatchData.gws[0] = gws[0];
        if (n_dims == 8) {
            dispatchData.gws[1] = gws[1] * gws[2] * gws[3] * gws[4] * gws[5];  // y*z*w*u*v
            dispatchData.gws[2] = gws[6] * gws[7];
        } else if (n_dims == 7) {
            dispatchData.gws[1] = gws[1] * gws[2] * gws[3] * gws[4];  // y*z*w*u
            dispatchData.gws[2] = gws[5] * gws[6];
        } else if (n_dims == 6) {
            dispatchData.gws[1] = gws[1] * gws[2] * gws[3];  // y*z*w
            dispatchData.gws[2] = gws[4] * gws[5];
        } else if (n_dims == 5) {
            dispatchData.gws[1] = gws[1] * gws[2];  // y*z
            dispatchData.gws[2] = gws[3] * gws[4];
        } else {
            dispatchData.gws[1] = gws[1];
            dispatchData.gws[2] = gws[2] * gws[3];
        }
    }

    auto local = GetOptimalLocalWorkGroupSizes({dispatchData.gws[0], dispatchData.gws[1], dispatchData.gws[2]}, params.engineInfo);

    // TODO: can be potentially improved for GPUs with support of LWS > 256
    const size_t optimal_lws_values[] = { 256, 224, 192, 160, 128, 96, 64, 32, 16 };

    if (dispatchData.gws[2] % 16 == 0 &&
        params.outputs[0].Batch().v % 16 == 0 &&
        params.outputs[0].Feature().v % 16 == 0 &&
        dispatchData.gws[1] % 16 == 0 &&
        (params.outputs[0].GetLayout() == DataLayout::bs_fs_yx_bsv32_fsv16 ||
        params.outputs[0].GetLayout() == DataLayout::bs_fs_yx_bsv16_fsv16 ||
        params.outputs[0].GetLayout() == DataLayout::bs_fs_zyx_bsv32_fsv16 ||
        params.outputs[0].GetLayout() == DataLayout::bs_fs_zyx_bsv16_fsv16 ||
        params.outputs[0].GetLayout() == DataLayout::bs_fs_zyx_bsv16_fsv32 ||
        params.outputs[0].GetLayout() == DataLayout::bs_fs_zyx_bsv32_fsv32)) {
        dispatchData.lws[0] = 1;
        //dispatchData.gws[1] = ???; calc it below
        dispatchData.lws[2] = 16;
        for (auto lws : optimal_lws_values) {
            if (dispatchData.gws[1] % lws == 0 && lws * dispatchData.lws[2] <= params.engineInfo.maxWorkGroupSize) {
                dispatchData.lws[1] = lws;
                break;
            }
        }
    } else if ((params.outputs[0].GetLayout() == DataLayout::b_fs_yx_fsv16 ||
         params.outputs[0].GetLayout() == DataLayout::b_fs_zyx_fsv16 ||
         params.outputs[0].GetLayout() == DataLayout::bs_fs_yx_bsv32_fsv16 ||
         params.outputs[0].GetLayout() == DataLayout::bs_fs_yx_bsv16_fsv16 ||
         params.outputs[0].GetLayout() == DataLayout::bs_fs_zyx_bsv32_fsv16 ||
         params.outputs[0].GetLayout() == DataLayout::bs_fs_zyx_bsv16_fsv16) &&
        params.outputs[0].Feature().v % 16 == 0 && dispatchData.gws[1] % 16 == 0) {
        dispatchData.lws[0] = 1;
        for (auto lws : optimal_lws_values) {
            if (dispatchData.gws[1] % lws == 0) {
                dispatchData.lws[1] = lws;
                break;
            }
        }
        dispatchData.lws[2] = 1;
    } else if (params.outputs[0].GetLayout() == DataLayout::fs_b_yx_fsv32) {
        dispatchData.gws[2] = Align(dispatchData.gws[2], 32);
        dispatchData.lws[0] = 1;
        dispatchData.lws[1] = 1;
        dispatchData.lws[2] = 32;
    } else if ((params.outputs[0].GetLayout() == DataLayout::b_fs_yx_fsv32 ||
                params.outputs[0].GetLayout() == DataLayout::b_fs_zyx_fsv32 ||
                params.outputs[0].GetLayout() == DataLayout::bs_fs_yx_bsv16_fsv32 ||
                params.outputs[0].GetLayout() == DataLayout::bs_fs_yx_bsv32_fsv32 ||
                params.outputs[0].GetLayout() == DataLayout::bs_fs_zyx_bsv16_fsv32 ||
                params.outputs[0].GetLayout() == DataLayout::bs_fs_zyx_bsv32_fsv32)) {
        if (params.layoutBased || params.int8_quantization || params.broadcast) {
            auto bs_fsv32_local = GetLimitedOptimalLocalWorkGroupSizes({dispatchData.gws[1], dispatchData.gws[2], dispatchData.gws[0]},
                                                                        params.engineInfo, {32, 32, 1024});
            dispatchData.lws[0] = bs_fsv32_local[2];
            dispatchData.lws[1] = bs_fsv32_local[0];
            dispatchData.lws[2] = bs_fsv32_local[1];
        } else if (dispatchData.gws[0] == params.outputs[0].LogicalSize()) {
            dispatchData.lws = local;
        } else {
            auto bs_fsv32_local = GetOptimalLocalWorkGroupSizes({dispatchData.gws[2], dispatchData.gws[0], dispatchData.gws[1]}, params.engineInfo);
            dispatchData.lws[0] = bs_fsv32_local[1];
            dispatchData.lws[1] = bs_fsv32_local[2];
            dispatchData.lws[2] = bs_fsv32_local[0];
        }
    } else if ((params.outputs[0].GetLayout() == DataLayout::bs_fs_yx_bsv32_fsv16 ||
                params.outputs[0].GetLayout() == DataLayout::bs_fs_yx_bsv16_fsv16 ||
                params.outputs[0].GetLayout() == DataLayout::bs_fs_zyx_bsv16_fsv16 ||
                params.outputs[0].GetLayout() == DataLayout::bs_fs_zyx_bsv32_fsv16) &&
                (params.outputs[0].Feature().v % 16 != 0 || dispatchData.gws[1] % 16 != 0)) {
            auto bs_fsv16_local = GetLimitedOptimalLocalWorkGroupSizes({dispatchData.gws[2], dispatchData.gws[0], dispatchData.gws[1]},
                                                                        params.engineInfo, {32 * 16, 1024, 1024});
            dispatchData.lws[0] = bs_fsv16_local[1];
            dispatchData.lws[1] = bs_fsv16_local[2];
            dispatchData.lws[2] = bs_fsv16_local[0];
    } else {
        dispatchData.lws[0] = local[0];
        dispatchData.lws[1] = local[1];
        dispatchData.lws[2] = local[2];
    }

    return dispatchData;
}

void EltwiseKernelBase::GetUpdateDispatchDataFunc(KernelData& kd) const {
    kd.update_dispatch_data_func = [this](const Params& params, KernelData& kd) {
        const auto& prim_params = static_cast<const eltwise_params&>(params);
        auto dispatchData = SetDefault(prim_params);
        OPENVINO_ASSERT(kd.kernels.size() == 1, "[GPU] Invalid kernels size for update dispatch data func");
        kd.kernels[0].params.workGroups.global = dispatchData.gws;
        kd.kernels[0].params.workGroups.local = dispatchData.lws;
        kd.kernels[0].skip_execution = KernelData::SkipKernelExecution(prim_params);
    };
}

KernelsData EltwiseKernelBase::GetCommonKernelsData(const Params& params) const {
    if (!Validate(params)) {
        return {};
    }

    KernelData kd = KernelData::Default<eltwise_params>(params);
    eltwise_params& newParams = *static_cast<eltwise_params*>(kd.params.get());

    auto entry_point = GetEntryPoint(kernelName, newParams.layerID, params);
    auto cldnn_jit = GetJitConstants(newParams);
    auto jit = CreateJit(kernelName, cldnn_jit, entry_point);

    GetUpdateDispatchDataFunc(kd);

    DispatchData dispatchData = SetDefault(newParams);

    auto& kernel = kd.kernels[0];

    kernel.code.kernelString = GetKernelString(kernelName, jit, entry_point, params.engineInfo, EXE_MODE_DEFAULT);

    kernel.params.workGroups.global = dispatchData.gws;
    kernel.params.workGroups.local = dispatchData.lws;
    bool is_dynamic = newParams.is_shape_agnostic;
    kernel.params.arguments = GetArgsDesc((uint32_t)newParams.inputs.size(),
                                   false,
                                   false,
                                   GetFusedPrimitiveInputsCount(params),
                                   1,
                                   is_dynamic);
    if (params.is_shape_agnostic && newParams.operations[0].mode == EltwiseMode::ASSIGN) {
        kernel.params.arguments.push_back({ArgumentDescriptor::Types::SCALAR, 0});
        kernel_selector::ScalarDescriptor s;
        s.t = kernel_selector::ScalarDescriptor::Types::UINT32;
        s.v.u32 = 0;
        kernel.params.scalars.push_back(s);
    }
    return {kd};
}
}  // namespace kernel_selector
