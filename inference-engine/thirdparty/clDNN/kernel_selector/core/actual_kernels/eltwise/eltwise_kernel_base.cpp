// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

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

    if (!stride.empty()) {
        k.EnableEltwiseStride();
    }

    if (broadcast) {
        k.EnableEltwiseBroadcast();
    }

    return k;
}

Datatype EltwiseKernelBase::GetAccumulatorType(const eltwise_params &params) const {
    if (params.int8_quantization)
        return Datatype::INT32;

    Datatype types[] = { Datatype::F32, Datatype::F16, Datatype::INT64, Datatype::INT32, Datatype::UINT32};

    for (Datatype type : types)
        for (auto& in : params.inputs)
            if (in.GetDType() == type)
                return type;

    return Datatype::F32;
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
        const std::string op_num_str = std::to_string(op_num);
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
            case EltwiseMode::FLOOR_MOD:
                op += "(" + input0_str + " - floor(" + input0_str + " / " + input1_str + ") * " + input1_str + ")";
                break;
            case EltwiseMode::ASSIGN:
                op += input0_str;
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
        const std::string op_num_str = std::to_string(op_num);
        const auto &ew = params.operations[op_num];
        for (size_t input_idx = 0; input_idx < ew.inputs.size(); input_idx++) {
            const auto &input = ew.inputs[input_idx];
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
                                                        "," + idx_order + ")]"));
                    break;
                case EltwiseInputMode::OUTPUT_BUFFER:
                    jit.AddConstant(MakeJitConstant(name, "output[GET_INDEX(OUTPUT,,OUTPUT_IDX_ORDER)]"));
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

    if (useVload8) {
        for (size_t i = 0; i < params.inputs.size(); i++) {
            vload_decls += "\\\n\tconst " + toCLType(params.inputs[i].GetDType()) + "8 in" + std::to_string(i);
            if (params.inputs[i].PhysicalSize() == 1)  // Scalar case
                vload_decls += " = (" + toCLType(params.inputs[i].GetDType()) + "8)(input" + std::to_string(i) + "[0]";
            else  // Buffer case
                vload_decls += " = vload8(global_id, input" + std::to_string(i);
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
        inputs_decls += const_str + " __global " + toCLType(params.inputs[i].GetDType()) + "* input" + std::to_string(i) + ", ";
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

    std::string out_idx_order = "OUTPUT_IDX_ORDER";
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
                                                                                          {1, 1, 1})));
            } else if (out_c == 5) {
                jit.AddConstant(MakeJitConstant(out_idx_order, "d5,d4,d3,d2,d1"));
            } else if (out_c == 6) {
                jit.AddConstant(MakeJitConstant(out_idx_order, "d6,d5,d4,d3,d2,d1"));
            } else {
                assert(0);
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
            jit.AddConstant(MakeJitConstant("INPUT" + std::to_string(i) + "_STRIDE_X", params.stride[i].x));
            jit.AddConstant(MakeJitConstant("INPUT" + std::to_string(i) + "_STRIDE_Y", params.stride[i].y));
            jit.AddConstant(MakeJitConstant("INPUT" + std::to_string(i) + "_STRIDE_Z", params.stride[i].z));
        }

        std::string idx_order = "INPUT" + std::to_string(i) + "_IDX_ORDER";
        if (useVload8) {
            jit.AddConstant(MakeJitConstant(idx_order, "d1"));
        } else {
            if (CheckInputsOutputNoPitchSameDims(params) &&
                !(params.layoutBased || params.int8_quantization || params.broadcast)) {
                jit.AddConstant(MakeJitConstant(idx_order, "d1"));
            } else {
                size_t in_c = DataTensor::ChannelsCount(params.inputs[i].GetLayout());
                size_t out_c = DataTensor::ChannelsCount(params.output.GetLayout());
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
        do_eltwise += "\\\n\tOPERATION" + std::to_string(op_num) + ";";
    }

    auto& updateInputs = params.updateInputIds;
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

    jit.Merge(MakeActivationJitConstants(params.activations, GetAccumulatorType(params), "_TYPED"));

    return jit;
}

JitConstants EltwiseKernelBase::GetJitConstants(const eltwise_params& params) const {
    return GetJitConstantsCommon(params, false);
}

EltwiseKernelBase::DispatchData EltwiseKernelBase::SetDefault(const eltwise_params& params) const {
    DispatchData dispatchData;

    if (params.layoutBased || params.int8_quantization || params.broadcast) {
        dispatchData.gws = GetTensorFriendlyWorkGroups(params.output);
    } else if (CheckInputsOutputNoPitchSameDims(params)) {
        dispatchData.gws[0] = params.output.LogicalSize();
        dispatchData.gws[1] = 1;
        dispatchData.gws[2] = 1;
    } else {
        const auto& out = params.output;

        std::vector<size_t> gws;
        for (const auto& o : out.GetDims()) {
            gws.push_back(o.v);
        }

        size_t n_dims = DataTensor::ChannelsCount(out.GetLayout());
        for (size_t i = gws.size(); i < n_dims; i++) {
            gws.push_back(1U);
        }

        dispatchData.gws[0] = gws[0];
        if (n_dims == 6) {
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

    const size_t optimal_lws_values[] = {256, 224, 192, 160, 128, 96, 64, 32, 16};
    if ((params.output.GetLayout() == DataLayout::b_fs_yx_fsv16 ||
         params.output.GetLayout() == DataLayout::b_fs_zyx_fsv16 ||
         params.output.GetLayout() == DataLayout::bs_fs_yx_bsv16_fsv16) &&
        params.output.Feature().v % 16 == 0 && dispatchData.gws[1] % 16 == 0) {
        dispatchData.lws[0] = 1;
        for (auto lws : optimal_lws_values) {
            if (dispatchData.gws[1] % lws == 0) {
                dispatchData.lws[1] = lws;
                break;
            }
        }
        dispatchData.lws[2] = 1;
    } else if (params.output.GetLayout() == DataLayout::fs_b_yx_fsv32) {
        dispatchData.gws[2] = Align(dispatchData.gws[2], 32);
        dispatchData.lws[0] = 1;
        dispatchData.lws[1] = 1;
        dispatchData.lws[2] = 32;
    } else if (params.output.GetLayout() == DataLayout::b_fs_yx_fsv32 && params.output.Feature().v % 32 == 0) {
        if (params.layoutBased || params.int8_quantization || params.broadcast) {
            dispatchData.lws[0] = 1;
            dispatchData.lws[1] = 32;
            dispatchData.lws[2] = 1;
        } else if (dispatchData.gws[0] == params.output.LogicalSize()) {
            dispatchData.lws = local;
        } else {
            dispatchData.lws[0] = 1;
            dispatchData.lws[1] = 1;
            dispatchData.lws[2] = 32;
        }
    } else {
        dispatchData.lws[0] = local[0];
        dispatchData.lws[1] = local[1];
        dispatchData.lws[2] = local[2];
    }

    return dispatchData;
}

KernelsData EltwiseKernelBase::GetCommonKernelsData(const Params& params, const optional_params& options) const {
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
}  // namespace kernel_selector
