/*
// Copyright (c) 2016 Intel Corporation
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

#include "eltwise_kernel_base.h"
#include "kernel_selector_utils.h" 

namespace kernel_selector
{
    static uint32_t GetNumberOfInputs(EltwiseMode m)
    {
        switch (m)
        {
        case EltwiseMode::ADD:
        case EltwiseMode::SUB:
        case EltwiseMode::MUL:
        case EltwiseMode::DIV:
        case EltwiseMode::MIN:
        case EltwiseMode::MAX:
        case EltwiseMode::POW:
        case EltwiseMode::MODULU:
            return 2;
        case EltwiseMode::SQRT:
        case EltwiseMode::RSQRT:
        case EltwiseMode::ASSIGN:
            return 1;
        default:
            return 0;
        }
    }

    ParamsKey eltwise_params::GetParamsKey() const
    {
        ParamsKey k = base_params::GetParamsKey();
        if (int8_quantization)
        {
            k.EnableInt8Quantization();
        }

        if (output_calibration)
        {
            k.EnableOutputCalibration();
        }

        return k;
    }

    bool EltwiseKernelBase::Validate(const Params& p, const optional_params& o) const
    {
        if (p.GetType() != KernelType::ELTWISE ||
            o.GetType() != KernelType::ELTWISE)
        {
            return false;
        }

        const eltwise_params& params = static_cast<const eltwise_params&>(p);

        if (params.inputs.size() == 0)
        {
            return false;
        }

        auto& operations = params.operations;

        if (operations.size() == 0)
        {
            return false;
        }

        for (size_t op_num = 0; op_num < operations.size(); op_num++)
        {
            const auto& ew = operations[op_num];

            if (ew.inputs.size() != GetNumberOfInputs(ew.mode))
            {
                return false;
            }

            for (size_t input_idx = 0; input_idx < ew.inputs.size(); input_idx++)
            {
                const auto& input = ew.inputs[input_idx];
                if (input.mode == EltwiseInputMode::INPUT_BUFFER &&
                    input.index >= params.inputs.size())
                {
                    return false;
                }
            }
        }

        return true;
    }

    JitConstants EltwiseKernelBase::GetJitConstantsCommon(const eltwise_params& params, bool useVload8) const
    {
        JitConstants jit = MakeBaseParamsJitConstants(params);

        jit.AddConstants({
            MakeJitConstant("ELTWISE_LAYOUT_BASED", params.layoutBased),
            MakeJitConstant("QUANTIZATION_TERM",    params.int8_quantization),
        });

        if (params.int8_quantization)
        {
            if (params.output_calibration)
            {
                jit.AddConstant(MakeJitConstant("CALIBRATION_TERM", params.output_calibration));
                jit.AddConstant(MakeJitConstant("O_QF", params.output_calibration_factors[0]));

            }
            else
                jit.AddConstants({ MakeJitConstant("O_QF",       params.output_quantization_factor) });
        }

        std::string inputs_decls, vload_decls;
        auto& updateInputs = params.updateInputIds;

        for (size_t i = 0; i < params.inputs.size(); i++)
        {
            //const should be added only to inputs which will not be updated
            std::string const_str = "const";
            for (size_t update_input_idx = 0; update_input_idx < updateInputs.size(); update_input_idx++)
            {
                if (updateInputs[update_input_idx].inputId == i)
                {
                    const_str = "";
                    break;
                }
            }

            inputs_decls += const_str + " __global " + toCLType(params.inputs[i].GetDType()) + "* input" + std::to_string(i) + ", ";
            if (useVload8)
            {
                vload_decls += "\\\n\tconst " + toCLType(params.inputs[i].GetDType()) + "8 in" + std::to_string(i);
                if (params.inputs[i].PhysicalSize() == 1) //Scalar case
                    vload_decls += " = (" + toCLType(params.inputs[i].GetDType()) + "8)(input" + std::to_string(i) + "[0]";
                else //Buffer case
                    vload_decls += " = vload8(global_id, input" + std::to_string(i);
                vload_decls += ");";
            }
        }

        jit.AddConstant(MakeJitConstant("INPUTS_DECLS", inputs_decls));
        jit.AddConstant(MakeJitConstant("ELTWISE_NO_PITCH_SAME_DIMS", CheckInputsOutputNoPitchSameDims(params)));

        if (useVload8)
            jit.AddConstant(MakeJitConstant("VLOAD_DECLS", vload_decls));

        std::string do_eltwise;

        auto& operations   = params.operations;
        auto& coefficients = params.coefficients;

        for (size_t op_num = 0; op_num < operations.size(); op_num++)
        {
            const std::string op_num_str = std::to_string(op_num);
            const auto& ew = operations[op_num];

            for (size_t input_idx = 0; input_idx < ew.inputs.size(); input_idx++)
            {
                const auto& input = ew.inputs[input_idx];
                const std::string name = "INPUT_" + op_num_str + "_" + std::to_string(input_idx);
                switch (input.mode)
                {
                case EltwiseInputMode::SCALAR:
                    jit.AddConstant(MakeJitConstant(name, input.scalar));
                    break;
                case EltwiseInputMode::INPUT_BUFFER:
                    if(useVload8)
                        jit.AddConstant(MakeJitConstant(name, "in" + std::to_string(input.index)));
                    else
                        jit.AddConstant(MakeJitConstant(name, "input" + std::to_string(input.index) + "[GET_INDEX(INPUT, " + std::to_string(input.index) + ")]"));
                    break;
                case EltwiseInputMode::OUTPUT_BUFFER:
                    jit.AddConstant(MakeJitConstant(name, "output[GET_INDEX(OUTPUT, )]"));
                    break;
                case EltwiseInputMode::UNORDERED_ACCESS_INPUT_BUFFER:
                    jit.AddConstant(MakeJitConstant(name, "input" + std::to_string(input.index) + "[(size_t)tmp" + std::to_string(input.tmpIndex) + "]"));
                    break;
                case EltwiseInputMode::INTERMEDIATE_RESULTS_INDEX:
                    jit.AddConstant(MakeJitConstant(name, "tmp" + std::to_string(input.tmpIndex)));
                    break;
                default:
                    break;
                }
            }

            std::string input0_str, input1_str, cast_type, op;

            if (useVload8)
            {
                cast_type = "(MAKE_VECTOR_TYPE(UNIT_TYPE, 8))";
                op = "const MAKE_VECTOR_TYPE(UNIT_TYPE, 8) tmp" + op_num_str + " = ";
            }
            else if(params.int8_quantization)
            {
                cast_type = "(int)";
                op = "const int tmp" + op_num_str + " = ";
            }
            else
            {
                cast_type = "(UNIT_TYPE)";
                op = "const UNIT_TYPE tmp" + op_num_str + " = ";
            }

            input0_str = cast_type + "INPUT_" + op_num_str + "_0";
            input1_str = cast_type + "INPUT_" + op_num_str + "_1";

            if (ew.mode == EltwiseMode::ADD)
            {
                std::vector<std::string> coeff_strings(ew.inputs.size(), "");
                for (size_t input_idx = 0; input_idx < ew.inputs.size(); input_idx++)
                {
                    const auto& input = ew.inputs[input_idx];
                    if (input.mode == EltwiseInputMode::INPUT_BUFFER && input.index < coefficients.size())
                    {
                        const float c = coefficients[input.index];
                        if (c != 1.0f)
                            coeff_strings[input_idx] = cast_type + "(" + std::to_string(c) + ")*";
                    }
                }

                input0_str = coeff_strings[0] + input0_str;
                input1_str = coeff_strings[1] + input1_str;
            }


            switch (ew.mode)
            {
            case EltwiseMode::ADD:      op += input0_str + " + " + input1_str; break;
            case EltwiseMode::SUB:      op += input0_str + " - " + input1_str; break;
            case EltwiseMode::MUL:      op += input0_str + " * " + input1_str; break;
            case EltwiseMode::DIV:      op += input0_str + " / " + input1_str; break;
            case EltwiseMode::MODULU:   op += cast_type + "fmod(" + input0_str + ", " + input1_str + ")"; break;
            case EltwiseMode::MIN:      op += cast_type + "fmin(" + input0_str + ", " + input1_str + ")"; break;
            case EltwiseMode::MAX:      op += cast_type + "fmax(" + input0_str + ", " + input1_str + ")"; break;
            case EltwiseMode::POW:      op += cast_type + "pow(" + input0_str + ", " + input1_str + ")"; break;
            case EltwiseMode::SQRT:     op += cast_type + "sqrt(" + input0_str + ")"; break;
            case EltwiseMode::RSQRT:    op += cast_type + "1/sqrt(" + input0_str + ")"; break;
            case EltwiseMode::ASSIGN:   op += input0_str; break;
            default:
                break;
            }

            std::string opname = "OPERATION" + op_num_str;
            jit.AddConstant(MakeJitConstant(opname, op));
            do_eltwise += "\\\n\t" + opname + ";";
        }

        for (size_t update_input_idx = 0; update_input_idx < updateInputs.size(); update_input_idx++)
            do_eltwise += "\\\n\tinput" + std::to_string(updateInputs[update_input_idx].inputId) + 
            "[GET_INDEX(INPUT, " + std::to_string(updateInputs[update_input_idx].inputId) +
            ")] = tmp" + std::to_string(updateInputs[update_input_idx].tmpId) + ";";

        do_eltwise += "\\\n\tres = tmp" + std::to_string(operations.size() - 1) + ";";

        jit.AddConstant(MakeJitConstant("DO_ELTWISE", do_eltwise));

        if (params.layoutBased || params.int8_quantization)
        {
            jit.Merge(GetTensorFriendlyWorkGroupsJit(params.inputs[0]));
        }

        return jit;
    }

    JitConstants EltwiseKernelBase::GetJitConstants(const eltwise_params& params) const
    {
        return GetJitConstantsCommon(params, false);
    }

    EltwiseKernelBase::DispatchData EltwiseKernelBase::SetDefault(const eltwise_params& params) const
    {
        DispatchData kd;

        if (params.layoutBased || params.int8_quantization)
        {
            auto global = GetTensorFriendlyWorkGroups(params.inputs[0]);
            kd.gws0 = global[0];
            kd.gws1 = global[1];
            kd.gws2 = global[2];
        }
        else if (CheckInputsOutputNoPitchSameDims(params))
        {
            kd.gws0 = params.inputs[0].LogicalSize();
            kd.gws1 = 1;
            kd.gws2 = 1;
        }
        else
        {
            const auto& out = params.output;

            std::vector<size_t> gws;
            for (const auto& o : out.GetDims())
            {
                gws.push_back(o.v);
            }

            for (size_t i = gws.size(); i < 4; i++)
            {
                gws.push_back(1U);
            }

            kd.gws0 = gws[0];
            kd.gws1 = gws[1];
            kd.gws2 = gws[2] * gws[3];
        }

        auto local = GetOptimalLocalWorkGroupSizes( { kd.gws0, kd.gws1, kd.gws2 } );
        kd.lws0 = local[0];
        kd.lws1 = local[1];
        kd.lws2 = local[2];

        return kd;
    }

    KernelsData EltwiseKernelBase::GetCommonKernelsData(const Params& params, const optional_params& options) const
    {
        if (!Validate(params, options))
        {
            return{};
        }

        KernelData kd = KernelData::Default<eltwise_params>(params);
        eltwise_params& newParams = *static_cast<eltwise_params*>(kd.params.get());

        auto entry_point = GetEntryPoint(kernelName, newParams.layerID, options);
        auto cldnn_jit = GetJitConstants(newParams);
        std::string jit = CreateJit(kernelName, cldnn_jit, entry_point);

        DispatchData runInfo = SetDefault(newParams);

        auto& kernel = kd.kernels[0];

        kernel.workGroups.global = { runInfo.gws0, runInfo.gws1, runInfo.gws2 };
        kernel.workGroups.local = { runInfo.lws0, runInfo.lws1, runInfo.lws2 };

        kernel.kernelString = GetKernelString(kernelName, jit, entry_point, params.engineInfo, ROUND_ROBIN);
        kernel.arguments = GetArgsDesc((uint32_t)newParams.inputs.size(), false, false, newParams.int8_quantization, newParams.output_calibration);

        kd.estimatedTime = DONT_USE_IF_HAVE_SOMETHING_ELSE;

        return{ kd };
    }
}
