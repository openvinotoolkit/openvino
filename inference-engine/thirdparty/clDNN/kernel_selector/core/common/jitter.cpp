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

#include "jitter.h"
#include "tensor_type.h"

namespace kernel_selector {

    std::string toCLType(WeightsType wType)
    {
        switch (wType)
        {
        case WeightsType::INT8: return GetTypeName<int8_t>();
        case WeightsType::UINT8: return GetTypeName<uint8_t>();
        case WeightsType::F16:  return "half";
        case WeightsType::F32:  return GetTypeName<float>();
        default: return "";
        }
    }

    std::string toCLType(Datatype dType)
    {
        switch (dType)
        {
        case Datatype::INT8:    return GetTypeName<int8_t>();
        case Datatype::UINT8:   return GetTypeName<uint8_t>();
        case Datatype::INT16:   return GetTypeName<int16_t>();
        case Datatype::UINT16:  return GetTypeName<uint16_t>();
        case Datatype::INT32:   return GetTypeName<int32_t>();
        case Datatype::UINT32:  return GetTypeName<uint32_t>();
        case Datatype::INT64:   return GetTypeName<int64_t>();
        case Datatype::F16:     return "half";
        case Datatype::F32:     return GetTypeName<float>();
        default: return "";
        }
    }

    std::string getMeanOpString(MeanOp op)
    {
        switch (op)
        {
        case MeanOp::NONE:   return "val";
        case MeanOp::DIV:    return "val/mean_val";
        case MeanOp::MUL:    return "val*mean_val";
        case MeanOp::SUB:    return "val-mean_val";
        default: return "";
        }
    }

    std::string toCodeString(float val) {
        if (std::isinf(val))
            return std::signbit(val) ? "-INFINITY" : "INFINITY";
        std::stringstream ss;
        // Workaround GCC compiler/STL bug
        ss << "as_float(0x" << std::hex << *reinterpret_cast<uint32_t*>(&val) << ")";

        ss << " /*" << std::scientific << val << "*/";
        return ss.str();
    }

    std::string toCodeString(double val) {
        if (std::isinf(val))
            return std::signbit(val) ? "-INFINITY" : "INFINITY";
        std::stringstream ss;
        // Workaround GCC compiler/STL bug
        ss << "as_double(0x" << std::hex << *reinterpret_cast<uint64_t*>(&val) << ")";

        ss << " /*" << std::scientific << val << "*/";
        return ss.str();
    }

    JitDefinitions JitConstants::GetDefinitions() const
    {
        JitDefinitions definitons;
        definitons.reserve(_constants.size() * 6); //assuming max 6 pairs per jit_constant

        for (auto& constant : _constants) {
            auto def = constant->GetDefinitions();
            definitons.insert(definitons.end(), def.begin(), def.end());
        }
        return definitons;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // TensorBaseTJitConstant
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    template<typename DType, typename Layout>
    class TensorBaseTJitConstant : public JitConstant
    {
    protected:
        TensorBaseTJitConstant(const std::string& name) : JitConstant(name) {}

    public:

        JitDefinitions GetDefinitions(const Tensor::TensorBaseT<DType, Layout>& t) const
        {
            JitDefinitions definitions{
                { _name + "_TYPE",          toCLType(t.GetDType()) },
            { _name + "_OFFSET",        toCodeString(t.GetFirstElementOffset()) },
            { _name + "_VIEW_OFFSET",   toCodeString(t.GetViewOffset()) },
            { _name + "_LENGTH",        toCodeString(t.LogicalSize()) },
            { _name + "_DIMS",          toCodeString(t.GetDims().size()) },
            { _name + "_SIMPLE",        toCodeString(t.SimpleLayout()) },
            { "TO_" + _name + "_TYPE",  "convert_" + toCLType(t.GetDType()) },
            { _name + "_LAYOUT_" + toString(t.GetLayout()), "1" },
            };

            definitions.push_back({ _name + "_SIZE",        toCodeString(t.GetDims().size()) });
            definitions.push_back({ _name + "_SIZES",       toVectorString(t.GetDims(), "size_t", KERNEL_SELECTOR_TENSOR_DIM_MAX, 1, [](const Tensor::Dim& d) { return d.v; }) });
            definitions.push_back({ _name + "_PITCHES",     toVectorString(t.GetDims(), "size_t", KERNEL_SELECTOR_TENSOR_DIM_MAX, 1, [](const Tensor::Dim& d) { return d.pitch; }) });
            definitions.push_back({ _name + "_PAD_BEFORE",  toVectorString(t.GetDims(), "size_t", KERNEL_SELECTOR_TENSOR_DIM_MAX, 0, [](const Tensor::Dim& d) { return d.pad.before; }) });
            definitions.push_back({ _name + "_PAD_AFTER",   toVectorString(t.GetDims(), "size_t", KERNEL_SELECTOR_TENSOR_DIM_MAX, 0, [](const Tensor::Dim& d) { return d.pad.after; }) });

            return definitions;
        }
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // DataTensorJitConstant
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    class DataTensorJitConstant : public TensorBaseTJitConstant<Datatype, DataLayout>
    {
        const DataTensor _tensor;

    public:
        DataTensorJitConstant(const std::string& name, const DataTensor& t) : TensorBaseTJitConstant(name), _tensor(t) {}

        JitDefinitions GetDefinitions() const override;
    };

    JitDefinitions DataTensorJitConstant::GetDefinitions() const
    {
        JitDefinitions baseDefinitions = TensorBaseTJitConstant::GetDefinitions(_tensor);

        JitDefinitions definitions{
            { _name + "_SIZE_X",        toCodeString(_tensor.X().v) },
        { _name + "_SIZE_Y",        toCodeString(_tensor.Y().v) },
        { _name + "_FEATURE_NUM",   toCodeString(_tensor.Feature().v) },
        { _name + "_ROI_NUM",       toCodeString(_tensor.ROI().v) },
        { _name + "_BATCH_NUM",     toCodeString(_tensor.Batch().v) },
        { _name + "_X_PITCH",       toCodeString(_tensor.X().pitch) },
        { _name + "_Y_PITCH",       toCodeString(_tensor.Y().pitch) },
        { _name + "_FEATURE_PITCH", toCodeString(_tensor.Feature().pitch) },
        { _name + "_ROI_PITCH",     toCodeString(_tensor.ROI().pitch) },
        { _name + "_BATCH_PITCH",   toCodeString(_tensor.Batch().pitch) },
        { _name + "_PAD_BEFORE_SIZE_X",        toCodeString(_tensor.X().pad.before) },
        { _name + "_PAD_BEFORE_SIZE_Y",        toCodeString(_tensor.Y().pad.before) },
        { _name + "_PAD_BEFORE_FEATURE_NUM",   toCodeString(_tensor.Feature().pad.before) },
        { _name + "_PAD_BEFORE_BATCH_NUM",     toCodeString(_tensor.Batch().pad.before) },
        { _name + "_PAD_AFTER_SIZE_X",         toCodeString(_tensor.X().pad.after) },
        { _name + "_PAD_AFTER_SIZE_Y",         toCodeString(_tensor.Y().pad.after) },
        { _name + "_PAD_AFTER_FEATURE_NUM",    toCodeString(_tensor.Feature().pad.after) },
        { _name + "_PAD_AFTER_BATCH_NUM",      toCodeString(_tensor.Batch().pad.after) },
        };

        definitions.insert(definitions.end(), baseDefinitions.begin(), baseDefinitions.end());

        return definitions;
    }

    std::shared_ptr<JitConstant> MakeJitConstant(const std::string& name, const DataTensor& value)
    {
        return std::static_pointer_cast<JitConstant>(std::make_shared<DataTensorJitConstant>(name, value));
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // WeightTensorJitConstant
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    class WeightTensorJitConstant : public TensorBaseTJitConstant<WeightsType, WeightsLayout>
    {
        const WeightsTensor _tensor;

    public:
        WeightTensorJitConstant(const std::string& name, const WeightsTensor& t) : TensorBaseTJitConstant(name), _tensor(t) {}

        JitDefinitions GetDefinitions() const override;
    };

    JitDefinitions WeightTensorJitConstant::GetDefinitions() const
    {
        JitDefinitions baseDefinitions = TensorBaseTJitConstant::GetDefinitions(_tensor);

        JitDefinitions definitions{
            { _name + "_SIZE_X",        toCodeString(_tensor.X().v) },
            { _name + "_SIZE_Y",        toCodeString(_tensor.Y().v) },
            { _name + "_IFM_NUM",       toCodeString(_tensor.IFM().v) },
            { _name + "_OFM_NUM",       toCodeString(_tensor.OFM().v) },
            { _name + "_X_PITCH",       toCodeString(_tensor.X().pitch) },
            { _name + "_Y_PITCH",       toCodeString(_tensor.Y().pitch) },
            { _name + "_IFM_PITCH",     toCodeString(_tensor.IFM().pitch) },
            { _name + "_OFM_PITCH",     toCodeString(_tensor.OFM().pitch) },
        };

        definitions.insert(definitions.end(), baseDefinitions.begin(), baseDefinitions.end());

        return definitions;
    }

    std::shared_ptr<JitConstant> MakeJitConstant(const std::string& name, const WeightsTensor& value)
    {
        return std::static_pointer_cast<JitConstant>(std::make_shared<WeightTensorJitConstant>(name, value));
    }

    std::shared_ptr<JitConstant> MakeActivationJitConstants(ActivationFunction activation_function, const std::string& suffix)
    {
        std::string name = "ACTIVATION" + suffix;
        // TODO: use native_exp and use cast for APL
        switch (activation_function)
        {
        case ActivationFunction::LOGISTIC:
            return MakeJitConstant(name + "(input, m, n)", "(UNIT_VAL_ONE/(UNIT_VAL_ONE + exp(-input)))");
        case ActivationFunction::HYPERBOLIC_TAN:
            return MakeJitConstant(name + "(input, m, n)", "(tanh(input))");
        case ActivationFunction::RELU:
            return MakeJitConstant(name + "(input, m, n)", "(UNIT_MAX_FUNC(UNIT_VAL_ZERO, input))");
        case ActivationFunction::RELU_NEGATIVE_SLOPE:
            return MakeJitConstant(name + "(input, slope, n)", "isinf(TO_UNIT_TYPE(slope)) ? ((input >= UNIT_VAL_ZERO) ? \
                                                        input : -TO_UNIT_TYPE(slope)) : \
                                                        (UNIT_MAX_FUNC(input, UNIT_VAL_ZERO) + TO_UNIT_TYPE(slope) * UNIT_MIN_FUNC(input, UNIT_VAL_ZERO))");
        case ActivationFunction::ELU:
            return MakeJitConstant(name + "(input, alpha, n)", "(UNIT_MAX_FUNC(input, UNIT_VAL_ZERO) +  \
                                                        TO_UNIT_TYPE(alpha) * (exp(UNIT_MIN_FUNC(input, UNIT_VAL_ZERO)) - UNIT_VAL_ONE));");
        case ActivationFunction::CLAMP:
            return MakeJitConstant(name + "(input, m, n)", "(UNIT_MAX_FUNC(TO_UNIT_TYPE(m), UNIT_MIN_FUNC(TO_UNIT_TYPE(n), input)))");
        case ActivationFunction::SOFTRELU:
            return MakeJitConstant(name + "(input, m, n)", "(log(UNIT_VAL_ONE + exp(input)))");
        case ActivationFunction::ABS:
            return MakeJitConstant(name + "(input, m, n)", "(fabs(input))");
        case ActivationFunction::LINEAR:
            return MakeJitConstant(name + "(input, m, n)", "(m*input + n)");
        case ActivationFunction::SQUARE:
            return MakeJitConstant(name + "(input, m, n)", "(input*input)");
        case ActivationFunction::SQRT:
            return MakeJitConstant(name + "(input, m, n)", "(sqrt(input))");
        case ActivationFunction::SIN:
            return MakeJitConstant(name + "(input, m, n)", "(sin(input))");
        case ActivationFunction::ASIN:
            return MakeJitConstant(name + "(input, m, n)", "(asin(input))");
        case ActivationFunction::SINH:
            return MakeJitConstant(name + "(input, m, n)", "(sinh(input))");
        case ActivationFunction::COS:
            return MakeJitConstant(name + "(input, m, n)", "(cos(input))");
        case ActivationFunction::ACOS:
            return MakeJitConstant(name + "(input, m, n)", "(acos(input))");
        case ActivationFunction::COSH:
            return MakeJitConstant(name + "(input, m, n)", "(cosh(input))");
        case ActivationFunction::LOG:
            return MakeJitConstant(name + "(input, m, n)", "(log(input))");
        case ActivationFunction::LOG2:
            return MakeJitConstant(name + "(input, m, n)", "(log2(input))");
        case ActivationFunction::EXP:
            return MakeJitConstant(name + "(input, m, n)", "(exp(input))");
        case ActivationFunction::NOT:
            return MakeJitConstant(name + "(input, m, n)", "((input != 0) ? UNIT_VAL_ZERO : UNIT_VAL_ONE)");
        case ActivationFunction::RELU_GRAD:
            return MakeJitConstant(name + "(input_grad, input, m, n)", "(input_grad * (input > UNIT_VAL_ZERO ? TO_UNIT_TYPE(1) : TO_UNIT_TYPE(0)))");
        case ActivationFunction::RELU_NEGATIVE_SLOPE_GRAD:
            return MakeJitConstant(name + "(input_grad, input, slope, n)", "(input_grad * ((input > UNIT_VAL_ZERO ? TO_UNIT_TYPE(1) : TO_UNIT_TYPE(0)) + TO_UNIT_TYPE(slope) * (input <= 0 ? TO_UNIT_TYPE(1) : TO_UNIT_TYPE(0))))");
        case ActivationFunction::NONE_GRAD:
            return MakeJitConstant(name + "(input_grad, input, m, n)", "input_grad");
        case ActivationFunction::NONE:
        default:
            return MakeJitConstant(name + "(input, m, n)", "input");
        }
    }

    JitConstants MakeUnitTypeJitConstants(Datatype dataType)
    {
        std::string unit_type;
        std::string unit_max_val;
        std::string unit_min_val;
        std::string unit_val_one;
        std::string unit_val_zero;
        std::string to_unit_type;
        std::string unit_max_func;
        std::string unit_min_func;
        switch (dataType)
        {
        case Datatype::INT8:
            unit_type = "char";
            unit_max_val = "CHAR_MAX";
            unit_min_val = "CHAR_MIN";
            unit_val_one = "(char) 1";
            unit_val_zero = "(char) 0";
            to_unit_type = "convert_char(v)";
            unit_max_func = "max";
            unit_min_func = "min";
            break;
        case Datatype::UINT8:
            unit_type = "uchar";
            unit_max_val = "UCHAR_MAX";
            unit_min_val = "0";
            unit_val_one = "(uchar) 1";
            unit_val_zero = "(uchar) 0";
            to_unit_type = "convert_uchar(v)";
            unit_max_func = "max";
            unit_min_func = "min";
            break;
        case Datatype::INT32:
            unit_type = "int";
            unit_max_val = "INT_MAX";
            unit_min_val = "INT_MIN";
            unit_val_one = "(int) 1";
            unit_val_zero = "(int) 0";
            to_unit_type = "convert_int(v)";
            unit_max_func = "max";
            unit_min_func = "min";
            break;
        case Datatype::UINT32:
            unit_type = "uint";
            unit_max_val = "UINT_MAX";
            unit_min_val = "0";
            unit_val_one = "(uint) 1";
            unit_val_zero = "(uint) 0";
            to_unit_type = "convert_uint(v)";
            unit_max_func = "max";
            unit_min_func = "min";
            break;
        case Datatype::INT64:
            unit_type = "long";
            unit_max_val = "LONG_MAX";
            unit_min_val = "LONG_MIN";
            unit_val_one = "(long) 1";
            unit_val_zero = "(long) 0";
            to_unit_type = "convert_long(v)";
            unit_max_func = "max";
            unit_min_func = "min";
            break;
        case Datatype::F16:
            unit_type = "half";
            unit_max_val = "HALF_MAX";
            unit_min_val = "-UNIT_VAL_MAX";
            unit_val_one = "1.0h";
            unit_val_zero = "0.0h";
            to_unit_type = "convert_half(v)";
            unit_max_func = "fmax";
            unit_min_func = "fmin";
            break;
        default:
            unit_type = "float";
            unit_max_val = "FLT_MAX";
            unit_min_val = "-UNIT_VAL_MAX";
            unit_val_one = "1.0f";
            unit_val_zero = "0.0f";
            to_unit_type = "(float)(v)";
            unit_max_func = "fmax";
            unit_min_func = "fmin";
            break;
        }

        return JitConstants
        {
            MakeJitConstant("UNIT_TYPE",            unit_type),
            MakeJitConstant("UNIT_VAL_MAX",         unit_max_val),
            MakeJitConstant("UNIT_VAL_MIN",         unit_min_val),
            MakeJitConstant("UNIT_VAL_ONE",         unit_val_one),
            MakeJitConstant("UNIT_VAL_ZERO",        unit_val_zero),
            MakeJitConstant("TO_UNIT_TYPE(v)",      to_unit_type),
            MakeJitConstant("UNIT_MAX_FUNC",        unit_max_func),
            MakeJitConstant("UNIT_MIN_FUNC",        unit_min_func),
        };
    }

    JitConstants MakeActivationJitConstants(const base_activation_params& params, const std::string& suffix)
    {
        return JitConstants{
            MakeJitConstant("NL_M" + suffix, params.m),
            MakeJitConstant("NL_N" + suffix, params.n),
            MakeActivationJitConstants(params.function, suffix)
        };
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // MakeBaseParamsJitConstants
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    JitConstants MakeBaseParamsJitConstants(const base_params& params)
    {
        bool bFP16Used = params.output.GetDType() == Datatype::F16;
        bool bInt8Used = params.output.GetDType() == Datatype::INT8;
        bool bInt32Used = params.output.GetDType() == Datatype::INT32;
        bool bInt64Used = params.output.GetDType() == Datatype::INT64;
        bool bUInt8Used = params.output.GetDType() == Datatype::UINT8;
        bool bUInt32Used = params.output.GetDType() == Datatype::INT32;
        for (const auto& i : params.inputs)
        {
            bFP16Used |= i.GetDType() == Datatype::F16;
            bInt8Used |= i.GetDType() == Datatype::INT8;
            bInt32Used |= i.GetDType() == Datatype::INT32;
            bInt64Used |= i.GetDType() == Datatype::INT64;
            bUInt8Used |= i.GetDType() == Datatype::UINT8;
            bUInt32Used |= i.GetDType() == Datatype::UINT32;
        }

        JitConstants jit{
            MakeJitConstant("OUTPUT",               params.output),
            MakeJitConstant("FP64_SUPPORTED",       params.engineInfo.bFP64Support),
            MakeJitConstant("FP16_SUPPORTED",       params.engineInfo.bFP16Support),
            MakeJitConstant("FP16_UNIT_USED",       bFP16Used),
            MakeJitConstant("INT8_UNIT_USED",       bInt8Used),
            MakeJitConstant("INT32_UNIT_USED",      bInt32Used),
            MakeJitConstant("INT64_UNIT_USED",      bInt64Used),
            MakeJitConstant("UINT8_UNIT_USED",      bUInt8Used),
            MakeJitConstant("UINT32_UNIT_USED",     bUInt32Used),
            MakeJitConstant("GRADIENT",             params.gradient),
        };

        if (bInt8Used)
        {
            jit.Merge(MakeUnitTypeJitConstants(Datatype::INT8));
        }
        else if (bFP16Used)
        {
            jit.Merge(MakeUnitTypeJitConstants(Datatype::F16));
        }
        else if (bInt32Used)
        {
            jit.Merge(MakeUnitTypeJitConstants(Datatype::INT32));
        }
        else if (bInt64Used)
        {
            jit.Merge(MakeUnitTypeJitConstants(Datatype::INT64));
        }
        else if (bUInt8Used)
        {
            jit.Merge(MakeUnitTypeJitConstants(Datatype::UINT8));
        }
        else if (bUInt32Used)
        {
            jit.Merge(MakeUnitTypeJitConstants(Datatype::UINT32));
        }
        else
        {
            jit.Merge(MakeUnitTypeJitConstants(Datatype::F32));
        }

        // for activation function
        jit.Merge(MakeActivationJitConstants(params.activation));

        for (size_t i = 0; i < params.inputs.size(); i++)
        {
            jit.AddConstant(MakeJitConstant("INPUT" + toCodeString(i), params.inputs[i]));
        }

        return jit;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // MakeLoopUnrollParamsJitConstants
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    JitConstants MakeLoopUnrollParamsJitConstants(uint32_t loopCount)
    {
        JitConstants jit{
            MakeJitConstant("LOOP0(VAR, STMT)", ""),
            MakeJitConstant("LOOP1(VAR, STMT)", "(STMT); (VAR)++;"),
        };

        for (uint32_t i = 2; i <= loopCount + 1; i++)
        {
            jit.AddConstant({
                MakeJitConstant("LOOP" + toCodeString(i) + "(VAR, STMT)", "LOOP" + toCodeString(i - 1) + "(VAR, STMT); (STMT); (VAR)++;"),
                });
        }

        jit.AddConstant({
            MakeJitConstant("LOOP(N, VAR, STMT)", "CAT(LOOP, N)((VAR), (STMT))"),
            });

        return jit;
    }

}
