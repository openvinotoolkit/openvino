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
#include <string>
#include <memory>
#include <utility>

namespace {
class JitTerm {
public:
    explicit JitTerm(std::string text)
        : text(std::move(text)) {}

    std::string str() const { return text; }

    JitTerm gt(const JitTerm& rhs) const {
        JitTerm jit_term { "(" + text + ">" + rhs.str() + ")" };
        return jit_term;
    }

    JitTerm ge(const JitTerm& rhs) const {
        JitTerm jit_term {"(" + text + ">=" + rhs.str() + ")"};
        return jit_term;
    }

    JitTerm le(const JitTerm& rhs) const {
        JitTerm jit_term {"(" + text + "<=" + rhs.str() + ")"};
        return jit_term;
    }

    JitTerm eq(const JitTerm& rhs) const {
        JitTerm jit_term {"(" + text + "==" + rhs.str() + ")"};
        return jit_term;
    }

private:
    std::string text;
};

JitTerm operator+(const JitTerm& lhs, const JitTerm& rhs) {
    JitTerm jit_term{"(" + lhs.str() + " + " + rhs.str() + ")"};
    return jit_term;
}

JitTerm operator-(const JitTerm& lhs, const JitTerm& rhs) {
    JitTerm jit_term{"(" + lhs.str() + " - " + rhs.str() + ")"};
    return jit_term;
}

JitTerm operator*(const JitTerm& lhs, const JitTerm& rhs) {
    JitTerm jit_term{"(" + lhs.str() + " * " + rhs.str() + ")"};
    return jit_term;
}

JitTerm operator/(const JitTerm& lhs, const JitTerm& rhs) {
    JitTerm jit_term{"(" + lhs.str() + " / " + rhs.str() + ")"};
    return jit_term;
}

JitTerm neg(const JitTerm& arg) {
    JitTerm jit_term{"(-" + arg.str() + ")"};
    return jit_term;
}

JitTerm ternary(const JitTerm& condition,
                const JitTerm& true_expr,
                const JitTerm& false_expr) {
    JitTerm jit_term{"(" + condition.str() + " ? " + true_expr.str() + " : " + false_expr.str() + ")"};
    return jit_term;
}
JitTerm isinf(const JitTerm& arg) {
    JitTerm jit_term{"(isinf(" + arg.str() + "))"};
    return jit_term;
}

JitTerm exp(const JitTerm& arg) {
    JitTerm jit_term{"(exp(" + arg.str() + "))"};
    return jit_term;
}

JitTerm log(const JitTerm& arg) {
    JitTerm jit_term{"(log(" + arg.str() + "))"};
    return jit_term;
}

JitTerm operator"" _jit(const char* str, size_t) {
    JitTerm jit_term{str};
    return jit_term;
}
}  // namespace

namespace kernel_selector {

std::string toCLType(WeightsType wType) {
    switch (wType) {
        case WeightsType::BINARY:
            return GetTypeName<uint32_t>();
        case WeightsType::INT8:
            return GetTypeName<int8_t>();
        case WeightsType::UINT8:
            return GetTypeName<uint8_t>();
        case WeightsType::F16:
            return "half";
        case WeightsType::F32:
            return GetTypeName<float>();
        default:
            return "";
    }
}

std::string toCLType(Datatype dType) {
    switch (dType) {
        case Datatype::BINARY:
            return GetTypeName<uint32_t>();
        case Datatype::INT8:
            return GetTypeName<int8_t>();
        case Datatype::UINT8:
            return GetTypeName<uint8_t>();
        case Datatype::INT16:
            return GetTypeName<int16_t>();
        case Datatype::UINT16:
            return GetTypeName<uint16_t>();
        case Datatype::INT32:
            return GetTypeName<int32_t>();
        case Datatype::UINT32:
            return GetTypeName<uint32_t>();
        case Datatype::INT64:
            return GetTypeName<int64_t>();
        case Datatype::F16:
            return "half";
        case Datatype::F32:
            return GetTypeName<float>();
        default:
            return "";
    }
}

std::string getMeanOpString(MeanOp op) {
    switch (op) {
        case MeanOp::NONE:
            return "val";
        case MeanOp::DIV:
            return "val/mean_val";
        case MeanOp::MUL:
            return "val*mean_val";
        case MeanOp::SUB:
            return "val-mean_val";
        default:
            return "";
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

JitDefinitions JitConstants::GetDefinitions() const {
    JitDefinitions definitons;
    definitons.reserve(_constants.size() * 6);  // assuming max 6 pairs per jit_constant

    for (auto& constant : _constants) {
        auto def = constant->GetDefinitions();
        definitons.insert(definitons.end(), def.begin(), def.end());
    }
    return definitons;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// TensorBaseTJitConstant
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename DType, typename Layout>
class TensorBaseTJitConstant : public JitConstant {
protected:
    explicit TensorBaseTJitConstant(const std::string& name) : JitConstant(name) {}

public:
    JitDefinitions GetDefinitions(const Tensor::TensorBaseT<DType, Layout>& t) const {
        JitDefinitions definitions{
            {_name + "_OFFSET", toCodeString(t.GetFirstElementOffset())},
            {_name + "_VIEW_OFFSET", toCodeString(t.GetViewOffset())},
            {_name + "_LENGTH", toCodeString(t.LogicalSize())},
            {_name + "_DIMS", toCodeString(t.GetDims().size())},
            {_name + "_SIMPLE", toCodeString(t.SimpleLayout())},
            {_name + "_LAYOUT_" + toString(t.GetLayout()), "1"},
        };

        auto type_defs = MakeTypeJitConstants(t.GetDType(), _name).GetDefinitions();
        definitions.insert(definitions.end(), type_defs.begin(), type_defs.end());

        definitions.push_back({_name + "_SIZE", toCodeString(t.GetDims().size())});
        definitions.push_back(
            {_name + "_SIZES",
             toVectorString(t.GetDims(), "size_t", KERNEL_SELECTOR_TENSOR_DIM_MAX, 1, [](const Tensor::Dim& d) { return d.v; })});
        definitions.push_back(
            {_name + "_PITCHES",
             toVectorString(t.GetDims(), "size_t", KERNEL_SELECTOR_TENSOR_DIM_MAX, 1, [](const Tensor::Dim& d) { return d.pitch; })});
        definitions.push_back(
            {_name + "_PAD_BEFORE",
             toVectorString(t.GetDims(), "size_t", KERNEL_SELECTOR_TENSOR_DIM_MAX, 0, [](const Tensor::Dim& d) { return d.pad.before; })});
        definitions.push_back(
            {_name + "_PAD_AFTER",
             toVectorString(t.GetDims(), "size_t", KERNEL_SELECTOR_TENSOR_DIM_MAX, 0, [](const Tensor::Dim& d) { return d.pad.after; })});

        return definitions;
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// DataTensorJitConstant
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class DataTensorJitConstant : public TensorBaseTJitConstant<Datatype, DataLayout> {
    const DataTensor _tensor;

public:
    DataTensorJitConstant(const std::string& name, const DataTensor& t) : TensorBaseTJitConstant(name), _tensor(t) {}

    JitDefinitions GetDefinitions() const override;
};

JitDefinitions DataTensorJitConstant::GetDefinitions() const {
    JitDefinitions baseDefinitions = TensorBaseTJitConstant::GetDefinitions(_tensor);

    JitDefinitions definitions{
        {_name + "_SIZE_X", toCodeString(_tensor.X().v)},
        {_name + "_SIZE_Y", toCodeString(_tensor.Y().v)},
        {_name + "_SIZE_Z", toCodeString(_tensor.Z().v)},
        {_name + "_SIZE_W", toCodeString(_tensor.W().v)},
        {_name + "_FEATURE_NUM", toCodeString(_tensor.Feature().v)},
        {_name + "_BATCH_NUM", toCodeString(_tensor.Batch().v)},
        {_name + "_X_PITCH", toCodeString(_tensor.X().pitch)},
        {_name + "_Y_PITCH", toCodeString(_tensor.Y().pitch)},
        {_name + "_Z_PITCH", toCodeString(_tensor.Z().pitch)},
        {_name + "_W_PITCH", toCodeString(_tensor.W().pitch)},
        {_name + "_FEATURE_PITCH", toCodeString(_tensor.Feature().pitch)},
        {_name + "_BATCH_PITCH", toCodeString(_tensor.Batch().pitch)},
        {_name + "_PAD_BEFORE_SIZE_X", toCodeString(_tensor.X().pad.before)},
        {_name + "_PAD_BEFORE_SIZE_Y", toCodeString(_tensor.Y().pad.before)},
        {_name + "_PAD_BEFORE_SIZE_Z", toCodeString(_tensor.Z().pad.before)},
        {_name + "_PAD_BEFORE_SIZE_W", toCodeString(_tensor.W().pad.before)},
        {_name + "_PAD_BEFORE_FEATURE_NUM", toCodeString(_tensor.Feature().pad.before)},
        {_name + "_PAD_BEFORE_BATCH_NUM", toCodeString(_tensor.Batch().pad.before)},
        {_name + "_PAD_AFTER_SIZE_X", toCodeString(_tensor.X().pad.after)},
        {_name + "_PAD_AFTER_SIZE_Y", toCodeString(_tensor.Y().pad.after)},
        {_name + "_PAD_AFTER_SIZE_Z", toCodeString(_tensor.Z().pad.after)},
        {_name + "_PAD_AFTER_SIZE_W", toCodeString(_tensor.W().pad.after)},
        {_name + "_PAD_AFTER_FEATURE_NUM", toCodeString(_tensor.Feature().pad.after)},
        {_name + "_PAD_AFTER_BATCH_NUM", toCodeString(_tensor.Batch().pad.after)},
    };

    auto is_common_nd_layout = [](std::vector<Tensor::DataChannelName> common_channels, DataLayout l) -> bool {
        for (size_t c = 0; c < static_cast<size_t>(Tensor::DataChannelName::COUNT); c++) {
            auto channel = static_cast<Tensor::DataChannelName>(c);
            if (DataTensor::Channelndex(l, channel) != -1) {
                if (std::find(common_channels.begin(), common_channels.end(), channel) == common_channels.end()) {
                    return false;
                }
            }
        }
        return true;
    };

    // TODO: add support for other layouts
    auto layout = _tensor.GetLayout();
    if (DataTensor::ChannelsCount(layout) <= 4) {
        std::vector<Tensor::DataChannelName> base_4d_channels = {
                Tensor::DataChannelName::BATCH,
                Tensor::DataChannelName::FEATURE,
                Tensor::DataChannelName::Y,
                Tensor::DataChannelName::X,
        };
        bool is_common_4d_layout = is_common_nd_layout(base_4d_channels, layout);
        if (is_common_4d_layout) {
            auto index_func_name = _name + "_GET_INDEX(b, f, y, x)";
            auto safe_index_func_name = _name + "_GET_INDEX_SAFE(b, f, y, x)";
            auto raw_index_func_name = _name + "_GET_INDEX_RAW(b, f, y, x)";
            if (_tensor.SimpleLayout()) {
                definitions.push_back({ index_func_name, "GET_DATA_INDEX("+_name+", b, f, y, x)" });
                definitions.push_back({ safe_index_func_name, "GET_DATA_INDEX_SAFE("+_name+", b, f, y, x)" });
                definitions.push_back({ raw_index_func_name, "GET_DATA_INDEX_RAW("+_name+", b, f, y, x)" });
            } else if (layout == DataLayout::bfyx_f16 ||
                       layout == DataLayout::byxf_af32 ||
                       layout == DataLayout::fs_bs_yx_bsv4_fsv32 ||
                       layout == DataLayout::b_fs_yx_fsv4 ||
                       layout == DataLayout::fs_b_yx_fsv32) {
                auto layout_str = toString(layout);
                definitions.push_back({ index_func_name, "GET_DATA_"+layout_str+"_INDEX("+_name+", b, f, y, x)" });
                definitions.push_back({ raw_index_func_name, "GET_DATA_"+layout_str+"_INDEX("+_name+", b, f, y, x)" });
                if (layout == DataLayout::bfyx_f16)
                    definitions.push_back({ safe_index_func_name, "GET_DATA_"+layout_str+"_INDEX_SAFE("+_name+", b, f, y, x)" });
                else
                    definitions.push_back({ safe_index_func_name, "GET_DATA_"+layout_str+"_INDEX("+_name+", b, f, y, x)" });
            } else if (layout == DataLayout::bs_f_bsv8__af8 ||
                       layout == DataLayout::bs_f_bsv16__af8) {
                size_t sub_group_size = layout == DataLayout::bs_f_bsv16__af8 ? 16 : 8;
                definitions.push_back({ index_func_name, "GET_DATA_BS_FYX_BSV8_INDEX("+_name+
                                                          ", b, f, y, x"+std::to_string(sub_group_size)+")" });
                definitions.push_back({ safe_index_func_name, "GET_DATA_BS_FYX_BSV8_INDEX("+_name+
                                                              ", b, f, y, x"+std::to_string(sub_group_size)+")" });
                definitions.push_back({ raw_index_func_name, "GET_DATA_BS_FYX_BSV8_INDEX("+_name+
                                                             ", b, f, y, x"+std::to_string(sub_group_size)+")" });
            } else {
                definitions.push_back({ index_func_name,  "GET_DATA_INDEX_RAW("+_name+", b, f, y, x)" });
                definitions.push_back({ safe_index_func_name,  "GET_DATA_INDEX_RAW("+_name+", b, f, y, x)" });
                definitions.push_back({ raw_index_func_name,  "GET_DATA_INDEX_RAW("+_name+", b, f, y, x)" });
            }
        } else {
            // TODO: implement support of non-default layouts with 4 channels
            assert(0);
        }
    } else if (DataTensor::ChannelsCount(layout) == 5) {
        std::vector<Tensor::DataChannelName> base_5d_channels = {
                Tensor::DataChannelName::BATCH,
                Tensor::DataChannelName::FEATURE,
                Tensor::DataChannelName::Z,
                Tensor::DataChannelName::Y,
                Tensor::DataChannelName::X,
        };
        bool is_common_5d_layout = is_common_nd_layout(base_5d_channels, layout);
        if (is_common_5d_layout) {
            auto index_func_name = _name + "_GET_INDEX(b, f, z, y, x)";
            auto safe_index_func_name = _name + "_GET_INDEX_SAFE(b, f, z, y, x)";
            auto raw_index_func_name = _name + "_GET_INDEX_RAW(b, f, z, y, x)";
            if (_tensor.SimpleLayout()) {
                definitions.push_back({ index_func_name,  "GET_DATA_INDEX_5D("+_name+", b, f, z, y, x)" });
                definitions.push_back({ safe_index_func_name,  "GET_DATA_INDEX_5D_SAFE("+_name+", b, f, z, y, x)" });
                definitions.push_back({ raw_index_func_name,  "GET_DATA_INDEX_5D_RAW("+_name+", b, f, z, y, x)" });
            } else if (layout == DataLayout::bfzyx_f16) {
                definitions.push_back({ index_func_name, "GET_DATA_BFZYX_F16_INDEX(" + _name + ", b, f, z, y, x)" });
                definitions.push_back({ raw_index_func_name, "GET_DATA_BFZYX_F16_INDEX(" + _name + ", b, f, z, y, x)" });
                definitions.push_back({ safe_index_func_name, "GET_DATA_BFZYX_F16_INDEX(" + _name + ", b, f, z, y, x)" });
            } else {
                definitions.push_back({ index_func_name,  "GET_DATA_INDEX_5D_RAW(" + _name + ", b, f, z, y, x)" });
                definitions.push_back({ safe_index_func_name,  "GET_DATA_INDEX_5D_RAW(" + _name + ", b, f, z, y, x)" });
                definitions.push_back({ raw_index_func_name,  "GET_DATA_INDEX_5D_RAW(" + _name + ", b, f, z, y, x)" });
            }
        } else {
            // TODO: implement support of non-default layouts with 5 channels
            assert(0);
        }
    } else if (DataTensor::ChannelsCount(layout) == 6) {
        std::vector<Tensor::DataChannelName> base_6d_channels = {
                Tensor::DataChannelName::BATCH,
                Tensor::DataChannelName::FEATURE,
                Tensor::DataChannelName::W,
                Tensor::DataChannelName::Z,
                Tensor::DataChannelName::Y,
                Tensor::DataChannelName::X,
        };
        bool is_common_6d_layout = is_common_nd_layout(base_6d_channels, layout);
        if (is_common_6d_layout) {
            definitions.push_back({ _name + "_GET_INDEX(b, f, w, z, y, x)",  "GET_DATA_INDEX_6D("+_name+", b, f, w, z, y, x)" });
            definitions.push_back({ _name + "_GET_INDEX_SAFE(b, f, w, z, y, x)",  "GET_DATA_INDEX_6D_SAFE("+_name+", b, f, w, z, y, x)" });
            definitions.push_back({ _name + "_GET_INDEX_RAW(b, f, w, z, y, x)",  "GET_DATA_INDEX_6D_RAW("+_name+", b, f, w, z, y, x)" });
        } else {
            // TODO: implement support of non-default layouts with 4 channels
            assert(0);
        }
    } else {
        throw std::runtime_error("Unsupported channels count(" + std::to_string(DataTensor::ChannelsCount(layout)) +
                                 ") in layout: " + toString(layout));
    }

    definitions.insert(definitions.end(), baseDefinitions.begin(), baseDefinitions.end());

    return definitions;
}

std::shared_ptr<JitConstant> MakeJitConstant(const std::string& name, const DataTensor& value) {
    return std::static_pointer_cast<JitConstant>(std::make_shared<DataTensorJitConstant>(name, value));
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// WeightTensorJitConstant
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class WeightTensorJitConstant : public TensorBaseTJitConstant<WeightsType, WeightsLayout> {
    const WeightsTensor _tensor;

public:
    WeightTensorJitConstant(const std::string& name, const WeightsTensor& t) : TensorBaseTJitConstant(name), _tensor(t) {}

    JitDefinitions GetDefinitions() const override;
};

JitDefinitions WeightTensorJitConstant::GetDefinitions() const {
    JitDefinitions baseDefinitions = TensorBaseTJitConstant::GetDefinitions(_tensor);

    JitDefinitions definitions{
        {_name + "_SIZE_X", toCodeString(_tensor.X().v)},
        {_name + "_SIZE_Y", toCodeString(_tensor.Y().v)},
        {_name + "_SIZE_Z", toCodeString(_tensor.Z().v)},
        {_name + "_IFM_NUM", toCodeString(_tensor.IFM().v)},
        {_name + "_OFM_NUM", toCodeString(_tensor.OFM().v)},
        {_name + "_X_PITCH", toCodeString(_tensor.X().pitch)},
        {_name + "_Y_PITCH", toCodeString(_tensor.Y().pitch)},
        {_name + "_Z_PITCH", toCodeString(_tensor.Z().pitch)},
        {_name + "_IFM_PITCH", toCodeString(_tensor.IFM().pitch)},
        {_name + "_OFM_PITCH", toCodeString(_tensor.OFM().pitch)},
    };

    definitions.insert(definitions.end(), baseDefinitions.begin(), baseDefinitions.end());

    return definitions;
}

std::shared_ptr<JitConstant> MakeJitConstant(const std::string& name, const WeightsTensor& value) {
    return std::static_pointer_cast<JitConstant>(std::make_shared<WeightTensorJitConstant>(name, value));
}

JitConstants MakeActivationJitConstants(ActivationFunction activation_function,
                                        const std::string& suffix,
                                        bool use_type_parameter,
                                        bool disable_type_conversion) {
    std::string name = "ACTIVATION_FUNC" + suffix;
    JitConstants jitConstants = {};

    // See the comment in the jitter.h regarding `use_type_parameter`.
    // The "CAT" macro is expected to be defined through the inlcusion of
    // 'common.cl' in the kernel.
    auto type_handler =
        [use_type_parameter](const std::string& prefix,
                             const std::string& suffix) -> std::string {
        if (!use_type_parameter)
            return prefix + "UNIT" + suffix;

        std::string result = "jit_type";

        // Process the prefix first, otherwise when doing "CAT(TO_,
        // CAT(NAME, _TYPE))" the second concatenation will be expanded
        // fully first resulting in something like "TO_float".
        if (!prefix.empty())
            result = "CAT(" + prefix + ", " + result + ")";

        if (!suffix.empty())
            result = "CAT(" + result + ", " + suffix + ")";

        return result;
    };

    const JitTerm one{type_handler("", "_VAL_ONE")};
    const JitTerm zero{type_handler("", "_VAL_ZERO")};
    const JitTerm input{"input"};
    auto max_func = [type_handler](const JitTerm& lhs,
                                   const JitTerm& rhs) -> JitTerm {
        JitTerm jit_term{"(" + type_handler("", "_MAX_FUNC") + "(" + lhs.str() + ", " + rhs.str() + "))"};
        return jit_term;
    };
    auto min_func = [type_handler](const JitTerm& lhs,
                                   const JitTerm& rhs) -> JitTerm {
        JitTerm jit_term{"(" + type_handler("", "_MIN_FUNC") + "(" + lhs.str() + ", " + rhs.str() + "))"};
        return jit_term;
    };
    auto abs_func = [type_handler](const JitTerm& hs) -> JitTerm {
        JitTerm jit_term{"(" + type_handler("", "_ABS_FUNC") + "(" + hs.str() + "))"};
        return jit_term;
    };
    auto to_type = [type_handler](const JitTerm& arg) -> JitTerm {
        JitTerm jit_term{type_handler("TO_", "_TYPE") + "(" + arg.str() + ")"};
        return jit_term;
    };

    std::string macro_def = name + (use_type_parameter ? "(jit_type, input, m, n)" : "(input, m, n)");
    std::string macro_def_grad = name + (use_type_parameter ? "(jit_type, input_grad, input, m, n)"
                                                            : "(input_grad, input, m, n)");
    std::string macro_def_params = use_type_parameter ? "(jit_type, input, params)" : "(input, params)";

    jitConstants.AddConstant(MakeJitConstant("ACTIVATION_PARAMS" + suffix, "NL_M" + suffix + ", NL_N" + suffix));

    // TODO: use native_exp and use cast for APL
    switch (activation_function) {
        case ActivationFunction::LOGISTIC:
            jitConstants.AddConstant(MakeJitConstant(macro_def, (one / (one + exp(neg(input)))).str()));
            break;
        case ActivationFunction::HYPERBOLIC_TAN:
            jitConstants.AddConstant(MakeJitConstant(macro_def, "(tanh(input))"));
            break;
        case ActivationFunction::RELU:
            jitConstants.AddConstant(MakeJitConstant(macro_def, max_func(zero, input).str()));
            break;
        case ActivationFunction::RELU_NEGATIVE_SLOPE: {
            const JitTerm slope = disable_type_conversion ? "m"_jit : to_type("m"_jit);
            jitConstants.AddConstant(MakeJitConstant(
                macro_def,
                ternary(isinf(slope),
                        ternary(input.ge(zero), input, neg(slope)),
                        max_func(input, zero) + (slope * min_func(input, zero)))
                    .str()));
            break;
        }
        case ActivationFunction::ELU: {
            auto alpha = disable_type_conversion ? "m"_jit : to_type("m"_jit);
            jitConstants.AddConstant(MakeJitConstant(
                macro_def,
                (max_func(input, zero) + (alpha * (exp(min_func(input, zero)) - one)))
                    .str()));
            break;
        }
        case ActivationFunction::CLAMP: {
            const JitTerm m = disable_type_conversion ? "m"_jit : to_type("m"_jit);
            const JitTerm n = disable_type_conversion ? "n"_jit : to_type("n"_jit);
            jitConstants.AddConstant(MakeJitConstant(
                 macro_def,
                 max_func(m, min_func(n, input)).str()));
            break;
        }
        case ActivationFunction::SOFTRELU:
            jitConstants.AddConstant(MakeJitConstant(macro_def, log(one + exp(input)).str()));
            break;
        case ActivationFunction::ABS:
            jitConstants.AddConstant(MakeJitConstant(macro_def, "(fabs(input))"));
            break;
        case ActivationFunction::LINEAR: {
            const JitTerm m = disable_type_conversion ? "m"_jit : to_type("m"_jit);
            const JitTerm n = disable_type_conversion ? "n"_jit : to_type("n"_jit);
            jitConstants.AddConstant(MakeJitConstant(macro_def, (m * input + n).str()));
            break;
        }
        case ActivationFunction::SQUARE:
            jitConstants.AddConstant(MakeJitConstant(macro_def, "(input*input)"));
            break;
        case ActivationFunction::SQRT:
            jitConstants.AddConstant(MakeJitConstant(macro_def, "(sqrt(input))"));
            break;
        case ActivationFunction::SIN:
            jitConstants.AddConstant(MakeJitConstant(macro_def, "(sin(input))"));
            break;
        case ActivationFunction::ASIN:
            jitConstants.AddConstant(MakeJitConstant(macro_def, "(asin(input))"));
            break;
        case ActivationFunction::SINH:
            jitConstants.AddConstant(MakeJitConstant(macro_def, "(sinh(input))"));
            break;
        case ActivationFunction::ASINH:
            jitConstants.AddConstant(MakeJitConstant(macro_def, "(asinh(input))"));
            break;
        case ActivationFunction::COS:
            jitConstants.AddConstant(MakeJitConstant(macro_def, "(cos(input))"));
            break;
        case ActivationFunction::ACOS:
            jitConstants.AddConstant(MakeJitConstant(macro_def, "(acos(input))"));
            break;
        case ActivationFunction::COSH:
            jitConstants.AddConstant(MakeJitConstant(macro_def, "(cosh(input))"));
            break;
        case ActivationFunction::ACOSH:
            jitConstants.AddConstant(MakeJitConstant(macro_def, "(acosh(input))"));
            break;
        case ActivationFunction::LOG:
            jitConstants.AddConstant(MakeJitConstant(macro_def, "(log(input))"));
            break;
        case ActivationFunction::LOG2:
            jitConstants.AddConstant(MakeJitConstant(macro_def, "(log2(input))"));
            break;
        case ActivationFunction::EXP:
            jitConstants.AddConstant(MakeJitConstant(macro_def, "(exp(input))"));
            break;
        case ActivationFunction::POW: {
            const JitTerm m = disable_type_conversion ? "m"_jit : to_type("m"_jit);
            jitConstants.AddConstant(MakeJitConstant(macro_def, "(pow(input," + m.str() + "))"));
            break;
        }
        case ActivationFunction::RELU_GRAD:
            jitConstants.AddConstant(MakeJitConstant(
                macro_def_grad,
                ("input_grad"_jit * ternary(input.gt(zero), one, zero)).str()));
            macro_def_params = use_type_parameter ? "(jit_type, input_grad, input, params)" : "(input_grad, input, params)";
            break;
        case ActivationFunction::RELU_NEGATIVE_SLOPE_GRAD: {
            const JitTerm slope = disable_type_conversion ? "m"_jit : to_type("m"_jit);
            jitConstants.AddConstant(MakeJitConstant(
                macro_def_grad,
                ("input_grad"_jit * (ternary(input.gt(zero), one, zero) + (to_type(slope) * ternary(input.le(zero), one, zero))))
                    .str()));
            macro_def_params = use_type_parameter ? "(jit_type, input_grad, input, params)" : "(input_grad, input, params)";
            break;
        }
        case ActivationFunction::NONE_GRAD:
            jitConstants.AddConstant(MakeJitConstant(macro_def_grad, "input_grad"));
            macro_def_params = use_type_parameter ? "(jit_type, input_grad, input, params)" : "(input_grad, input, params)";
            break;
        case ActivationFunction::TAN:
            jitConstants.AddConstant(MakeJitConstant(macro_def, "(tan(input))"));
            break;
        case ActivationFunction::ATAN:
            jitConstants.AddConstant(MakeJitConstant(macro_def, "(atan(input))"));
            break;
        case ActivationFunction::ATANH:
            jitConstants.AddConstant(MakeJitConstant(macro_def, "(atanh(input))"));
            break;
        case ActivationFunction::FLOOR:
            jitConstants.AddConstant(MakeJitConstant(macro_def, "(floor(input))"));
            break;
        case ActivationFunction::CEIL:
            jitConstants.AddConstant(MakeJitConstant(macro_def, "(ceil(input))"));
            break;
        case ActivationFunction::NEGATIVE:
            jitConstants.AddConstant(MakeJitConstant(macro_def, "(-input)"));
            break;
        case ActivationFunction::ERF:
            jitConstants.AddConstant(MakeJitConstant(macro_def, "erf(input)"));
            break;
        case ActivationFunction::HARD_SIGMOID: {
            auto alpha = disable_type_conversion ? "m"_jit : to_type("m"_jit);
            auto beta =  disable_type_conversion ? "n"_jit : to_type("n"_jit);
            jitConstants.AddConstant(MakeJitConstant(
                    macro_def,
                    max_func(zero, min_func(one, (JitTerm)((alpha * input + beta).str()))).str()));
            break;
        }
        case ActivationFunction::SIGN:
            jitConstants.AddConstant(MakeJitConstant(
                    macro_def,
                    ternary(input.gt(zero), one, ternary(input.eq(zero), zero, neg(one))).str()));
            break;
        case ActivationFunction::RECIPROCAL:
            jitConstants.AddConstant(MakeJitConstant(macro_def, (one / input).str()));
            break;
        case ActivationFunction::SELU: {
            auto alpha = disable_type_conversion ? "m"_jit : to_type("m"_jit);
            auto gamma = disable_type_conversion ? "n"_jit : to_type("n"_jit);
            jitConstants.AddConstant(MakeJitConstant(
                    macro_def,
                    ternary(input.le(zero), gamma * (alpha * exp(input) - alpha), gamma * input).str()));
            break;
        }
        case ActivationFunction::SOFTPLUS: {
            jitConstants.AddConstant(MakeJitConstant(
                    macro_def,
                    log(exp(input) + one).str()));
            break;
        }
        case ActivationFunction::SOFTSIGN: {
            jitConstants.AddConstant(MakeJitConstant(
                    macro_def,
                    (input / (one + abs_func(input))).str()));
            break;
        }
        case ActivationFunction::NOT:
            jitConstants.AddConstant(MakeJitConstant(
                macro_def,
                ternary(input.eq(zero), one, zero)
                    .str()));  // the workaround for OpenCL's vector type result (!input)
            break;
        case ActivationFunction::NONE:
        default:
            jitConstants.AddConstant(MakeJitConstant(macro_def, "input"));
            break;
    }

    jitConstants.AddConstant(MakeJitConstant("ACTIVATION" + suffix + macro_def_params, name + macro_def_params));

    return jitConstants;
}

JitConstants MakeTypeJitConstants(Datatype dataType, const std::string& macroName) {
    std::string type;
    std::string max_val;
    std::string min_val;
    std::string val_one;
    std::string val_zero;
    std::string to_type;
    std::string to_type_sat;
    std::string max_func;
    std::string min_func;
    std::string abs_func;
    std::string type_size;
    bool is_fp;
    switch (dataType) {
        case Datatype::INT8:
            type = "char";
            max_val = "CHAR_MAX";
            min_val = "CHAR_MIN";
            val_one = "(char) 1";
            val_zero = "(char) 0";
            to_type = "convert_char(v)";
            to_type_sat = "convert_char_sat(v)";
            max_func = "max";
            min_func = "min";
            abs_func = "abs";
            type_size = "1";
            is_fp = false;
            break;
        case Datatype::UINT8:
            type = "uchar";
            max_val = "UCHAR_MAX";
            min_val = "0";
            val_one = "(uchar) 1";
            val_zero = "(uchar) 0";
            to_type = "convert_uchar(v)";
            to_type_sat = "convert_uchar_sat(v)";
            max_func = "max";
            min_func = "min";
            abs_func = "abs";
            type_size = "1";
            is_fp = false;
            break;
        case Datatype::INT16:
            type = "short";
            max_val = "SHRT_MAX";
            min_val = "SHRT_MIN";
            val_one = "(short) 1";
            val_zero = "(short) 0";
            to_type = "convert_short(v)";
            to_type_sat = "convert_short_sat(v)";
            max_func = "max";
            min_func = "min";
            abs_func = "abs";
            type_size = "2";
            is_fp = false;
            break;
        case Datatype::UINT16:
            type = "ushort";
            max_val = "USHRT_MAX";
            min_val = "0";
            val_one = "(ushort) 1";
            val_zero = "(ushort) 0";
            to_type = "convert_ushort(v)";
            to_type_sat = "convert_ushort_sat(v)";
            max_func = "max";
            min_func = "min";
            abs_func = "abs";
            type_size = "2";
            is_fp = false;
            break;
        case Datatype::INT32:
            type = "int";
            max_val = "INT_MAX";
            min_val = "INT_MIN";
            val_one = "(int) 1";
            val_zero = "(int) 0";
            to_type = "convert_int(v)";
            to_type_sat = "convert_int_sat(v)";
            max_func = "max";
            min_func = "min";
            abs_func = "abs";
            type_size = "4";
            is_fp = false;
            break;
        case Datatype::UINT32:
        case Datatype::BINARY:
            type = "uint";
            max_val = "UINT_MAX";
            min_val = "0";
            val_one = "(uint) 1";
            val_zero = "(uint) 0";
            to_type = "convert_uint(v)";
            to_type_sat = "convert_uint_sat(v)";
            max_func = "max";
            min_func = "min";
            abs_func = "abs";
            type_size = "4";
            is_fp = false;
            break;
        case Datatype::INT64:
            type = "long";
            max_val = "LONG_MAX";
            min_val = "LONG_MIN";
            val_one = "(long) 1";
            val_zero = "(long) 0";
            to_type = "convert_long(v)";
            to_type_sat = "convert_long_sat(v)";
            max_func = "max";
            min_func = "min";
            abs_func = "abs";
            type_size = "8";
            is_fp = false;
            break;
        case Datatype::F16:
            type = "half";
            max_val = "HALF_MAX";
            min_val = "-" + macroName + "_VAL_MAX";
            val_one = "1.0h";
            val_zero = "0.0h";
            to_type = "convert_half(v)";
            to_type_sat = "convert_half(v)";
            max_func = "fmax";
            min_func = "fmin";
            abs_func = "fabs";
            type_size = "2";
            is_fp = true;
            break;
        default:
            type = "float";
            max_val = "FLT_MAX";
            min_val = "-" + macroName + "_VAL_MAX";
            val_one = "1.0f";
            val_zero = "0.0f";
            to_type = "convert_float(v)";
            to_type_sat = "convert_float(v)";
            max_func = "fmax";
            min_func = "fmin";
            abs_func = "fabs";
            type_size = "4";
            is_fp = true;
            break;
    }

    return JitConstants{
        MakeJitConstant(macroName + "_TYPE", type),
        MakeJitConstant(macroName + "_VAL_MAX", max_val),
        MakeJitConstant(macroName + "_VAL_MIN", min_val),
        MakeJitConstant(macroName + "_VAL_ONE", val_one),
        MakeJitConstant(macroName + "_VAL_ZERO", val_zero),
        MakeJitConstant("TO_" + macroName + "_TYPE(v)", to_type),
        MakeJitConstant("TO_" + macroName + "_TYPE_SAT(v)", to_type_sat),
        MakeJitConstant(macroName + "_MAX_FUNC", max_func),
        MakeJitConstant(macroName + "_MIN_FUNC", min_func),
        MakeJitConstant(macroName + "_ABS_FUNC", abs_func),
        MakeJitConstant(macroName + "_TYPE_SIZE", type_size),
        MakeJitConstant(macroName + "_IS_FP", is_fp),
    };
}
JitConstants MakeTypeJitConstants(WeightsType weightsType, const std::string& macroName) {
    switch (weightsType) {
        case WeightsType::UNSUPPORTED:
            return MakeTypeJitConstants(Datatype::UNSUPPORTED, macroName);
        case WeightsType::F16:
            return MakeTypeJitConstants(Datatype::F16, macroName);
        case WeightsType::F32:
            return MakeTypeJitConstants(Datatype::F32, macroName);
        case WeightsType::INT8:
            return MakeTypeJitConstants(Datatype::INT8, macroName);
        case WeightsType::UINT8:
            return MakeTypeJitConstants(Datatype::UINT8, macroName);
        case WeightsType::BINARY:
            return MakeTypeJitConstants(Datatype::UINT32, macroName);
    }
    assert(false || "Unreachable!");
    // FIXME: Is there some builtin_unreachable available?
    return MakeTypeJitConstants(Datatype::UNSUPPORTED, macroName);
}

JitConstants MakeActivationJitConstants(const base_activation_params& params,
                                        const std::string& suffix,
                                        bool use_type_parameter,
                                        bool disable_type_conversion) {
    auto jitConstants = JitConstants{MakeJitConstant("NL_M" + suffix, params.m),
                                     MakeJitConstant("NL_N" + suffix, params.n)};
    jitConstants.Merge(MakeActivationJitConstants(
        params.function, suffix, use_type_parameter, disable_type_conversion));
    return jitConstants;
}

JitConstants MakeActivationJitConstants(std::vector<kernel_selector::base_activation_params> params,
                                        const std::string& suffix,
                                        bool use_type_parameter,
                                        bool disable_type_conversion) {
    JitConstants res = {};
    if (params.empty()) {
        return MakeActivationJitConstants({ActivationFunction::NONE, 0.f, 0.f}, suffix, use_type_parameter, disable_type_conversion);
    }
    std::string res_activation = "";
    std::string activation_params = "";
    for (size_t i = 0; i < params.size(); i++) {
        std::string activation_suffix = suffix + "_" + std::to_string(i);
        auto jitConstants = JitConstants{MakeJitConstant("NL_M" + activation_suffix, params[i].m),
                                         MakeJitConstant("NL_N" + activation_suffix, params[i].n)};
        jitConstants.Merge(MakeActivationJitConstants(
                params[i].function, activation_suffix, use_type_parameter, disable_type_conversion));
        res.Merge(jitConstants);

        if (i == 0) {
            if (params[i].gradient) {
                activation_params = use_type_parameter ? "(jit_type, input_grad, input, params)" : "(input_grad, input, params)";
            } else {
                activation_params = use_type_parameter ? "(jit_type, input, params)" : "(input, params)";
            }
            res_activation = "ACTIVATION_FUNC" + activation_suffix + activation_params;
        } else {
            res_activation = "ACTIVATION" + activation_suffix + "(" + (use_type_parameter ? "jit_type, " : "") +
                             (params[i].gradient ? "input_grad, " : "") +
                             res_activation + ", ACTIVATION_PARAMS" + activation_suffix + ")";
        }
    }
    if (params[params.size() - 1].gradient) {
        activation_params = use_type_parameter ? "(jit_type, input_grad, input, params)" : "(input_grad, input, params)";
    } else {
        activation_params = use_type_parameter ? "(jit_type, input, params)" : "(input, params)";
    }
    res.AddConstant(MakeJitConstant("ACTIVATION_PARAMS" + suffix, "ACTIVATION_PARAMS" + suffix + "_0"));
    res.AddConstant(MakeJitConstant("ACTIVATION" + suffix + activation_params, res_activation));
    return res;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// MakeLoopUnrollParamsJitConstants
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
JitConstants MakeLoopUnrollParamsJitConstants(uint32_t loopCount) {
    JitConstants jit{
        MakeJitConstant("LOOP0(VAR, STMT)", ""),
        MakeJitConstant("LOOP1(VAR, STMT)", "(STMT); (VAR)++;"),
    };

    for (uint32_t i = 2; i <= loopCount + 1; i++) {
        jit.AddConstant({
            MakeJitConstant("LOOP" + toCodeString(i) + "(VAR, STMT)", "LOOP" + toCodeString(i - 1) + "(VAR, STMT); (STMT); (VAR)++;"),
        });
    }

    jit.AddConstant({
        MakeJitConstant("LOOP(N, VAR, STMT)", "CAT(LOOP, N)((VAR), (STMT))"),
    });

    return jit;
}

}  // namespace kernel_selector
