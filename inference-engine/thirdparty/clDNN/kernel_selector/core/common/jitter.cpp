/*
// Copyright (c) 2019-2020 Intel Corporation
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

#include <quantize/quantize_kernel_params.h>
#include <eltwise/eltwise_kernel_base.h>
#include <activation/activation_kernel_base.h>

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

JitTerm erf(const JitTerm& arg) {
    JitTerm jit_term{"(erf(" + arg.str() + "))"};
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
            {_name + "_GROUPED", toCodeString(t.GroupedLayout())},
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

    std::string index_func_name;
    std::string safe_index_func_name;
    std::string raw_index_func_name;
    std::string index_func_val;
    std::string safe_index_func_val;
    std::string raw_index_func_val;

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
            index_func_name = _name + "_GET_INDEX(b, f, y, x)";
            safe_index_func_name = _name + "_GET_INDEX_SAFE(b, f, y, x)";
            raw_index_func_name = _name + "_GET_INDEX_RAW(b, f, y, x)";

            if (_tensor.SimpleLayout()) {
                index_func_val = "GET_DATA_INDEX(" + _name + ", b, f, y, x)";
                safe_index_func_val = "GET_DATA_INDEX_SAFE(" + _name + ", b, f, y, x)";
                raw_index_func_val = "GET_DATA_INDEX_RAW(" + _name + ", b, f, y, x)";
            } else if (layout == DataLayout::b_fs_yx_fsv16 ||
                       layout == DataLayout::b_fs_yx_fsv32 ||
                       layout == DataLayout::byxf_af32 ||
                       layout == DataLayout::fs_bs_yx_bsv4_fsv32 ||
                       layout == DataLayout::b_fs_yx_fsv4 ||
                       layout == DataLayout::fs_b_yx_fsv32 ||
                       layout == DataLayout::bs_fs_yx_bsv16_fsv16) {
                auto layout_str = toString(layout);
                index_func_val = "GET_DATA_" + layout_str + "_INDEX(" + _name + ", b, f, y, x)";
                raw_index_func_val = "GET_DATA_" + layout_str + "_INDEX(" + _name + ", b, f, y, x)";
                if (layout == DataLayout::b_fs_yx_fsv16 ||
                    layout == DataLayout::b_fs_yx_fsv32 ||
                    layout == DataLayout::b_fs_yx_fsv4  ||
                    layout == DataLayout::bs_fs_yx_bsv16_fsv16)
                    safe_index_func_val = "GET_DATA_" + layout_str + "_INDEX_SAFE(" + _name + ", b, f, y, x)";
                else
                    safe_index_func_val = "GET_DATA_" + layout_str + "_INDEX(" + _name + ", b, f, y, x)";
            } else if (layout == DataLayout::bs_f_bsv8__af8 ||
                       layout == DataLayout::bs_f_bsv16__af8) {
                size_t sub_group_size = layout == DataLayout::bs_f_bsv16__af8 ? 16 : 8;
                index_func_val = "GET_DATA_BS_FYX_BSV8_INDEX(" + _name + ", b, f, y, x," + std::to_string(sub_group_size) + ")";
                safe_index_func_val = "GET_DATA_BS_FYX_BSV8_INDEX(" + _name + ", b, f, y, x," + std::to_string(sub_group_size) + ")";
                raw_index_func_val = "GET_DATA_BS_FYX_BSV8_INDEX(" + _name + ", b, f, y, x," + std::to_string(sub_group_size) + ")";
            } else {
                index_func_val =  "GET_DATA_INDEX_RAW(" + _name + ", b, f, y, x)";
                safe_index_func_val = "GET_DATA_INDEX_RAW(" + _name + ", b, f, y, x)";
                raw_index_func_val = "GET_DATA_INDEX_RAW(" + _name + ", b, f, y, x)";
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
            index_func_name = _name + "_GET_INDEX(b, f, z, y, x)";
            safe_index_func_name = _name + "_GET_INDEX_SAFE(b, f, z, y, x)";
            raw_index_func_name = _name + "_GET_INDEX_RAW(b, f, z, y, x)";

            if (_tensor.SimpleLayout()) {
                index_func_val = "GET_DATA_INDEX_5D(" + _name + ", b, f, z, y, x)";
                safe_index_func_val = "GET_DATA_INDEX_5D_SAFE("+ _name + ", b, f, z, y, x)";
                raw_index_func_val = "GET_DATA_INDEX_5D_RAW("+ _name + ", b, f, z, y, x)";
            } else if (layout == DataLayout::b_fs_zyx_fsv16) {
                index_func_val = "GET_DATA_B_FS_ZYX_FSV16_INDEX(" + _name + ", b, f, z, y, x)";
                raw_index_func_val = "GET_DATA_B_FS_ZYX_FSV16_INDEX(" + _name + ", b, f, z, y, x)";
                safe_index_func_val = "GET_DATA_B_FS_ZYX_FSV16_INDEX_SAFE(" + _name + ", b, f, z, y, x)";
            } else if (layout == DataLayout::bs_fs_zyx_bsv16_fsv16) {
                index_func_val = "GET_DATA_BS_FS_ZYX_BSV16_FSV16_INDEX(" + _name + ", b, f, z, y, x)";
                raw_index_func_val = "GET_DATA_BS_FS_ZYX_BSV16_FSV16_INDEX(" + _name + ", b, f, z, y, x)";
                safe_index_func_val = "GET_DATA_BS_FS_ZYX_BSV16_FSV16_INDEX_SAFE(" + _name + ", b, f, z, y, x)";
            } else if (layout == DataLayout::b_fs_zyx_fsv32) {
                index_func_val = "GET_DATA_B_FS_ZYX_FSV32_INDEX(" + _name + ", b, f, z, y, x)";
                raw_index_func_val = "GET_DATA_B_FS_ZYX_FSV32_INDEX(" + _name + ", b, f, z, y, x)";
                safe_index_func_val = "GET_DATA_B_FS_ZYX_FSV32_INDEX_SAFE(" + _name + ", b, f, z, y, x)";
            } else {
                index_func_val = "GET_DATA_INDEX_5D_RAW(" + _name + ", b, f, z, y, x)";
                safe_index_func_val = "GET_DATA_INDEX_5D_RAW(" + _name + ", b, f, z, y, x)";
                raw_index_func_val = "GET_DATA_INDEX_5D_RAW(" + _name + ", b, f, z, y, x)";
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
            index_func_name = _name + "_GET_INDEX(b, f, w, z, y, x)";
            safe_index_func_name = _name + "_GET_INDEX_SAFE(b, f, w, z, y, x)";
            raw_index_func_name = _name + "_GET_INDEX_RAW(b, f, w, z, y, x)";

            index_func_val = "GET_DATA_INDEX_6D(" + _name + ", b, f, w, z, y, x)";
            safe_index_func_val = "GET_DATA_INDEX_6D_SAFE(" + _name + ", b, f, w, z, y, x)";
            raw_index_func_val = "GET_DATA_INDEX_6D_RAW(" + _name + ", b, f, w, z, y, x)";
        } else {
            // TODO: implement support of non-default layouts with 6 channels
            assert(0);
        }
    } else {
        throw std::runtime_error("Unsupported channels count(" + std::to_string(DataTensor::ChannelsCount(layout)) +
                                 ") in layout: " + toString(layout));
    }

    std::string offset = std::to_string(_tensor.GetFirstElementOffset());
    if (_tensor.LogicalSize() == 1) {
        // if tensor contains single element we can always return 0 for safe function
        if (_tensor.PitchesDifferFromLogicalDims()) {
            definitions.push_back({ safe_index_func_name, offset });
            definitions.push_back({ index_func_name, offset });
        } else {
            definitions.push_back({ safe_index_func_name, "0" });
            definitions.push_back({ index_func_name, "0" });
        }
    } else if (_tensor.LogicalSize() == _tensor.Feature().v) {
        // We support broadcast only if corresponding dimension is equal to 1.
        // Otherwise, dimensions should be equal and using "f" should be safe.
        if (_tensor.PitchesDifferFromLogicalDims() && _tensor.SimpleLayout()) {
            std::string f_pitch = std::to_string(_tensor.Feature().pitch);
            definitions.push_back({ safe_index_func_name, "(" + offset + " + (f) * " + f_pitch + ")" });
            definitions.push_back({ index_func_name, "(" + offset + " + (f) * " + f_pitch + ")" });
        } else if (_tensor.PitchesDifferFromLogicalDims()) {
            // TODO This should be solved differently, by setting the macro arguments to zero
            definitions.push_back({ safe_index_func_name, safe_index_func_val });
            definitions.push_back({ index_func_name, index_func_val });
        } else {
            definitions.push_back({ safe_index_func_name, "(" + std::to_string(_tensor.Feature().pad.before) + " + (f))" });
            definitions.push_back({ index_func_name, "(" + std::to_string(_tensor.Feature().pad.before) + " + (f))" });
        }
    } else {
        definitions.push_back({ safe_index_func_name, safe_index_func_val });
        definitions.push_back({ index_func_name, index_func_val });
    }
    definitions.push_back({ raw_index_func_name, raw_index_func_val });

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
        {_name + "_GROUPS_NUM", toCodeString(_tensor.G().v)},
        {_name + "_X_PITCH", toCodeString(_tensor.X().pitch)},
        {_name + "_Y_PITCH", toCodeString(_tensor.Y().pitch)},
        {_name + "_Z_PITCH", toCodeString(_tensor.Z().pitch)},
        {_name + "_IFM_PITCH", toCodeString(_tensor.IFM().pitch)},
        {_name + "_OFM_PITCH", toCodeString(_tensor.OFM().pitch)},
        {_name + "_GROUPS_PITCH", toCodeString(_tensor.G().pitch)},
    };

    definitions.insert(definitions.end(), baseDefinitions.begin(), baseDefinitions.end());

    return definitions;
}

std::shared_ptr<JitConstant> MakeJitConstant(const std::string& name, const WeightsTensor& value) {
    return std::static_pointer_cast<JitConstant>(std::make_shared<WeightTensorJitConstant>(name, value));
}

JitConstants MakeActivationJitConstants(ActivationFunction activation_function,
                                        Datatype out_dt,
                                        const std::string& suffix,
                                        bool use_type_parameter,
                                        bool disable_type_conversion) {
    std::string name = "ACTIVATION_FUNC" + suffix;
    JitConstants jitConstants = {};

    jitConstants.Merge(MakeTypeJitConstants(out_dt, name));
    // See the comment in the jitter.h regarding `use_type_parameter`.
    // The "CAT" macro is expected to be defined through the inlcusion of
    // 'common.cl' in the kernel.
    auto type_handler =
        [use_type_parameter, name](const std::string& prefix,
                                   const std::string& suffix) -> std::string {
        if (!use_type_parameter)
            return prefix + name + suffix;

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
            jitConstants.AddConstant(MakeJitConstant(macro_def, erf(input).str()));
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
        case ActivationFunction::SWISH: {
            jitConstants.AddConstant(MakeJitConstant(
                    macro_def,
                    (input / (one + exp(neg(input)))).str()));
            break;
        }
        case ActivationFunction::HSWISH: {
            std::string type_suffix = out_dt == Datatype::F32 ? "f" : "h";
            const JitTerm three("3." + type_suffix);
            const JitTerm six("6." + type_suffix);
            jitConstants.AddConstant(MakeJitConstant(
                    macro_def,
                    (input * min_func(max_func(zero, input + three), six) / six).str()));
            break;
        }
        case ActivationFunction::MISH: {
            std::string type_suffix = out_dt == Datatype::F32 ? "f" : "h";
            auto bound = out_dt == Datatype::F32 ? "9.9f"_jit : "4.75h"_jit;
            const JitTerm two("2." + type_suffix);
            const JitTerm n((exp(input) + two) * exp(input));
            const JitTerm common_mish_formula((input * n) / (n + two));

            jitConstants.AddConstant(MakeJitConstant(
                macro_def,
                ternary(input.ge(bound), input, common_mish_formula).str()));
            break;
        }
        case ActivationFunction::GELU: {
            std::string type_suffix = out_dt == Datatype::F32 ? "f" : "h";
            const JitTerm half{"0.5" + type_suffix};
            const JitTerm mult{std::to_string(1.0f / std::sqrt(2.0f)) + type_suffix};
            jitConstants.AddConstant(MakeJitConstant(
                    macro_def,
                    (half * input * (one + erf((input * mult)))).str()));
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
    std::string as_type;
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
            as_type = "as_char(v)";
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
            as_type = "as_uchar(v)";
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
            as_type = "as_short(v)";
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
            as_type = "as_ushort(v)";
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
            as_type = "as_int(v)";
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
            as_type = "as_uint(v)";
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
            as_type = "as_long(v)";
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
            as_type = "as_half(v)";
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
            as_type = "as_float(v)";
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
        MakeJitConstant("AS_" + macroName + "_TYPE(v)", as_type),
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
                                        Datatype out_dt,
                                        const std::string& suffix,
                                        bool use_type_parameter,
                                        bool disable_type_conversion) {
    auto jitConstants = JitConstants{MakeJitConstant("NL_M" + suffix, params.m),
                                     MakeJitConstant("NL_N" + suffix, params.n)};
    jitConstants.Merge(MakeActivationJitConstants(
        params.function, out_dt, suffix, use_type_parameter, disable_type_conversion));
    return jitConstants;
}

JitConstants MakeActivationJitConstants(std::vector<kernel_selector::base_activation_params> params,
                                        Datatype out_dt,
                                        const std::string& suffix,
                                        bool use_type_parameter,
                                        bool disable_type_conversion) {
    JitConstants res = {};
    if (params.empty()) {
        return MakeActivationJitConstants({ActivationFunction::NONE, 0.f, 0.f}, out_dt,
                                          suffix, use_type_parameter, disable_type_conversion);
    }
    std::string res_activation = "";
    std::string activation_params = "";
    for (size_t i = 0; i < params.size(); i++) {
        std::string activation_suffix = suffix + "_" + std::to_string(i);
        auto jitConstants = JitConstants{MakeJitConstant("NL_M" + activation_suffix, params[i].m),
                                         MakeJitConstant("NL_N" + activation_suffix, params[i].n)};
        jitConstants.Merge(MakeActivationJitConstants(
                params[i].function, out_dt, activation_suffix, use_type_parameter, disable_type_conversion));
        res.Merge(jitConstants);

        if (i == 0) {
            activation_params = use_type_parameter ? "(jit_type, input, params)" : "(input, params)";
            res_activation = "ACTIVATION_FUNC" + activation_suffix + activation_params;
        } else {
            res_activation = "ACTIVATION" + activation_suffix + "(" + (use_type_parameter ? "jit_type, " : "") +
                             res_activation + ", ACTIVATION_PARAMS" + activation_suffix + ")";
        }
    }
    activation_params = use_type_parameter ? "(jit_type, input, params)" : "(input, params)";
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

bool FusedOpsCodeGenerator::CanPreloadData(const FusedOpsConfiguration& conf) const {
    if (conf.loop_axes.empty())
        return true;

    bool can_preload = true;
    // Check that tensor offset doesn't have dependency from the loop dimensions
    for (auto& d : conf.loop_axes) {
        for (auto& t : desc.tensors) {
            auto idx = idx_desc{conf.bfzyx_idx_order, t};
            switch (d) {
                case Tensor::DataChannelName::BATCH:   can_preload &= idx.b == "0"; break;
                case Tensor::DataChannelName::FEATURE: can_preload &= idx.f == "0"; break;
                case Tensor::DataChannelName::Z:       can_preload &= idx.z == "0"; break;
                case Tensor::DataChannelName::Y:       can_preload &= idx.y == "0"; break;
                case Tensor::DataChannelName::X:       can_preload &= idx.x == "0"; break;
                default: return false;
            }
        }
    }

    return can_preload;
}

std::string FusedOpsCodeGenerator::GetTypeStr() const {
    switch (desc.GetType()) {
        case KernelType::ELTWISE: return "eltwise";
        case KernelType::SCALE: return "scale";
        case KernelType::QUANTIZE: return "quantize";
        case KernelType::ACTIVATION: return "activation";
        case KernelType::UNKNOWN: throw std::runtime_error("Invalid type of fused operation. Fused op can't have type UNKNOWN");
        default: return "";
    }
}

JitConstants FusedOpsCodeGenerator::MakeFusedTensorJitConstants(const FusedOpsConfiguration& /*conf*/) const {
    JitConstants jit{};
    for (size_t op_input_id = 0; op_input_id < desc.tensors.size(); op_input_id++) {
        std::string name = GetInputTensorName(op_input_id);
        jit.AddConstant(MakeJitConstant(name, desc.tensors[op_input_id]));
    }
    jit.AddConstant(MakeJitConstant(GetOutputTensorName(), desc.output_tensor));
    return jit;
}

JitConstants FusedOpsCodeGenerator::MakeInputDeclsJitConstants(const FusedOpsConfiguration& /*conf*/) const {
    JitConstants jit = {};

    std::string input_decls = "";
    for (size_t op_input_id = 0; op_input_id < desc.tensors.size(); op_input_id++) {
        std::string ptr_name = GetInputPtrName(op_input_id);
        input_decls += "\\\n\tconst __global " + toCLType(desc.tensors[op_input_id].GetDType()) +
                       "* " + ptr_name + (op_input_id == desc.tensors.size() - 1 ? "" : ",");
    }

    jit.AddConstant(MakeJitConstant("FUSED_OP"+std::to_string(desc.op_id)+"_DECLS", input_decls));
    return jit;
}

JitConstants FusedOpsCodeGenerator::MakeLoadJitConstants(const FusedOpsConfiguration& conf, const DataTensor prim_output) const {
    JitConstants jit = {};

    auto vec_size = conf.vec_size;
    auto idx = conf.bfzyx_idx_order;
    auto fused_op_config = conf;

    std::string load_decls = "";
    static int i = 0;
    // TODO: check if there is a use case for index reuse or it can be removed
    bool reuse_index = false;
    bool safe_load = conf.boundary_check == FusedOpsConfiguration::BoundaryCheck::ENABLED;
    std::string reused_idx = "reused_idx_" + std::to_string(i++);
    if (reuse_index) {
        load_decls += "\\\n\tint " + reused_idx + " = " +  GetIdx(0, idx_desc{idx, desc.tensors[0]}, safe_load) + ";";
    }
    // TODO: add some generic way to support shuffled feature, lets say possibility to add separate config for each fused op
    if (desc.GetType() == KernelType::ELTWISE && conf.load_type == FusedOpsConfiguration::LoadType::FEATURE_SHUFFLE) {
        std::string sub_group_local_id_str = "get_sub_group_local_id()";
        size_t found_sub = conf.bfzyx_idx_order[1].rfind(sub_group_local_id_str);
        if (found_sub != std::string::npos)
            fused_op_config.bfzyx_idx_order[1].replace(found_sub, sub_group_local_id_str.length(), fused_op_config.shuffle_var_name);
    }

    for (auto op_input_id : GetRequiredInputs()) {
        load_decls += "\\\n\t" + GetInputTypeName(op_input_id, vec_size) + " " + GetInputVarName(op_input_id) + " = " +
                      GetJitLoad(fused_op_config, op_input_id, prim_output, reuse_index, reused_idx) + ";";
    }

    jit.AddConstant(MakeJitConstant("FUSED_OP"+std::to_string(desc.op_id)+"_LOAD" + conf.suffix, load_decls));

    return jit;
}

JitConstants FusedOpsCodeGenerator::MakeOpJitConstants(const FusedOpsConfiguration& conf,
                                                       const std::string in_var, const Datatype in_type,
                                                       std::string& out_var, Datatype& out_type) const {
    JitConstants jit = {};

    std::string op_decls = "";
    auto vec_size = conf.vec_size;
    auto idx = conf.bfzyx_idx_order;
    std::string shuffle_var = conf.shuffle_var_name;
    bool is_shuffled = false;

    out_var = GetOutputVarName(in_var);
    out_type = desc.output_tensor.GetDType();

    if (conf.load_type == FusedOpsConfiguration::LoadType::FEATURE_SHUFFLE &&
        (desc.GetType() == KernelType::SCALE || desc.GetType() == KernelType::QUANTIZE)) {
        is_shuffled = true;
    }

    std::vector<std::string> in_vars_converted;
    for (size_t i = 0; i < desc.tensors.size(); i++) {
        auto in_name = GetInputVarName(i, is_shuffled, shuffle_var);
        if (desc.tensors[0].GetDType() != desc.output_tensor.GetDType()) {
            in_name = ConvertToOutputType(in_name, vec_size);
        }
        in_vars_converted.push_back(in_name);
    }

    switch (desc.GetType()) {
        case KernelType::SCALE: {
            auto get_acc_t = [&]() -> Datatype {
                std::vector<Datatype> tensor_types = {desc.output_tensor.GetDType()};
                for (auto& in : desc.tensors) {
                    tensor_types.push_back(in.GetDType());
                }

                std::vector<Datatype> types_prioritized = { Datatype::F32, Datatype::F16 };

                for (auto& type : types_prioritized) {
                    if (std::any_of(tensor_types.begin(), tensor_types.end(), [=](const Datatype& t) -> bool { return t == type; })) {
                        return type;
                    }
                }

                return Datatype::F32;
            };

            auto get_input = [&](size_t index) -> std::string {
                auto in_name = index == 0 ? in_var : GetInputVarName(index - 1, is_shuffled, shuffle_var);
                auto tensor_type = index == 0 ? in_type : desc.tensors[index - 1].GetDType();
                auto acc_t = get_acc_t();

                if (tensor_type != acc_t)
                    return ConvertToType(in_name, acc_t, vec_size);
                else
                    return in_name;
            };

            auto tmp_var = out_var + "_tmp";
            if (desc.tensors.size() > 1) {
                op_decls += "\\\n\t" + GetType(get_acc_t(), vec_size) + " " + tmp_var + " = "
                          + get_input(0) + " * " + get_input(1) + " + " + get_input(2) + ";";
            } else {
                op_decls += "\\\n\t" + GetType(get_acc_t(), vec_size) + " " + tmp_var + " = "
                          + get_input(0) + " * " + get_input(1) + ";";
            }
            op_decls += "\\\n\t" + GetOutputType(vec_size) + " " + out_var + " = " + ConvertToOutputType(tmp_var, vec_size) + ";";
            break;
        }
        case KernelType::ELTWISE: {
            auto p = desc.GetOpParams<eltwise_fuse_params>();
            if (!p)
                throw std::runtime_error("[clDNN] Eltwise fuse params can't be nullptr");

            std::string op = "";
            switch (p->mode)
            {
            case kernel_selector::EltwiseMode::ADD:
                op = "+";
                break;
            case kernel_selector::EltwiseMode::MUL:
                op = "*";
                break;
            default:
                throw std::runtime_error("[clDNN] Eltwise mode is not supported in fused ops codegen");
            }

            op_decls += "\\\n\t" + GetOutputType(vec_size) + " " + out_var + " = " + in_vars_converted[0] +
                        op + ConvertToOutputType(in_var, vec_size) + ";";
            break;
        }
        case KernelType::QUANTIZE: {
            auto p = desc.GetOpParams<quantize_fuse_params>();
            if (!p)
                throw std::runtime_error("[clDNN] Quantize fuse params can't be nullptr");

            std::string in_converted = in_var;
            Datatype tmp_type = Datatype::F32;
            std::string tmp_type_str = GetType(tmp_type, vec_size);
            std::string tmp_var = out_var + "_tmp";

            if (in_type != tmp_type) {
                in_converted = ConvertToType(in_var, tmp_type, vec_size);
            }

            auto post_scale = p->per_tensor_output_scale ? Broadcast(std::to_string(p->out_scale), tmp_type, vec_size)
                                                         : ConvertToType(GetInputVarName(p->out_scale_idx, is_shuffled, shuffle_var), tmp_type, vec_size);
            auto post_shift = p->per_tensor_output_shift ? Broadcast(std::to_string(p->out_shift), tmp_type, vec_size)
                                                         : ConvertToType(GetInputVarName(p->out_shift_idx, is_shuffled, shuffle_var), tmp_type, vec_size);
            auto pre_scale = p->per_tensor_input_scale ? Broadcast(std::to_string(p->in_scale), tmp_type, vec_size)
                                                       : ConvertToType(GetInputVarName(p->in_scale_idx, is_shuffled, shuffle_var), tmp_type, vec_size);
            auto pre_shift = p->per_tensor_input_shift ? Broadcast(std::to_string(p->in_shift), tmp_type, vec_size)
                                                       : ConvertToType(GetInputVarName(p->in_shift_idx, is_shuffled, shuffle_var), tmp_type, vec_size);
            auto in_lo = p->per_tensor_input_range ? Broadcast(std::to_string(p->in_lo), tmp_type, vec_size)
                                                   : ConvertToType(GetInputVarName(p->in_range_lo_idx, is_shuffled, shuffle_var), tmp_type, vec_size);
            auto in_hi = p->per_tensor_input_range ? Broadcast(std::to_string(p->in_hi), tmp_type, vec_size)
                                                   : ConvertToType(GetInputVarName(p->in_range_hi_idx, is_shuffled, shuffle_var), tmp_type, vec_size);

            if (p->has_clamp) {
                op_decls += "\\\n\t" + tmp_type_str + " " + tmp_var + " = min(max(" + in_lo + ", " + in_converted + "), " + in_hi + ");";
            } else {
                op_decls += "\\\n\t" + tmp_type_str + " " + tmp_var + " = " + in_converted + ";";
            }
            op_decls += "\\\n\t" + tmp_var + " = " + tmp_var + "*" + pre_scale + ";";
            if (p->has_pre_shift)
                op_decls += "\\\n\t" + tmp_var + " = " + tmp_var + " + " + pre_shift + ";";

            op_decls += "\\\n\t" + tmp_var + " = round(" + tmp_var + ");";

            bool need_round = (p->has_post_scale || p->has_post_shift) &&
                              (desc.output_tensor.GetDType() == Datatype::UINT8 || desc.output_tensor.GetDType() == Datatype::INT8);
            if (p->has_post_scale)
                op_decls += "\\\n\t" + tmp_var + " = (" + tmp_var + "*" + post_scale + ");";
            if (p->has_post_shift)
                op_decls += "\\\n\t" + tmp_var + " = (" + tmp_var + " + " + post_shift + ");";
            if (need_round)
                op_decls += "\\\n\t" + tmp_var + " = round(" + tmp_var + ");";

            op_decls += "\\\n\t" + GetOutputType(vec_size) + " " + out_var + " = " + ConvertToOutputTypeSat(tmp_var, vec_size) +";";
            break;
        }
        case KernelType::ACTIVATION: {
            auto p = desc.GetOpParams<activation_fuse_params>();
            base_activation_params activation_p = p->param;
            op_decls += "\\\n\t" + GetOutputType(vec_size) + " " + out_var + " = " + in_var + ";";
            if (activation_p.function != ActivationFunction::NONE) {
                auto suffix = "_FUSED_OP"+std::to_string(desc.op_id) + conf.suffix;
                std::string nl_m = std::to_string(activation_p.m);
                std::string nl_n = std::to_string(activation_p.n);

                if (desc.tensors.size() == 1) {
                    if (desc.tensors[0].GetDType() != out_type) {
                        nl_m = ConvertToOutputType(GetInputVarName(0), vec_size);
                    } else {
                        nl_m = GetInputVarName(0);
                    }
                } else {
                    nl_m = Broadcast(nl_m, out_type, vec_size);
                }

                nl_n = Broadcast(nl_n, out_type, vec_size);

                // Disable type casts in activation, since current jit generator for activation don't respect vector size of parameters.
                // So conversion is explicitly done in params declaration
                jit.Merge(MakeActivationJitConstants(activation_p.function, out_type, suffix, false, true));
                std::string params = nl_m + ","+ nl_n;
                op_decls += "\\\n\t" + out_var + " = ACTIVATION_FUNC" + suffix + "(" + out_var + ", " + params + ");";
            }
            break;
        }
        default: break;
    }

    jit.AddConstant(MakeJitConstant("FUSED_OP"+std::to_string(desc.op_id)+"_ACTION" + conf.suffix, op_decls));

    return jit;
}

std::string FusedOpsCodeGenerator::GetInputTensorName(size_t input_id) const {
    return "FUSED_OP_" + std::to_string(desc.op_id) + "_INPUT" + std::to_string(input_id);
}

std::string FusedOpsCodeGenerator::GetOutputTensorName() const {
    return "FUSED_OP_" + std::to_string(desc.op_id) + "_OUTPUT";
}

std::string FusedOpsCodeGenerator::GetInputTypeName(size_t input_id, size_t vec_size) const {
    if (vec_size == 0 || vec_size > 8)
        throw std::invalid_argument("Invalid vector size in jit definitions: " + std::to_string(vec_size));
    std::string scalar_type = GetInputTensorName(input_id) + "_TYPE";
    if (vec_size > 1)
        return "MAKE_VECTOR_TYPE(" + scalar_type + "," + std::to_string(vec_size) + ")";
    else
        return scalar_type;
}

std::string FusedOpsCodeGenerator::GetIdx(size_t input_id, idx_desc idx, bool should_be_safe) const {
    std::string idx_order = "";

    if (DataTensor::ChannelsCount(desc.tensors[input_id].GetLayout()) <= 4) {
        idx_order = idx.b + "," + idx.f + "," + idx.y + "," + idx.x;
    } else if (DataTensor::ChannelsCount(desc.tensors[input_id].GetLayout()) == 5) {
        idx_order = idx.b + "," + idx.f + "," + idx.z + "," + idx.y + "," + idx.x;
    }

    if (should_be_safe) {
        return GetInputTensorName(input_id) + "_GET_INDEX_SAFE(" + idx_order + ")";
    } else {
        return GetInputTensorName(input_id) + "_GET_INDEX(" + idx_order + ")";
    }
}

std::string FusedOpsCodeGenerator::GetJitLoad(const FusedOpsConfiguration& conf, size_t input_id, const DataTensor prim_output,
                                                          bool reuse_index, std::string reused_idx) const {
    auto& input_tensor = desc.tensors[input_id];
    size_t vec_size = 1;
    auto input_dt = input_tensor.GetDType();

    if (desc.GetType() == KernelType::ELTWISE &&
        input_tensor.GetLayout() != prim_output.GetLayout() && conf.vec_size > 1) {
        throw std::runtime_error("[clDNN] Mixed layouts of input tensors are not supported in fused eltwise");
    }

    if (conf.vec_axis != Tensor::DataChannelName::COUNT &&
        DataTensor::Extract(input_tensor.GetLayout(), conf.vec_axis, input_tensor.GetDims()).v != 1) {
        vec_size = conf.vec_size;
    }

    auto idx = conf.bfzyx_idx_order;
    if (vec_size == 0 || vec_size > 8)
        throw std::invalid_argument("Invalid vector size in jit definitions: " + std::to_string(vec_size));

    bool safe_load = conf.boundary_check == FusedOpsConfiguration::BoundaryCheck::ENABLED;

    std::string index_func_call_vec = reuse_index ? reused_idx : GetIdx(input_id, idx_desc{idx, desc.tensors[input_id]}, safe_load);
    std::string index_func_call = reuse_index ? reused_idx : GetIdx(input_id, idx_desc{idx, desc.tensors[input_id]}, safe_load);
    if (conf.index_type == FusedOpsConfiguration::IndexType::LINEAR_OFFSET) {
        std::string offset = conf.bfzyx_idx_order[0];
        if (safe_load)
            offset = "(" + offset + " % " + std::to_string(input_tensor.LogicalSize()) + ")";
        if (vec_size > 1)
            return "((const __global " + toCLType(input_dt) + std::to_string(vec_size) + "*)(" +
                   GetInputPtrName(input_id) + " + " + offset + "))[0]";
        else
            return GetInputPtrName(input_id) + "[" + offset + "]";
    } else {
        // TODO: Need to add smarter vectors handling:
        // 1. Boundary checks for safe load
        // 2. If in given configuration data can't be loaded by a simple UNIT_BLOCK_READx call or load from casted ptr,
        //    we can gather the data to vector
        if (conf.load_type == FusedOpsConfiguration::LoadType::LT_ALIGNED_READ) {
            std::string vs = vec_size > 1 ? std::to_string(vec_size)  : "";
            std::string block_read;

            if (input_dt == Datatype::F32) {
                block_read = CastToType(" intel_sub_group_block_read" + vs + "("
                                        + "(const __global uint*)(" + GetInputPtrName(input_id) + " + " + index_func_call_vec + "))",
                                        input_dt, vec_size);
            } else if (input_dt == Datatype::F16) {
                block_read = CastToType(" intel_sub_group_block_read_us" + vs + "("
                                        + "(const __global ushort*)(" + GetInputPtrName(input_id) + " + " + index_func_call_vec + "))",
                                        input_dt, vec_size);
            } else if (input_dt == Datatype::UINT8 || input_dt == Datatype::INT8) {
                block_read = CastToType("BLOCK_READ_UC_" + std::to_string(vec_size) + "("
                                        + "(const __global uchar*)(" + GetInputPtrName(input_id) + " + " + index_func_call_vec + "))",
                                        input_dt, vec_size);
            } else {
                throw std::runtime_error("Aligned load is not supported yet for " + toCLType(input_dt) + " data type");
            }

            if (vec_size > 1) {
                return block_read;
            } else if (input_tensor.LogicalSize() > 1) {
                // Currently we assume that in such scenario we can safely load sub_group_size elements from the pointer
                return Broadcast(block_read, input_dt, conf.vec_size);
            } else {
                // Input has only one element, so broadcast it for the whole vector size
                return Broadcast(GetInputPtrName(input_id) + "[" + index_func_call + "]", input_dt, conf.vec_size);
            }
        } else {
            if (vec_size > 1) {
                return "((const __global " + toCLType(input_dt) + std::to_string(vec_size) + "*)(" +
                       GetInputPtrName(input_id) + " + " + index_func_call_vec + "))[0]";
            } else {
                return GetInputPtrName(input_id) + "[" + index_func_call + "]";
            }
        }
    }
}

std::string FusedOpsCodeGenerator::GetInputPtrName(size_t input_id) const {
    return GetTypeStr() + std::to_string(desc.op_id) + "_input" + std::to_string(input_id);
}

std::string FusedOpsCodeGenerator::GetInputVarName(size_t input_id, bool is_shuffled, std::string shuffle_var) const {
    if (is_shuffled)
        return "intel_sub_group_shuffle(" + GetTypeStr() + std::to_string(desc.op_id) + "_data" +
               std::to_string(input_id) + ", " + shuffle_var + ")";
    return GetTypeStr() + std::to_string(desc.op_id) + "_data" + std::to_string(input_id);
}

std::string FusedOpsCodeGenerator::GetOutputVarName(std::string input_var) const {
    std::replace(input_var.begin(), input_var.end(), '[', '_');
    std::replace(input_var.begin(), input_var.end(), ']', '_');
    std::replace(input_var.begin(), input_var.end(), ' ', '_');
    return input_var + "_out";
}

std::string FusedOpsCodeGenerator::GetType(Datatype dt, size_t vec_size) const {
    if (vec_size > 1)
        return toCLType(dt) + std::to_string(vec_size);
    else
        return toCLType(dt);
}

std::string FusedOpsCodeGenerator::GetOutputType(size_t vec_size) const {
    return GetType(desc.output_tensor.GetDType(), vec_size);
}

std::string FusedOpsCodeGenerator::ConvertToType(std::string var, Datatype dt, size_t vec_size) const {
    return "convert_" + GetType(dt, vec_size) + "(" + var + ")";
}

std::string FusedOpsCodeGenerator::CastToType(std::string var, Datatype dt, size_t vec_size) const {
    return "as_" + GetType(dt, vec_size) + "(" + var + ")";
}

std::string FusedOpsCodeGenerator::ConvertToOutputType(std::string var, size_t vec_size) const {
    return ConvertToType(var, desc.output_tensor.GetDType(), vec_size);
}

std::string FusedOpsCodeGenerator::Broadcast(std::string var, Datatype dt, size_t vec_size) const {
    return "(" + GetType(dt, vec_size) + ")(" + var + ")";
}

std::string FusedOpsCodeGenerator::ConvertToOutputTypeSat(std::string var, size_t vec_size) const {
    if (desc.output_tensor.GetDType() == Datatype::F32 || desc.output_tensor.GetDType() == Datatype::F16)
        return "convert_" + GetOutputType(vec_size) + "(" + var + ")";
    else
        return "convert_" + GetOutputType(vec_size) + "_sat(" + var + ")";
}

std::vector<size_t> FusedOpsCodeGenerator::GetRequiredInputs() const {
    switch (desc.GetType()) {
        case KernelType::QUANTIZE: {
            auto p = std::dynamic_pointer_cast<quantize_fuse_params>(desc.op_params);
            if (p) {
                std::vector<size_t> res = {};
                if (!p->per_tensor_input_range && p->has_clamp) {
                    res.push_back(p->in_range_lo_idx);
                    res.push_back(p->in_range_hi_idx);
                }
                if (!p->per_tensor_input_scale)
                    res.push_back(p->in_scale_idx);
                if (p->has_pre_shift && !p->per_tensor_input_shift)
                    res.push_back(p->in_shift_idx);
                if (p->has_post_scale && !p->per_tensor_output_scale)
                    res.push_back(p->out_scale_idx);
                if (p->has_post_shift && !p->per_tensor_output_shift)
                    res.push_back(p->out_shift_idx);

                return res;
            }
            return {};
        }
        default: {
            std::vector<size_t> res;
            for (size_t i = 0; i < desc.tensors.size(); i++) {
                res.push_back(i);
            }
            return res;
        }
    }
}

}  // namespace kernel_selector
