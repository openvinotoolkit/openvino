// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cinttypes>

#include "jitter.h"
#include "kernel_selector_utils.h"
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

JitTerm tanh(const JitTerm& arg) {
    JitTerm jit_term{"(tanh(" + arg.str() + "))"};
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
        case WeightsType::INT4:
        case WeightsType::INT8:
            return GetTypeName<int8_t>();
        case WeightsType::UINT4:
        case WeightsType::UINT8:
            return GetTypeName<uint8_t>();
        case WeightsType::F16:
            return "half";
        case WeightsType::F32:
            return GetTypeName<float>();
        case WeightsType::INT32:
            return GetTypeName<int32_t>();
        default:
            return "";
    }
}

std::string toCLType(Datatype dType) {
    switch (dType) {
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
// Longest notation for value represented by double type has 24 chars
static thread_local char buf[24 + 24 + 18] = "";

std::string toCodeString(uint8_t val) {
    snprintf(buf, sizeof(buf), "%d", static_cast<int>(val));
    return buf;
}

std::string toCodeString(int8_t val) {
    snprintf(buf, sizeof(buf), "%d", static_cast<int>(val));
    return buf;
}

std::string toCodeString(size_t val) {
    snprintf(buf, sizeof(buf), "%zu", val);
    return buf;
}

std::string toCodeString(const Tensor::Dim& dim, size_t offset, bool padded, bool pad_is_dynamic, size_t pad_offset) {
    std::string pad_str = "";
    if (padded) {
        if (pad_is_dynamic) {
            pad_str = " + (shape_info[" + std::to_string(pad_offset) + "] + shape_info[" +
                      std::to_string(pad_offset + 1) + "])";
        } else {
            pad_str = " + " + std::to_string(dim.pad.Total());
        }
    }
    if (dim.is_dynamic) {
        snprintf(buf, sizeof(buf), "(shape_info[%zu] %s)", offset, pad_str.c_str());
    } else {
        if (pad_is_dynamic) {
            snprintf(buf, sizeof(buf), "(%zu %s)", dim.v, pad_str.c_str()); // Static dim, dynamic padding
        } else {
            snprintf(buf, sizeof(buf), "%zu", dim.v + (padded ? dim.pad.Total() : 0));  // Static dim, static padding
        }
    }
    return buf;
}

std::string toCodeString(float val) {
    if (std::isinf(val))
        return std::signbit(val) ? "-INFINITY" : "INFINITY";
    // Workaround GCC compiler/STL bug
    snprintf(buf, sizeof(buf), "as_float(0x%" PRIx32 ")/*%.6e*/", *reinterpret_cast<uint32_t*>(&val), val);
    return buf;
}

std::string toCodeString(double val) {
    if (std::isinf(val))
        return std::signbit(val) ? "-INFINITY" : "INFINITY";
    // Workaround GCC compiler/STL bug
    snprintf(buf, sizeof(buf), "as_double(0x%" PRIx64 ")/*%.6e*/", *reinterpret_cast<uint64_t*>(&val), val);
    return buf;
}

std::string toShapeInfoString(size_t arg_idx, size_t data_idx, bool is_output, size_t num_of_inputs) {
    auto dims_rank = DataTensor::max_rank();
    size_t actual_idx = (num_of_inputs * dims_rank * (is_output ? 1 : 0)) + (dims_rank * arg_idx) + data_idx;
    snprintf(buf, sizeof(buf), "shape_info[%zu]", actual_idx);
    return buf;
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
    using JitConstant::GetDefinitions;

    JitDefinitions GetDefinitions(const Tensor::TensorBaseT<DType, Layout>& t) const {
        JitDefinitions definitions{
            {_name + "_VIEW_OFFSET", toCodeString(t.GetViewOffset())},
            {_name + "_LENGTH", toCodeString(t.LogicalSize())},
            {_name + "_DIMS", toCodeString(t.GetDims().size())},
            {_name + "_SIMPLE", toCodeString(t.SimpleLayout())},
            {_name + "_GROUPED", toCodeString(t.GroupedLayout())},
            {_name + "_LAYOUT_" + toString(t.GetLayout()), "1"},
        };

        auto type_defs = MakeTypeJitConstants(t.GetDType(), _name).GetDefinitions();
        definitions.insert(definitions.end(), type_defs.begin(), type_defs.end());

        if (!t.is_dynamic()) {
            definitions.push_back({_name + "_OFFSET", toCodeString(t.GetFirstElementOffset())});
            definitions.push_back(
                {_name + "_SIZES_DATA",
                toVectorString(t.GetDims(), "", KERNEL_SELECTOR_TENSOR_DIM_MAX, 1, [](const Tensor::Dim& d) { return d.v; })});
            definitions.push_back(
                {_name + "_PITCHES",
                toVectorString(t.GetDims(), "size_t", KERNEL_SELECTOR_TENSOR_DIM_MAX, 1, [](const Tensor::Dim& d) { return d.pitch; })});
        } else {
            // calculate tensor offset
            std::vector<std::string> padded_pitches = {
                toVectorMulString({_name + "_X_PITCH", _name + "_PAD_BEFORE_SIZE_X"}),
                toVectorMulString({_name + "_Y_PITCH", _name + "_PAD_BEFORE_SIZE_Y"}),
                toVectorMulString({_name + "_Z_PITCH", _name + "_PAD_BEFORE_SIZE_Z"}),
                toVectorMulString({_name + "_W_PITCH", _name + "_PAD_BEFORE_SIZE_W"}),
                toVectorMulString({_name + "_FEATURE_PITCH", _name + "_PAD_BEFORE_FEATURE_NUM"}),
                toVectorMulString({_name + "_BATCH_PITCH", _name + "_PAD_BEFORE_BATCH_NUM"})};
            std::string offset_str = "(";
            for (size_t i = 0; i < padded_pitches.size(); ++i) {
                offset_str += padded_pitches[i];
                if (i < padded_pitches.size() - 1)
                    offset_str += " + ";
            }
            offset_str += ")";
            definitions.push_back({_name + "_OFFSET", offset_str});
        }
        definitions.push_back(
            {_name + "_PAD_BEFORE",
             toVectorString(t.GetDims(), "size_t", KERNEL_SELECTOR_TENSOR_DIM_MAX, 0, [](const Tensor::Dim& d) {
                 return d.pad.before;
             })});
        definitions.push_back(
            {_name + "_PAD_AFTER",
             toVectorString(t.GetDims(), "size_t", KERNEL_SELECTOR_TENSOR_DIM_MAX, 0, [](const Tensor::Dim& d) {
                 return d.pad.after;
             })});

        return definitions;
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// DataTensorJitConstant
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class DataTensorJitConstant : public TensorBaseTJitConstant<Datatype, DataLayout> {
    const DataTensor _tensor;

public:
    DataTensorJitConstant(const std::string& name, const DataTensor& t)
    : TensorBaseTJitConstant(name)
    , _tensor(t) {}

    JitDefinitions GetDefinitions() const override;
};

JitDefinitions DataTensorJitConstant::GetDefinitions() const {
    JitDefinitions baseDefinitions = TensorBaseTJitConstant::GetDefinitions(_tensor);

    JitDefinitions definitions{};
    DimensionAccessHelper dims(_tensor);
    DimensionAccessHelper dims_padded(_tensor, true);
    // shape_info layout
    // if only y has dynamic padding:
    // [dim_b, dim_f, dim_v, dim_u, dim_w, dim_z, dim_y, dim_x, pad_before_y, pad_after_y]
    // if only x has dynamic padding:
    // [dim_b, dim_f, dim_v, dim_u, dim_w, dim_z, dim_y, dim_x, pad_before_x, pad_after_x]

    definitions = {
        {_name + "_SIZE_X", dims.x()},
        {_name + "_SIZE_Y", dims.y()},
        {_name + "_SIZE_Z", dims.z()},
        {_name + "_SIZE_W", dims.w()},
        {_name + "_SIZE_U", dims.u()},
        {_name + "_SIZE_V", dims.v()},
        {_name + "_FEATURE_NUM", dims.f()},
        {_name + "_BATCH_NUM", dims.b()},
        {_name + "_PAD_BEFORE_SIZE_X", dims_padded.x_pad().first},
        {_name + "_PAD_BEFORE_SIZE_Y", dims_padded.y_pad().first},
        {_name + "_PAD_BEFORE_SIZE_Z", dims_padded.z_pad().first},
        {_name + "_PAD_BEFORE_SIZE_W", dims_padded.w_pad().first},
        {_name + "_PAD_BEFORE_SIZE_U", dims_padded.u_pad().first},
        {_name + "_PAD_BEFORE_SIZE_V", dims_padded.v_pad().first},
        {_name + "_PAD_BEFORE_FEATURE_NUM", dims_padded.f_pad().first},
        {_name + "_PAD_BEFORE_BATCH_NUM", dims_padded.b_pad().first},
        {_name + "_PAD_AFTER_SIZE_X", dims_padded.x_pad().second},
        {_name + "_PAD_AFTER_SIZE_Y", dims_padded.y_pad().second},
        {_name + "_PAD_AFTER_SIZE_Z", dims_padded.z_pad().second},
        {_name + "_PAD_AFTER_SIZE_W", dims_padded.w_pad().second},
        {_name + "_PAD_AFTER_SIZE_U", dims_padded.u_pad().second},
        {_name + "_PAD_AFTER_SIZE_V", dims_padded.v_pad().second},
        {_name + "_PAD_AFTER_FEATURE_NUM", dims_padded.f_pad().second},
        {_name + "_PAD_AFTER_BATCH_NUM", dims_padded.b_pad().second},
    };
    if (_tensor.is_dynamic()) {
        if (_tensor.GetLayout() == DataLayout::bf || _tensor.GetLayout() == DataLayout::bfyx ||
            _tensor.GetLayout() == DataLayout::bfzyx || _tensor.GetLayout() == DataLayout::bfwzyx ||
            _tensor.GetLayout() == DataLayout::bfuwzyx || _tensor.GetLayout() == DataLayout::bfvuwzyx) {
            definitions.push_back({_name + "_X_PITCH", "1"});
            definitions.push_back({_name + "_Y_PITCH", dims_padded.x()});
            definitions.push_back({_name + "_Z_PITCH", toVectorMulString({dims_padded.x(), dims_padded.y()})});
            definitions.push_back(
                {_name + "_W_PITCH", toVectorMulString({dims_padded.x(), dims_padded.y(), dims_padded.z()})});
            definitions.push_back(
                {_name + "_U_PITCH", toVectorMulString({dims_padded.x(), dims_padded.y(), dims_padded.z(), dims_padded.w()})});
            definitions.push_back(
                {_name + "_V_PITCH",
                 toVectorMulString({dims_padded.x(), dims_padded.y(), dims_padded.z(), dims_padded.w(), dims_padded.u()})});
            definitions.push_back(
                {_name + "_FEATURE_PITCH",
                 toVectorMulString(
                     {dims_padded.x(), dims_padded.y(), dims_padded.z(), dims_padded.w(), dims_padded.u(), dims_padded.v()})});
            definitions.push_back({_name + "_BATCH_PITCH",
                                   toVectorMulString({dims_padded.x(),
                                                      dims_padded.y(),
                                                      dims_padded.z(),
                                                      dims_padded.w(),
                                                      dims_padded.u(),
                                                      dims_padded.v(),
                                                      dims_padded.f()})});
        } else {
            OPENVINO_ASSERT(false, "[GPU] Jitter couldn't generate dynamic pitches for given layout");
        }
    } else {
        // static dim
        definitions.push_back({_name + "_X_PITCH", toCodeString(_tensor.X().pitch)});
        definitions.push_back({_name + "_Y_PITCH", toCodeString(_tensor.Y().pitch)});
        definitions.push_back({_name + "_Z_PITCH", toCodeString(_tensor.Z().pitch)});
        definitions.push_back({_name + "_W_PITCH", toCodeString(_tensor.W().pitch)});
        definitions.push_back({_name + "_U_PITCH", toCodeString(_tensor.U().pitch)});
        definitions.push_back({_name + "_V_PITCH", toCodeString(_tensor.V().pitch)});
        definitions.push_back({_name + "_FEATURE_PITCH", toCodeString(_tensor.Feature().pitch)});
        definitions.push_back({_name + "_BATCH_PITCH", toCodeString(_tensor.Batch().pitch)});
    }
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
                       layout == DataLayout::b_fs_yx_fsv2 ||
                       layout == DataLayout::b_fs_yx_fsv4 ||
                       layout == DataLayout::b_fs_yx_fsv8 ||
                       layout == DataLayout::fs_b_yx_fsv32 ||
                       layout == DataLayout::bs_fs_yx_bsv16_fsv16 ||
                       layout == DataLayout::bs_fs_yx_bsv16_fsv32 ||
                       layout == DataLayout::bs_fs_yx_bsv4_fsv4 ||
                       layout == DataLayout::bs_fs_yx_bsv16_fsv8 ||
                       layout == DataLayout::bs_fs_yx_bsv16_fsv4 ||
                       layout == DataLayout::bs_fs_yx_bsv16_fsv2 ||
                       layout == DataLayout::bs_fs_yx_bsv8_fsv4 ||
                       layout == DataLayout::bs_fs_yx_bsv8_fsv2 ||
                       layout == DataLayout::bs_fs_yx_bsv4_fsv2 ||
                       layout == DataLayout::bs_fs_yx_bsv32_fsv16 ||
                       layout == DataLayout::bs_fs_yx_bsv32_fsv32) {
                auto layout_str = toString(layout);
                index_func_val = "GET_DATA_" + layout_str + "_INDEX(" + _name + ", b, f, y, x)";
                raw_index_func_val = "GET_DATA_" + layout_str + "_INDEX(" + _name + ", b, f, y, x)";
                safe_index_func_val = "GET_DATA_" + layout_str + "_INDEX_SAFE(" + _name + ", b, f, y, x)";
            } else if (layout == DataLayout::bs_f_bsv8__af8 ||
                       layout == DataLayout::bs_f_bsv16__af8) {
                size_t sub_group_size = layout == DataLayout::bs_f_bsv16__af8 ? 16 : 8;
                index_func_val = "GET_DATA_BS_FYX_BSV8_INDEX(" + _name + ", b, f, y, x," + toCodeString(sub_group_size) + ")";
                safe_index_func_val = "GET_DATA_BS_FYX_BSV8_INDEX(" + _name + ", b, f, y, x," + toCodeString(sub_group_size) + ")";
                raw_index_func_val = "GET_DATA_BS_FYX_BSV8_INDEX(" + _name + ", b, f, y, x," + toCodeString(sub_group_size) + ")";
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
            } else if (layout == DataLayout::bs_fs_zyx_bsv32_fsv32) {
                index_func_val = "GET_DATA_BS_FS_ZYX_BSV32_FSV32_INDEX(" + _name + ", b, f, z, y, x)";
                raw_index_func_val = "GET_DATA_BS_FS_ZYX_BSV32_FSV32_INDEX(" + _name + ", b, f, z, y, x)";
                safe_index_func_val = "GET_DATA_BS_FS_ZYX_BSV32_FSV32_INDEX_SAFE(" + _name + ", b, f, z, y, x)";
            } else if (layout == DataLayout::bs_fs_zyx_bsv32_fsv16) {
                index_func_val = "GET_DATA_BS_FS_ZYX_BSV32_FSV16_INDEX(" + _name + ", b, f, z, y, x)";
                raw_index_func_val = "GET_DATA_BS_FS_ZYX_BSV32_FSV16_INDEX(" + _name + ", b, f, z, y, x)";
                safe_index_func_val = "GET_DATA_BS_FS_ZYX_BSV32_FSV16_INDEX_SAFE(" + _name + ", b, f, z, y, x)";
            } else if (layout == DataLayout::bs_fs_zyx_bsv16_fsv32) {
                index_func_val = "GET_DATA_BS_FS_ZYX_BSV16_FSV32_INDEX(" + _name + ", b, f, z, y, x)";
                raw_index_func_val = "GET_DATA_BS_FS_ZYX_BSV16_FSV32_INDEX(" + _name + ", b, f, z, y, x)";
                safe_index_func_val = "GET_DATA_BS_FS_ZYX_BSV16_FSV32_INDEX_SAFE(" + _name + ", b, f, z, y, x)";
            } else if (layout == DataLayout::bs_fs_zyx_bsv16_fsv16) {
                index_func_val = "GET_DATA_BS_FS_ZYX_BSV16_FSV16_INDEX(" + _name + ", b, f, z, y, x)";
                raw_index_func_val = "GET_DATA_BS_FS_ZYX_BSV16_FSV16_INDEX(" + _name + ", b, f, z, y, x)";
                safe_index_func_val = "GET_DATA_BS_FS_ZYX_BSV16_FSV16_INDEX_SAFE(" + _name + ", b, f, z, y, x)";
            } else if (layout == DataLayout::bs_fs_zyx_bsv16_fsv8) {
                index_func_val = "GET_DATA_BS_FS_ZYX_BSV16_FSV8_INDEX(" + _name + ", b, f, z, y, x)";
                raw_index_func_val = "GET_DATA_BS_FS_ZYX_BSV16_FSV8_INDEX(" + _name + ", b, f, z, y, x)";
                safe_index_func_val = "GET_DATA_BS_FS_ZYX_BSV16_FSV8_INDEX_SAFE(" + _name + ", b, f, z, y, x)";
            } else if (layout == DataLayout::b_fs_zyx_fsv32) {
                index_func_val = "GET_DATA_B_FS_ZYX_FSV32_INDEX(" + _name + ", b, f, z, y, x)";
                raw_index_func_val = "GET_DATA_B_FS_ZYX_FSV32_INDEX(" + _name + ", b, f, z, y, x)";
                safe_index_func_val = "GET_DATA_B_FS_ZYX_FSV32_INDEX_SAFE(" + _name + ", b, f, z, y, x)";
            } else if (layout == DataLayout::bs_fs_zyx_bsv16_fsv4) {
                index_func_val = "GET_DATA_BS_FS_ZYX_BSV16_FSV4_INDEX(" + _name + ", b, f, z, y, x)";
                raw_index_func_val = "GET_DATA_BS_FS_ZYX_BSV16_FSV4_INDEX(" + _name + ", b, f, z, y, x)";
                safe_index_func_val = "GET_DATA_BS_FS_ZYX_BSV16_FSV4_INDEX_SAFE(" + _name + ", b, f, z, y, x)";
            } else if (layout == DataLayout::bs_fs_zyx_bsv8_fsv4) {
                index_func_val = "GET_DATA_BS_FS_ZYX_BSV8_FSV4_INDEX(" + _name + ", b, f, z, y, x)";
                raw_index_func_val = "GET_DATA_BS_FS_ZYX_BSV8_FSV4_INDEX(" + _name + ", b, f, z, y, x)";
                safe_index_func_val = "GET_DATA_BS_FS_ZYX_BSV8_FSV4_INDEX_SAFE(" + _name + ", b, f, z, y, x)";
            } else if (layout == DataLayout::bs_fs_zyx_bsv16_fsv2) {
                index_func_val = "GET_DATA_BS_FS_ZYX_BSV16_FSV2_INDEX(" + _name + ", b, f, z, y, x)";
                raw_index_func_val = "GET_DATA_BS_FS_ZYX_BSV16_FSV2_INDEX(" + _name + ", b, f, z, y, x)";
                safe_index_func_val = "GET_DATA_BS_FS_ZYX_BSV16_FSV2_INDEX_SAFE(" + _name + ", b, f, z, y, x)";
            } else if (layout == DataLayout::bs_fs_zyx_bsv8_fsv2) {
                index_func_val = "GET_DATA_BS_FS_ZYX_BSV8_FSV2_INDEX(" + _name + ", b, f, z, y, x)";
                raw_index_func_val = "GET_DATA_BS_FS_ZYX_BSV8_FSV2_INDEX(" + _name + ", b, f, z, y, x)";
                safe_index_func_val = "GET_DATA_BS_FS_ZYX_BSV8_FSV2_INDEX_SAFE(" + _name + ", b, f, z, y, x)";
            } else if (layout == DataLayout::b_fs_zyx_fsv2) {
                index_func_val = "GET_DATA_B_FS_ZYX_FSV2_INDEX(" + _name + ", b, f, z, y, x)";
                raw_index_func_val = "GET_DATA_B_FS_ZYX_FSV2_INDEX(" + _name + ", b, f, z, y, x)";
                safe_index_func_val = "GET_DATA_B_FS_ZYX_FSV2_INDEX_SAFE(" + _name + ", b, f, z, y, x)";
            } else if (layout == DataLayout::b_fs_zyx_fsv4) {
                index_func_val = "GET_DATA_B_FS_ZYX_FSV4_INDEX(" + _name + ", b, f, z, y, x)";
                raw_index_func_val = "GET_DATA_B_FS_ZYX_FSV4_INDEX(" + _name + ", b, f, z, y, x)";
                safe_index_func_val = "GET_DATA_B_FS_ZYX_FSV4_INDEX_SAFE(" + _name + ", b, f, z, y, x)";
            } else if (layout == DataLayout::b_fs_zyx_fsv8) {
                index_func_val = "GET_DATA_B_FS_ZYX_FSV8_INDEX(" + _name + ", b, f, z, y, x)";
                raw_index_func_val = "GET_DATA_B_FS_ZYX_FSV8_INDEX(" + _name + ", b, f, z, y, x)";
                safe_index_func_val = "GET_DATA_B_FS_ZYX_FSV8_INDEX_SAFE(" + _name + ", b, f, z, y, x)";
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
        OPENVINO_ASSERT(is_common_6d_layout, "[GPU] Unhandled 6d format in jitter: ", toString(layout));
        index_func_name = _name + "_GET_INDEX(b, f, w, z, y, x)";
        safe_index_func_name = _name + "_GET_INDEX_SAFE(b, f, w, z, y, x)";
        raw_index_func_name = _name + "_GET_INDEX_RAW(b, f, w, z, y, x)";

        index_func_val = "GET_DATA_INDEX_6D(" + _name + ", b, f, w, z, y, x)";
        safe_index_func_val = "GET_DATA_INDEX_6D_SAFE(" + _name + ", b, f, w, z, y, x)";
        raw_index_func_val = "GET_DATA_INDEX_6D_RAW(" + _name + ", b, f, w, z, y, x)";
    } else if (DataTensor::ChannelsCount(layout) == 7) {
        std::vector<Tensor::DataChannelName> base_7d_channels = {
                Tensor::DataChannelName::BATCH,
                Tensor::DataChannelName::FEATURE,
                Tensor::DataChannelName::U,
                Tensor::DataChannelName::W,
                Tensor::DataChannelName::Z,
                Tensor::DataChannelName::Y,
                Tensor::DataChannelName::X,
        };
        bool is_common_7d_layout = is_common_nd_layout(base_7d_channels, layout);
        OPENVINO_ASSERT(is_common_7d_layout, "[GPU] Unhandled 7d format in jitter: ", toString(layout));
        index_func_name = _name + "_GET_INDEX(b, f, u, w, z, y, x)";
        safe_index_func_name = _name + "_GET_INDEX_SAFE(b, f, u, w, z, y, x)";
        raw_index_func_name = _name + "_GET_INDEX_RAW(b, f, u, w, z, y, x)";

        index_func_val = "GET_DATA_INDEX_7D(" + _name + ", b, f, u, w, z, y, x)";
        safe_index_func_val = "GET_DATA_INDEX_7D_SAFE(" + _name + ", b, f, u, w, z, y, x)";
        raw_index_func_val = "GET_DATA_INDEX_7D_RAW(" + _name + ", b, f, u, w, z, y, x)";
    } else if (DataTensor::ChannelsCount(layout) == 8) {
        std::vector<Tensor::DataChannelName> base_8d_channels = {
                Tensor::DataChannelName::BATCH,
                Tensor::DataChannelName::FEATURE,
                Tensor::DataChannelName::V,
                Tensor::DataChannelName::U,
                Tensor::DataChannelName::W,
                Tensor::DataChannelName::Z,
                Tensor::DataChannelName::Y,
                Tensor::DataChannelName::X,
        };
        bool is_common_8d_layout = is_common_nd_layout(base_8d_channels, layout);
        OPENVINO_ASSERT(is_common_8d_layout, "[GPU] Unhandled 8d format in jitter: ", toString(layout));
        index_func_name = _name + "_GET_INDEX(b, f, v, u, w, z, y, x)";
        safe_index_func_name = _name + "_GET_INDEX_SAFE(b, f, v, u, w, z, y, x)";
        raw_index_func_name = _name + "_GET_INDEX_RAW(b, f, v, u, w, z, y, x)";

        index_func_val = "GET_DATA_INDEX_8D(" + _name + ", b, f, v, u, w, z, y, x)";
        safe_index_func_val = "GET_DATA_INDEX_8D_SAFE(" + _name + ", b, f, v, u, w, z, y, x)";
        raw_index_func_val = "GET_DATA_INDEX_8D_RAW(" + _name + ", b, f, v, u, w, z, y, x)";
    } else {
        throw std::runtime_error("Unsupported channels count(" + std::to_string(DataTensor::ChannelsCount(layout)) +
                                 ") in layout: " + toString(layout));
    }

    std::string offset = toCodeString(_tensor.GetFirstElementOffset());
    if (_tensor.LogicalSize() == 1 && !_tensor.is_dynamic()) {
        // if tensor contains single element we can always return first element offset for safe function
        definitions.push_back({ safe_index_func_name, offset });
        definitions.push_back({ index_func_name, offset });
    } else if (_tensor.LogicalSize() == _tensor.Feature().v && !_tensor.is_dynamic()) {
        // We support broadcast only if corresponding dimension is equal to 1.
        // Otherwise, dimensions should be equal and using "f" should be safe.
        if (_tensor.PitchesDifferFromLogicalDims() && _tensor.SimpleLayout()) {
            auto f_pitch = toCodeString(_tensor.Feature().pitch);
            auto f_size = toCodeString(_tensor.Feature().v);
            definitions.push_back({ safe_index_func_name, "(" + offset + " + ((f) % " + f_size + ")  * " + f_pitch + ")" });
            definitions.push_back({ index_func_name, "(" + offset + " + (f) * " + f_pitch + ")" });
        } else if (_tensor.PitchesDifferFromLogicalDims() || _tensor.DoubleBlockedLayout()) {
            // TODO This should be solved differently, by setting the macro arguments to zero
            definitions.push_back({ safe_index_func_name, safe_index_func_val });
            definitions.push_back({ index_func_name, index_func_val });
        } else {
            auto f_pad = toCodeString(_tensor.Feature().pad.before);
            auto f_size = toCodeString(_tensor.Feature().v);
            definitions.push_back({ safe_index_func_name, "((" + offset + " + (f)) % " + f_size + ")" });
            definitions.push_back({ index_func_name, "(" + offset + " + (f))" });
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
    struct WeightIndexFuncDesc {
        std::string macroName;
        std::string macroBody;
        std::string calcFunction;

        WeightIndexFuncDesc() = default;
        WeightIndexFuncDesc(std::string tensor_name, const WeightsLayout l) {
            if (tensor_name == "FILTER")
                return;
            const auto layout_name = toString(l);
            using args = std::initializer_list<std::string>;
            if (l == WeightsLayout::oiyx ||
                l == WeightsLayout::ioyx ||
                l == WeightsLayout::oyxi ||
                l == WeightsLayout::oyix ||
                l == WeightsLayout::oxiy ||
                l == WeightsLayout::iyxo ||
                l == WeightsLayout::yxio ||
                l == WeightsLayout::iozyx ||
                l == WeightsLayout::oizyx ||
                l == WeightsLayout::goiyx ||
                l == WeightsLayout::gioyx ||
                l == WeightsLayout::giozyx ||
                l == WeightsLayout::goizyx) {
                args macroNameArgs = {"prefix", "g", "o", "i", "z", "y", "x"};
                this->calcFunction = FuncBody(layout_name);
                this->macroName = MacroName(tensor_name, layout_name, macroNameArgs);
                this->macroBody = R"V0G0N( \
    CAT(prefix, _OFFSET) + \
    (x)*CAT(prefix, _X_PITCH) + \
    (y)*CAT(prefix, _Y_PITCH) + \
    (z)*CAT(prefix, _Z_PITCH) + \
    (i)*CAT(prefix, _IFM_PITCH) + \
    (o)*CAT(prefix, _OFM_PITCH) + \
    (g)*CAT(prefix, _GROUPS_PITCH)
                )V0G0N";
            } else if (l == WeightsLayout::os_is_yx_isv16_osv16 || l == WeightsLayout::os_is_zyx_isv16_osv16 ||
                       l == WeightsLayout::g_os_is_yx_isv16_osv16 || l == WeightsLayout::g_os_is_zyx_isv16_osv16) {
                args macroNameArgs = {"prefix", "g", "o", "i", "z", "y", "x", "sub_group_size"};
                this->calcFunction = FuncBody(layout_name);
                this->macroName = MacroName(tensor_name, layout_name, macroNameArgs);
                this->macroBody = R"V0G0N( \
    CAT(prefix, _OFFSET) + \
    (g)*CAT(prefix, _GROUPS_PITCH) + \
    ((o) % (sub_group_size)) + \
    (sub_group_size)*( \
        (x)*(sub_group_size)*CAT(prefix, _X_PITCH) + \
        (y)*(sub_group_size)*CAT(prefix, _Y_PITCH) + \
        (z)*(sub_group_size)*CAT(prefix, _Z_PITCH) + \
        ((i) % (sub_group_size)) + \
        ((i) / (sub_group_size))*(sub_group_size)*CAT(prefix, _IFM_PITCH) + \
        ((o) / (sub_group_size))*CAT(prefix, _OFM_PITCH) \
    )
                )V0G0N";
            } else if (l == WeightsLayout::os_iyx_osv16 || l == WeightsLayout::os_iyx_osv32 ||
                       l == WeightsLayout::os_iyx_osv32__ai32 || l == WeightsLayout::g_os_iyx_osv8 || l == WeightsLayout::g_os_iyx_osv16 ||
                       l == WeightsLayout::g_os_iyx_osv32) {
                args macroNameArgs = {"prefix", "g", "o", "i", "y", "x", "sub_group_size"};
                this->calcFunction = FuncBody(layout_name);
                this->macroName = MacroName(tensor_name, layout_name, macroNameArgs);
                this->macroBody = R"V0G0N( \
    CAT(prefix, _OFFSET) + \
    (g * CAT(prefix, _GROUPS_PITCH)) + \
    ((o) % (sub_group_size)) + \
    (sub_group_size)*( \
        (x)*CAT(prefix, _X_PITCH) + \
        (y)*CAT(prefix, _Y_PITCH) +  \
        (i)*CAT(prefix, _IFM_PITCH) + \
        ((o) / (sub_group_size))*CAT(prefix, _OFM_PITCH) \
    )
                )V0G0N";
            } else if (l == WeightsLayout::is_os_yx_isv16_osv16 || l == WeightsLayout::is_os_zyx_isv16_osv16 ||
                       l == WeightsLayout::g_is_os_yx_isv16_osv16 || l == WeightsLayout::g_is_os_zyx_isv16_osv16) {
                args macroNameArgs = {"prefix", "g", "o", "i", "z", "y", "x", "sub_group_size"};
                this->calcFunction = FuncBody(layout_name);
                this->macroName = MacroName(tensor_name, layout_name, macroNameArgs);
                this->macroBody = R"V0G0N( \
    CAT(prefix, _OFFSET) + \
    (g)*CAT(prefix, _GROUPS_PITCH) + \
    ((o) % (sub_group_size)) + \
    (sub_group_size)*( \
        (x)*(sub_group_size)*CAT(prefix, _X_PITCH) + \
        (y)*(sub_group_size)*CAT(prefix, _Y_PITCH) + \
        (z)*(sub_group_size)*CAT(prefix, _Z_PITCH) + \
        ((i) % (sub_group_size)) + \
        ((o) / (sub_group_size))*(sub_group_size)*CAT(prefix, _OFM_PITCH) + \
        ((i) / (sub_group_size))*CAT(prefix, _IFM_PITCH) \
    )
                )V0G0N";
            } else if (l == WeightsLayout::os_is_yx_osv16_isv16 || l == WeightsLayout::os_is_zyx_osv32_isv16 ||
                       l == WeightsLayout::os_is_zyx_osv64_isv16) {
                args macroNameArgs = {"prefix", "o", "i", "z", "y", "x"};
                args funcArgs = {"o", "i", "z", "y", "x", "x_size", "y_size", "z_size", "i_size", "o_size", "osv_size", "isv_size"};
                const auto body = R"V0G0N( \
    const uint isv = i % isv_size; \
    const uint osv = o % osv_size; \
    const uint is = i / isv_size; \
    const uint os = o / osv_size; \
    const uint x_pitch = osv_size * isv_size; \
    const uint y_pitch = x_pitch * x_size; \
    const uint z_pitch = y_pitch * y_size; \
    const uint is_pitch = z_pitch * z_size; \
    const uint os_pitch = is_pitch * ((i_size + isv_size - 1) / isv_size); \
    const uint output_offset = \
        isv + \
        osv * isv_size + \
        x * x_pitch + \
        y * y_pitch + \
        z * z_pitch + \
        is * is_pitch + \
        os * os_pitch; \
    return output_offset; \
                )V0G0N";
                this->macroName = MacroName(tensor_name, layout_name, macroNameArgs);
                this->calcFunction = FuncBody(layout_name, funcArgs, body);
                if (l == WeightsLayout::os_is_yx_osv16_isv16)
                    this->macroBody = FuncCall(layout_name, {"o", "i", "0", "y", "x",
                                               Cat("_SIZE_X"), Cat("_SIZE_Y"), "1", Cat("_IFM_NUM"), Cat("_OFM_NUM"), "16", "16"});
                else if (l == WeightsLayout::os_is_zyx_osv32_isv16)
                    this->macroBody = FuncCall(layout_name, {"o", "i", "z", "y", "x",
                                               Cat("_SIZE_X"), Cat("_SIZE_Y"), Cat("_SIZE_Z"), Cat("_IFM_NUM"), Cat("_OFM_NUM"), "32", "16"});
                else if (l == WeightsLayout::os_is_zyx_osv64_isv16)
                    this->macroBody = FuncCall(layout_name, {"o", "i", "z", "y", "x",
                                               Cat("_SIZE_X"), Cat("_SIZE_Y"), Cat("_SIZE_Z"), Cat("_IFM_NUM"), Cat("_OFM_NUM"), "64", "16"});
            } else if (l == WeightsLayout::g_os_zyx_is_osv16_isv16 || l == WeightsLayout::g_os_zyx_is_osv16_isv32 ||
                       l == WeightsLayout::g_os_zyx_is_osv32_isv16 || l == WeightsLayout::g_os_zyx_is_osv32_isv32) {
                args macroNameArgs = {"prefix", "g", "o", "i", "z", "y", "x"};
                args funcArgs = {"g", "o", "i", "z", "y", "x", "g_size", "o_size", "i_size", "z_size", "y_size", "x_size", "osv", "isv"};
                const auto body = R"V0G0N( \
    uint is_size = (i_size + isv - 1) / isv; \
    uint os_size = (o_size + osv - 1) / osv; \
    uint isv_index = i % isv; \
    uint osv_index = o % osv; \
    uint is_index = i / isv; \
    uint os_index = o / osv; \
    uint isv_pitch = 1; \
    uint osv_pitch = isv_pitch * isv; \
    uint is_pitch = osv_pitch * osv; \
    uint x_pitch = is_pitch * is_size; \
    uint y_pitch = x_pitch * x_size; \
    uint z_pitch = y_pitch * y_size; \
    uint os_pitch = z_pitch * z_size; \
    uint g_pitch = os_pitch * os_size; \
    uint index = 0; \
    index += isv_index * isv_pitch; \
    index += osv_index * osv_pitch; \
    index += is_index * is_pitch; \
    index += x * x_pitch; \
    index += y * y_pitch; \
    index += z * z_pitch; \
    index += os_index * os_pitch; \
    index += g * g_pitch; \
    return index; \
                )V0G0N";
                this->macroName = MacroName(tensor_name, layout_name, macroNameArgs);
                this->calcFunction = FuncBody(layout_name, funcArgs, body);
                std::string osv = "16", isv = "16";
                if (l == WeightsLayout::g_os_zyx_is_osv16_isv16) {
                    osv = "16"; isv = "16";
                } else if (l == WeightsLayout::g_os_zyx_is_osv16_isv32) {
                    osv = "16"; isv = "32";
                } else if (l == WeightsLayout::g_os_zyx_is_osv32_isv16) {
                    osv = "32"; isv = "16";
                } else if (l == WeightsLayout::g_os_zyx_is_osv32_isv32) {
                    osv = "32"; isv = "32";
                }
                this->macroBody = FuncCall(layout_name, {"g", "o", "i", "z", "y", "x", Cat("_GROUPS_NUM"), Cat("_OFM_NUM"), Cat("_IFM_NUM"), Cat("_SIZE_Z"),
                                                         Cat("_SIZE_Y"), Cat("_SIZE_X"), osv, isv});
            } else if (l == WeightsLayout::os_is_yx_osv16_isv4 || l == WeightsLayout::os_is_yx_osv32_isv4) {
                args macroNameArgs = {"prefix", "o", "i", "y", "x"};
                args funcArgs = {"o", "i", "y", "x", "i_size", "o_size", "x_size", "otd"};
                const auto body = R"V0G0N( \
    uint out_depth_tile = o / otd; \
    uint od             = o - out_depth_tile * otd; \
    const uint tile = 4; \
    uint id_tile = i / tile; \
    uint id      = i - id_tile * tile; \
    uint idx = out_depth_tile * (o_size / tile) * otd * tile \
            + id_tile               * i_size * otd * tile \
            + y                     * x_size * otd * tile \
            + x                              * otd * tile \
            + od                                   * tile \
            + id; \
    return idx; \
                )V0G0N";
                this->macroName = MacroName(tensor_name, layout_name, macroNameArgs);
                this->calcFunction = FuncBody(layout_name, funcArgs, body);
                if (l == WeightsLayout::os_is_yx_osv16_isv4)
                    this->macroBody = FuncCall(layout_name, {"o", "i", "y", "x", Cat("_IFM_PITCH"), Cat("_OFM_PITCH"), Cat("_SIZE_X"), "16"});
                else if (l == WeightsLayout::os_is_yx_osv32_isv4)
                    this->macroBody = FuncCall(layout_name, {"o", "i", "y", "x", Cat("_IFM_PITCH"), Cat("_OFM_PITCH"), Cat("_SIZE_X"), "32"});
            } else {
                // throw error?
            }
        }

        static const std::string Cat(std::string name, std::string prefix = "prefix") {
            return "CAT(" + prefix + ", " + name + ")";
        }

        static const std::string FuncCall(std::string name, std::initializer_list<std::string> args) {
            std::string args_str = "";
            size_t counter = 0;
            for (auto& arg : args)
                args_str += (++counter == args.size()) ? (arg) : (arg + ", ");
            return "FUNC_CALL(" + name + ")(" + args_str + ")";
        }

        static const std::string MacroName(std::string tensor_name, std::string layout_name, std::initializer_list<std::string> args) {
            std::string args_str = "";
            size_t counter = 0;
            for (auto& arg : args)
                args_str += (++counter == args.size()) ? (arg) : (arg + ", ");
            return "GET_" + tensor_name + "_" + layout_name + "_INDEX(" + args_str + ")";
        }

        static const std::string FuncBody(std::string name, std::initializer_list<std::string> args = {}, std::string body = "return 0;") {
            std::string args_str = "";
            size_t counter = 0;
            for (auto& arg : args)
                args_str += (++counter == args.size()) ? (arg) : (arg + ", ");
            return "inline uint FUNC(" + name + ")(" + args_str + "){" + body + "}";
        }
    };

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

    auto is_common_nd_layout = [](std::vector<Tensor::WeightsChannelName> common_channels, WeightsLayout l) -> bool {
        for (size_t c = 0; c < static_cast<size_t>(Tensor::WeightsChannelName::COUNT); c++) {
            auto channel = static_cast<Tensor::WeightsChannelName>(c);
            if (WeightsTensor::Channelndex(l, channel) != -1) {
                if (std::find(common_channels.begin(), common_channels.end(), channel) == common_channels.end()) {
                    return false;
                }
            }
        }
        return true;
    };

    std::string index_func_name = _name + "_INDEX_FUNC";
    std::string index_macro_name;
    std::string index_func_val;

    auto layout = _tensor.GetLayout();
    auto layout_str = toString(layout);
    WeightIndexFuncDesc indexFuncDesc{_name, layout};
    std::string called_func_name = "GET_" + _name + "_" + layout_str + "_INDEX";
    if (WeightsTensor::DoesGroupDimExist(layout)) {
        if (WeightsTensor::ChannelsCount(layout) <= 5) {
            std::vector<Tensor::WeightsChannelName> grouped_4d_channels = {
                    Tensor::WeightsChannelName::G,
                    Tensor::WeightsChannelName::OFM,
                    Tensor::WeightsChannelName::IFM,
                    Tensor::WeightsChannelName::Y,
                    Tensor::WeightsChannelName::X,
            };
            bool is_grouped_4d_layout = is_common_nd_layout(grouped_4d_channels, layout);
            if (is_grouped_4d_layout) {
                index_macro_name = _name + "_GET_INDEX(g, o, i, y, x)";
                if (layout == WeightsLayout::goiyx || layout == WeightsLayout::gioyx)
                    index_func_val = called_func_name + "(" + _name + ", g, o, i, 0, y, x)";
                else if (layout == WeightsLayout::g_os_is_yx_isv16_osv16)
                    index_func_val = called_func_name + "(" + _name + ", g, o, i, 0, y, x, 16)";
                else if (layout == WeightsLayout::g_os_iyx_osv8)
                    index_func_val = called_func_name + "(" + _name + ", g, o, i, y, x, 8)";
                else if (layout == WeightsLayout::g_os_iyx_osv16)
                    index_func_val = called_func_name + "(" + _name + ", g, o, i, y, x, 16)";
                else if (layout == WeightsLayout::g_is_os_yx_isv16_osv16)
                    index_func_val = called_func_name + "(" + _name + ", g, o, i, 0, y, x, 16)";
            } else {
                assert(0);
            }
        } else if (WeightsTensor::ChannelsCount(layout) == 6) {
            std::vector<Tensor::WeightsChannelName> grouped_5d_channels = {
                    Tensor::WeightsChannelName::G,
                    Tensor::WeightsChannelName::OFM,
                    Tensor::WeightsChannelName::IFM,
                    Tensor::WeightsChannelName::Z,
                    Tensor::WeightsChannelName::Y,
                    Tensor::WeightsChannelName::X,
            };
            bool is_grouped_5d_layout = is_common_nd_layout(grouped_5d_channels, layout);
            if (is_grouped_5d_layout) {
                index_macro_name = _name + "_GET_INDEX(g, o, i, z, y, x)";
                if (layout == WeightsLayout::goizyx || layout == WeightsLayout::giozyx)
                    index_func_val = called_func_name + "(" + _name + ", g, o, i, z, y, x)";
                else if (layout == WeightsLayout::g_os_is_zyx_isv16_osv16)
                    index_func_val = called_func_name + "(" + _name + ", g, o, i, z, y, x, 16)";
                else if (layout == WeightsLayout::g_is_os_zyx_isv16_osv16)
                    index_func_val = called_func_name + "(" + _name + ", g, o, i, z, y, x, 16)";
            } else {
                assert(0);
            }
        }
    } else {
        if (WeightsTensor::ChannelsCount(layout) <= 4) {
            std::vector<Tensor::WeightsChannelName> base_4d_channels = {
                    Tensor::WeightsChannelName::OFM,
                    Tensor::WeightsChannelName::IFM,
                    Tensor::WeightsChannelName::Y,
                    Tensor::WeightsChannelName::X,
            };
            bool is_common_4d_layout = is_common_nd_layout(base_4d_channels, layout);
            if (is_common_4d_layout) {
                index_macro_name = _name + "_GET_INDEX(o, i, y, x)";
                if (layout == WeightsLayout::oiyx || layout == WeightsLayout::ioyx ||
                    layout == WeightsLayout::oyxi || layout == WeightsLayout::oyix ||
                    layout == WeightsLayout::oxiy || layout == WeightsLayout::iyxo ||
                    layout == WeightsLayout::yxio)
                    index_func_val = called_func_name + "(" + _name + ", 0, o, i, 0, y, x)";
                else if (layout == WeightsLayout::os_is_yx_isv16_osv16)
                    index_func_val = called_func_name + "(" + _name + ", 0, o, i, 0, y, x, 16)";
                else if (layout == WeightsLayout::os_iyx_osv16)
                    index_func_val = called_func_name + "(" + _name + ", 0, o, i, y, x, 16)";
                else if (layout == WeightsLayout::os_iyx_osv32 || layout == WeightsLayout::os_iyx_osv32__ai32)
                    index_func_val = called_func_name + "(" + _name + ", 0, o, i, y, x, 32)";
                else if (layout == WeightsLayout::is_os_yx_isv16_osv16)
                    index_func_val = called_func_name + "(" + _name + ", 0, o, i, 0, y, x, 16)";
                else if (layout == WeightsLayout::os_is_yx_osv16_isv16)
                    index_func_val = called_func_name + "(" + _name + ", o, i, 0, y, x)";
            } else {
                assert(0);
            }
        } else if (WeightsTensor::ChannelsCount(layout) == 5) {
            std::vector<Tensor::WeightsChannelName> base_5d_channels = {
                    Tensor::WeightsChannelName::OFM,
                    Tensor::WeightsChannelName::IFM,
                    Tensor::WeightsChannelName::Z,
                    Tensor::WeightsChannelName::Y,
                    Tensor::WeightsChannelName::X,
            };
            bool is_common_5d_layout = is_common_nd_layout(base_5d_channels, layout);
            if (is_common_5d_layout) {
                index_macro_name = _name + "_GET_INDEX(o, i, z, y, x)";
                if (layout == WeightsLayout::oizyx || layout == WeightsLayout::iozyx)
                    index_func_val = called_func_name + "(" + _name + ", 0, o, i, z, y, x)";
                else if (layout == WeightsLayout::os_is_zyx_isv16_osv16)
                    index_func_val = called_func_name + "(" + _name + ", 0, o, i, z, y, x, 16)";
                else if (layout == WeightsLayout::is_os_zyx_isv16_osv16)
                    index_func_val = called_func_name + "(" + _name + ", 0, o, i, z, y, x, 16)";
                else if (layout == WeightsLayout::os_is_zyx_osv32_isv16 || layout == WeightsLayout::os_is_zyx_osv64_isv16)
                    index_func_val = called_func_name + "(" + _name + ", o, i, z, y, x)";
            } else {
                assert(0);
            }
        }
    }

    if (!indexFuncDesc.macroName.empty()) {
        definitions.push_back({ index_func_name, indexFuncDesc.calcFunction });
        definitions.push_back({ "INIT_" + index_func_name + "_HERE", index_func_name });
        definitions.push_back({ indexFuncDesc.macroName, indexFuncDesc.macroBody });
        definitions.push_back({ index_macro_name, index_func_val });
    }
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
            if (out_dt == Datatype::F32 || out_dt == Datatype::F16)
                jitConstants.AddConstant(MakeJitConstant(macro_def, "(fabs(input))"));
            else
                jitConstants.AddConstant(MakeJitConstant(macro_def, "(abs(input))"));
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
            if (out_dt == Datatype::F32 || out_dt == Datatype::F16)
                jitConstants.AddConstant(MakeJitConstant(macro_def, "(floor(input))"));
            else
                jitConstants.AddConstant(MakeJitConstant(macro_def, "(input)"));
            break;
        case ActivationFunction::CEIL:
            if (out_dt == Datatype::F32 || out_dt == Datatype::F16)
                jitConstants.AddConstant(MakeJitConstant(macro_def, "(ceil(input))"));
            else
                jitConstants.AddConstant(MakeJitConstant(macro_def, "(input)"));
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
        case ActivationFunction::HSIGMOID: {
            std::string type_suffix = out_dt == Datatype::F32 ? "f" : "h";
            const JitTerm three("3." + type_suffix);
            const JitTerm six("6." + type_suffix);
            jitConstants.AddConstant(MakeJitConstant(
                    macro_def,
                    (min_func(max_func(zero, input + three), six) / six).str()));
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
            auto beta = disable_type_conversion ? "m"_jit : to_type("m"_jit);
            jitConstants.AddConstant(MakeJitConstant(
                    macro_def,
                    (input / (one + exp(neg(beta * input)))).str()));
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
            const JitTerm mult{"0.7071067811865475" + type_suffix}; // (1 / sqrt(2))
            jitConstants.AddConstant(MakeJitConstant(
                    macro_def,
                    (half * input * (one + erf((input * mult)))).str()));
            break;
        }
        case ActivationFunction::GELU_TANH: {
            const std::string type_suffix = out_dt == Datatype::F32 ? "f" : "h";
            const JitTerm half{"0.5" + type_suffix};
            const JitTerm mult{"0.044715" + type_suffix};
            const JitTerm sqrt_2_over_pi{"0.79788458347320556640625" + type_suffix};
            jitConstants.AddConstant(MakeJitConstant(
                    macro_def,
                    (half * input * (one + tanh(sqrt_2_over_pi * input * (one + mult * input * input)))).str()));
            break;
        }
        case ActivationFunction::NOT:
            jitConstants.AddConstant(MakeJitConstant(
                macro_def,
                ternary(input.eq(zero), one, zero)
                    .str()));  // the workaround for OpenCL's vector type result (!input)
            break;
        case ActivationFunction::ROUND_HALF_TO_EVEN:
            jitConstants.AddConstant(MakeJitConstant(macro_def, "rint(input)"));
            break;
        case ActivationFunction::ROUND_HALF_AWAY_FROM_ZERO:
            jitConstants.AddConstant(MakeJitConstant(macro_def, "(round(input))"));
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
    std::string type = "undefined";
    std::string max_val = "undefined";
    std::string min_val = "undefined";
    std::string val_one = "undefined";
    std::string val_zero = "undefined";
    std::string to_type = "undefined";
    std::string to_type_sat = "undefined";
    std::string as_type = "undefined";
    std::string max_func = "undefined";
    std::string min_func = "undefined";
    std::string abs_func = "undefined";
    std::string type_size = "undefined";
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
        case Datatype::INT4:
            type = "char";
            type_size = "0.5f";
            is_fp = false;
            break;
        case Datatype::UINT4:
            type = "uchar";
            type_size = "0.5f";
            is_fp = false;
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
        case WeightsType::INT4:
            return MakeTypeJitConstants(Datatype::INT4, macroName);
        case WeightsType::UINT4:
            return MakeTypeJitConstants(Datatype::UINT4, macroName);
        case WeightsType::INT32:
            return MakeTypeJitConstants(Datatype::INT32, macroName);
    }
    assert(false || "Unreachable!");
    // FIXME: Is there some builtin_unreachable available?
    return MakeTypeJitConstants(Datatype::UNSUPPORTED, macroName);
}

JitConstants make_int4_packed_type_jit_constant(const std::string& macro_name, WeightsType wt, size_t pack_size) {
    OPENVINO_ASSERT(pack_size % 2 == 0 && pack_size != 0 && pack_size <= 16);
    std::string type_string = "";
    switch (wt) {
        case WeightsType::UINT4: type_string = "uint4x"; break;
        case WeightsType::INT4: type_string = "int4x"; break;
        default: OPENVINO_THROW("[GPU] Unsupported compressed type");
    }
    return { MakeJitConstant(macro_name, type_string + std::to_string(pack_size) + "_t") };
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
                                        bool disable_type_conversion,
                                        bool convert_input_to_output_dt) {
    JitConstants res = {};
    if (params.empty()) {
        return MakeActivationJitConstants({ActivationFunction::NONE, 0.f, 0.f}, out_dt,
                                          suffix, use_type_parameter, disable_type_conversion);
    }
    std::string res_activation = "";
    std::string activation_params = "";
    for (size_t i = 0; i < params.size(); i++) {
        std::string activation_suffix = suffix + "_" + toCodeString(i);
        std::string nl_m = toCodeString(params[i].m);
        std::string nl_n = toCodeString(params[i].n);
        if (params[i].function == ActivationFunction::CLAMP) {
            if (out_dt == Datatype::INT8) {
                nl_m = toCodeString(std::max<float>(params[i].m, std::numeric_limits<signed char>::min()));
                nl_n = toCodeString(std::min<float>(params[i].n, std::numeric_limits<signed char>::max()));
            } else if (out_dt == Datatype::UINT8) {
                nl_m = toCodeString(std::max(params[i].m, 0.0f));
                nl_n = toCodeString(std::min<float>(params[i].n, std::numeric_limits<unsigned char>::max()));
            }
        }
        auto jitConstants = JitConstants{MakeJitConstant("NL_M" + activation_suffix, nl_m),
                                         MakeJitConstant("NL_N" + activation_suffix, nl_n)};
        jitConstants.Merge(MakeActivationJitConstants(
                params[i].function, out_dt, activation_suffix, use_type_parameter, disable_type_conversion));
        res.Merge(jitConstants);

        if (i == 0) {
            activation_params = use_type_parameter ? "(jit_type, input, params)" : "(input, params)";
            if (convert_input_to_output_dt) {
                // Convert the input to the output data type to fix that cl kernel build failed for an ambiguous issue of the fmax/fmin functions
                // occurring by the different data types between input and output.
                res_activation = "ACTIVATION_FUNC" + activation_suffix
                                + "(" + (use_type_parameter? "jit_type, ":"") + "convert_" + toCLType(out_dt) + "(input), params)";
            } else {
                res_activation = "ACTIVATION_FUNC" + activation_suffix + activation_params;
            }
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

JitConstants MakeConstantLoopUnrollJitConstants(uint32_t loopCount) {
    JitConstants jit{
        MakeJitConstant("CONST_LOOP_CALL(macro, idx)", "macro(idx)"),
        MakeJitConstant("CONST_LOOP_1(macro)", "CONST_LOOP_CALL(macro, 0)")
    };

    for (uint32_t i = 2; i <= loopCount; ++i) {
        jit.AddConstant(MakeJitConstant("CONST_LOOP_" + toCodeString(i) + "(macro)",
                                        "CONST_LOOP_" + toCodeString(i - 1) + "(macro); CONST_LOOP_CALL(macro," + toCodeString(i - 1) + ")"));
    }

    jit.AddConstant(MakeJitConstant("CONST_LOOP(count, macro)", "CAT(CONST_LOOP_, count)(macro)"));

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
                case Tensor::DataChannelName::W:       can_preload &= idx.w == "0"; break;
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
    // Use shape_ids from output tensor as won't support fused ops which changes out shape for now
    jit.AddConstant(MakeJitConstant(GetOutputTensorName(), desc.output_tensor));
    return jit;
}

JitConstants FusedOpsCodeGenerator::MakeInputDeclsJitConstants(const FusedOpsConfiguration& /*conf*/) const {
    JitConstants jit = {};

    std::string input_decls = "";
    std::string input_args = "";
    for (size_t op_input_id = 0; op_input_id < desc.tensors.size(); op_input_id++) {
        std::string ptr_name = GetInputPtrName(op_input_id);
        input_decls += "\\\n\tconst __global " + toCLType(desc.tensors[op_input_id].GetDType()) +
                       "* " + ptr_name + (op_input_id == desc.tensors.size() - 1 ? "" : ",");
        input_args += "\\\n\t" + ptr_name + (op_input_id == desc.tensors.size() - 1 ? "" : ",");
    }

    jit.AddConstant(MakeJitConstant("FUSED_OP" + toCodeString(desc.op_id) + "_DECLS", input_decls));
    jit.AddConstant(MakeJitConstant("FUSED_OP" + toCodeString(desc.op_id) + "_ARGS", input_args));
    return jit;
}

JitConstants FusedOpsCodeGenerator::MakeLoadJitConstants(const FusedOpsConfiguration& conf, const DataTensor prim_output) const {
    JitConstants jit = {};

    auto vec_size = conf.vec_size;
    auto idx = conf.bfzyx_idx_order;
    auto fused_op_config = conf;

    std::string load_decls = "";
    static thread_local int i = 0;
    // TODO: check if there is a use case for index reuse or it can be removed
    bool reuse_index = false;
    bool safe_load = conf.boundary_check == FusedOpsConfiguration::BoundaryCheck::ENABLED;
    std::string reused_idx = "reused_idx_" + toCodeString(i++);
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

    jit.AddConstant(MakeJitConstant("FUSED_OP"+toCodeString(desc.op_id)+"_LOAD" + conf.suffix, load_decls));

    return jit;
}

JitConstants FusedOpsCodeGenerator::MakeOpJitConstants(const FusedOpsConfiguration& conf,
                                                       const std::string in_var, const Datatype in_type,
                                                       std::string& out_var) const {
    JitConstants jit = {};

    std::string op_decls = "";
    auto vec_size = conf.vec_size;
    std::string shuffle_var = conf.shuffle_var_name;
    bool is_shuffled = false;
    bool floor_integer_div = false;

    auto& dep_data = desc.dep_data;
    int first_fused_ops_idx = -1;
    size_t dep_idx = 0;
    for (auto dep : dep_data) {
        if (dep.dep_type == kernel_selector::DepType::INTERNAL) {
            first_fused_ops_idx = static_cast<int>(dep_idx);
            break;
        }
        dep_idx++;
    }

    std::vector<std::string> input_vars;

    out_var = GetOutputVarName(in_var, desc.op_id);
    const auto& out_type = desc.output_tensor.GetDType();

    if (conf.load_type == FusedOpsConfiguration::LoadType::FEATURE_SHUFFLE &&
        desc.GetType() == KernelType::QUANTIZE) {
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

    if (desc.GetType() == KernelType::ELTWISE) {
        auto p = desc.GetOpParams<eltwise_fuse_params>();
        OPENVINO_ASSERT(p != nullptr, "[GPU] Eltwise fuse params can't be nullptr");

        if (p->mode == kernel_selector::EltwiseMode::DIV) {
            if (p->m_pythondiv)
                floor_integer_div = true;
        }
    }

    auto get_acc_t = [&]() -> Datatype {
        std::vector<Datatype> input_types = {desc.output_tensor.GetDType()};
        for (auto& dep : dep_data) {
            input_types.push_back(dep.data_type);
        }

        std::vector<Datatype> types_prioritized = { };
        if (floor_integer_div) {
            if (std::all_of(input_types.begin(), input_types.end(),
                            [=](const Datatype& t) -> bool { return (t != Datatype::F32 && t != Datatype::F16); })) {
                types_prioritized = { Datatype::INT64, Datatype::INT32, Datatype::UINT32, Datatype::INT16, Datatype::UINT16, Datatype::INT8, Datatype::UINT8 };
                for (auto& type : types_prioritized) {
                    if (std::any_of(input_types.begin(), input_types.end(),
                                [=](const Datatype& t) -> bool { return (t == type); })) {
                        return type;
                    }
                }
            }
        }

        floor_integer_div = false;
        types_prioritized.clear();
        types_prioritized = { Datatype::F32, Datatype::F16 };
        for (auto& type : types_prioritized) {
            if (std::any_of(input_types.begin(), input_types.end(), [=](const Datatype& t) -> bool { return t == type; })) {
                return type;
            }
        }

        return Datatype::F32;
    };

    auto get_input = [&](size_t index) -> std::string {
        const auto dep = dep_data[index];
        auto input_name = (dep.dep_type == kernel_selector::DepType::ORIGINAL)? in_var :
                            (dep.dep_type == kernel_selector::DepType::INTERNAL)? GetOutputVarName(in_var, dep.op_id)
                                : GetInputVarName(dep.op_id, is_shuffled, shuffle_var);
        auto input_type = (dep.dep_type == kernel_selector::DepType::ORIGINAL)? in_type : dep.data_type;
        auto acc_t = get_acc_t();

        if (input_type != acc_t)
            return ConvertToType(input_name, acc_t, vec_size);
        else
            return input_name;
    };

    for (size_t i = 0; i < dep_data.size(); i++) {
        input_vars.push_back(get_input(i));
    }

    switch (desc.GetType()) {
        case KernelType::ELTWISE: {
            auto p = desc.GetOpParams<eltwise_fuse_params>();
            std::string op = "";
            switch (p->mode) {
            case kernel_selector::EltwiseMode::ADD:
                op = "+";
                break;
            case kernel_selector::EltwiseMode::MUL:
                op = "*";
                break;
            case kernel_selector::EltwiseMode::SUB:
                op = "-";
                break;
            case kernel_selector::EltwiseMode::DIV:
                op = "/";
                break;
            default:
                throw std::runtime_error("[clDNN] Eltwise mode is not supported in fused ops codegen");
            }

            auto tmp_var = out_var + "_tmp";
            auto acc_t_type = GetType(get_acc_t(), vec_size);
            op_decls += "\\\n\t" + acc_t_type + " " + tmp_var + " = " + input_vars[0] + op + input_vars[1] + ";";
            if (floor_integer_div) {
                auto tmp_var_rem = tmp_var + "_rem";
                op_decls += "\\\n\t" + acc_t_type + " " + tmp_var_rem + " = " + input_vars[0] + " % " + input_vars[1] + ";";
                op_decls += "\\\n\t" + tmp_var + " -= " + "((" + tmp_var_rem + " != 0 && (" + input_vars[0] + " < 0) != (" + input_vars[1] + " < 0)) ? 1 : 0);";
            }
            op_decls += "\\\n\t" + GetOutputType(vec_size) + " " + out_var + " = " + ConvertToOutputType(tmp_var, vec_size) + ";";
            break;
        }
        case KernelType::QUANTIZE: {
            auto p = desc.GetOpParams<quantize_fuse_params>();
            if (!p)
                throw std::runtime_error("[clDNN] Quantize fuse params can't be nullptr");

            std::string in_converted = (first_fused_ops_idx < 0) ? in_var : GetOutputVarName(in_var, dep_data[first_fused_ops_idx].op_id);
            Datatype input_type = (first_fused_ops_idx < 0) ? in_type : dep_data[first_fused_ops_idx].data_type;
            Datatype tmp_type = Datatype::F32;
            std::string tmp_type_str = GetType(tmp_type, vec_size);
            std::string tmp_var = out_var + "_tmp";

            if (input_type != tmp_type) {
                in_converted = ConvertToType(in_converted, tmp_type, vec_size);
            }

            auto post_scale = p->per_tensor_output_scale ? Broadcast(toCodeString(p->out_scale), tmp_type, vec_size)
                                                         : ConvertToType(GetInputVarName(p->out_scale_idx, is_shuffled, shuffle_var), tmp_type, vec_size);
            auto post_shift = p->per_tensor_output_shift ? Broadcast(toCodeString(p->out_shift), tmp_type, vec_size)
                                                         : ConvertToType(GetInputVarName(p->out_shift_idx, is_shuffled, shuffle_var), tmp_type, vec_size);
            auto pre_scale = p->per_tensor_input_scale ? Broadcast(toCodeString(p->in_scale), tmp_type, vec_size)
                                                       : ConvertToType(GetInputVarName(p->in_scale_idx, is_shuffled, shuffle_var), tmp_type, vec_size);
            auto pre_shift = p->per_tensor_input_shift ? Broadcast(toCodeString(p->in_shift), tmp_type, vec_size)
                                                       : ConvertToType(GetInputVarName(p->in_shift_idx, is_shuffled, shuffle_var), tmp_type, vec_size);

            if (p->per_tensor_output_range && p->out_lo < p->out_hi) {
                // Input scale
                op_decls += "\\\n\t" + tmp_type_str + " " + tmp_var + " = " + in_converted + " * " + pre_scale + ";";

                // Input shift
                if (p->has_pre_shift)
                    op_decls += "\\\n\t" + tmp_var + " = " + tmp_var + " + " + pre_shift + ";";

                // Round operation isn't needed if output type is int8/uint8 and scale coefficient in all output channels is equal to 1.0
                bool output_type_is_int8 = desc.output_tensor.GetDType() == Datatype::UINT8 || desc.output_tensor.GetDType() == Datatype::INT8;
                if (((p->has_post_scale || p->has_post_shift) && output_type_is_int8) || !output_type_is_int8)
                    op_decls += "\\\n\t" + tmp_var + " = round(" + tmp_var + ");";

                // Output scale
                if (p->has_post_scale)
                    op_decls += "\\\n\t" + tmp_var + " = (" + tmp_var + " * " + post_scale + ");";

                // Output shift
                if (p->has_post_shift)
                    op_decls += "\\\n\t" + tmp_var + " = (" + tmp_var + " + " + post_shift + ");";

                // Output range
                auto out_lo = Broadcast(std::to_string(p->out_lo), tmp_type, vec_size);
                auto out_hi = Broadcast(std::to_string(p->out_hi), tmp_type, vec_size);

                // Output clamp
                if (p->has_clamp) {
                    if (p->has_min_clamp && p->has_max_clamp)
                        op_decls += "\\\n\t" + tmp_var + " = clamp(" + tmp_var + ", " + out_lo + ", " + out_hi + ");";
                    else if (p->has_min_clamp)
                        op_decls += "\\\n\t" + tmp_var + " = max(" + tmp_var + ", " + out_lo + ");";
                    else
                        op_decls += "\\\n\t" + tmp_var + " = min(" + tmp_var + ", " + out_hi + ");";
                }

                // Output conversion with rounding and saturation
                op_decls += "\\\n\t" + GetOutputType(vec_size) + " " + out_var + " = " + ConvertToOutputTypeSat(tmp_var, vec_size) + ";";
                break;
            } else {
                // Input range
                auto in_lo = p->per_tensor_input_range ? Broadcast(std::to_string(p->in_lo), tmp_type, vec_size)
                                                       : ConvertToType(GetInputVarName(p->in_range_lo_idx, is_shuffled, shuffle_var), tmp_type, vec_size);
                auto in_hi = p->per_tensor_input_range ? Broadcast(std::to_string(p->in_hi), tmp_type, vec_size)
                                                       : ConvertToType(GetInputVarName(p->in_range_hi_idx, is_shuffled, shuffle_var), tmp_type, vec_size);

                // Input clamp
                if (p->has_clamp) {
                    if (p->has_min_clamp && p->has_max_clamp)
                        op_decls += "\\\n\t" + tmp_type_str + " " + tmp_var + " = clamp(" + in_converted + ", " + in_lo + ", " + in_hi + ");";
                    else if (p->has_min_clamp)
                        op_decls += "\\\n\t" + tmp_type_str + " " + tmp_var + " = max(" + in_converted + ", " + in_lo + ");";
                    else
                        op_decls += "\\\n\t" + tmp_type_str + " " + tmp_var + " = min(" + in_converted + ", " + in_hi + ");";
                } else {
                    op_decls += "\\\n\t" + tmp_type_str + " " + tmp_var + " = " + in_converted + ";";
                }

                // Input scale
                op_decls += "\\\n\t" + tmp_var + " = " + tmp_var + " * " + pre_scale + ";";

                // Input shift
                if (p->has_pre_shift)
                    op_decls += "\\\n\t" + tmp_var + " = " + tmp_var + " + " + pre_shift + ";";

                // Round operation isn't needed if output type is int8/uint8 and scale coefficient in all output channels is equal to 1.0
                bool output_type_is_int8 = desc.output_tensor.GetDType() == Datatype::UINT8 || desc.output_tensor.GetDType() == Datatype::INT8;
                if (((p->has_post_scale || p->has_post_shift) && output_type_is_int8) || !output_type_is_int8)
                    op_decls += "\\\n\t" + tmp_var + " = round(" + tmp_var + ");";

                // Output scale
                if (p->has_post_scale)
                    op_decls += "\\\n\t" + tmp_var + " = (" + tmp_var + " * " + post_scale + ");";

                // Output shift
                if (p->has_post_shift)
                    op_decls += "\\\n\t" + tmp_var + " = (" + tmp_var + " + " + post_shift + ");";

                // Output conversion with rounding and saturation
                op_decls += "\\\n\t" + GetOutputType(vec_size) + " " + out_var + " = " + ConvertToOutputTypeSat(tmp_var, vec_size) + ";";
                break;
            }
        }
        case KernelType::ACTIVATION: {
            auto p = desc.GetOpParams<activation_fuse_params>();
            base_activation_params activation_p = p->param;
            std::string new_in_var = (first_fused_ops_idx < 0) ? in_var : GetOutputVarName(in_var, dep_data[first_fused_ops_idx].op_id);
            op_decls += "\\\n\t" + GetOutputType(vec_size) + " " + out_var + " = " + ConvertToOutputType(new_in_var, vec_size) + ";";
            if (activation_p.function != ActivationFunction::NONE) {
                auto suffix = "_FUSED_OP"+toCodeString(desc.op_id) + conf.suffix;
                std::string nl_m = toCodeString(activation_p.m);
                std::string nl_n = toCodeString(activation_p.n);

                if (activation_p.function == ActivationFunction::CLAMP) {
                    if (out_type == Datatype::INT8) {
                        nl_m = toCodeString(std::max<float>(activation_p.m, std::numeric_limits<signed char>::min()));
                        nl_n = toCodeString(std::min<float>(activation_p.n, std::numeric_limits<signed char>::max()));
                    } else if (out_type == Datatype::UINT8) {
                        nl_m = toCodeString(std::max(activation_p.m, 0.0f));
                        nl_n = toCodeString(std::min<float>(activation_p.n, std::numeric_limits<unsigned char>::max()));
                    }
                }

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

    jit.AddConstant(MakeJitConstant("FUSED_OP"+toCodeString(desc.op_id)+"_ACTION" + conf.suffix, op_decls));

    return jit;
}

std::string FusedOpsCodeGenerator::GetInputTensorName(size_t input_id) const {
    return "FUSED_OP_" + toCodeString(desc.op_id) + "_INPUT" + toCodeString(input_id);
}

std::string FusedOpsCodeGenerator::GetOutputTensorName() const {
    return "FUSED_OP_" + toCodeString(desc.op_id) + "_OUTPUT";
}

std::string FusedOpsCodeGenerator::GetInputTypeName(size_t input_id, size_t vec_size) const {
    if (vec_size == 0 || vec_size > 8)
        throw std::invalid_argument("Invalid vector size in jit definitions: " + std::to_string(vec_size));
    std::string scalar_type = GetInputTensorName(input_id) + "_TYPE";
    if (vec_size > 1)
        return "MAKE_VECTOR_TYPE(" + scalar_type + "," + toCodeString(vec_size) + ")";
    else
        return scalar_type;
}

std::string FusedOpsCodeGenerator::GetIdx(size_t input_id, idx_desc idx, bool should_be_safe) const {
    std::string idx_order = "";

    if (DataTensor::ChannelsCount(desc.tensors[input_id].GetLayout()) <= 4) {
        idx_order = idx.b + "," + idx.f + "," + idx.y + "," + idx.x;
    } else if (DataTensor::ChannelsCount(desc.tensors[input_id].GetLayout()) == 5) {
        idx_order = idx.b + "," + idx.f + "," + idx.z + "," + idx.y + "," + idx.x;
    } else if (DataTensor::ChannelsCount(desc.tensors[input_id].GetLayout()) == 6) {
        idx_order = idx.b + "," + idx.f + "," + idx.w + "," + idx.z + "," + idx.y + "," + idx.x;
    } else if (DataTensor::ChannelsCount(desc.tensors[input_id].GetLayout()) == 7) {
        idx_order = idx.b + "," + idx.f + "," + idx.u + "," + idx.w + "," + idx.z + "," + idx.y + "," + idx.x;
    } else if (DataTensor::ChannelsCount(desc.tensors[input_id].GetLayout()) == 8) {
        idx_order = idx.b + "," + idx.f + "," + idx.v + "," + idx.u + "," + idx.w + "," + idx.z + "," + idx.y + "," + idx.x;
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

    bool valid_broadcast_case = input_tensor.LogicalSize() == prim_output.Feature().v ||
                                input_tensor.LogicalSize() == 1;

    // Eltwise fused op can't have full tensor argument when requested vec_size > 1, since it might require
    // splitting load into several parts and some kind of index recalculation which is not supported
    DataLayout orig_output_layout = conf.IsPostReorderFused() ? conf.orig_output_layout : prim_output.GetLayout();

    if (desc.GetType() == KernelType::ELTWISE && !valid_broadcast_case &&
        input_tensor.GetLayout() != orig_output_layout && conf.vec_size > 1) {
        throw std::runtime_error("[clDNN] Mixed layouts of input tensors are not supported in fused eltwise:"
                                 "\nfused_input: " + toString_v2(input_tensor) +
                                 "\noutput: " + toString_v2(prim_output));
    }

    if (conf.vec_axis != Tensor::DataChannelName::COUNT &&
        DataTensor::Extract(input_tensor.GetLayout(), conf.vec_axis, input_tensor.GetDims()).v != 1) {
        vec_size = conf.vec_size;
    }

    auto idx = conf.bfzyx_idx_order;
    if (vec_size == 0 || vec_size > 8)
        throw std::invalid_argument("Invalid vector size in jit definitions: " + toCodeString(vec_size));

    bool safe_load = conf.boundary_check == FusedOpsConfiguration::BoundaryCheck::ENABLED;

    // Fsv16 Eltwise whcih requires f axis broadcast such as input[1,1,z,1,1], output[b,f,z,y,x] need to use LT unligned read.
    // In this case, intel_sub_group_block_read() introduces increasing index in feature block.
    bool f_axis_broadcast = (input_tensor.Feature().v != prim_output.Feature().v) && (input_tensor.Feature().v == 1);
    // Change JitLoad to ignore LT_ALIGNED_READ LoadType if this input tensor has a planar format(SimpleLayout)
    if (desc.GetType() == KernelType::ELTWISE &&
        conf.load_type == FusedOpsConfiguration::LoadType::LT_ALIGNED_READ &&
        ((input_tensor.SimpleLayout() && input_tensor.GetLayout() != orig_output_layout) || f_axis_broadcast) &&
        (input_tensor.SameDimsSizes(prim_output) || f_axis_broadcast) &&
        input_tensor.LogicalSize() != 1) {
        std::string sub_group_local_id_str = "get_sub_group_local_id";
        size_t found_sub = conf.bfzyx_idx_order[1].rfind(sub_group_local_id_str);
        OPENVINO_ASSERT(found_sub == std::string::npos, "[GPU] LT_ALIGNED_READ LoadType is used with get_sub_group_local_id.");

        auto new_idx_order = conf.bfzyx_idx_order;
        new_idx_order[1] = "(" + conf.bfzyx_idx_order[1] + " + " + sub_group_local_id_str + "()" + ")";
        if (vec_size > 1) {
            auto vec_axis_idx = conf.GetDimIndexFromOrder(conf.vec_axis);
            OPENVINO_ASSERT(vec_axis_idx != -1, "[GPU] Incorrect vec_axis value ", static_cast<int>(conf.vec_axis),
                                                " for bfzyx_idx_order order");
            new_idx_order[vec_axis_idx] = "((" + conf.bfzyx_idx_order[vec_axis_idx] + ") + loop_var)";
        }
        std::string new_index_func_call = GetIdx(input_id, idx_desc{new_idx_order, desc.tensors[input_id]}, safe_load);

        if (vec_size > 1) {
            std::string load_str = "0;"; // Assign zero to initial variable (GetInputVarName(input_id)) and modify it in the loop below
            load_str += "for (uint loop_var = 0; loop_var < " + std::to_string(vec_size)  + "; loop_var++) { ";
            load_str += GetInputVarName(input_id) + "[loop_var] = " + GetInputPtrName(input_id) + "[" + new_index_func_call + "]; }";
            return load_str;
        } else {
            return GetInputPtrName(input_id) + "[" + new_index_func_call + "]";
        }
    }

    std::string index_func_call_vec = reuse_index ? reused_idx : GetIdx(input_id, idx_desc{idx, desc.tensors[input_id]}, safe_load);
    std::string index_func_call = reuse_index ? reused_idx : GetIdx(input_id, idx_desc{idx, desc.tensors[input_id]}, safe_load);
    if (conf.index_type == FusedOpsConfiguration::IndexType::LINEAR_OFFSET) {
        std::string offset = conf.bfzyx_idx_order[0];
        if (safe_load)
            offset = "(" + offset + " % " + toCodeString(input_tensor.LogicalSize()) + ")";
        if (vec_size > 1)
            return "((const __global " + toCLType(input_dt) + toCodeString(vec_size) + "*)(" +
                   GetInputPtrName(input_id) + " + " + offset + "))[0]";
        else
            return GetInputPtrName(input_id) + "[" + offset + "]";
    } else {
        // TODO: Need to add smarter vectors handling:
        // 1. Boundary checks for safe load
        // 2. If in given configuration data can't be loaded by a simple UNIT_BLOCK_READx call or load from casted ptr,
        //    we can gather the data to vector
        if (conf.load_type == FusedOpsConfiguration::LoadType::LT_ALIGNED_READ) {
            std::string vs = vec_size > 1 ? toCodeString(vec_size)  : "";
            std::string block_read;

            if (input_dt == Datatype::F32 || input_dt == Datatype::INT32 || input_dt == Datatype::UINT32) {
                block_read = CastToType(" _sub_group_block_read" + vs + "("
                                        + "(const __global uint*)(" + GetInputPtrName(input_id) + " + " + index_func_call_vec + "))",
                                        input_dt, vec_size);
            } else if (input_dt == Datatype::F16) {
                block_read = CastToType(" _sub_group_block_read_us" + vs + "("
                                        + "(const __global ushort*)(" + GetInputPtrName(input_id) + " + " + index_func_call_vec + "))",
                                        input_dt, vec_size);
            } else if (input_dt == Datatype::UINT8 || input_dt == Datatype::INT8) {
                block_read = CastToType(" _sub_group_block_read_uc" + vs + "("
                                        + "(const __global uchar*)(" + GetInputPtrName(input_id) + " + " + index_func_call_vec + "))",
                                        input_dt, vec_size);
            } else {
                throw std::runtime_error("Aligned load is not supported yet for " + toCLType(input_dt) + " data type");
            }

            if (vec_size > 1) {
                return block_read;
            } else if (input_tensor.LogicalSize() > 1) {
                // Currently we assume that in such scenario we can safely load sub_group_size elements from the pointer
                return Broadcast(block_read, input_dt, vec_size);
            } else {
                // Input has only one element, so broadcast it for the whole vector size
                return Broadcast(GetInputPtrName(input_id) + "[" + index_func_call + "]", input_dt, vec_size);
            }
        } else {
            if (vec_size > 1) {
                return "((const __global " + toCLType(input_dt) + toCodeString(vec_size) + "*)(" +
                       GetInputPtrName(input_id) + " + " + index_func_call_vec + "))[0]";
            } else {
                return GetInputPtrName(input_id) + "[" + index_func_call + "]";
            }
        }
    }
}

std::string FusedOpsCodeGenerator::GetInputPtrName(size_t input_id) const {
    return GetTypeStr() + toCodeString(desc.op_id) + "_input" + toCodeString(input_id);
}

std::string FusedOpsCodeGenerator::GetInputVarName(size_t input_id, bool is_shuffled, std::string shuffle_var) const {
    if (is_shuffled)
        return "_sub_group_shuffle(" + GetTypeStr() + toCodeString(desc.op_id) + "_data" +
               toCodeString(input_id) + ", " + shuffle_var + ")";
    return GetTypeStr() + toCodeString(desc.op_id) + "_data" + toCodeString(input_id);
}

std::string FusedOpsCodeGenerator::GetOutputVarName(std::string input_var, size_t op_id) const {
    std::replace(input_var.begin(), input_var.end(), '[', '_');
    std::replace(input_var.begin(), input_var.end(), ']', '_');
    std::replace(input_var.begin(), input_var.end(), ' ', '_');
    std::replace(input_var.begin(), input_var.end(), '.', '_');
    return input_var + "_out_" + toCodeString(op_id);
}

std::string FusedOpsCodeGenerator::GetType(Datatype dt, size_t vec_size) const {
    if (vec_size > 1)
        return toCLType(dt) + toCodeString(vec_size);
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
        return "convert_" + GetOutputType(vec_size) + "_sat_rte(" + var + ")";
}

std::vector<size_t> FusedOpsCodeGenerator::GetRequiredInputs() const {
    switch (desc.GetType()) {
        case KernelType::QUANTIZE: {
            auto p = std::dynamic_pointer_cast<quantize_fuse_params>(desc.op_params);
            if (p) {
                std::vector<size_t> res = {};
                bool out_range_usage = p->per_tensor_output_range && p->out_lo < p->out_hi;
                if (!out_range_usage && p->has_clamp) {
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
