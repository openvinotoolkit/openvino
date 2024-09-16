// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jitter.hpp"

#include <string>
#include "openvino/core/except.hpp"

namespace ov {
namespace intel_gpu {
namespace ocl {

namespace {

std::string format_string(format fmt) {
    auto str = fmt.to_string();
    for (auto& s : str) {
        s = std::toupper(s);
    }

    return str;
}

std::vector<DataChannelName> get_data_channels_order(size_t rank) {
    using channel = DataChannelName;
    switch (rank) {
        case 1: return {channel::BATCH};
        case 2: return {channel::BATCH, channel::FEATURE};
        case 3: return {channel::BATCH, channel::FEATURE, channel::Y};
        case 4: return {channel::BATCH, channel::FEATURE, channel::Y, channel::X};
        case 5: return {channel::BATCH, channel::FEATURE, channel::Z, channel::Y, channel::X};
        case 6: return {channel::BATCH, channel::FEATURE, channel::W, channel::Z, channel::Y, channel::X};
        case 7: return {channel::BATCH, channel::FEATURE, channel::U, channel::W, channel::Z, channel::Y, channel::X};
        case 8: return {channel::BATCH, channel::FEATURE, channel::V, channel::U, channel::W, channel::Z, channel::Y, channel::X};
    }

    return {};
}

std::vector<WeightsChannelName> get_weights_channels_order(size_t rank, bool is_grouped) {
    using channel = WeightsChannelName;
    if (is_grouped) {
        switch (rank) {
            case 4: return {channel::G, channel::OFM, channel::IFM, channel::Y, channel::X};
            case 5: return {channel::G, channel::OFM, channel::IFM, channel::Z, channel::Y, channel::X};
            case 6: return {channel::G, channel::OFM, channel::IFM, channel::Z, channel::Y, channel::X};
            default: break;
        }
    } else {
        switch (rank) {
            case 3: return {channel::OFM, channel::IFM, channel::X};
            case 4: return {channel::OFM, channel::IFM, channel::Y, channel::X};
            case 5: return {channel::OFM, channel::IFM, channel::Z, channel::Y, channel::X};
            default: break;
        }
    }

    return {};
}

int get_channel_index_impl(DataChannelName channel_name, format fmt, size_t rank) {
    auto order = get_data_channels_order(rank);
    auto it = std::find(order.begin(), order.end(), channel_name);
    if (it == order.end())
        return -1;

    return std::distance(order.begin(), it);
}

int get_channel_index_impl(WeightsChannelName channel_name, format fmt, size_t rank) {
    auto order = get_weights_channels_order(rank, format::is_grouped(fmt));

    auto it = std::find(order.begin(), order.end(), channel_name);
    if (it == order.end())
        return -1;

    return std::distance(order.begin(), it);
}


template <typename ChannelName>
int get_channel_index(ChannelName channel_name, format fmt, size_t rank) {
    return get_channel_index_impl(channel_name, fmt, rank);
}

class LayoutJitterBase {
public:
    virtual ~LayoutJitterBase() = default;
};

class DataLayoutJitter : public LayoutJitterBase {
public:
    using channel_type = DataChannelName;
    DataLayoutJitter(const layout& l, size_t shape_info_idx) {
        OPENVINO_ASSERT(!format::is_weights_format(l.format));
        make_definitions(l, shape_info_idx);
    }

    // definition of tensor element accessors in the following order:
    // data tensor: b, f, u, v, w, z, y, x
    std::vector<std::string> dims;
    std::vector<std::string> strides;
    std::vector<std::string> pad_lower;
    std::vector<std::string> pad_upper;

    std::map<channel_type, size_t> channels_map;

    std::string dim(DataChannelName channel) const {
        return dims[channels_map.at(channel)];
    }

    std::string pad_l(DataChannelName channel) const {
        return pad_lower[channels_map.at(channel)];
    }

    std::string pad_u(DataChannelName channel) const {
        return pad_upper[channels_map.at(channel)];
    }

    std::string stride(DataChannelName channel) const {
        return strides[channels_map.at(channel)];
    }

private:
    void make_definitions(const layout& l, size_t shape_info_tensor_idx) {
        const auto fmt = l.format;
        const auto& pshape = l.get_partial_shape();
        const auto& pad = l.data_padding;
        bool is_static = l.is_static();

        ov::PartialShape vals_ordered;
        const auto& axis_order = fmt.dims_order();
        for (size_t i = 0; i < axis_order.size(); i++) {
            if (axis_order[i] >= pshape.size())
                vals_ordered.push_back(ov::Dimension(1));
            else
                vals_ordered.push_back(pshape[axis_order[i]]);
        };

        ov::Strides strides_v{};
        if (is_static) {
            auto pitches = l.get_pitches();
            strides_v = ov::Strides(pitches.begin(), pitches.end());
        }

        const auto complete_channels_order = get_data_channels_order(layout::max_rank());
        const size_t rank = pshape.size();
        size_t dyn_pad_offset = shape_info_tensor_idx * (layout::max_rank() + 1);

        dims.resize(layout::max_rank());
        pad_lower.resize(layout::max_rank());
        pad_upper.resize(layout::max_rank());
        strides.resize(layout::max_rank());

        for (size_t i = 0; i < complete_channels_order.size(); i++) {
            int channel_index = get_channel_index(complete_channels_order[i], fmt, rank);
            const size_t shape_info_dim_offset = shape_info_tensor_idx * layout::max_rank() + i;
            channels_map[complete_channels_order[i]] = i;

            bool invalid_channel = ((channel_index < 0) || (channel_index >= static_cast<int>(rank)));
            if (invalid_channel) {
                dims[i] = "1";
                pad_lower[i] = "0";
                pad_upper[i] = "0";
                strides[i] = "0";
            } else {
                const auto& dim = pshape[channel_index];
                const auto& pad_l = pad._lower_size[channel_index];
                const auto& pad_u = pad._upper_size[channel_index];
                const auto& pad_dynamic = pad._dynamic_dims_mask[channel_index];

                if (dim.is_static()) {
                    dims[i] = to_code_string(dim.get_length());
                } else {
                    dims[i] = "(shape_info[" + to_code_string(shape_info_dim_offset) + "])";
                }

                if (pad_dynamic) {
                    pad_lower[i] = "(shape_info[" + to_code_string(dyn_pad_offset++) + "])";
                    pad_upper[i] = "(shape_info[" + to_code_string(dyn_pad_offset++) + "])";
                } else {
                    pad_lower[i] = to_code_string(pad_l);
                    pad_upper[i] = to_code_string(pad_u);
                }

                if (is_static) {
                    strides[i] = to_code_string(strides_v[channel_index]);
                } else {
                    OPENVINO_NOT_IMPLEMENTED;
                }
            }
        }
    }
};


class WeightsLayoutJitter : public LayoutJitterBase {
public:
    using channel_type = WeightsChannelName;
    WeightsLayoutJitter(const layout& l, size_t shape_info_idx) {
        OPENVINO_ASSERT(!format::is_weights_format(l.format));
        make_definitions(l, shape_info_idx);
    }

    // definition of tensor element accessors in the following order:
    // weights tensor: g, ofm, ifm, z, y, x
    std::vector<std::string> dims;
    std::vector<std::string> strides;
    std::vector<std::string> pad_lower;
    std::vector<std::string> pad_upper;

    std::map<channel_type, size_t> channels_map;

    std::string dim(WeightsChannelName channel) const {
        return dims[channels_map.at(channel)];
    }

    std::string pad_l(WeightsChannelName channel) const {
        return pad_lower[channels_map.at(channel)];
    }

    std::string pad_u(WeightsChannelName channel) const {
        return pad_upper[channels_map.at(channel)];
    }

    std::string stride(WeightsChannelName channel) const {
        return strides[channels_map.at(channel)];
    }

private:
    void make_definitions(const layout& l, size_t shape_info_tensor_idx) {
        const auto fmt = l.format;
        const auto& pshape = l.get_partial_shape();
        const auto& pad = l.data_padding;
        bool is_static = l.is_static();

        ov::PartialShape vals_ordered;
        const auto& axis_order = fmt.dims_order();
        for (size_t i = 0; i < axis_order.size(); i++) {
            if (axis_order[i] >= pshape.size())
                vals_ordered.push_back(ov::Dimension(1));
            else
                vals_ordered.push_back(pshape[axis_order[i]]);
        };

        ov::Strides strides_v{};
        if (is_static) {
            auto pitches = l.get_pitches();
            strides_v = ov::Strides(pitches.begin(), pitches.end());
        }

        const auto complete_channels_order = get_weights_channels_order(layout::max_rank(), format::is_grouped(fmt));
        const size_t rank = pshape.size();
        size_t dyn_pad_offset = shape_info_tensor_idx * (layout::max_rank() + 1);

        for (size_t i = 0; i < complete_channels_order.size(); i++) {
            int channel_index = get_channel_index(complete_channels_order[i], fmt, rank);
            const size_t shape_info_dim_offset = shape_info_tensor_idx * layout::max_rank() + i;
            channels_map[complete_channels_order[i]] = i;

            bool invalid_channel = ((channel_index < 0) || (channel_index >= static_cast<int>(rank)));
            if (invalid_channel) {
                dims[i] = "1";
                pad_lower[i] = "0";
                pad_upper[i] = "0";
                strides[i] = "0";
            } else {
                const auto& dim = pshape[channel_index];
                const auto& pad_l = pad._lower_size[channel_index];
                const auto& pad_u = pad._upper_size[channel_index];
                const auto& pad_dynamic = pad._dynamic_dims_mask[channel_index];

                if (dim.is_static()) {
                    dims[i] = to_code_string(dim.get_length());
                } else {
                    dims[i] = "(shape_info[" + to_code_string(shape_info_dim_offset) + "])";
                }

                if (pad_dynamic) {
                    pad_lower[i] = "(shape_info[" + to_code_string(dyn_pad_offset++) + "])";
                    pad_upper[i] = "(shape_info[" + to_code_string(dyn_pad_offset++) + "])";
                } else {
                    pad_lower[i] = to_code_string(pad_l);
                    pad_upper[i] = to_code_string(pad_u);
                }

                if (is_static) {
                    strides[i] = to_code_string(strides_v[channel_index]);
                }
            }
        }
    }
};

}  // namespace

JitConstants make_jit_constants(const std::string& name, const ov::element::Type& value) {
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
    switch (value) {
        case ov::element::i8:
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
        case ov::element::u8:
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
        case ov::element::i16:
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
        case ov::element::u16:
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
        case ov::element::i32:
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
        case ov::element::u32:
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
        case ov::element::i64:
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
        case ov::element::f16:
            type = "half";
            max_val = "HALF_MAX";
            min_val = "-" + name + "_VAL_MAX";
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
        case ov::element::i4:
            type = "char";
            type_size = "0.5f";
            is_fp = false;
            break;
        case ov::element::u4:
            type = "uchar";
            type_size = "0.5f";
            is_fp = false;
            break;
        case ov::element::bf16:
            type = "ushort";
            val_one = "(ushort) 1";
            val_zero = "(ushort) 0";
            to_type = "_convert_bfloat16_as_ushort(v)";
            to_type_sat = "_convert_bfloat16_as_ushort(v)";
            type_size = "2";
            is_fp = false;
            break;
        case ov::element::f32:
            type = "float";
            max_val = "FLT_MAX";
            min_val = "-" + name + "_VAL_MAX";
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
        default:
            OPENVINO_THROW("[GPU] Jitter: unsupported data type: ", value);
    }

    return {
        make_jit_constant(name + "_TYPE", type),
        make_jit_constant(name + "_VAL_MAX", max_val),
        make_jit_constant(name + "_VAL_MIN", min_val),
        make_jit_constant(name + "_VAL_ONE", val_one),
        make_jit_constant(name + "_VAL_ZERO", val_zero),
        make_jit_constant("TO_" + name + "_TYPE(v)", to_type),
        make_jit_constant("TO_" + name + "_TYPE_SAT(v)", to_type_sat),
        make_jit_constant("AS_" + name + "_TYPE(v)", as_type),
        make_jit_constant(name + "_MAX_FUNC", max_func),
        make_jit_constant(name + "_MIN_FUNC", min_func),
        make_jit_constant(name + "_ABS_FUNC", abs_func),
        make_jit_constant(name + "_TYPE_SIZE", type_size),
        make_jit_constant(name + "_IS_FP", is_fp),
    };
}

JitConstants make_indexing_jit_functions(const std::string& name, const layout& l) {
    JitConstants definitions;
    auto fmt = l.format;
    std::string args = "";
    switch (fmt.dimension()) {
        case 8: args = "b, f, u, v, w, z, y, x"; break;
        case 7: args = "b, f, v, w, z, y, x"; break;
        case 6: args = "b, f, w, z, y, x"; break;
        case 5: args = "b, f, z, y, x"; break;
        default: args = "b, f, y, x"; break;
    }

    std::string layout_suffix = format::is_simple_data_format(fmt) ? "DATA" : format_string(fmt);

    std::string index_func_name = name + "_GET_INDEX(" + args + ")";
    std::string safe_index_func_name = name + "_GET_INDEX_SAFE(" + args + ")";
    std::string raw_index_func_name = name + "_GET_INDEX_RAW(" + args + ")";

    std::string index_func_val = "GET_" + layout_suffix + "_INDEX(" + name + ", " + args + ")";
    std::string safe_index_func_val = "GET_" + layout_suffix + "_INDEX_SAFE(" + name + ", " + args + ")";
    std::string raw_index_func_val = "GET_" + layout_suffix + "_INDEX_RAW(" + name + ", " + args + ")";

    std::string offset = to_code_string(l.get_linear_offset());
    if (l.count() == 1 && l.is_static()) {
        // if tensor contains single element we can always return first element offset for safe function
        safe_index_func_val = offset;
        index_func_val = offset;
    } else if (l.count() == static_cast<size_t>(l.feature()) && l.is_static()) {
        // We support broadcast only if corresponding dimension is equal to 1.
        // Otherwise, dimensions should be equal and using "f" should be safe.
        if (l.data_padding && format::is_simple_data_format(fmt)) {
            auto f_pitch = to_code_string(0/* _tensor.Feature().pitch */);
            auto f_size = to_code_string(l.feature());
            safe_index_func_val = "(" + offset + " + ((f) % " + f_size + ")  * " + f_pitch + ")";
            index_func_val =  "(" + offset + " + (f) * " + f_pitch + ")";
        } else if (!l.data_padding &&  !format::is_multi_blocked(fmt)) {
            auto f_pad = to_code_string(0/* _tensor.Feature().pad.before */);
            auto f_size = to_code_string(l.feature());
            safe_index_func_val = "((" + offset + " + (f)) % " + f_size + ")";
            index_func_val = "(" + offset + " + (f))";
        }
    }

    definitions.make(index_func_name, index_func_val);
    definitions.make(safe_index_func_name, safe_index_func_val);
    definitions.make(raw_index_func_name, raw_index_func_val);

    return definitions;
}

JitConstants make_jit_constants(const std::string& name, const cldnn::layout& value) {
    JitConstants definitions{
        {name + "_VIEW_OFFSET", to_code_string(0)}, // FIXME
        {name + "_LENGTH", to_code_string(value.count())},
        {name + "_DIMS", to_code_string(value.get_rank())},
        {name + "_SIMPLE", to_code_string(cldnn::format::is_simple_data_format(value.format))},
        {name + "_GROUPED", to_code_string(cldnn::format::is_grouped(value.format))},
        {name + "_LAYOUT_" + to_code_string(format_string(value.format)), "1"},
    };

    definitions.add(make_jit_constants(name, value.data_type));


    if (value.is_static()) {
        definitions.push_back({name + "_OFFSET", to_code_string(0)}); // FIXME
        // definitions.push_back(
        //     {name + "_SIZES_DATA",
        //     toVectorString(t.GetDims(), "", KERNEL_SELECTOR_TENSOR_DIM_MAX, 1, [](const Tensor::Dim& d) { return d.v; })});
        // definitions.push_back(
        //     {name + "_PITCHES",
        //     toVectorString(t.GetDims(), "size_t", KERNEL_SELECTOR_TENSOR_DIM_MAX, 1, [](const Tensor::Dim& d) { return d.pitch; })});
    } else {
        // calculate tensor offset
        // std::vector<std::string> padded_pitches = {
        //     toVectorMulString({name + "_X_PITCH", name + "_PAD_BEFORE_SIZE_X"}),
        //     toVectorMulString({name + "_Y_PITCH", name + "_PAD_BEFORE_SIZE_Y"}),
        //     toVectorMulString({name + "_Z_PITCH", name + "_PAD_BEFORE_SIZE_Z"}),
        //     toVectorMulString({name + "_W_PITCH", name + "_PAD_BEFORE_SIZE_W"}),
        //     toVectorMulString({name + "_FEATURE_PITCH", name + "_PAD_BEFORE_FEATURE_NUM"}),
        //     toVectorMulString({name + "_BATCH_PITCH", name + "_PAD_BEFORE_BATCH_NUM"})};
        // std::string offset_str = "(";
        // for (size_t i = 0; i < padded_pitches.size(); ++i) {
        //     offset_str += padded_pitches[i];
        //     if (i < padded_pitches.size() - 1)
        //         offset_str += " + ";
        // }
        // offset_str += ")";
        // definitions.push_back({name + "_OFFSET", offset_str});
    }
    // definitions.push_back(
    //     {name + "_PAD_BEFORE",
    //         toVectorString(t.GetDims(), "size_t", layout::max_rank(), 0, [](const Tensor::Dim& d) {
    //             return d.pad.before;
    //         })});
    // definitions.push_back(
    //     {name + "_PAD_AFTER",
    //         toVectorString(t.GetDims(), "size_t", layout::max_rank(), 0, [](const Tensor::Dim& d) {
    //             return d.pad.after;
    //         })});


    if (format::is_weights_format(value.format)) {
        WeightsLayoutJitter jitter(value, 0);
        definitions.add({
            // make_jit_constant(name + "_SIZE_X", extract_dim(WeightsChannelName::X, fmt, vals_ordered).get_length()),
            // make_jit_constant(name + "_SIZE_Y", extract_dim(WeightsChannelName::Y, fmt, vals_ordered).get_length()),
            // make_jit_constant(name + "_SIZE_Z", extract_dim(WeightsChannelName::Z, fmt, vals_ordered).get_length()),
            // make_jit_constant(name + "_IFM_NUM", extract_dim(WeightsChannelName::IFM, fmt, vals_ordered).get_length()),
            // make_jit_constant(name + "_OFM_NUM", extract_dim(WeightsChannelName::OFM, fmt, vals_ordered).get_length()),
            // make_jit_constant(name + "_GROUPS_NUM", extract_dim(WeightsChannelName::G, fmt, vals_ordered).get_length()),

            // make_jit_constant(name + "_PAD_BEFORE_SIZE_X", extract_pad(WeightsChannelName::X, fmt, value._upper_size)),
            // make_jit_constant(name + "_PAD_BEFORE_SIZE_Y", extract_pad(WeightsChannelName::Y, fmt, value._upper_size)),
            // make_jit_constant(name + "_PAD_BEFORE_SIZE_Z", extract_pad(WeightsChannelName::Z, fmt, value._upper_size)),
            // make_jit_constant(name + "_PAD_BEFORE_IFM_NUM", extract_pad(WeightsChannelName::IFM, fmt, value._upper_size)),
            // make_jit_constant(name + "_PAD_BEFORE_OFM_NUM", extract_pad(WeightsChannelName::OFM, fmt, value._upper_size)),
            // make_jit_constant(name + "_PAD_BEFORE_GROUP_NUM", extract_pad(WeightsChannelName::G, fmt, value._upper_size)),

            // make_jit_constant(name + "_PAD_AFTER_SIZE_X", extract_pad(WeightsChannelName::X, fmt, value._lower_size)),
            // make_jit_constant(name + "_PAD_AFTER_SIZE_Y", extract_pad(WeightsChannelName::Y, fmt, value._lower_size)),
            // make_jit_constant(name + "_PAD_AFTER_SIZE_Z", extract_pad(WeightsChannelName::Z, fmt, value._lower_size)),
            // make_jit_constant(name + "_PAD_AFTER_IFM_NUM", extract_pad(WeightsChannelName::IFM, fmt, value._lower_size)),
            // make_jit_constant(name + "_PAD_AFTER_OFM_NUM", extract_pad(WeightsChannelName::OFM, fmt, value._lower_size)),
            // make_jit_constant(name + "_PAD_AFTER_GROUPS_NUM", extract_pad(WeightsChannelName::G, fmt, value._lower_size)),

            // make_jit_constant(name + "_X_PITCH", extract_stride(WeightsChannelName::X, fmt, vals_ordered)),
            // make_jit_constant(name + "_Y_PITCH", extract_stride(WeightsChannelName::Y, fmt, vals_ordered)),
            // make_jit_constant(name + "_Z_PITCH", extract_stride(WeightsChannelName::Z, fmt, vals_ordered)),
            // make_jit_constant(name + "_IFM_PITCH", extract_stride(WeightsChannelName::IFM, fmt, vals_ordered)),
            // make_jit_constant(name + "_OFM_PITCH", extract_stride(WeightsChannelName::OFM, fmt, vals_ordered)),
            // make_jit_constant(name + "_GROUPS_PITCH", extract_stride(WeightsChannelName::G, fmt, vals_ordered)),
        });
    } else {
        DataLayoutJitter jitter(value, 0);

        definitions.add(make_indexing_jit_functions(name, value));
        definitions.add({
            make_jit_constant(name + "_SIZE_X", jitter.dim(DataChannelName::X)),
            make_jit_constant(name + "_SIZE_Y", jitter.dim(DataChannelName::Y)),
            make_jit_constant(name + "_SIZE_Z", jitter.dim(DataChannelName::Z)),
            make_jit_constant(name + "_SIZE_W", jitter.dim(DataChannelName::W)),
            make_jit_constant(name + "_SIZE_U", jitter.dim(DataChannelName::U)),
            make_jit_constant(name + "_SIZE_V", jitter.dim(DataChannelName::V)),
            make_jit_constant(name + "_FEATURE_NUM", jitter.dim(DataChannelName::FEATURE)),
            make_jit_constant(name + "_BATCH_NUM", jitter.dim(DataChannelName::BATCH)),

            make_jit_constant(name + "_PAD_BEFORE_SIZE_X", jitter.pad_l(DataChannelName::X)),
            make_jit_constant(name + "_PAD_BEFORE_SIZE_Y", jitter.pad_l(DataChannelName::Y)),
            make_jit_constant(name + "_PAD_BEFORE_SIZE_Z", jitter.pad_l(DataChannelName::Z)),
            make_jit_constant(name + "_PAD_BEFORE_SIZE_W", jitter.pad_l(DataChannelName::W)),
            make_jit_constant(name + "_PAD_BEFORE_SIZE_U", jitter.pad_l(DataChannelName::U)),
            make_jit_constant(name + "_PAD_BEFORE_SIZE_V", jitter.pad_l(DataChannelName::V)),
            make_jit_constant(name + "_PAD_BEFORE_FEATURE_NUM", jitter.pad_l(DataChannelName::FEATURE)),
            make_jit_constant(name + "_PAD_BEFORE_BATCH_NUM", jitter.pad_l(DataChannelName::BATCH)),

            make_jit_constant(name + "_PAD_AFTER_SIZE_X", jitter.pad_u(DataChannelName::X)),
            make_jit_constant(name + "_PAD_AFTER_SIZE_Y", jitter.pad_u(DataChannelName::Y)),
            make_jit_constant(name + "_PAD_AFTER_SIZE_Z", jitter.pad_u(DataChannelName::Z)),
            make_jit_constant(name + "_PAD_AFTER_SIZE_W", jitter.pad_u(DataChannelName::W)),
            make_jit_constant(name + "_PAD_AFTER_SIZE_U", jitter.pad_u(DataChannelName::U)),
            make_jit_constant(name + "_PAD_AFTER_SIZE_V", jitter.pad_u(DataChannelName::V)),
            make_jit_constant(name + "_PAD_AFTER_FEATURE_NUM", jitter.pad_u(DataChannelName::FEATURE)),
            make_jit_constant(name + "_PAD_AFTER_BATCH_NUM", jitter.pad_u(DataChannelName::BATCH)),

            make_jit_constant(name + "_X_PITCH", jitter.stride(DataChannelName::X)),
            make_jit_constant(name + "_Y_PITCH", jitter.stride(DataChannelName::Y)),
            make_jit_constant(name + "_Z_PITCH", jitter.stride(DataChannelName::Z)),
            make_jit_constant(name + "_W_PITCH", jitter.stride(DataChannelName::W)),
            make_jit_constant(name + "_U_PITCH", jitter.stride(DataChannelName::U)),
            make_jit_constant(name + "_V_PITCH", jitter.stride(DataChannelName::V)),
            make_jit_constant(name + "_FEATURE_PITCH", jitter.stride(DataChannelName::FEATURE)),
            make_jit_constant(name + "_BATCH_PITCH", jitter.stride(DataChannelName::BATCH))
        });
    }

    return definitions;
}

}  // namespace ocl
}  // namespace intel_gpu
}  // namespace ov
