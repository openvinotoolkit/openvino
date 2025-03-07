// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jitter.hpp"

#include <string>
#include "intel_gpu/primitives/activation.hpp"
#include "openvino/core/dimension.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/type/element_type.hpp"

namespace ov::intel_gpu {
namespace {

using namespace cldnn;

std::string format_string(format fmt) {
    auto str = fmt.to_string();
    for (auto& s : str) {
        s = std::toupper(s);
    }

    return str;
}

std::vector<ChannelName> get_data_channels_order(size_t rank) {
    using channel = ChannelName;
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

std::vector<ChannelName> get_weights_channels_order(size_t rank, bool is_grouped) {
    using channel = ChannelName;
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

std::vector<ChannelName> get_default_channels_order(size_t rank, bool is_weights_fmt = false, bool is_grouped = false) {
    if (is_weights_fmt)
        return get_weights_channels_order(rank, is_grouped);
    else
        return get_data_channels_order(rank);
}

int get_channel_index(ChannelName channel_name, size_t rank, bool is_weights_fmt = false, bool is_grouped = false) {
    auto order = get_default_channels_order(rank, is_weights_fmt, is_grouped);
    auto it = std::find(order.begin(), order.end(), channel_name);
    if (it == order.end())
        return -1;

    return std::distance(order.begin(), it);
}

}  // namespace

void LayoutJitter::make_definitions(const layout& l, size_t shape_info_offset) {
    const auto fmt = l.format;
    const bool is_weights_fmt = format::is_weights_format(fmt);
    const bool is_grouped = format::is_grouped(fmt);

    const auto& pshape = l.get_partial_shape();
    const size_t rank = pshape.size();
    const auto& pad = l.data_padding;
    const auto& dims_order = l.get_dims_order();
    bool is_static = l.is_static();

    ov::PartialShape vals_ordered;
    const auto& axis_order = fmt.dims_order();
    for (size_t i = 0; i < axis_order.size(); i++) {
        if (axis_order[i] >= pshape.size())
            vals_ordered.push_back(ov::Dimension(1));
        else
            vals_ordered.push_back(pshape[axis_order[i]]);
    }

    ov::Strides strides{};
    if (is_static) {
        auto pitches = l.get_pitches();
        strides = ov::Strides(pitches.begin(), pitches.end());
    }

    const size_t max_rank = is_weights_fmt ? 6 : layout::max_rank();
    const auto complete_channels_order = get_default_channels_order(max_rank, is_weights_fmt, is_grouped);
    const auto default_channels_order = get_default_channels_order(rank, is_weights_fmt, is_grouped);
    std::vector<ChannelName> actual_channels_order;
    for (size_t i = 0; i < rank; i++) {
        actual_channels_order.push_back(default_channels_order[dims_order[i]]);
    }

    size_t dyn_pad_offset = shape_info_offset + max_rank;

    m_dims.resize(max_rank);
    m_pad_lower.resize(max_rank);
    m_pad_upper.resize(max_rank);
    m_strides.resize(max_rank);

    const JitTerm one{"1"};
    const JitTerm zero{"0"};
    const JitTerm shape_info{"shape_info"};

    for (size_t i = 0; i < complete_channels_order.size(); i++) {
        auto target_channel = complete_channels_order[i];
        int channel_index = get_channel_index(target_channel, rank, is_weights_fmt, is_grouped);
        const size_t shape_info_dim_offset = shape_info_offset + i;
        channels_map[target_channel] = i;

        bool invalid_channel = ((channel_index < 0) || (channel_index >= static_cast<int>(rank)));
        if (invalid_channel) {
            m_dims[i] = one;
            m_pad_lower[i] = zero;
            m_pad_upper[i] = zero;
            m_strides[i] = zero;
        } else {
            const auto& dim = pshape[channel_index];
            const auto& pad_l = pad._lower_size[channel_index];
            const auto& pad_u = pad._upper_size[channel_index];
            const auto& pad_dynamic = pad._dynamic_dims_mask[channel_index];

            if (dim.is_static()) {
                m_dims[i] = JitTerm{to_code_string(dim.get_length())};
            } else {
                m_dims[i] = shape_info[shape_info_dim_offset];
            }

            if (pad_dynamic) {
                m_pad_lower[i] = shape_info[dyn_pad_offset++];
                m_pad_upper[i] = shape_info[dyn_pad_offset++];
            } else {
                m_pad_lower[i] = JitTerm{to_code_string(pad_l)};
                m_pad_upper[i] = JitTerm{to_code_string(pad_u)};
            }
        }
    }

    for (size_t i = 0; i < complete_channels_order.size(); i++) {
        auto target_channel = complete_channels_order[i];
        int channel_index = get_channel_index(target_channel, rank, is_weights_fmt, is_grouped);
        bool valid_channel = ((channel_index >= 0) && (channel_index < static_cast<int>(rank)));
        if (valid_channel) {
            if (is_static && !pad.is_dynamic()) {
                m_strides[i] = JitTerm{to_code_string(strides[channel_index])};
            } else if (format::is_simple_data_format(fmt)) {
                auto channel_it = std::find(actual_channels_order.begin(), actual_channels_order.end(), target_channel);
                m_strides[i] = JitTerm{"1"};
                for (channel_it++; channel_it != actual_channels_order.end(); channel_it++) {
                    auto idx = std::distance(default_channels_order.begin(),
                                             std::find(default_channels_order.begin(), default_channels_order.end(), *channel_it));
                    auto idx_ext = channels_map[*channel_it];
                    if (pad._lower_size[idx] || pad._upper_size[idx] || pad._dynamic_dims_mask[idx])
                        m_strides[i] = m_strides[i] * (m_dims[idx_ext] + m_pad_lower[idx_ext] + m_pad_upper[idx_ext]);
                    else
                        m_strides[i] = m_strides[i] * m_dims[idx_ext];
                }
            } else {
                m_strides[i] = JitTerm{"NOT_IMPLEMENTED"};
            }
        }
    }

    std::vector<JitTerm> padded_pitches = {
            m_strides[channels_map.at(ChannelName::X)] * m_pad_lower[channels_map.at(ChannelName::X)],
            m_strides[channels_map.at(ChannelName::Y)] * m_pad_lower[channels_map.at(ChannelName::Y)],
            m_strides[channels_map.at(ChannelName::Z)] * m_pad_lower[channels_map.at(ChannelName::Z)],
            m_strides[channels_map.at(ChannelName::W)] * m_pad_lower[channels_map.at(ChannelName::W)],
            m_strides[channels_map.at(ChannelName::U)] * m_pad_lower[channels_map.at(ChannelName::U)],
            m_strides[channels_map.at(ChannelName::V)] * m_pad_lower[channels_map.at(ChannelName::V)],
            m_strides[channels_map.at(ChannelName::FEATURE)] * m_pad_lower[channels_map.at(ChannelName::FEATURE)],
            m_strides[channels_map.at(ChannelName::BATCH)] * m_pad_lower[channels_map.at(ChannelName::BATCH)],
    };

    m_offset = JitTerm{"0"};
    for (size_t i = 0; i < padded_pitches.size(); ++i) {
        m_offset = m_offset + padded_pitches[i];
    }
}

JitConstants make_type_jit_constants(const std::string& name, const ov::element::Type& value) {
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
    JitTerm args{""};
    JitTerm rank_suffix{""};
    switch (fmt.dimension()) {
        case 8: args = JitTerm{"b, f, u, v, w, z, y, x"}; rank_suffix = JitTerm{"_8D"}; break;
        case 7: args = JitTerm{"b, f, v, w, z, y, x"}; rank_suffix = JitTerm{"_7D"}; break;
        case 6: args = JitTerm{"b, f, w, z, y, x"}; rank_suffix = JitTerm{"_6D"}; break;
        case 5: args = JitTerm{"b, f, z, y, x"}; rank_suffix = JitTerm{"_5D"}; break;
        default: args = JitTerm{"b, f, y, x"}; rank_suffix = JitTerm{""}; break;
    }

    const JitTerm tensor_name{name};
    bool simple_format = format::is_simple_data_format(fmt);

    const JitTerm layout_suffix{format::is_simple_data_format(fmt) ? "DATA" : format_string(fmt)};
    if (!simple_format) {
        rank_suffix = JitTerm{""};
    }

    const JitTerm base_func_name{ name + "_GET_INDEX"};
    const JitTerm base_val_name{ "GET_" + layout_suffix.str() + "_INDEX" + rank_suffix.str()};
    const JitTerm safe_suffix{"_SAFE"};
    const JitTerm raw_suffix{"_RAW"};

    JitTerm index_func_name = base_func_name(args);
    JitTerm safe_index_func_name = concat(base_func_name, safe_suffix)(args);
    JitTerm raw_index_func_name = concat(base_func_name, raw_suffix)(args);

    JitTerm index_func_val = base_val_name(tensor_name, args);
    JitTerm safe_index_func_val = concat(base_val_name, safe_suffix)(tensor_name, args);
    JitTerm raw_index_func_val = concat(base_val_name, raw_suffix)(tensor_name, args);

    if (l.is_static()) {
        const JitTerm offset {to_code_string(l.get_linear_offset())};
        if (l.count() == 1) {
            // if tensor contains single element we can always return first element offset for safe function
            safe_index_func_val = offset;
            index_func_val = offset;
        } else if (l.count() == static_cast<size_t>(l.feature())) {
            // We support broadcast only if corresponding dimension is equal to 1.
            // Otherwise, dimensions should be equal and using "f" should be safe.
            const JitTerm f_size{ to_code_string(l.feature()) };
            const JitTerm f{"(f)"};
            if (l.data_padding && format::is_simple_data_format(fmt)) {
                const JitTerm f_pitch{ to_code_string(0/* _tensor.Feature().pitch */) };
                safe_index_func_val = offset + (f % f_size) * f_pitch;
                index_func_val =  offset + (f * f_pitch);
            } else if (!l.data_padding && !format::is_multi_blocked(fmt)) {
                safe_index_func_val = (offset + f) % f_size;
                index_func_val = offset + f;
            }
        }
    }

    definitions.make(index_func_name.str(), index_func_val.str());
    definitions.make(safe_index_func_name.str(), safe_index_func_val.str());
    definitions.make(raw_index_func_name.str(), raw_index_func_val.str());

    return definitions;
}

JitConstants make_layout_jit_constants(const std::string& name, const cldnn::layout& value, size_t shape_info_offset) {
    JitConstants definitions{
        {name + "_VIEW_OFFSET", to_code_string(0)}, // FIXME
        {name + "_LENGTH", to_code_string(value.is_static() ? value.count() : 0)},
        {name + "_DIMS", to_code_string(value.get_rank())},
        {name + "_SIMPLE", to_code_string(cldnn::format::is_simple_data_format(value.format))},
        {name + "_GROUPED", to_code_string(cldnn::format::is_grouped(value.format))},
        {name + "_LAYOUT_" + to_code_string(format_string(value.format)), "1"},
    };

    definitions.add(make_type_jit_constants(name, value.data_type));

    if (format::is_weights_format(value.format)) {
        LayoutJitter jitter(value, shape_info_offset);
        definitions.add({
            make_jit_constant(name + "_OFFSET", jitter.offset()),

            make_jit_constant(name + "_SIZE_X", jitter.dim(ChannelName::X)),
            make_jit_constant(name + "_SIZE_Y", jitter.dim(ChannelName::Y)),
            make_jit_constant(name + "_SIZE_Z", jitter.dim(ChannelName::Z)),
            make_jit_constant(name + "_IFM_NUM", jitter.dim(ChannelName::IFM)),
            make_jit_constant(name + "_OFM_NUM", jitter.dim(ChannelName::OFM)),
            make_jit_constant(name + "_GROUPS_NUM", jitter.dim(ChannelName::G)),

            make_jit_constant(name + "_PAD_BEFORE_SIZE_X", jitter.pad_l(ChannelName::X)),
            make_jit_constant(name + "_PAD_BEFORE_SIZE_Y", jitter.pad_l(ChannelName::Y)),
            make_jit_constant(name + "_PAD_BEFORE_SIZE_Z", jitter.pad_l(ChannelName::Z)),
            make_jit_constant(name + "_PAD_BEFORE_IFM_NUM", jitter.pad_l(ChannelName::IFM)),
            make_jit_constant(name + "_PAD_BEFORE_OFM_NUM", jitter.pad_l(ChannelName::OFM)),
            make_jit_constant(name + "_PAD_BEFORE_GROUP_NUM", jitter.pad_l(ChannelName::G)),

            make_jit_constant(name + "_PAD_AFTER_SIZE_X", jitter.pad_u(ChannelName::X)),
            make_jit_constant(name + "_PAD_AFTER_SIZE_Y", jitter.pad_u(ChannelName::Y)),
            make_jit_constant(name + "_PAD_AFTER_SIZE_Z", jitter.pad_u(ChannelName::Z)),
            make_jit_constant(name + "_PAD_AFTER_IFM_NUM", jitter.pad_u(ChannelName::IFM)),
            make_jit_constant(name + "_PAD_AFTER_OFM_NUM", jitter.pad_u(ChannelName::OFM)),
            make_jit_constant(name + "_PAD_AFTER_GROUPS_NUM", jitter.pad_u(ChannelName::G)),

            make_jit_constant(name + "_X_PITCH", jitter.stride(ChannelName::X)),
            make_jit_constant(name + "_Y_PITCH", jitter.stride(ChannelName::Y)),
            make_jit_constant(name + "_Z_PITCH", jitter.stride(ChannelName::Z)),
            make_jit_constant(name + "_IFM_PITCH", jitter.stride(ChannelName::IFM)),
            make_jit_constant(name + "_OFM_PITCH", jitter.stride(ChannelName::OFM)),
            make_jit_constant(name + "_GROUPS_PITCH", jitter.stride(ChannelName::G)),
        });
    } else {
        LayoutJitter jitter(value, shape_info_offset);
        definitions.add(make_indexing_jit_functions(name, value));
        definitions.add({
            make_jit_constant(name + "_OFFSET", jitter.offset()),

            make_jit_constant(name + "_SIZE_X", jitter.dim(ChannelName::X)),
            make_jit_constant(name + "_SIZE_Y", jitter.dim(ChannelName::Y)),
            make_jit_constant(name + "_SIZE_Z", jitter.dim(ChannelName::Z)),
            make_jit_constant(name + "_SIZE_W", jitter.dim(ChannelName::W)),
            make_jit_constant(name + "_SIZE_U", jitter.dim(ChannelName::U)),
            make_jit_constant(name + "_SIZE_V", jitter.dim(ChannelName::V)),
            make_jit_constant(name + "_FEATURE_NUM", jitter.dim(ChannelName::FEATURE)),
            make_jit_constant(name + "_BATCH_NUM", jitter.dim(ChannelName::BATCH)),

            make_jit_constant(name + "_PAD_BEFORE_SIZE_X", jitter.pad_l(ChannelName::X)),
            make_jit_constant(name + "_PAD_BEFORE_SIZE_Y", jitter.pad_l(ChannelName::Y)),
            make_jit_constant(name + "_PAD_BEFORE_SIZE_Z", jitter.pad_l(ChannelName::Z)),
            make_jit_constant(name + "_PAD_BEFORE_SIZE_W", jitter.pad_l(ChannelName::W)),
            make_jit_constant(name + "_PAD_BEFORE_SIZE_U", jitter.pad_l(ChannelName::U)),
            make_jit_constant(name + "_PAD_BEFORE_SIZE_V", jitter.pad_l(ChannelName::V)),
            make_jit_constant(name + "_PAD_BEFORE_FEATURE_NUM", jitter.pad_l(ChannelName::FEATURE)),
            make_jit_constant(name + "_PAD_BEFORE_BATCH_NUM", jitter.pad_l(ChannelName::BATCH)),

            make_jit_constant(name + "_PAD_AFTER_SIZE_X", jitter.pad_u(ChannelName::X)),
            make_jit_constant(name + "_PAD_AFTER_SIZE_Y", jitter.pad_u(ChannelName::Y)),
            make_jit_constant(name + "_PAD_AFTER_SIZE_Z", jitter.pad_u(ChannelName::Z)),
            make_jit_constant(name + "_PAD_AFTER_SIZE_W", jitter.pad_u(ChannelName::W)),
            make_jit_constant(name + "_PAD_AFTER_SIZE_U", jitter.pad_u(ChannelName::U)),
            make_jit_constant(name + "_PAD_AFTER_SIZE_V", jitter.pad_u(ChannelName::V)),
            make_jit_constant(name + "_PAD_AFTER_FEATURE_NUM", jitter.pad_u(ChannelName::FEATURE)),
            make_jit_constant(name + "_PAD_AFTER_BATCH_NUM", jitter.pad_u(ChannelName::BATCH)),

            make_jit_constant(name + "_X_PITCH", jitter.stride(ChannelName::X)),
            make_jit_constant(name + "_Y_PITCH", jitter.stride(ChannelName::Y)),
            make_jit_constant(name + "_Z_PITCH", jitter.stride(ChannelName::Z)),
            make_jit_constant(name + "_W_PITCH", jitter.stride(ChannelName::W)),
            make_jit_constant(name + "_U_PITCH", jitter.stride(ChannelName::U)),
            make_jit_constant(name + "_V_PITCH", jitter.stride(ChannelName::V)),
            make_jit_constant(name + "_FEATURE_PITCH", jitter.stride(ChannelName::FEATURE)),
            make_jit_constant(name + "_BATCH_PITCH", jitter.stride(ChannelName::BATCH))
        });
    }

    return definitions;
}

JitConstants make_activation_jit_constants(activation_func activation_function,
                                           ov::element::Type_t out_dt,
                                           const std::string& suffix,
                                           bool use_type_parameter,
                                           bool disable_type_conversion) {
    std::string name = "ACTIVATION_FUNC" + suffix;
    JitConstants jit = {};

    if (activation_function != activation_func::none)
        jit.add(make_type_jit_constants(name, out_dt));

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
    auto max_func = [type_handler](const JitTerm& lhs, const JitTerm& rhs) -> JitTerm {
        return JitTerm{"(" + type_handler("", "_MAX_FUNC") + "(" + lhs.str() + ", " + rhs.str() + "))"};
    };
    auto min_func = [type_handler](const JitTerm& lhs, const JitTerm& rhs) -> JitTerm {
        return JitTerm{"(" + type_handler("", "_MIN_FUNC") + "(" + lhs.str() + ", " + rhs.str() + "))"};
    };
    auto abs_func = [type_handler](const JitTerm& hs) -> JitTerm {
        return JitTerm{"(" + type_handler("", "_ABS_FUNC") + "(" + hs.str() + "))"};
    };
    auto to_type = [type_handler](const JitTerm& arg) -> JitTerm {
        return JitTerm{type_handler("TO_", "_TYPE") + "(" + arg.str() + ")"};
    };

    std::string macro_def = name + (use_type_parameter ? "(jit_type, input, m, n)" : "(input, m, n)");
    std::string macro_def_params = use_type_parameter ? "(jit_type, input, params)" : "(input, params)";

    jit.add(make_jit_constant("ACTIVATION_PARAMS" + suffix, "NL_M" + suffix + ", NL_N" + suffix));

    // TODO: use native_exp and use cast for APL
    switch (activation_function) {
        case activation_func::logistic:
            jit.add(make_jit_constant(macro_def, (one / (one + exp(neg(input)))).str()));
            break;
        case activation_func::hyperbolic_tan:
            jit.add(make_jit_constant(macro_def, "(tanh(input))"));
            break;
        case activation_func::relu:
            jit.add(make_jit_constant(macro_def, max_func(zero, input).str()));
            break;
        case activation_func::relu_negative_slope: {
            const JitTerm slope = disable_type_conversion ? "m"_jit : to_type("m"_jit);
            jit.add(make_jit_constant(
                macro_def,
                ternary(isinf(slope),
                        ternary(input.ge(zero), input, neg(slope)),
                        max_func(input, zero) + (slope * min_func(input, zero)))
                    .str()));
            break;
        }
        case activation_func::elu: {
            auto alpha = disable_type_conversion ? "m"_jit : to_type("m"_jit);
            jit.add(make_jit_constant(
                macro_def,
                (max_func(input, zero) + (alpha * (exp(min_func(input, zero)) - one)))
                    .str()));
            break;
        }
        case activation_func::clamp: {
            const JitTerm m = disable_type_conversion ? "m"_jit : to_type("m"_jit);
            const JitTerm n = disable_type_conversion ? "n"_jit : to_type("n"_jit);
            jit.add(make_jit_constant(
                 macro_def,
                 max_func(m, min_func(n, input)).str()));
            break;
        }
        case activation_func::softrelu:
            jit.add(make_jit_constant(macro_def, log(one + exp(input)).str()));
            break;
        case activation_func::abs:
            if (out_dt == ov::element::f32 || out_dt == ov::element::f16)
                jit.add(make_jit_constant(macro_def, "(fabs(input))"));
            else
                jit.add(make_jit_constant(macro_def, "(abs(input))"));
            break;
        case activation_func::linear: {
            const JitTerm m = disable_type_conversion ? "m"_jit : to_type("m"_jit);
            const JitTerm n = disable_type_conversion ? "n"_jit : to_type("n"_jit);
            jit.add(make_jit_constant(macro_def, (m * input + n).str()));
            break;
        }
        case activation_func::square:
            jit.add(make_jit_constant(macro_def, "(input*input)"));
            break;
        case activation_func::sqrt:
            jit.add(make_jit_constant(macro_def, "(sqrt(input))"));
            break;
        case activation_func::sin:
            jit.add(make_jit_constant(macro_def, "(sin(input))"));
            break;
        case activation_func::asin:
            jit.add(make_jit_constant(macro_def, "(asin(input))"));
            break;
        case activation_func::sinh:
            jit.add(make_jit_constant(macro_def, "(sinh(input))"));
            break;
        case activation_func::asinh:
            jit.add(make_jit_constant(macro_def, "(asinh(input))"));
            break;
        case activation_func::cos:
            jit.add(make_jit_constant(macro_def, "(cos(input))"));
            break;
        case activation_func::acos:
            jit.add(make_jit_constant(macro_def, "(acos(input))"));
            break;
        case activation_func::cosh:
            jit.add(make_jit_constant(macro_def, "(cosh(input))"));
            break;
        case activation_func::acosh:
            jit.add(make_jit_constant(macro_def, "(acosh(input))"));
            break;
        case activation_func::log:
            jit.add(make_jit_constant(macro_def, "(log(input))"));
            break;
        case activation_func::log2:
            jit.add(make_jit_constant(macro_def, "(log2(input))"));
            break;
        case activation_func::exp:
            jit.add(make_jit_constant(macro_def, "(exp(input))"));
            break;
        case activation_func::pow: {
            const JitTerm m = disable_type_conversion ? "m"_jit : to_type("m"_jit);
            jit.add(make_jit_constant(macro_def, "(pow(input," + m.str() + "))"));
            break;
        }
        case activation_func::tan:
            jit.add(make_jit_constant(macro_def, "(tan(input))"));
            break;
        case activation_func::atan:
            jit.add(make_jit_constant(macro_def, "(atan(input))"));
            break;
        case activation_func::atanh:
            jit.add(make_jit_constant(macro_def, "(atanh(input))"));
            break;
        case activation_func::floor:
            if (out_dt == ov::element::f32 || out_dt == ov::element::f16)
                jit.add(make_jit_constant(macro_def, "(floor(input))"));
            else
                jit.add(make_jit_constant(macro_def, "(input)"));
            break;
        case activation_func::ceil:
            if (out_dt == ov::element::f32 || out_dt == ov::element::f16)
                jit.add(make_jit_constant(macro_def, "(ceil(input))"));
            else
                jit.add(make_jit_constant(macro_def, "(input)"));
            break;
        case activation_func::negative:
            jit.add(make_jit_constant(macro_def, "(-input)"));
            break;
        case activation_func::erf:
            jit.add(make_jit_constant(macro_def, erf(input).str()));
            break;
        case activation_func::hard_sigmoid: {
            auto alpha = disable_type_conversion ? "m"_jit : to_type("m"_jit);
            auto beta =  disable_type_conversion ? "n"_jit : to_type("n"_jit);
            jit.add(make_jit_constant(
                    macro_def,
                    max_func(zero, min_func(one, (JitTerm)((alpha * input + beta).str()))).str()));
            break;
        }
        case activation_func::hsigmoid: {
            std::string type_suffix = out_dt == ov::element::f32 ? "f" : "h";
            const JitTerm three("3." + type_suffix);
            const JitTerm six("6." + type_suffix);
            jit.add(make_jit_constant(
                    macro_def,
                    (min_func(max_func(zero, input + three), six) / six).str()));
            break;
        }
        case activation_func::sign:
            jit.add(make_jit_constant(
                    macro_def,
                    ternary(input.gt(zero), one, ternary(input.eq(zero), zero, neg(one))).str()));
            break;
        case activation_func::reciprocal:
            jit.add(make_jit_constant(macro_def, (one / input).str()));
            break;
        case activation_func::selu: {
            auto alpha = disable_type_conversion ? "m"_jit : to_type("m"_jit);
            auto gamma = disable_type_conversion ? "n"_jit : to_type("n"_jit);
            jit.add(make_jit_constant(
                    macro_def,
                    ternary(input.le(zero), gamma * (alpha * exp(input) - alpha), gamma * input).str()));
            break;
        }
        case activation_func::softplus: {
            jit.add(make_jit_constant(
                    macro_def,
                    log(exp(input) + one).str()));
            break;
        }
        case activation_func::softsign: {
            jit.add(make_jit_constant(
                    macro_def,
                    (input / (one + abs_func(input))).str()));
            break;
        }
        case activation_func::swish: {
            auto beta = disable_type_conversion ? "m"_jit : to_type("m"_jit);
            jit.add(make_jit_constant(
                    macro_def,
                    (input / (one + exp(neg(beta * input)))).str()));
            break;
        }
        case activation_func::hswish: {
            std::string type_suffix = out_dt == ov::element::f32 ? "f" : "h";
            const JitTerm three("3." + type_suffix);
            const JitTerm six("6." + type_suffix);
            jit.add(make_jit_constant(
                    macro_def,
                    (input * min_func(max_func(zero, input + three), six) / six).str()));
            break;
        }
        case activation_func::mish: {
            std::string type_suffix = out_dt == ov::element::f32 ? "f" : "h";
            auto bound = out_dt == ov::element::f32 ? "9.9f"_jit : "4.75h"_jit;
            const JitTerm two("2." + type_suffix);
            const JitTerm n((exp(input) + two) * exp(input));
            const JitTerm common_mish_formula((input * n) / (n + two));

            jit.add(make_jit_constant(
                macro_def,
                ternary(input.ge(bound), input, common_mish_formula).str()));
            break;
        }
        case activation_func::gelu: {
            std::string type_suffix = out_dt == ov::element::f32 ? "f" : "h";
            const JitTerm half{"0.5" + type_suffix};
            const JitTerm mult{"0.7071067811865475" + type_suffix}; // (1 / sqrt(2))
            jit.add(make_jit_constant(
                    macro_def,
                    (half * input * (one + erf((input * mult)))).str()));
            break;
        }
        case activation_func::gelu_tanh: {
            const std::string type_suffix = out_dt == ov::element::f32 ? "f" : "h";
            const JitTerm half{"0.5" + type_suffix};
            const JitTerm mult{"0.044715" + type_suffix};
            const JitTerm sqrt_2_over_pi{"0.79788458347320556640625" + type_suffix};
            jit.add(make_jit_constant(
                    macro_def,
                    (half * input * (one + tanh(sqrt_2_over_pi * input * (one + mult * input * input)))).str()));
            break;
        }
        case activation_func::negation:
            jit.add(make_jit_constant(
                macro_def,
                ternary(input.eq(zero), one, zero)
                    .str()));  // the workaround for OpenCL's vector type result (!input)
            break;
        case activation_func::round_half_to_even:
            jit.add(make_jit_constant(macro_def, "rint(input)"));
            break;
        case activation_func::round_half_away_from_zero:
            jit.add(make_jit_constant(macro_def, "(round(input))"));
            break;
        case activation_func::none:
        default:
            jit.add(make_jit_constant(macro_def, "input"));
            break;
    }

    jit.add(make_jit_constant("ACTIVATION" + suffix + macro_def_params, name + macro_def_params));

    return jit;
}

size_t extract_channel(ChannelName channel, const layout& l) {
    const auto& pshape = l.get_partial_shape();
    auto idx = get_channel_index(channel, pshape.size(), format::is_weights_format(l.format), format::is_grouped(l.format));
    return (idx < 0 || idx >= static_cast<int>(pshape.size())) ? 1 : static_cast<size_t>(pshape[idx].get_length());
}

std::string to_ocl_type(ov::element::Type_t et) {
    switch (et) {
        case ov::element::Type_t::i8:
            return get_ocl_type_name<int8_t>();
        case ov::element::Type_t::u8:
            return get_ocl_type_name<uint8_t>();
        case ov::element::Type_t::i16:
            return get_ocl_type_name<int16_t>();
        case ov::element::Type_t::u16:
            return get_ocl_type_name<uint16_t>();
        case ov::element::Type_t::i32:
            return get_ocl_type_name<int32_t>();
        case ov::element::Type_t::u32:
            return get_ocl_type_name<uint32_t>();
        case ov::element::Type_t::i64:
            return get_ocl_type_name<int64_t>();
        case ov::element::Type_t::f16:
            return "half";
        case ov::element::Type_t::f32:
            return get_ocl_type_name<float>();
        default:
            return "";
    }
}

}  // namespace ov::intel_gpu
