// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/runtime/layout.hpp"

#include <list>
#include <vector>
#include <algorithm>

namespace cldnn {
namespace {
// pair.first tells whether l1 and l2 are absolutely identical
// pair.second tells whether l1 and l2 can be reinterpreted to each other without need of reordering
// note: layouts can only be considered identical if data size described by both layouts match (so no data are genereted
// nor dropped) note: if layouts describe two buffers with different size, consider them not to be identical even if
// smaller buffer can be considered to hold subsequence of larger buffer,
//       this behavior is required to force buffer allocation for smaller buffer which, currently, should always be
//       performed
std::pair<bool, bool> are_layouts_identical(layout const& l1, layout const& l2) {
    const auto& l1_pad = l1.data_padding;
    const auto& l2_pad = l2.data_padding;

    if (l1.is_dynamic() || l2.is_dynamic())
        return {false, false};

    auto l1_size = l1.get_tensor();
    auto l2_size = l2.get_tensor();
    int64_t offset_last_element_l1 = l1.get_linear_offset(l1_size - tensor{1});
    int64_t offset_last_element_l2 = l2.get_linear_offset(l2_size - tensor{1});
    if (l1 == l2)
        return {true, true};
    if (l1.data_type != l2.data_type)
        return {false, false};
    // Reorders between bfyx, bfzyx, bfwzyx can pe reinterpeted as reshape when
    // there is no padding and both hold same number of elements.
    if ((l1.format == format::bfyx || l1.format == format::bfzyx || l1.format == format::bfwzyx) &&
        (l2.format == format::bfyx || l2.format == format::bfzyx || l2.format == format::bfwzyx) && !l1_pad &&
        !l2_pad && l1.get_linear_size() == l2.get_linear_size())
        return {false, true};
    if (l1_size != l2_size)
        return {false, false};
    if (l1.get_linear_size() != l2.get_linear_size())
        return {false, false};

    auto check_format = [&l1, &l2](cldnn::format format) {
        return (l1.format == format && l2.format != format) ||
               (l2.format == format && l1.format != format);
    };

    if (check_format(format::b_fs_yx_fsv2) ||
        check_format(format::b_fs_yx_fsv4) ||
        check_format(format::fs_b_yx_fsv32) ||
        check_format(format::b_fs_yx_fsv16) ||
        check_format(format::b_fs_yx_fsv32) ||
        check_format(format::b_fs_zyx_fsv2) ||
        check_format(format::b_fs_zyx_fsv4) ||
        check_format(format::b_fs_zyx_fsv32) ||
        check_format(format::b_fs_zyx_fsv16) ||
        check_format(format::bs_fs_yx_bsv4_fsv4) ||
        check_format(format::bs_fs_yx_bsv8_fsv4) ||
        check_format(format::bs_fs_zyx_bsv8_fsv4) ||
        check_format(format::bs_fs_yx_bsv8_fsv2) ||
        check_format(format::bs_fs_zyx_bsv8_fsv2) ||
        check_format(format::bs_fs_yx_bsv4_fsv2) ||
        check_format(format::bs_fs_yx_bsv32_fsv16) ||
        check_format(format::bs_fs_yx_bsv32_fsv32) ||
        check_format(format::bs_fs_yx_bsv16_fsv16) ||
        check_format(format::bs_fs_zyx_bsv16_fsv32) ||
        check_format(format::bs_fs_zyx_bsv16_fsv16) ||
        check_format(format::bs_fs_zyx_bsv32_fsv16) ||
        check_format(format::bs_fs_zyx_bsv32_fsv32))
        return {false, false};

    // If data is actually 1d along f and dense, the layouts are identical
    if (l1.data_type == l2.data_type && l1_size == l2_size && !l1_pad && !l2_pad && l1_size.batch[0] == 1 &&
        ((l1.format.spatial_num() == 2 && l1_size.spatial[0] == 1 && l1_size.spatial[1] == 1) ||
        ((l1.format.spatial_num() == 3 && l1_size.spatial[0] == 1 && l1_size.spatial[1] == 1 && l1_size.spatial[2] == 1))) &&
        (offset_last_element_l1 + 1 == l1_size.feature[0] && offset_last_element_l2 + 1 == l2_size.feature[0]))
        return {false, true};

    auto l1_pitch = l1.get_pitches();
    auto l2_pitch = l2.get_pitches();

    // ignore pitches which will never be used (for dims with size == 1)
    for (size_t i = 0; i < tensor_dim_max; ++i)
        if (l1_size.raw[i] == 1)
            l1_pitch.raw[i] = 0;
    for (size_t i = 0; i < tensor_dim_max; ++i)
        if (l2_size.raw[i] == 1)
            l2_pitch.raw[i] = 0;

    auto l1_offset = l1.get_linear_offset();
    auto l2_offset = l2.get_linear_offset();
    if (l1_pitch == l2_pitch && l1_offset == l2_offset)
        return {false, true};

    return {false, false};
}

}  // namespace

// The definitions below are needed to follow ODR
// Otherwise statements like
//     optional_value ov = type_to_data_type<float>::value;
//     optional_value ov(type_to_data_type<float>::value);
// violate ODR and leads to undefined behavior
const data_types type_to_data_type<int8_t>::value;
const data_types type_to_data_type<uint8_t>::value;
const data_types type_to_data_type<int32_t>::value;
const data_types type_to_data_type<int64_t>::value;
const data_types type_to_data_type<half_t>::value;
const data_types type_to_data_type<float>::value;

size_t layout::get_rank() const {
    return format.dimension();
}

size_t layout::get_spatial_rank() const {
    return format.spatial_num();
}

tensor::value_type layout::get_dim(size_t idx) const {
    auto dims = get_dims();
    return dims[idx];
}

tensor::value_type layout::batch() const {
    auto dims = get_dims();
    const size_t dim_idx = 0;
    return dims[dim_idx];
}

tensor::value_type layout::feature() const {
    auto dims = get_dims();
    const size_t dim_idx = 1;
    return dims[dim_idx];
}

tensor::value_type layout::spatial(size_t spatial_idx) const {
    if (spatial_idx >= format.spatial_num() )
        return 1;
    auto dims = get_dims();
    const size_t dim_idx = (format::is_grouped(format) ? 3 : 2) + (format.spatial_num() - 1 - spatial_idx);
    return dims[dim_idx];
}

tensor::value_type layout::group() const {
    auto dims = get_dims();
    if (!format::is_weights_format(format)) {
        throw std::logic_error("[GPU] can't get group dimension for data layout");
    }

    if (!format::is_grouped(format))
        return 1;

    return dims[0];
}

tensor::value_type layout::ofm() const {
    if (!format::is_weights_format(format)) {
        throw std::logic_error("[GPU] can't get OFM dimension for data layout");
    }
    auto dims = get_dims();
    const size_t dim_idx = format::is_grouped(format) ? 1 : 0;

    return dims[dim_idx];
}

tensor::value_type layout::ifm() const {
    if (!format::is_weights_format(format)) {
        throw std::logic_error("[GPU] can't get IFM dimension for data layout");
    }
    auto dims = get_dims();
    const size_t dim_idx = format::is_grouped(format) ? 2 : 1;
    return dims[dim_idx];
}

std::vector<tensor::value_type> layout::get_dims() const {
    if (is_dynamic())
        throw std::runtime_error("[GPU] get_dims() is called for dynamic shape");
    auto shape = size.to_shape();
    std::vector<tensor::value_type> res(shape.begin(), shape.end());

    if (res.size() < format.dimension())
        res.insert(res.end(), format.dimension() - res.size(), 1);

    return res;
}

std::vector<tensor::value_type> layout::get_padded_dims() const {
    if (is_dynamic())
        throw std::runtime_error("[GPU] get_padded_dims() is called for dynamic shape");

    auto default_fmt = format::get_default_format(format.dimension(), format::is_weights_format(format), format::is_grouped(format));
    auto t = get_tensor();
    auto padded_size = t.add(data_padding.lower_size()).add(data_padding.upper_size());
    return padded_size.sizes(default_fmt);
}

static format to_weights_format(format f, bool is_grouped) {
    if (format::is_weights_format(f))
        return f;

    switch (f) {
        case format::bfyx:
            return format::oiyx;
        case format::fyxb:
            return format::iyxo;
        case format::byxf:
            return format::oyxi;
        case format::yxfb:
            return format::yxio;
        case format::bfzyx:
            return is_grouped ? format::goiyx : format::oizyx;
        case format::bfwzyx: {
            if (!is_grouped)
                throw std::runtime_error("Invalid conversion of data format to weights format. bfwzyx can't be non-grouped as 4D spatials are not supported");
            return format::goizyx;
        }
        case format::b_fs_yx_fsv16:
            return format::o_is_yx_isv16;
        case format::bs_xs_xsv8_bsv8:
            return format::os_i_osv8__ai8;
        default:
            throw std::invalid_argument("Unable to convert data format " + f.to_string() + " to weights format");
    }
}

layout layout::convert_to_weights_layout(bool is_grouped) const {
    auto fmt = to_weights_format(format, is_grouped);

    return layout{size, data_type, fmt};
}

std::vector<tensor::value_type> layout::get_ordered_dims() const {
    if (is_dynamic())
        throw std::runtime_error("[GPU] get_ordered_dims() is called for dynamic shape");

    auto t = get_tensor();
    return t.sizes(format);
}

std::vector<size_t> layout::get_dims_order() const {
    return format::traits(format)._order;
}

std::string layout::to_string() const {
    std::stringstream s;
    s << "\n{\n"
      << "\tdata_type=" << data_type_traits::name(data_type) << ";\n"
      << "\tformat=" << format.to_string() << ";\n"
      << "\tshape=" << size << ";\n"
      << "\tpad_l=" << data_padding.lower_size().to_string() << ";\n"
      << "\tpad_u=" << data_padding.upper_size().to_string() << ";\n"
      << "}";
    return s.str();
}

std::string layout::to_short_string() const {
    std::stringstream s;
    auto dump_shape = [](std::stringstream& stream, const ov::PartialShape& shape) {
        for (size_t i = 0; i < shape.size(); i++) {
            stream << shape[i];
            if (i != shape.size() - 1)
                stream << "x";
        }
    };

    s << data_type_traits::name(data_type) << ":" << format.to_string() << ":";
    dump_shape(s, size);
    if (data_padding)
        s << ":pad";
    else
        s << ":nopad";
    return s.str();
}

size_t layout::count() const {
    if (is_dynamic())
        throw std::runtime_error("[GPU] Count is called for dynamic shape");

    return ov::shape_size(size.to_shape());
}

bool layout::is_dynamic() const {
    return size.is_dynamic();
}

bool layout::is_static() const {
    return !is_dynamic();
}

ov::PartialShape layout::get_partial_shape() const {
    return size;
}

ov::Shape layout::get_shape() const {
    return size.to_shape();
}

tensor layout::get_tensor() const {
    if (is_dynamic())
        throw std::runtime_error("[GPU] get_tensor() is called for dynamic shape");

    auto shape = size.to_shape();
    std::vector<tensor::value_type> dims(shape.begin(), shape.end());

    auto rank = std::max(format.dimension(), dims.size());
    auto default_fmt = format::get_default_format(rank, format::is_weights_format(format), format::is_grouped(format));
    if (default_fmt.dimension() > dims.size()) {
        dims.insert(dims.end(), default_fmt.dimension() - dims.size(), 1);
    }

    while (dims.size() > default_fmt.dimension()) {
        dims.pop_back();
    }

    tensor t(default_fmt, dims);
    return t;
}

template<typename T>
T layout::get() const {
    static_assert(meta::always_false<T>::value, "Unexpected layout::get() template speciaization");
}

template<>
ov::PartialShape layout::get<ov::PartialShape>() const {
    return size;
}

void layout::set_tensor(const tensor& size) {
    auto sizes = format == format::any ? size.sizes() : size.sizes(format::get_default_format(format.dimension(),
                                                                                              format::is_weights_format(format),
                                                                                              format::is_grouped(format)));
    ov::Shape shape(sizes.begin(), sizes.end());
    this->size = ov::PartialShape(shape);
}

void layout::set_partial_shape(const ov::PartialShape& size) {
    this->size = size;
}

tensor layout::get_buffer_size() const {
    if (is_dynamic())
        throw std::runtime_error("[GPU] get_buffer_size() is called for dynamic shape");

    auto t = get_tensor();

    return t.add(data_padding.lower_size()).add(data_padding.upper_size());
}

tensor layout::get_pitches() const {
    auto sizes = get_buffer_size().sizes(format);

    std::vector<tensor::value_type> pitches(sizes.size(), tensor::value_type(1));
    std::partial_sum(sizes.rbegin(), sizes.rend() - 1, pitches.rbegin() + 1, std::multiplies<tensor::value_type>());
    return {format, pitches};
}

size_t layout::get_linear_offset(tensor element) const {
    auto l_padd = data_padding.lower_size();
    auto u_padd = data_padding.upper_size();

    auto t = get_tensor();

    if ((element.batch[0] < 0 && -element.batch[0] > l_padd.batch[0]) ||
        (element.feature[0] < 0 && -element.feature[0] > l_padd.feature[0]) ||
        (element.spatial[0] < 0 && -element.spatial[0] > l_padd.spatial[0]) ||
        (element.spatial[1] < 0 && -element.spatial[1] > l_padd.spatial[1]) ||
        (element.spatial[2] < 0 && -element.spatial[2] > l_padd.spatial[2]) ||
        (element.spatial[3] < 0 && -element.spatial[3] > l_padd.spatial[3]) ||
        (element.batch[0] >= t.batch[0] + u_padd.batch[0]) ||
        (element.feature[0] >= t.feature[0] + u_padd.feature[0]) ||
        (element.spatial[0] >= t.spatial[0] + u_padd.spatial[0]) ||
        (element.spatial[1] >= t.spatial[1] + u_padd.spatial[1]) ||
        (element.spatial[2] >= t.spatial[2] + u_padd.spatial[2]) ||
        (element.spatial[3] >= t.spatial[3] + u_padd.spatial[3]))
        throw std::invalid_argument("Requested to calculate linear offset for an element which lies outside of the buffer range.");

    auto padded_size = t + l_padd + u_padd;
    auto padded_element = element + l_padd;

    return padded_size.get_linear_offset(padded_element, format);
}

/// @brief Get aligned linear size calculated as multiplication of all elements.
size_t layout::get_linear_size() const {
    auto sizes = get_buffer_size().sizes();

    std::set<size_t> processed_dims;
    const auto& blocks = format.block_sizes();
    for (size_t i = 0; i < blocks.size(); i++) {
        if (processed_dims.count(blocks[i].first))
            continue;

        auto block_axis = blocks[i].first;
        auto block_size = blocks[i].second;

        for (size_t j = i + 1; j < blocks.size(); j++) {
            if (blocks[j].first != block_axis)
                continue;

            block_size *= blocks[j].second;
        }

        sizes[block_axis] = align_to(sizes[block_axis], block_size);
        processed_dims.insert(block_axis);
    }

    if (this->format == cldnn::format::os_is_yx_isa8_osv8_isv4 && (!(is_aligned_to(sizes[0], 8)) || !(is_aligned_to(sizes[1], 32)))) {
        sizes[0] = align_to(sizes[0], 8);
        sizes[1] = align_to(sizes[1], 32);
    } else if (this->format == cldnn::format::os_is_yx_isa8_osv16_isv4 && (!(is_aligned_to(sizes[0], 16)) || !(is_aligned_to(sizes[1], 32)))) {
        sizes[0] = align_to(sizes[0], 16);
        sizes[1] = align_to(sizes[1], 32);
    } else if (this->format == cldnn::format::os_is_yx_isa8_osv8_isv4_swizzled_by_4 && (!(is_aligned_to(sizes[0], 32)) || !(is_aligned_to(sizes[1], 32)))) {
        sizes[0] = align_to(sizes[0], 32);
        sizes[1] = align_to(sizes[1], 32);
    } else if (this->format == cldnn::format::is_o32_yx_isv32_swizzled_by_4 && (!is_aligned_to(sizes[1], 32) || !(is_aligned_to(sizes[0], 32)))) {
        sizes[0] = align_to(sizes[0], 32);
        sizes[1] = align_to(sizes[1], 32);
    } else if (this->format == cldnn::format::os_is_y_x8_osv8_isv4 || this->format == cldnn::format::os_is_y_x8_osv8_isv4_swizzled_by_4) {
        sizes[1] = align_to(sizes[1], 4);
        sizes[0] = align_to(sizes[0], 8);
        sizes[2] = align_to(sizes[2], 8);
    } else if (this->format == cldnn::format::b_fs_yx_32fp) {
        sizes[1] = align_to(sizes[1], 32);
    } else if (this->format == cldnn::format::os_is_yx_osv32_isv32p) {
        sizes[0] = align_to(sizes[0], 32);
        sizes[1] = align_to(sizes[1], 32);
    } else if (this->format == cldnn::format::image_2d_rgba) {
        sizes[1] = 4;
    } else if (this->format == cldnn::format::gs_oi_yxs_gsv4_yxsv4 ||
                this->format == cldnn::format::gs_oi_yxs_gsv16_yxsv4 ||
                this->format == cldnn::format::gs_oi_yxs_gsv32_yxsv4) {
        sizes[3] = align_to(sizes[2] * sizes[3], 4);
        sizes[2] = 1;
    } else if (this->format == cldnn::format::os_iyx_osv32__ai32 && !is_aligned_to(sizes[1], 32)) {
        sizes[1] = align_to(sizes[1], 32);
    } else if ((this->format == cldnn::format::iy_xs_os_xsv2_osv8__ao32 ||
                this->format == cldnn::format::iy_xs_os_xsv2_osv16__ao32 ||
                this->format == cldnn::format::giy_xs_os_xsv2_osv8__ao32 ||
                this->format == cldnn::format::giy_xs_os_xsv2_osv16__ao32) && !is_aligned_to(sizes[0], 32))  {
        sizes[0] = align_to(sizes[0], 32);
        sizes[3] = align_to(sizes[2] * sizes[3], 2);
        sizes[2] = 1;
    } else if (this->format == cldnn::format::i_yxs_os_yxsv2_osv16 || this->format == cldnn::format::gi_yxs_os_yxsv2_osv16) {
        sizes[3] = align_to(sizes[2] * sizes[3], 2);
        sizes[2] = 1;
    } else if (this->format == cldnn::format::os_i_yxs_osv4_yxsv4) {
        sizes[3] = align_to(sizes[2] * sizes[3], 4);
        sizes[2] = 1;
    }
    size_t total = std::accumulate(
        sizes.begin(),
        sizes.end(),
        static_cast<size_t>(1),
        std::multiplies<size_t>());

    return (this->data_type == data_types::bin) ? ceil_div(total, 32) : total;
}

layout layout::with_padding(padding const& padd) const {
    layout ret = *this;
    ret.data_padding = padd;
    return ret;
}

bool layout::compatible(const layout& other) const {
    return are_layouts_identical(*this, other).second;
}

bool layout::identical(const layout& other) const {
    return are_layouts_identical(*this, other).first;
}

}  // namespace cldnn
