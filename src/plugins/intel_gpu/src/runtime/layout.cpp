// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/runtime/layout.hpp"

#include <list>
#include <vector>
#include <algorithm>

namespace cldnn {
static inline bool check_redundant_1d_along_feature(layout const& l1, layout const& l2);
namespace {

std::vector<int32_t> convert_dimensions(const std::vector<int32_t>& sizes, std::string in_order, std::string out_order) {
    std::vector<int32_t> new_sizes(out_order.size(), {-1});
    for (size_t out_idx = 0; out_idx < out_order.size(); ++out_idx) {
        auto channel = out_order[out_idx];
        if (channel == '?')
            continue;

        auto in_idx = in_order.find(channel);
        if (in_idx != in_order.npos) {
            if (in_idx < sizes.size())
                new_sizes[out_idx] = sizes[in_idx];
        }
    }
    return new_sizes;
}

}  // namespace

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
    std::vector<tensor::value_type> res;
    for (auto dim : shape) {
        res.push_back(static_cast<tensor::value_type>(dim));
    }

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
        case format::fbyx:
            return format::ioyx;
        case format::fyxb:
            return format::iyxo;
        case format::byxf:
            return format::oyxi;
        case format::byfx:
            return format::oyix;
        case format::bxfy:
            return format::oxiy;
        case format::yxfb:
            return format::yxio;
        case format::bfzyx:
            return is_grouped ? format::goiyx : format::oizyx;
        case format::bfwzyx: {
            if (!is_grouped)
                throw std::runtime_error("Invalid conversion of data format to weights format. bfwzyx can't be non-grouped as 4D spatials are not supported");
            return format::goizyx;
        }
        case format::b_fs_yx_fsv4:
            return format::o_is_yx_isv4;
        case format::b_fs_yx_fsv16:
            return format::o_is_yx_isv16;
        case format::bs_fs_fsv8_bsv8:
            return format::os_i_osv8__ai8;
        case format::b_fs_yx_32fp:
            return format::os_is_yx_osv32_isv32p;
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
      << "\tdata_type=" << ov::element::Type(data_type) << ";\n"
      << "\tformat=" << format.to_string() << ";\n"
      << "\tshape=" << size << ";\n"
      << "\tpad_l=" << data_padding.lower_size().to_string() << ";\n"
      << "\tpad_u=" << data_padding.upper_size().to_string() << ";\n"
      << "\tdyn_pad_dims" << data_padding.get_dynamic_pad_dims().to_string() << ";\n"
      << "}";
    return s.str();
}

std::string layout::to_short_string() const {
    std::stringstream s;
    auto dump_shape = [](std::stringstream& stream, const ov::PartialShape& shape) {
        if (shape.rank().is_dynamic()) {
            stream << "...";
        } else {
            for (size_t i = 0; i < shape.size(); i++) {
                stream << shape[i];
                if (i != shape.size() - 1)
                    stream << "x";
            }
        }
    };

    s << ov::element::Type(data_type) << ":" << format.to_string() << ":";
    dump_shape(s, size);

    if (data_padding.get_dynamic_pad_dims() != tensor(0)) {
        s << ":dyn_pad_dims";
    } else {
        if (data_padding)
            s << ":pad";
        else
            s << ":nopad";
    }

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
    OPENVINO_ASSERT(!is_dynamic() || has_upper_bound(), "[GPU] get_tensor() is called for dynamic shape without upper bound");
    ov::Shape shape;
    if (is_dynamic() && has_upper_bound()) {
        for (auto dim : size) {
                shape.push_back(dim.get_max_length());
        }
    } else {
        shape = size.to_shape();
    }

    std::vector<tensor::value_type> dims;
    for (auto dim : shape) {
        dims.push_back(static_cast<tensor::value_type>(dim));
    }

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
    if (is_dynamic() && !has_upper_bound()) {
            throw std::runtime_error("[GPU] get_buffer_size() is called for dynamic shape");
    }

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

    return total;
}

layout layout::with_padding(padding const& padd) const {
    layout ret = *this;
    ret.data_padding = padd;
    return ret;
}

// tells whether l1 and l2 can be reinterpreted to each other without need of reordering
// note: layouts can only be considered identical if data size described by both layouts match (so no data are genereted
// nor dropped) note: if layouts describe two buffers with different size, consider them not to be identical even if
// smaller buffer can be considered to hold subsequence of larger buffer,
//       this behavior is required to force buffer allocation for smaller buffer which, currently, should always be
//       performed
bool layout::compatible(const layout& other) const {
    auto& l1 = *this;
    auto& l2 = other;
    const auto& l1_pad = l1.data_padding;
    const auto& l2_pad = l2.data_padding;

    if (l1.is_dynamic() || l2.is_dynamic())
        return false;

    auto l1_size = l1.get_tensor();
    auto l2_size = l2.get_tensor();
    if (l1 == l2)
        return true;
    if (check_redundant_1d_along_feature(l1, l2))
        return true;
    if (l1.data_type != l2.data_type)
        return false;
    // Reorders between bfyx, bfzyx, bfwzyx can be reinterpeted as reshape when
    // there is no padding and both hold same number of elements.
    if (format::is_default_format(l1.format) && format::is_default_format(l2.format) &&
        !l1_pad && !l2_pad && l1.get_linear_size() == l2.get_linear_size())
        return true;
    if (l1_size != l2_size)
        return false;
    if (l1.get_linear_size() != l2.get_linear_size())
        return false;

    auto check_format = [&l1, &l2](cldnn::format format) {
        return (l1.format == format && l2.format != format) ||
               (l2.format == format && l1.format != format);
    };

    const auto& blocks1 = format::block_sizes(l1.format);
    const auto& blocks2 = format::block_sizes(l2.format);

    // TODO: Relax restrictions below
    if (blocks1 != blocks2 ||
        (!blocks1.empty() && format::traits(l1.format)._order != format::traits(l2.format)._order))
        return false;

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
        check_format(format::bs_fs_yx_bsv16_fsv32) ||
        check_format(format::bs_fs_zyx_bsv16_fsv32) ||
        check_format(format::bs_fs_zyx_bsv16_fsv16) ||
        check_format(format::bs_fs_zyx_bsv32_fsv16) ||
        check_format(format::bs_fs_zyx_bsv32_fsv32))
        return false;

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
        return true;

    return false;
}

bool layout::identical(const layout& other) const {
    if (is_dynamic() || other.is_dynamic())
        return false;
    return *this == other;
}

ov::PartialShape layout::transform(const ov::PartialShape& pshape, cldnn::format old_fmt, cldnn::format new_fmt) {
    if (old_fmt == new_fmt) {
        return pshape;
    }

    int32_t default_size = -1;
    auto shape = pshape.to_shape();
    std::vector<int32_t> dims;
    for (auto dim : shape) {
        dims.push_back(static_cast<int32_t>(dim));
    }

    const cldnn::format default_fmt = cldnn::format::bfvuwzyx;
    auto old_sizes = convert_dimensions(dims, old_fmt.order(), default_fmt.internal_order()); // convert to internal order (bfxyzwuv)

    auto val_order = default_fmt.internal_order();
    auto new_order = new_fmt.internal_order();
    const auto& new_traits = format::traits(new_fmt);

    std::vector<int32_t> new_sizes(old_sizes.size(), {default_size});

    static const std::map<char, char> flatten_mapping = {
        { 'v', 'u'},
        { 'u', 'w'},
        { 'w', 'z'},
        { 'z', 'y'}
    };

    for (size_t i = 0; i < default_fmt.order().size(); i++) {
        auto target_dim = val_order[i]; //bfxywzuv
        while (!new_traits.has_dimension(target_dim)) {
            if (flatten_mapping.find(target_dim) != flatten_mapping.end()) {
                target_dim = flatten_mapping.at(target_dim);
            } else {
                target_dim = new_fmt.order().back();
            }
        }

        auto new_pos = new_order.find(target_dim);
        if (new_pos != std::string::npos) {
            if (new_sizes[new_pos] == -1) {
                new_sizes[new_pos] = old_sizes[i];
            } else {
                new_sizes[new_pos] *= old_sizes[i];
            }
        }
    }

    for (size_t i = 0; i < new_order.size(); i++) {
        auto c = new_order[i]; //bfxywz
        if (c == '?')
            continue;
        if (new_sizes[i] == -1) {
            new_sizes[i] = 1;
        }
    }

    auto new_dims = convert_dimensions(new_sizes, default_fmt.internal_order(), new_fmt.order());
    for (int idx = static_cast<int>(new_dims.size() - 1); idx >= 0; idx--) {
        if (new_dims[idx] == -1)
            new_dims.erase((new_dims.begin() + idx));
        else if (new_dims[idx] < 0)
            new_dims[idx] *= -1;
    }

    ov::Shape new_shape(new_dims.begin(), new_dims.end());
    return ov::PartialShape(new_shape);
}

// Check a reorder is 1d along feature axis. Or feature size fits to inner block size of feature axis
static inline bool check_redundant_1d_along_feature(layout const& l1, layout const& l2) {
    // No padding, double blocked format and different data_type
    if (!l1.data_padding && !l2.data_padding && !format::is_multi_blocked(l1.format) && !format::is_multi_blocked(l2.format) &&
        l2.data_type == l1.data_type && l2.count() == l1.count()) {
        auto l1_inner_blk = format::is_single_blocked(l1.format) ? format::traits(l1.format).block_sizes.at(0).second : 1;
        auto l2_inner_blk = format::is_single_blocked(l2.format) ? format::traits(l2.format).block_sizes.at(0).second : 1;
        auto max_inner_blk = std::max(l1_inner_blk, l2_inner_blk);
        if (static_cast<size_t>(l2.feature()) == l1.count() && l2.feature() == l1.feature() &&
           (l2.feature() % max_inner_blk == 0)) {
            return true;
        }

        // Acceptable if a feature size of l2 'byxf' fits to l1's inner block size of 'b_fs_yx_fsv'
        if ((l2.format == format::byxf && (l1.format == format::b_fs_yx_fsv16 ||  l1.format == format::b_fs_yx_fsv32) &&
            l2.feature() == l1_inner_blk) ||
            (l1.format == format::byxf && (l2.format == format::b_fs_yx_fsv16 ||  l2.format == format::b_fs_yx_fsv32) &&
            l1.feature() == l2_inner_blk)) {
            // each spatial axis should be same
            for (size_t i = 0 ; i < l2.get_spatial_rank() ; i++) {
                if (l2.spatial(i) != l1.spatial(i))
                    return false;
            }

            return true;
        }
    }

    return false;
}

}  // namespace cldnn
