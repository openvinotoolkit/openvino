// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/runtime/layout.hpp"

#include <list>
#include <vector>
#include <algorithm>
#include "openvino/core/dimension.hpp"
#include "openvino/core/partial_shape.hpp"
#include "intel_gpu/runtime/utils.hpp"

#include "intel_gpu/runtime/debug_configuration.hpp"

namespace cldnn {
/* c++11 requires to have a definition in cpp file */
constexpr padding::DynamicDimsMask padding::EMPTY_MASK;

static inline bool check_redundant_1d_along_feature(layout const& l1, layout const& l2);
namespace {

std::vector<int32_t> convert_dimensions(const std::vector<int32_t>& sizes, const std::string& in_order, const std::string& out_order) {
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
    const auto& dims = get_dims();
    return dims[idx];
}

tensor::value_type layout::batch() const {
    const auto& dims = get_dims();
    const size_t dim_idx = 0;
    return dims[dim_idx];
}

tensor::value_type layout::feature() const {
    const auto& dims = get_dims();
    const size_t dim_idx = 1;
    return dims[dim_idx];
}

tensor::value_type layout::spatial(size_t spatial_idx) const {
    if (spatial_idx >= format.spatial_num() )
        return 1;
    const auto& dims = get_dims();
    const size_t dim_idx = (format::is_grouped(format) ? 3 : 2) + (format.spatial_num() - 1 - spatial_idx);
    return dims[dim_idx];
}

tensor::value_type layout::group() const {
    const auto& dims = get_dims();
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
    const auto& dims = get_dims();
    const size_t dim_idx = format::is_grouped(format) ? 1 : 0;

    return dims[dim_idx];
}

tensor::value_type layout::ifm() const {
    if (!format::is_weights_format(format)) {
        throw std::logic_error("[GPU] can't get IFM dimension for data layout");
    }
    const auto& dims = get_dims();
    const size_t dim_idx = format::is_grouped(format) ? 2 : 1;
    return dims[dim_idx];
}

std::vector<tensor::value_type> layout::get_dims() const {
    if (is_dynamic())
        throw std::runtime_error("[GPU] get_dims() is called for dynamic shape");

    std::vector<tensor::value_type> res;
    for (const auto& dim : size) {
        res.push_back(static_cast<tensor::value_type>(dim.get_length()));
    }

    if (res.size() < format.dimension())
        res.insert(res.end(), format.dimension() - res.size(), 1);

    return res;
}

std::vector<tensor::value_type> layout::get_padded_dims() const {
    OPENVINO_ASSERT(!is_dynamic() || has_upper_bound(), "[GPU] get_tensor() is called for dynamic shape without upper bound");
    ov::Shape shape = is_dynamic() ? size.get_max_shape() : size.to_shape();

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

    std::vector<tensor::value_type> res(dims.size());
    for (size_t i = 0; i < res.size(); i++) {
        res[i] = dims[i] + data_padding._upper_size[i] + data_padding._lower_size[i];
    }

    return res;
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

    const auto& t = get_tensor();
    return t.sizes(format);
}

std::vector<size_t> layout::get_dims_order() const {
    return format.dims_order();
}

std::string layout::to_string() const {
    std::stringstream s;
    s << "\n{\n"
      << "\tdata_type=" << ov::element::Type(data_type) << ";\n"
      << "\tformat=" << format.to_string() << ";\n"
      << "\tshape=" << size << ";\n"
      << "\tpad_l=[";
    std::copy(std::begin(data_padding._lower_size), std::end(data_padding._lower_size), std::ostream_iterator<tensor::value_type>(s, ", "));
    s << "];\n"
      << "\tpad_u=[";
    std::copy(std::begin(data_padding._upper_size), std::end(data_padding._upper_size), std::ostream_iterator<tensor::value_type>(s, ", "));
    s << "];\n"
      << "\tdyn_pad_dims=[" << data_padding._dynamic_dims_mask.to_string() << "];\n"
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

    if (data_padding.is_dynamic()) {
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

const ov::PartialShape& layout::get_partial_shape() const {
    return size;
}

ov::Shape layout::get_shape() const {
    return size.to_shape();
}

tensor layout::get_tensor() const {
    OPENVINO_ASSERT(!is_dynamic() || has_upper_bound(), "[GPU] get_tensor() is called for dynamic shape without upper bound");
    ov::Shape shape;
    if (is_dynamic() && has_upper_bound()) {
        for (const auto& dim : size) {
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

std::vector<tensor::value_type> layout::get_pitches() const {
    const auto& padded_dims = get_padded_dims();
    auto sizes = format_sizes(padded_dims, format);

    std::vector<tensor::value_type> pitches_fmt(sizes.size(), size_t(1));
    std::partial_sum(sizes.rbegin(), sizes.rend() - 1, pitches_fmt.rbegin() + 1, std::multiplies<tensor::value_type>());

    // reorder back to default format order
    auto pitches = tensor(format, pitches_fmt).sizes(format::get_default_format(format.dimension(),
                                                                                format::is_weights_format(format),
                                                                                format::is_grouped(format)));

    return pitches;
}


size_t layout::get_linear_offset(tensor element) const {
    auto default_fmt = format::get_default_format(format.dimension(), format::is_weights_format(format), format::is_grouped(format));

    std::vector<tensor::value_type> lower_sizes, upper_sizes;
    lower_sizes.assign(data_padding._lower_size.begin(), data_padding._lower_size.begin() + format.dimension());
    upper_sizes.assign(data_padding._upper_size.begin(), data_padding._upper_size.begin() + format.dimension());
    const auto& l_padd = tensor(default_fmt, lower_sizes, 0);
    const auto& u_padd = tensor(default_fmt, upper_sizes, 0);

    const auto& t = get_tensor();

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
    auto sizes = get_padded_dims();

    std::set<size_t> processed_dims;
    const auto& blocks = format.logic_block_sizes();

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
        !l1.data_padding && !l2.data_padding && l1.get_linear_size() == l2.get_linear_size())
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
        (!blocks1.empty() && l1.format.dims_order() != l2.format.dims_order()))
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

    auto l1_offset = l1.get_linear_offset();
    auto l2_offset = l2.get_linear_offset();
    if (l1_pitch == l2_pitch && l1_offset == l2_offset)
        return true;

    return false;
}

bool layout::identical(const layout& other) const {
    if (is_dynamic() || other.is_dynamic())
        return false;
    bool ret = (*this == other);
    if (ret && this->format == cldnn::format::custom) {
        ret &= (this->format.traits().block_sizes == other.format.traits().block_sizes);
    }
    return ret;
}

ov::PartialShape layout::transform(const ov::PartialShape& pshape, const cldnn::format& old_fmt, const cldnn::format& new_fmt) {
    if (old_fmt == new_fmt) {
        return pshape;
    }

    // shortcut for transform to max rank default fmt which is used in fill_shape_info_data to improve perf
    if (format::is_default_format(old_fmt) && new_fmt == format::bfvuwzyx) {
        ov::PartialShape res = pshape;
        // This part is necessary because we treat 3D layouts as "bfy", not as "bfx".
        if (res.size() == 3)
            res.push_back(1);
        size_t num_to_insert = layout::max_rank() - res.size();
        size_t pos_to_insert = std::min<size_t>(res.size(), 2);
        res.insert(res.begin() + pos_to_insert, num_to_insert, 1);

        return res;
    }

    int32_t default_size = -1;
    std::vector<int32_t> dims;
    dims.reserve(pshape.size());
    for (const auto& dim : pshape) {
        dims.push_back(static_cast<int32_t>(dim.get_length()));
    }

    const cldnn::format default_fmt = cldnn::format::bfvuwzyx;
    const auto& old_sizes = convert_dimensions(dims, old_fmt.order(), default_fmt.internal_order()); // convert to internal order (bfxyzwuv)

    const auto& val_order = default_fmt.internal_order();
    const auto& new_order = new_fmt.internal_order();
    const auto& new_traits = new_fmt.traits();

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

    ov::PartialShape res;
    res.reserve(new_dims.size());
    for (size_t i = 0; i < new_dims.size(); i++) {
        res.push_back(ov::Dimension(new_dims[i]));
    }
    return res;
}

// Check a reorder is 1d along feature axis. Or feature size fits to inner block size of feature axis
static inline bool check_redundant_1d_along_feature(layout const& l1, layout const& l2) {
    // No padding, double blocked format and different data_type
    if ((l1.get_linear_size() == l2.get_linear_size()) && !l1.data_padding && !l2.data_padding &&
        !format::is_multi_blocked(l1.format) && !format::is_multi_blocked(l2.format) &&
        l2.data_type == l1.data_type && l2.count() == l1.count()) {
        auto l1_inner_blk = format::is_single_blocked(l1.format) ? l1.format.traits().block_sizes.at(0).second : 1;
        auto l2_inner_blk = format::is_single_blocked(l2.format) ? l2.format.traits().block_sizes.at(0).second : 1;
        auto max_inner_blk = std::max(l1_inner_blk, l2_inner_blk);
        auto has_batch_block = format::is_single_blocked(l1.format) && l1.format.traits().block_sizes.at(0).first == 0;
        has_batch_block |= format::is_single_blocked(l2.format) && l2.format.traits().block_sizes.at(0).first == 0;

        auto is_1x1_spatial = [](layout const& l) {
            for (size_t i = 0; i < l.get_spatial_rank(); ++i) {
                if (l.spatial(i) > 1)
                    return false;
            }
            return true;
        };

        if ((static_cast<size_t>(l2.feature()) == l1.count() ||
            (max_inner_blk > 1 && !has_batch_block && l1.batch() == l2.batch() &&
            l1.get_dims_order()[0] == 0 && l2.get_dims_order()[0] == 0 && is_1x1_spatial(l1) && is_1x1_spatial(l2))) &&
            l2.feature() == l1.feature() && (l2.feature() % max_inner_blk == 0)) {
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
