// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/runtime/layout.hpp"

#include <list>
#include <vector>
#include <algorithm>

namespace cldnn {

size_t layout::count() const {
    if (is_dynamic())
        throw std::runtime_error("[GPU] Count is called for dynamic shape");

    return ov::shape_size(size.to_shape());
}

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

format layout::get_default_format(size_t rank, bool is_weights, bool is_grouped) {
    auto default_fmt = cldnn::format::bfyx;
    if (is_weights) {
        if (is_grouped) {
            if (rank == 5) {
                default_fmt = cldnn::format::goiyx;
            } else if (rank == 6) {
                default_fmt = cldnn::format::goizyx;
            }
        } else {
            if (rank == 4) {
                default_fmt = cldnn::format::oiyx;
            } else if (rank == 5) {
                default_fmt = cldnn::format::oizyx;
            }
        }
    } else {
        if (rank == 5) {
            default_fmt = cldnn::format::bfzyx;
        } else if (rank == 6) {
            default_fmt = cldnn::format::bfwzyx;
        }
    }

    return default_fmt;
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

    auto default_fmt = get_default_format(format.dimension(), format::is_weights_format(format), format::is_grouped(format));
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
        case format::bs_xs_xsv8_bsv8:
            return format::os_i_osv8__ai8;
        default:
            throw std::invalid_argument("Unable to convert data format " + f.to_string() + " to weights format");
    }
}

layout layout::convert_to_weights_layout(bool is_grouped) const {
    auto fmt = to_weights_format(format, is_grouped);

    return layout{data_type, fmt, size};
}

std::vector<tensor::value_type> layout::get_ordered_dims() const {
    throw std::runtime_error("get_ordered_dims is not implemented yet");
}

std::vector<size_t> layout::get_dims_order() const {
    return format::traits(format)._order;
}

std::string layout::to_string() const {
    // TODO: Extend with format/data-type info
    std::stringstream s;
    s << format.to_string();
    s << size;
    return s.str();
}

bool layout::is_dynamic() const {
    return size.is_dynamic();
}

bool layout::is_static() const {
    return !is_dynamic();
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

// @brief Calculates position within buffer of the data element pointed by the provided tensor.
// element == { 0,0,0,0 } means first no-padding (i.e. data) element
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

/// Modify padding in layout
layout layout::with_padding(padding const& padd) const {
    layout ret = *this;
    ret.data_padding = padd;
    return ret;
}

tensor layout::get_tensor() const {
    if (is_dynamic())
        throw std::runtime_error("[GPU] get_tensor() is called for dynamic shape");

    auto shape = size.to_shape();
    std::vector<tensor::value_type> dims(shape.begin(), shape.end());

    auto default_fmt = get_default_format(format.dimension(), format::is_weights_format(format), format::is_grouped(format));
    if (default_fmt.dimension() > dims.size()) {
        dims.insert(dims.end(), default_fmt.dimension() - dims.size(), 1);
    }
    tensor t(default_fmt, dims);
    return t;
}

}  // namespace cldnn
