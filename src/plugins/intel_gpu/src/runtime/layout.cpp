// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/runtime/layout.hpp"

#include <list>
#include <vector>
#include <algorithm>

namespace cldnn {

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
    if (!format::is_weights_format(format) && dims[dim_idx] != size.batch[0]) {
        throw std::runtime_error("batch mismatch: " + std::to_string(dims[dim_idx]) + " vs " + std::to_string(size.batch[0]));
    }
    return dims[dim_idx];
}

tensor::value_type layout::feature() const {
    auto dims = get_dims();
    const size_t dim_idx = 1;
    if (!format::is_weights_format(format) && dims[dim_idx] != size.feature[0]) {
        throw std::runtime_error("feature mismatch: " + std::to_string(dims[dim_idx]) + " vs " + std::to_string(size.feature[0]));
    }
    return dims[dim_idx];
}

tensor::value_type layout::spatial(size_t spatial_idx) const {
    if (spatial_idx >= format.spatial_num() )
        return 1;
    auto dims = get_dims();
    const size_t dim_idx = (format::is_grouped(format) ? 3 : 2) + (format.spatial_num() - 1 - spatial_idx);
    if (dims[dim_idx] != size.spatial[spatial_idx]) {
        throw std::runtime_error("spatials mismatch: " + std::to_string(dims[dim_idx]) + " vs " + std::to_string(size.spatial[spatial_idx]));
    }
    return dims[dim_idx];
}

tensor::value_type layout::group() const {
    auto dims = get_dims();
    if (!format::is_weights_format(format)) {
        throw std::logic_error("[GPU] can't get group dimension for data layout");
    }

    if (!format::is_grouped(format))
        return 1;

    if (dims[0] != size.group[0]) {
        throw std::runtime_error("groups mismatch: " + std::to_string(dims[0]) + " vs " + std::to_string(size.group[0]));
    }
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
    auto default_fmt = format::get_default_format(format.dimension(), format::is_weights_format(format), format::is_grouped(format));
    return size.sizes(default_fmt);
}

std::vector<tensor::value_type> layout::get_padded_dims() const {
    auto default_fmt = format::get_default_format(format.dimension(), format::is_weights_format(format), format::is_grouped(format));
    auto padded_size = size.add(data_padding.lower_size()).add(data_padding.upper_size());
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
    auto dims = size.sizes(format);
    auto fmt = to_weights_format(format, is_grouped);

    return layout{data_type, fmt, tensor{fmt, dims}};
}

std::vector<tensor::value_type> layout::get_ordered_dims() const {
    return size.sizes(format);
}

std::vector<size_t> layout::get_dims_order() const {
    return format::traits(format)._order;
}

std::string layout::to_string() const {
    // TODO: Extend with format/data-type info
    return format.to_string() + size.to_string();
}

size_t layout::count() const {
    if (is_dynamic())
        throw std::runtime_error("[GPU] Count is called for dynamic shape");

    return size.count();
}

bool layout::is_dynamic() const {
    return false;
}

bool layout::is_static() const {
    return true;
}

tensor layout::get_tensor() const {
    if (is_dynamic())
        throw std::runtime_error("[GPU] get_tensor() is called for dynamic shape");

    return size;
}

void layout::set_tensor(const tensor& size) {
    this->size = size;
}

tensor layout::get_buffer_size() const {
    return size.add(data_padding.lower_size()).add(data_padding.upper_size());
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

    if ((element.batch[0] < 0 && -element.batch[0] > l_padd.batch[0]) ||
        (element.feature[0] < 0 && -element.feature[0] > l_padd.feature[0]) ||
        (element.spatial[0] < 0 && -element.spatial[0] > l_padd.spatial[0]) ||
        (element.spatial[1] < 0 && -element.spatial[1] > l_padd.spatial[1]) ||
        (element.spatial[2] < 0 && -element.spatial[2] > l_padd.spatial[2]) ||
        (element.spatial[3] < 0 && -element.spatial[3] > l_padd.spatial[3]) ||
        (element.batch[0] >= size.batch[0] + u_padd.batch[0]) ||
        (element.feature[0] >= size.feature[0] + u_padd.feature[0]) ||
        (element.spatial[0] >= size.spatial[0] + u_padd.spatial[0]) ||
        (element.spatial[1] >= size.spatial[1] + u_padd.spatial[1]) ||
        (element.spatial[2] >= size.spatial[2] + u_padd.spatial[2]) ||
        (element.spatial[3] >= size.spatial[3] + u_padd.spatial[3]))
        throw std::invalid_argument("Requested to calculate linear offset for an element which lies outside of the buffer range.");

    auto padded_size = size + l_padd + u_padd;
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

}  // namespace cldnn
