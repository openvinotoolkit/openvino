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
}  // namespace cldnn
