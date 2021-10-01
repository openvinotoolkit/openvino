// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/runtime/reference/slice.hpp"

#include <cstring>

#include "ngraph/check.hpp"
#include "ngraph/coordinate_range.hpp"

namespace ngraph {
namespace runtime {
namespace reference {

void normalize_indices(std::vector<int64_t>& start_ind,
                       std::vector<int64_t>& stop_ind,
                       std::vector<int64_t>& steps,
                       const Shape& data_shape) {
    for (size_t i = 0; i < start_ind.size(); ++i) {
        start_ind[i] = start_ind[i] >= 0
                           ? std::min<int64_t>(start_ind[i], steps[i] < 0 ? data_shape[i] - 1 : data_shape[i])
                           : std::min<int64_t>(std::max<int64_t>(0, start_ind[i] + data_shape[i]), data_shape[i] - 1);
        stop_ind[i] = stop_ind[i] >= 0
                          ? std::min<int64_t>(stop_ind[i], data_shape[i])
                          : std::min<int64_t>(std::max<int64_t>(-1, stop_ind[i] + data_shape[i]), data_shape[i]);
    }
}

void slice_v8(const char* data,
              const Shape& data_shape,
              char* out,
              const Shape& out_shape,
              size_t elem_size,
              std::vector<int64_t>& starts,
              std::vector<int64_t>& stops,
              std::vector<int64_t>& steps,
              std::vector<int64_t>& axes) {
    normalize_indices(starts, stops, steps, data_shape);
    const auto in_data_strides = row_major_strides(data_shape);
    const auto out_data_strides = row_major_strides(out_shape);

    std::vector<int64_t> in_data_coord(starts);
    for (size_t out_idx = 0; out_idx < shape_size(out_shape); ++out_idx) {
        for (size_t i = 0; i < in_data_coord.size(); ++i) {
            in_data_coord[i] = starts[i] + (out_idx / out_data_strides[i] % out_shape[i]) * steps[i];
        }
        const auto in_idx = std::inner_product(in_data_coord.begin(), in_data_coord.end(), in_data_strides.begin(), 0);
        const auto in_mem = data + in_idx * elem_size;
        std::memcpy(out, in_mem, elem_size);
        out += elem_size;
    }
}

void slice(const char* arg,
           char* out,
           const Shape& arg_shape,
           const Coordinate& lower_bounds,
           const Coordinate& upper_bounds,
           const Strides& strides,
           const Shape& out_shape,
           size_t elem_size) {
    NGRAPH_SUPPRESS_DEPRECATED_START
    const CoordinateTransform input_transform(arg_shape, lower_bounds, upper_bounds, strides);

    const CoordinateTransform output_transform(out_shape);
    NGRAPH_CHECK(shape_size(input_transform.get_target_shape()) == shape_size(output_transform.get_target_shape()));

    auto dst_mem = out;

    for (auto range : coordinates::slice(arg_shape, lower_bounds, upper_bounds, strides)) {
        auto src_index = range.begin_index;
        for (size_t i = 0; i < range.element_number; src_index += range.step, ++i) {
            const auto src_mem = arg + src_index * elem_size;
            std::memcpy(dst_mem, src_mem, elem_size);
            std::advance(dst_mem, elem_size);
        }
    }
    NGRAPH_SUPPRESS_DEPRECATED_END
}
}  // namespace reference
}  // namespace runtime
}  // namespace ngraph
