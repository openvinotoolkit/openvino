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

    constexpr int max_rank = 5;
    std::vector<int64_t> unsqueezed_shape(max_rank - data_shape.size(), 1);
    unsqueezed_shape.insert(unsqueezed_shape.end(), data_shape.begin(), data_shape.end());

    std::vector<int64_t> unsqueezed_starts(max_rank - data_shape.size(), 0);
    unsqueezed_starts.insert(unsqueezed_starts.end(), starts.begin(), starts.end());

    std::vector<int64_t> unsqueezed_stops(max_rank - data_shape.size(), 1);
    unsqueezed_stops.insert(unsqueezed_stops.end(), stops.begin(), stops.end());

    std::vector<int64_t> unsqueezed_steps(max_rank - data_shape.size(), 1);
    unsqueezed_steps.insert(unsqueezed_steps.end(), steps.begin(), steps.end());

    const auto unsqueezed_data_shape_strides = row_major_strides(unsqueezed_shape);
    auto dst_mem = out;

    for (int64_t i = unsqueezed_starts[0]; unsqueezed_steps[0] > 0 ? i < unsqueezed_stops[0] : i > unsqueezed_stops[0];
         i += unsqueezed_steps[0]) {
        for (int64_t j = unsqueezed_starts[1];
             unsqueezed_steps[1] > 0 ? j < unsqueezed_stops[1] : j > unsqueezed_stops[1];
             j += unsqueezed_steps[1]) {
            for (int64_t k = unsqueezed_starts[2];
                 unsqueezed_steps[2] > 0 ? k < unsqueezed_stops[2] : k > unsqueezed_stops[2];
                 k += unsqueezed_steps[2]) {
                for (int64_t l = unsqueezed_starts[3];
                     unsqueezed_steps[3] > 0 ? l < unsqueezed_stops[3] : l > unsqueezed_stops[3];
                     l += unsqueezed_steps[3]) {
                    for (int64_t m = unsqueezed_starts[4];
                         unsqueezed_steps[4] > 0 ? m < unsqueezed_stops[4] : m > unsqueezed_stops[4];
                         m += unsqueezed_steps[4]) {
                        std::vector<int64_t> coord{i, j, k, l, m};
                        const auto data_idx =
                            std::inner_product(coord.begin(), coord.end(), unsqueezed_data_shape_strides.begin(), 0);
                        const auto src_mem = data + data_idx * elem_size;
                        std::memcpy(dst_mem, src_mem, elem_size);
                        dst_mem += elem_size;
                    }
                }
            }
        }
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
