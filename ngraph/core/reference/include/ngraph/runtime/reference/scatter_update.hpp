// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstring>

#include "ngraph/check.hpp"
#include "ngraph/coordinate_transform.hpp"
#include "ngraph/shape.hpp"
#include "utils/span.hpp"

namespace ngraph {
namespace runtime {
namespace reference {
template <typename dataType>
void scatter_update(const dataType* input_data,
                    const int64_t* indices,
                    const dataType* updates,
                    const int64_t axis,
                    dataType* out_buf,
                    const size_t elem_size,
                    const Shape& data_shape,
                    const Shape& indices_shape,
                    const Shape& updates_shape) {
    std::memcpy(out_buf, input_data, elem_size * shape_size(data_shape));

    const auto num_of_updates = shape_size(indices_shape);

    const auto data_size_before_axis = shape_size(Shape(data_shape.begin(), data_shape.begin() + axis));
    const auto data_size_after_axis = shape_size(Shape(data_shape.begin() + axis + 1, data_shape.end()));

    const auto updates_size_before_axis = shape_size(Shape(updates_shape.begin(), updates_shape.begin() + axis));
    const auto updates_size_after_axis = shape_size(Shape(updates_shape.begin() + axis + 1, updates_shape.end()));

    const auto data_ndim = data_shape.size();
    const auto updates_ndim = updates_shape.size();

    const auto data_last_dim = data_ndim - axis == 1;
    const auto updates_last_dim = updates_ndim - axis == 1;

    auto updates_start_idx_step = axis == 0 ? updates_shape.back() : updates_shape[axis];
    auto updates_step = updates_size_before_axis == 1 ? 1 : updates_size_after_axis;
    if (updates_last_dim) {
        updates_start_idx_step = 1;
        updates_step = updates_shape[axis];
    }

    const auto data_start_idx_step = data_last_dim ? 1 : data_size_after_axis * data_size_before_axis;
    const auto data_step = data_last_dim ? data_shape.back() : 1;

    const auto data_num_swaps = data_size_after_axis * data_size_before_axis;
    const auto udpate_num_swaps = updates_size_before_axis * updates_size_after_axis;
    const auto num_of_swaps = std::min(data_num_swaps, udpate_num_swaps);

    for (size_t i = 0; i < num_of_updates; ++i) {
        const auto start_data_idx = *(indices + i) * data_start_idx_step;
        const auto updates_start_idx = i * updates_start_idx_step;
        for (size_t update_num = 0; update_num < num_of_swaps; ++update_num) {
            const auto out_idx = start_data_idx + update_num * data_step;
            const auto updates_idx = updates_start_idx + update_num * updates_step;
            out_buf[out_idx] = updates[updates_idx];
        }
    }
}
}  // namespace reference
}  // namespace runtime
}  // namespace ngraph
