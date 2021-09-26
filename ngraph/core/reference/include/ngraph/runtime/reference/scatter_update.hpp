// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstring>

#include "ngraph/shape.hpp"

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
    const auto size_before_axis = shape_size(Shape(data_shape.begin(), data_shape.begin() + axis));
    const auto size_after_axis = shape_size(Shape(data_shape.begin() + axis + 1, data_shape.end()));
    const auto axis_dim = data_shape[axis];

    const auto updates_axis = updates_shape[axis] == 1 ? axis + 1 : axis;
    const auto updates_axis_dim = updates_shape[updates_axis];
    const auto updates_axis_last_dim = updates_shape.size() - updates_axis == 1;
    const auto swaps = updates_axis_last_dim ? 1 : updates_shape.back();

    const auto updates_size_before_axis = shape_size(Shape(updates_shape.begin(), updates_shape.begin() + updates_axis));
    const auto updates_size_after_axis = shape_size(Shape(updates_shape.begin() + updates_axis + 1, updates_shape.end()));

    const auto updates_jump =  updates_size_after_axis == 1 ? 1 : updates_shape.back();
    const auto updates_move = updates_axis_last_dim ? updates_shape.back() : 1;

    std::vector<int64_t> updates_ids;
    const auto updates_space = num_of_updates * updates_size_before_axis * swaps;
    updates_ids.reserve(updates_space);
    std::vector<int64_t> data_ids;
    const auto data_space = num_of_updates * size_before_axis * size_after_axis;

    const auto max_iters = std::min(data_space, updates_space);
    data_ids.reserve(data_space);

    for (size_t k = 0; k < num_of_updates; ++k) {
        for (size_t i = 0; i < updates_size_before_axis; ++i) {
            const auto slice_idx = i * updates_axis_dim * updates_size_after_axis;
            for (size_t j = 0; j < swaps; ++j) {
                const auto seq_start_idx = slice_idx + j * updates_move;
                const auto updates_idx = seq_start_idx + k * updates_jump;
                updates_ids.push_back(updates_idx);
            }
        }
    }

    for (size_t k = 0; k < num_of_updates; ++k) {
        const auto ind_idx = *(indices + k);
        for (size_t i = 0; i < size_before_axis; ++i) {
            const auto slice_idx = i * axis_dim * size_after_axis;
            for (size_t j = 0; j < size_after_axis; ++j) {
                const auto sequence_start_idx = slice_idx + j;
                const auto element_idx = sequence_start_idx + (ind_idx * size_after_axis);
                data_ids.push_back(element_idx);
            }
        }
    }

    for (size_t i = 0; i < updates_ids.size(); ++i) {
        if (i == max_iters)
            break;
        out_buf[data_ids[i]] = updates[updates_ids[i]];
    }
}
}  // namespace reference
}  // namespace runtime
}  // namespace ngraph
