// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <numeric>

#include "openvino/core/shape.hpp"
#include "utils/span.hpp"

namespace ov {
namespace reference {
template <typename T, typename U>
void gather(const T* const data,
            const U* const indices,
            T* out,
            const Shape& data_shape,
            const Shape& indices_shape,
            const Shape& out_shape,
            size_t axis,
            size_t batch_dims = 0) {
    // flattened shapes
    int64_t batch_size = shape_size(span(data_shape).subspan(0, batch_dims));
    int64_t outer_size = shape_size(span(data_shape).subspan(batch_dims, axis - batch_dims));
    int64_t indices_size = shape_size(span(indices_shape).subspan(batch_dims));
    int64_t inner_size = shape_size(span(data_shape).subspan(axis + 1));

    int64_t batch_data_mul = shape_size(span(data_shape).subspan(batch_dims));
    int64_t batch_out_mul = shape_size(span(out_shape).subspan(batch_dims));

    int64_t axis_size = data_shape[axis];
    int64_t data_offset, out_offset, idx;
    // for out of bound indices is filled with zeros
    std::fill(out, out + shape_size(out_shape), T{0});

    for (int64_t batch = 0; batch < batch_size; batch++)
        for (int64_t outer_idx = 0; outer_idx < outer_size; outer_idx++) {
            data_offset = batch_data_mul * batch + inner_size * axis_size * outer_idx;
            out_offset = batch_out_mul * batch + indices_size * inner_size * outer_idx;
            for (int64_t i = 0; i < indices_size; i++) {
                idx = indices[i + indices_size * batch];
                if (idx < 0)
                    idx += axis_size;
                // for out of bound values have to be filled with zeros
                if (idx >= axis_size || idx < 0)
                    continue;

                const auto src_begin = std::next(data, data_offset + inner_size * idx);
                const auto out_ptr = std::next(out, out_offset + inner_size * i);
                std::copy_n(src_begin, inner_size, out_ptr);
            }
        }
}

}  // namespace reference
}  // namespace ov
