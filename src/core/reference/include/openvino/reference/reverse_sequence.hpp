// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cmath>
#include <numeric>

#include "openvino/reference/utils/coordinate_transform.hpp"

namespace ov {
namespace reference {
template <typename T, typename U>
void reverse_sequence(const T* arg,
                      T* out,
                      const Shape& arg_shape,
                      size_t batch_axis,
                      size_t sequence_axis,
                      const U* sequence_lengths) {
    const auto strides = row_major_strides(arg_shape);
    CoordinateTransformBasic input_transform(arg_shape);
    for (const Coordinate& in_coord : input_transform) {
        size_t batch_index = in_coord[batch_axis];
        auto orig_seq_index = static_cast<size_t>(sequence_lengths[batch_index]);

        OPENVINO_ASSERT(orig_seq_index <= arg_shape.at(sequence_axis),
                        "One of the elements of sequence lengths is greater than sequence axis dimension");

        if (orig_seq_index == 0) {
            orig_seq_index = 1;
        }

        size_t sequence_index = in_coord[sequence_axis] < orig_seq_index ? orig_seq_index - in_coord[sequence_axis] - 1
                                                                         : in_coord[sequence_axis];

        // make a copy of in_coord and update sequence_index
        Coordinate out_coord = in_coord;
        out_coord[sequence_axis] = sequence_index;

        const size_t in_idx = std::inner_product(in_coord.begin(), in_coord.end(), strides.begin(), size_t(0));
        const size_t out_idx = std::inner_product(out_coord.begin(), out_coord.end(), strides.begin(), size_t(0));
        out[out_idx] = arg[in_idx];
    }
}
}  // namespace reference
}  // namespace ov
