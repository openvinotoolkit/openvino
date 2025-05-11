// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/reference/tile.hpp"

#include <algorithm>
#include <cstdio>

namespace ov {
namespace reference {

/**
 * @brief Reference implementation of Tile operator
 *
 * @param arg        Pointer to input data.
 * @param out        Pointer to output data.
 * @param in_shape   Input data shape.
 * @param out_shape  Output data shape.
 * @param elem_size  Single data element size im bytes.
 * @param repeats    Vector with repeats values for axes (same rank as out_shape).
 */
void tile(const char* arg,
          char* out,
          const Shape& in_shape,
          const Shape& out_shape,
          const size_t elem_size,
          const std::vector<int64_t>& repeats) {
    if (std::any_of(repeats.begin(), repeats.end(), [](int64_t repeat) {
            return repeat == 0;
        })) {
        return;
    }

    decltype(arg) copy_from;
    typename std::decay<decltype(*in_shape.begin())>::type block_size;
    typename std::decay<decltype(*repeats.begin())>::type num_repeats;

    auto in_shape_expanded = in_shape;
    in_shape_expanded.insert(in_shape_expanded.begin(), out_shape.size() - in_shape.size(), 1);
    const auto last_dim = in_shape_expanded.back();
    const auto pitches = row_major_strides(out_shape);

    std::vector<size_t> indices(in_shape_expanded.size() - 1, 0);
    auto axis = indices.size();

    // Copy and repeat data for innermost axis as many times as described in the repeats parameter
    while (axis <= indices.size()) {
        block_size = last_dim * elem_size;
        std::memcpy(out, arg, block_size);
        out += block_size;
        arg += block_size;

        copy_from = out - block_size;
        num_repeats = repeats.back() - 1;
        for (int64_t i = 0; i < num_repeats; ++i) {
            std::memcpy(out, copy_from, block_size);
            out += block_size;
        }

        // Copy and repeat data for other axes as many times as described in the repeats parameter
        while (axis-- != 0) {
            if (++indices[axis] != in_shape_expanded[axis]) {
                axis = indices.size();
                break;
            }
            indices[axis] = 0;

            auto pitch = pitches[axis] * in_shape_expanded[axis];
            block_size = pitch * elem_size;
            copy_from = out - block_size;
            num_repeats = repeats[axis] - 1;
            for (int64_t i = 0; i < num_repeats; ++i) {
                std::memcpy(out, copy_from, block_size);
                out += block_size;
            }
        }
    }
}
}  // namespace reference
}  // namespace ov
