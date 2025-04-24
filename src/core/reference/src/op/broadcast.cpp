// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/reference/broadcast.hpp"

#include "openvino/reference/tile.hpp"

namespace ov {
namespace reference {
void broadcast(const char* arg,
               char* out,
               const Shape& in_shape,
               const Shape& out_shape,
               const AxisSet& broadcast_axes,
               size_t elem_size) {
    const auto output_rank = std::max(in_shape.size(), out_shape.size());
    Shape adjusted_in_shape = in_shape;
    for (const auto& axis : broadcast_axes) {
        if (adjusted_in_shape.size() < output_rank) {
            adjusted_in_shape.insert(adjusted_in_shape.begin() + axis, 1);
        }
    }
    Shape adjusted_out_shape = out_shape;
    adjusted_out_shape.insert(adjusted_out_shape.begin(), output_rank - adjusted_out_shape.size(), 1);
    std::vector<int64_t> repeats(output_rank);
    for (size_t i = 0; i < repeats.size(); ++i) {
        repeats[i] = adjusted_out_shape[i] / adjusted_in_shape[i];
    }

    return tile(arg, out, adjusted_in_shape, adjusted_out_shape, elem_size, repeats);
}
}  // namespace reference
}  // namespace ov
