// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/reference/concat.hpp"

#include <cstring>

namespace ov {
namespace reference {
namespace {
std::vector<size_t> calculate_shape_sizes(const std::vector<Shape>& in_shapes) {
    std::vector<size_t> sizes;
    sizes.reserve(in_shapes.size());
    std::transform(begin(in_shapes), end(in_shapes), std::back_inserter(sizes), [](const Shape& shape) {
        return shape_size(shape);
    });
    return sizes;
}
}  // namespace

void concat(const std::vector<const char*>& args,
            char* out,
            const std::vector<Shape>& in_shapes,
            const Shape& out_shape,
            int64_t concatenation_axis,
            size_t elem_size) {
    size_t steps = 1;
    for (int i = 0; i < concatenation_axis; ++i) {
        steps *= out_shape[i];
    }

    const auto& shape_sizes = calculate_shape_sizes(in_shapes);

    size_t out_offset = 0;
    for (size_t step = 0; step < steps; ++step) {
        for (size_t in_index = 0; in_index < args.size(); ++in_index) {
            const size_t size = shape_sizes[in_index] / steps;
            const size_t in_offset = step * size;

            std::memcpy(&out[out_offset * elem_size], &args[in_index][in_offset * elem_size], size * elem_size);

            out_offset += size;
        }
    }
}
}  // namespace reference
}  // namespace ov
