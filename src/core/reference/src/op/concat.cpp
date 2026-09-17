// Copyright (C) 2018-2026 Intel Corporation
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
    std::transform(begin(in_shapes), end(in_shapes), std::back_inserter(sizes), [](auto&& shape) {
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
    const auto steps = shape_size(out_shape.begin(), out_shape.begin() + concatenation_axis);
    const auto& shape_sizes = calculate_shape_sizes(in_shapes);

    size_t out_offset = 0;
    for (size_t step = 0; step < steps; ++step) {
        for (size_t in_index = 0; in_index < args.size(); ++in_index) {
            const size_t size = (shape_sizes[in_index] / steps) * elem_size;
            const size_t in_offset = step * size;
            std::memcpy(out + out_offset, args[in_index] + in_offset, size);

            out_offset += size;
        }
    }
}

void concat(const std::vector<const std::string*>& args,
            std::string* out,
            const std::vector<Shape>& in_shapes,
            const Shape& out_shape,
            int64_t concatenation_axis) {
    const auto steps = shape_size(out_shape.begin(), out_shape.begin() + concatenation_axis);
    const auto& shape_sizes = calculate_shape_sizes(in_shapes);

    size_t out_offset = 0;
    for (size_t step = 0; step < steps; ++step) {
        for (size_t in_index = 0; in_index < args.size(); ++in_index) {
            const size_t size = shape_sizes[in_index] / steps;
            const size_t in_offset = step * size;
            std::copy_n(std::next(args[in_index], in_offset), size, std::next(out, out_offset));

            out_offset += size;
        }
    }
}

void concat(const std::vector<const int8_t*>& args,
            int8_t* out,
            const std::vector<Shape>& in_shapes,
            const Shape& out_shape,
            int64_t concatenation_axis,
            size_t bitwidth) {
    const auto steps = shape_size(out_shape.begin(), out_shape.begin() + concatenation_axis);
    const auto& shape_sizes = calculate_shape_sizes(in_shapes);

    size_t out_offset = 0;
    for (size_t step = 0; step < steps; ++step) {
        for (size_t in_index = 0; in_index < args.size(); ++in_index) {
            const size_t num_elements = shape_sizes[in_index] / steps;
            // bitwidth < 8 always here, so this stays small regardless of tensor size.
            const size_t size = (num_elements * bitwidth) / 8;
            const size_t in_offset = step * size;
            std::memcpy(out + out_offset, args[in_index] + in_offset, size);

            out_offset += size;
        }
    }
}
}  // namespace reference
}  // namespace ov
