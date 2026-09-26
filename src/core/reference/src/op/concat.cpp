// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/reference/concat.hpp"

#include <cstring>

namespace ov::reference {
namespace {
std::vector<size_t> calculate_shape_sizes(const std::vector<Shape>& in_shapes) {
    std::vector<size_t> sizes;
    sizes.reserve(in_shapes.size());
    std::transform(begin(in_shapes), end(in_shapes), std::back_inserter(sizes), [](auto&& shape) {
        return shape_size(shape);
    });
    return sizes;
}

// Per-input byte size = op(total_elements / steps), shared by all concat() overloads below.
template <class UnaryOp>
std::vector<size_t> calculate_sizes(const std::vector<Shape>& in_shapes, size_t steps, UnaryOp op) {
    auto sizes = calculate_shape_sizes(in_shapes);
    std::transform(sizes.begin(), sizes.end(), sizes.begin(), [steps, op](size_t total) {
        return op(total / steps);
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
    if (steps == 0) {
        return;
    }
    const auto sizes = calculate_sizes(in_shapes, steps, [elem_size](auto&& count) {
        return count * elem_size;
    });

    for (size_t step = 0; step < steps; ++step) {
        for (size_t in_index = 0; in_index < args.size(); ++in_index) {
            const size_t size = sizes[in_index];
            std::memcpy(out, args[in_index] + step * size, size);
            out += size;
        }
    }
}

void concat(const std::vector<const std::string*>& args,
            std::string* out,
            const std::vector<Shape>& in_shapes,
            const Shape& out_shape,
            int64_t concatenation_axis) {
    const auto steps = shape_size(out_shape.begin(), out_shape.begin() + concatenation_axis);
    if (steps == 0) {
        return;
    }
    const auto sizes = calculate_sizes(in_shapes, steps, [](auto&& count) {
        return count;
    });

    for (size_t step = 0; step < steps; ++step) {
        for (size_t in_index = 0; in_index < args.size(); ++in_index) {
            const size_t size = sizes[in_index];
            out = std::copy_n(args[in_index] + step * size, size, out);
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
    if (steps == 0) {
        return;
    }
    // bitwidth < 8 always here, so this stays small regardless of tensor size.
    const auto sizes = calculate_sizes(in_shapes, steps, [bitwidth](auto&& count) {
        return (count * bitwidth) / 8;
    });

    for (size_t step = 0; step < steps; ++step) {
        for (size_t in_index = 0; in_index < args.size(); ++in_index) {
            const size_t size = sizes[in_index];
            std::memcpy(out, args[in_index] + step * size, size);
            out += size;
        }
    }
}
}  // namespace ov::reference
