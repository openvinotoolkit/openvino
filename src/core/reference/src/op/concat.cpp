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
    std::transform(begin(in_shapes), end(in_shapes), std::back_inserter(sizes), [](const Shape& shape) {
        return shape_size(shape);
    });
    return sizes;
}

void copy_elements(const char* arg,
                   char* out,
                   size_t in_offset,
                   size_t out_offset,
                   size_t num_of_elements,
                   size_t elem_size) {
    std::memcpy(out + (out_offset * elem_size), arg + (in_offset * elem_size), num_of_elements * elem_size);
}

void copy_string_elements(const char* arg,
                          char* out,
                          size_t in_offset,
                          size_t out_offset,
                          size_t num_of_elements,
                          size_t) {
    const auto src_begin = std::next(reinterpret_cast<const std::string*>(arg), in_offset);
    const auto out_ptr = std::next(reinterpret_cast<std::string*>(out), out_offset);
    std::copy_n(src_begin, num_of_elements, out_ptr);
}
}  // namespace

void concat(const std::vector<const char*>& args,
            char* out,
            const std::vector<Shape>& in_shapes,
            const Shape& out_shape,
            int64_t concatenation_axis,
            size_t elem_size,
            const ov::element::Type& elem_type) {
    const auto steps = shape_size(out_shape.begin(), out_shape.begin() + concatenation_axis);
    const auto& shape_sizes = calculate_shape_sizes(in_shapes);

    const auto copy_func = elem_type == ov::element::string ? copy_string_elements : copy_elements;

    // Sub-byte element types pack multiple values per byte.  The shape_sizes
    // and in_offset arithmetic below are in *elements*, but copy_elements
    // works in bytes (num_of_elements * elem_size, where elem_size == 1 for
    // all sub-byte types).  Divide element counts by the packing factor so
    // the memcpy length and source/destination byte offsets are correct.
    size_t elem_pack = 1;
    if (elem_type == ov::element::u2)
        elem_pack = 4;
    else if (elem_type == ov::element::u4 || elem_type == ov::element::i4)
        elem_pack = 2;

    size_t out_offset = 0;
    for (size_t step = 0; step < steps; ++step) {
        for (size_t in_index = 0; in_index < args.size(); ++in_index) {
            const size_t size = shape_sizes[in_index] / steps;
            const size_t in_offset = step * size / elem_pack;
            const size_t copy_size = size / elem_pack;
            copy_func(args[in_index], out, in_offset, out_offset, copy_size, elem_size);

            out_offset += copy_size;
        }
    }
}
}  // namespace reference
}  // namespace ov
