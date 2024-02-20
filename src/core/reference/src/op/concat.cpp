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
inline void copy_single_input_elements(const char* arg,
                                       char* out,
                                       size_t in_offset,
                                       size_t out_offset,
                                       size_t num_of_elements,
                                       size_t elem_size) {
    std::memcpy(&out[out_offset * elem_size], &arg[in_offset * elem_size], num_of_elements * elem_size);
}

inline void copy_single_input_elements(const std::string* arg,
                                       std::string* out,
                                       size_t in_offset,
                                       size_t out_offset,
                                       size_t num_of_elements,
                                       size_t elem_size) {
    const auto src_begin = std::next(arg, in_offset);
    const auto out_ptr = std::next(out, out_offset);
    std::copy_n(src_begin, num_of_elements, out_ptr);
}
}  // namespace

template <typename T>
void concat_common(const std::vector<const T*>& args,
                   T* out,
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

            copy_single_input_elements(args[in_index], out, in_offset, out_offset, size, elem_size);

            out_offset += size;
        }
    }
}

void concat(const std::vector<const char*>& args,
            char* out,
            const std::vector<Shape>& in_shapes,
            const Shape& out_shape,
            int64_t concatenation_axis,
            size_t elem_size) {
    reference::concat_common<char>(args, out, in_shapes, out_shape, concatenation_axis, elem_size);
}

void concat(const std::vector<const std::string*>& args,
            std::string* out,
            const std::vector<Shape>& in_shapes,
            const Shape& out_shape,
            int64_t concatenation_axis,
            size_t elem_size) {
    reference::concat_common<std::string>(args, out, in_shapes, out_shape, concatenation_axis, 1);
}
}  // namespace reference
}  // namespace ov
