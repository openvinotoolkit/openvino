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
                                       size_t num_of_elements) {
    const auto src_begin = std::next(arg, in_offset);
    const auto out_ptr = std::next(out, out_offset);
    std::copy_n(src_begin, num_of_elements, out_ptr);
}
}  // namespace

template <bool IS_STRING>
inline void copy_elements(const void* arg,
                          void* out,
                          size_t in_offset,
                          size_t out_offset,
                          size_t num_of_elements,
                          size_t elem_size) {
    return IS_STRING ? copy_single_input_elements(static_cast<const std::string*>(arg),
                                                  static_cast<std::string*>(out),
                                                  in_offset,
                                                  out_offset,
                                                  num_of_elements)
                     : copy_single_input_elements(static_cast<const char*>(arg),
                                                  static_cast<char*>(out),
                                                  in_offset,
                                                  out_offset,
                                                  num_of_elements,
                                                  elem_size);
}

void concat(const std::vector<const void*>& args,
            void* out,
            const std::vector<Shape>& in_shapes,
            const Shape& out_shape,
            int64_t concatenation_axis,
            const ov::element::Type& elem_type) {
    const auto steps = shape_size(out_shape.begin(), out_shape.begin() + concatenation_axis);
    const auto& shape_sizes = calculate_shape_sizes(in_shapes);

    const auto copy_func = elem_type == ov::element::string ? copy_elements<true> : copy_elements<false>;

    size_t out_offset = 0;
    for (size_t step = 0; step < steps; ++step) {
        for (size_t in_index = 0; in_index < args.size(); ++in_index) {
            const size_t size = shape_sizes[in_index] / steps;
            const size_t in_offset = step * size;

            copy_func(args[in_index], out, in_offset, out_offset, size, elem_type.size());

            out_offset += size;
        }
    }
}

}  // namespace reference
}  // namespace ov
