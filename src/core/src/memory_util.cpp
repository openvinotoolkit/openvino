// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/memory_util.hpp"

#include "openvino/core/shape_util.hpp"
#include "openvino/core/type/element_iterator.hpp"
#include "openvino/util/math_util.hpp"

namespace ov::util {
namespace {
size_t get_bit_memory_size(const element::Type& type, const size_t elements_count) {
    const auto bit_width = type.bitwidth();
    // Any 8 elements occupy exactly bit_width bytes, so ceil(elements_count * bit_width / 8) is taken in two steps
    const auto full_bytes = (elements_count / 8) * bit_width;
    const auto tail_bytes = ((elements_count % 8) * bit_width + 7) / 8;
    return full_bytes + tail_bytes;
}

size_t get_bit_elements_count(const element::Type& type, const size_t memory_size) {
    const auto bit_width = type.bitwidth();
    // Any bit_width bytes hold exactly 8 elements, so memory_size * 8 / bit_width is taken in two steps
    size_t elements_count;
    const auto tail_elements = ((memory_size % bit_width) * 8) / bit_width;
    OPENVINO_ASSERT(!mul_overflow<size_t>(memory_size / bit_width, 8, elements_count) &&
                    !add_overflow<size_t>(elements_count, tail_elements, elements_count));
    return elements_count;
}
}  // namespace

size_t get_memory_size(const element::Type& type, const size_t n) {
    if (n == 0) {
        return n;
    } else if (element::is_bit_type(type) || element::is_nibble_type(type)) {
        return get_bit_memory_size(type, n);
    } else {
        return (type.bitwidth() / 8) * n;
    }
}

std::optional<size_t> get_memory_size_safe(const element::Type& type, const size_t n) {
    if (auto s = type.bitwidth(); s >= 8) {
        return mul_overflow<size_t>(s / 8, n, s) ? std::nullopt : std::make_optional(s);
    } else {
        return std::make_optional(ov::util::get_memory_size(type, n));
    }
}

std::optional<size_t> get_memory_size_safe(const element::Type& type, const ov::Shape& shape) {
    auto byte_size = shape_size_safe(shape);
    return byte_size ? get_memory_size_safe(type, *byte_size) : byte_size;
}

size_t get_elements_capacity(const element::Type& type, const size_t memory_size) {
    if (type.bitwidth() == 0) {
        return 0;
    } else if (element::is_bit_type(type) || element::is_nibble_type(type)) {
        return get_bit_elements_count(type, memory_size);
    } else {
        const size_t bytes_per_element = type.bitwidth() / 8;
        return memory_size / bytes_per_element;
    }
}
}  // namespace ov::util
