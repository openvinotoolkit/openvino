// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/memory_util.hpp"

#include "openvino/core/shape_util.hpp"
#include "openvino/core/type/element_iterator.hpp"
#include "openvino/util/math_util.hpp"

namespace ov::util {
namespace {
// u3/u6 use a linear, LSB-first bit-stream layout (values may straddle byte boundaries).
size_t get_split_bit_memory_size(const element::Type& type, const size_t elements_count) {
    size_t used_bits;
    OPENVINO_ASSERT(!mul_overflow<size_t>(elements_count, type.bitwidth(), used_bits));
    return (used_bits + 7) / 8;
}

size_t get_split_elements_count(const element::Type& type, const size_t memory_size) {
    return (memory_size * 8) / type.bitwidth();
}

size_t get_bit_memory_size(const element::Type& type, const size_t shape_size) {
    const auto elements_per_byte = 8 / type.bitwidth();
    auto byte_size = shape_size / elements_per_byte;
    byte_size += static_cast<size_t>((byte_size * elements_per_byte) != shape_size);
    return byte_size;
}

size_t get_bit_elements_count(const element::Type& type, const size_t memory_size) {
    const size_t elements_per_byte = 8 / type.bitwidth();
    size_t elements_count;
    OPENVINO_ASSERT(!mul_overflow<size_t>(memory_size, elements_per_byte, elements_count));
    return elements_count;
}
}  // namespace

size_t get_memory_size(const element::Type& type, const size_t n) {
    if (n == 0) {
        return n;
    } else if (element::is_split_bit_type(type)) {
        return get_split_bit_memory_size(type, n);
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
    } else if (element::is_split_bit_type(type)) {
        return get_split_elements_count(type, memory_size);
    } else if (element::is_bit_type(type) || element::is_nibble_type(type)) {
        return get_bit_elements_count(type, memory_size);
    } else {
        const size_t bytes_per_element = type.bitwidth() / 8;
        return memory_size / bytes_per_element;
    }
}
}  // namespace ov::util
