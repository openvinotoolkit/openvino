// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/memory_util.hpp"

#include "openvino/core/shape_util.hpp"
#include "openvino/core/type/element_iterator.hpp"
#include "openvino/util/common_util.hpp"

namespace ov::util {

std::optional<size_t> get_memory_size_overflow(const element::Type& type, const size_t n) {
    auto byte_size = std::make_optional(n);

    if (n != 0) {
        if (element::is_split_bit_type(type)) {
            constexpr size_t storage_unit_size = 24;
            const size_t num_values = storage_unit_size / type.bitwidth();
            *byte_size /= num_values;
            *byte_size += static_cast<size_t>(*byte_size * num_values != n);
            *byte_size *= 3;
        } else if (element::is_bit_type(type) || element::is_nibble_type(type)) {
            constexpr size_t storage_unit_size = 8;
            const auto num_values = storage_unit_size / type.bitwidth();
            *byte_size /= num_values;
            *byte_size += static_cast<size_t>((*byte_size * num_values) != n);
        } else if (mul_overflow<size_t>(type.bitwidth() / 8, *byte_size, *byte_size)) {
            byte_size.reset();
        }
    }
    return byte_size;
}

std::optional<size_t> get_memory_size_overflow(const element::Type& type, const ov::Shape& shape) {
    auto byte_size = shape_size_overflow(shape);
    return byte_size ? get_memory_size_overflow(type, *byte_size) : byte_size;
}
}  // namespace ov::util
