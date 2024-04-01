// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/type/element_iterator.hpp"

namespace ov {

namespace element {

size_t get_byte_size(const element::Type& type, const size_t n) {
    auto byte_size = n * type.bitwidth();
    if (element::is_split_bit_type(type)) {
        constexpr size_t storage_unit_size = 24;
        byte_size += storage_unit_size - 1;
        byte_size /= storage_unit_size;
        byte_size *= 3;
    } else {
        constexpr size_t storage_unit_size = 8;
        byte_size += storage_unit_size - 1;
        byte_size /= storage_unit_size;
    }
    return byte_size;
}
}  // namespace element
}  // namespace ov
