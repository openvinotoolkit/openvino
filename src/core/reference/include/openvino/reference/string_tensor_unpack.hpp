// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "openvino/core/shape.hpp"

namespace ov {
namespace reference {
inline void string_tensor_unpack(const std::string* data,
                                 int32_t* out_begins,
                                 int32_t* out_ends,
                                 uint8_t* out_symbols,
                                 const size_t element_count) {
    int32_t offset = 0;
    for (size_t i = 0; i < element_count; ++i) {
        out_begins[i] = offset;
        out_symbols = std::copy(data[i].begin(), data[i].end(), out_symbols);
        offset += static_cast<int32_t>(data[i].length());
        out_ends[i] = offset;
    }
}
}  // namespace reference
}  // namespace ov
