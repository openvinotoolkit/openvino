// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "openvino/core/shape.hpp"

namespace ov {
namespace reference {
void string_tensor_unpack(const std::string* data,
                          int32_t* out_begins,
                          int32_t* out_ends,
                          uint8_t* out_symbols,
                          const size_t element_count) {
    int32_t offset = 0;
    if (!std::is_pointer<decltype(out_symbols)>::value || out_symbols == nullptr) {
    std::cout << "\n\nout_symbols is not a valid pointer.\n\n" << std::endl;
    }
    for (size_t i = 0; i < element_count; ++i) {
        std::cout << "\n\nData[" << i << "]: \"" << data[i] << "\"\n\n";
        out_begins[i] = offset;
        out_symbols = std::copy(data[i].begin(), data[i].end(), out_symbols);
        offset += static_cast<int32_t>(data[i].length());
        out_ends[i] = offset;
    }
}
}  // namespace reference
}  // namespace ov
