// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "openvino/core/shape.hpp"

namespace ov {
namespace reference {
template <typename T_idx>
void string_tensor_pack(const T_idx* begins,
                        const T_idx* ends,
                        const uint8_t* symbols,
                        std::string* out,
                        const int64_t symbol_count,
                        const int64_t string_count) {
    std::vector<char> chars(symbols, symbols + symbol_count);
    for (int64_t i = 0; i < string_count; ++i) {
        out[i].assign(chars.begin() + begins[i], chars.begin() + ends[i]);
    }
}
}  // namespace reference
}  // namespace ov
