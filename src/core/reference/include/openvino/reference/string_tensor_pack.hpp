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
                        const size_t string_count) {
    const char* chars = reinterpret_cast<const char*>(symbols);
    for (size_t i = 0; i < string_count; ++i) {
        out[i].assign(chars + begins[i], chars + ends[i]);
    }
}
}  // namespace reference
}  // namespace ov
