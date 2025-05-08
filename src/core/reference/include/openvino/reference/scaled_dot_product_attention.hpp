// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/shape.hpp"

namespace ov {
namespace reference {
template <typename T>
void scaled_dot_product_attention(const T* query,
                                 const T* key,
                                 const T* value,
                                 const T* mask,
                                 const T* scale,
                                 T* output,
                                 bool is_causal,
                                 const Shape& query_shape,
                                 const Shape& key_shape,
                                 const Shape& value_shape,
                                 const Shape& mask_shape,
                                 const Shape& output_shape) {
    // Stub impl.
}

}  // namespace reference
}  // namespace ov