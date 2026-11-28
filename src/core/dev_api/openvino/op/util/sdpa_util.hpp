// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/type/element_type.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"

namespace ov::op::util {
/**
 * @brief Tells whether an element type is accepted on the ScaledDotProductAttention key and value
 * operands as an integer type, whose values the operation converts to the query's type.
 *
 * Deliberately not element::Type::is_quantized(): that predicate is true for i32, which the
 * operation still rejects, and false for u4, which it accepts.
 */
OPENVINO_API bool is_sdpa_integer_kv_type(const element::Type& type);

/// @brief Tells whether the key or value operand of @p sdpa has an integer type.
OPENVINO_API bool sdpa_has_integer_kv(const ov::op::v13::ScaledDotProductAttention& sdpa);
}  // namespace ov::op::util
