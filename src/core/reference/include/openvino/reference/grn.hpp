// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/reference/normalize_l2.hpp"

namespace ov {
namespace reference {
template <typename T>
void grn(const T* data, T* out, float bias, const Shape& data_shape) {
    normalize_l2(data, out, data_shape, {1}, bias, op::EpsMode::ADD);
}
}  // namespace reference
}  // namespace ov
