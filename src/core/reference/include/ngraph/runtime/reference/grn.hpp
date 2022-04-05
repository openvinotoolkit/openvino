// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/runtime/reference/normalize_l2.hpp"

namespace ngraph {
namespace runtime {
namespace reference {
template <typename T>
void grn(const T* data, T* out, float bias, const Shape& data_shape) {
    normalize_l2(data, out, data_shape, {1}, bias, op::EpsMode::ADD);
}
}  // namespace reference
}  // namespace runtime
}  // namespace ngraph
