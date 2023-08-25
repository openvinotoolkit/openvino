// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/reference/softmax.hpp"

// Proxy call for dependant components transition to ov::reference namespace
namespace ngraph {
namespace runtime {
namespace reference {
template <typename T>
void softmax(const T* arg, T* out, const Shape& shape, const AxisSet& axes) {
    ov::reference::softmax(arg, out, shape, axes);
}
}  // namespace reference
}  // namespace runtime
}  // namespace ngraph
