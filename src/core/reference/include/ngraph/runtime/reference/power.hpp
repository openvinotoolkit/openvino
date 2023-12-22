// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/reference/power.hpp"

// Proxy calls for dependant components transition to ov::reference namespace
namespace ngraph {
namespace runtime {
namespace reference {

template <typename T>
void power(const T* arg0, const T* arg1, T* out, size_t count) {
    ov::reference::power(arg0, arg1, out, count);
}

template <typename T>
void power(const T* arg0,
           const T* arg1,
           T* out,
           const Shape& arg0_shape,
           const Shape& arg1_shape,
           const op::AutoBroadcastSpec& broadcast_spec) {
    ov::reference::power(arg0, arg1, out, arg0_shape, arg1_shape, broadcast_spec);
}

}  // namespace reference
}  // namespace runtime
}  // namespace ngraph
