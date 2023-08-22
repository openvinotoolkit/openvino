// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cmath>
#include <cstddef>

#include "ngraph/op/util/attr_types.hpp"
#include "ngraph/shape.hpp"
#include "openvino/reference/autobroadcast_binop.hpp"

namespace ov {
namespace reference {
template <typename T>
void power(const T* arg0, const T* arg1, T* out, size_t count) {
    for (size_t i = 0; i < count; i++) {
        out[i] = static_cast<T>(std::pow(arg0[i], arg1[i]));
    }
}

template <typename T>
void power(const T* arg0,
           const T* arg1,
           T* out,
           const Shape& arg0_shape,
           const Shape& arg1_shape,
           const op::AutoBroadcastSpec& broadcast_spec) {
    autobroadcast_binop(arg0, arg1, out, arg0_shape, arg1_shape, broadcast_spec, [](T x, T y) -> T {
        return static_cast<T>(std::pow(x, y));
    });
}
}  // namespace reference
}  // namespace ov

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
