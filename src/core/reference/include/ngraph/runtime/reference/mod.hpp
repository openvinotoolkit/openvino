// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cmath>
#include <cstddef>

#include "ngraph/runtime/reference/autobroadcast_binop.hpp"

namespace ngraph {
namespace runtime {
namespace reference {
template <typename T>
void mod(const T* arg0,
         const T* arg1,
         T* out,
         const Shape& arg_shape0,
         const Shape& arg_shape1,
         const op::AutoBroadcastSpec& broadcast_spec) {
    autobroadcast_binop(arg0, arg1, out, arg_shape0, arg_shape1, broadcast_spec, [](T x, T y) -> T {
        return static_cast<T>(x - std::truncf(static_cast<float>(x / y)) * y);
    });
}
}  // namespace reference
}  // namespace runtime
}  // namespace ngraph
