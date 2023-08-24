// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/reference/autobroadcast_binop.hpp"

// Proxy calls for dependant components transition to ov::reference namespace
namespace ngraph {
namespace runtime {
namespace reference {
template <typename T, typename U, typename Functor>
void autobroadcast_binop(const T* arg0,
                         const T* arg1,
                         U* out,
                         const Shape& arg0_shape,
                         const Shape& arg1_shape,
                         const op::AutoBroadcastSpec& broadcast_spec,
                         Functor elementwise_functor) {
    ov::reference::autobroadcast_binop(arg0, arg1, out, arg0_shape, arg1_shape, broadcast_spec, elementwise_functor);
}

template <typename T, typename U, typename Functor>
void autobroadcast_select(const U* arg0,
                          const T* arg1,
                          const T* arg2,
                          T* out,
                          const Shape& arg0_shape,
                          const Shape& arg1_shape,
                          const Shape& arg2_shape,
                          const op::AutoBroadcastSpec& broadcast_spec,
                          Functor elementwise_functor) {
    ov::reference::autobroadcast_select(arg0,
                                        arg1,
                                        arg2,
                                        out,
                                        arg0_shape,
                                        arg1_shape,
                                        arg2_shape,
                                        broadcast_spec,
                                        elementwise_functor);
}
}  // namespace reference
}  // namespace runtime
}  // namespace ngraph
