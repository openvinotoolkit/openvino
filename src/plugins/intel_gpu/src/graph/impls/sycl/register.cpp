// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "register.hpp"

namespace cldnn {
namespace sycl {

#define REGISTER_SYCL_IMPL(prim)                       \
    static detail::attach_##prim##_sycl attach_##prim

void register_implementations() {
    REGISTER_SYCL_IMPL(fully_connected);}

}  // namespace sycl
}  // namespace cldnn
