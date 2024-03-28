// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/primitives/fully_connected.hpp"

namespace cldnn {
namespace sycl {
void register_implementations();

namespace detail {

#define REGISTER_SYCL_IMPL(prim)  \
    struct attach_##prim##_sycl { \
        attach_##prim##_sycl();   \
    }

REGISTER_SYCL_IMPL(fully_connected);

#undef REGISTER_SYCL_IMPL

}  // namespace detail
}  // namespace sycl
}  // namespace cldnn
