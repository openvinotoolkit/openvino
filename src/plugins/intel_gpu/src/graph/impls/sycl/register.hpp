// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once


namespace cldnn {
namespace sycl {
void register_implementations();

namespace detail {

#define REGISTER_SYCL_IMPL(prim)  \
    struct attach_##prim##_sycl { \
        attach_##prim##_sycl();   \
    }

#undef REGISTER_SYCL_IMPL

}  // namespace detail
}  // namespace sycl
}  // namespace cldnn
