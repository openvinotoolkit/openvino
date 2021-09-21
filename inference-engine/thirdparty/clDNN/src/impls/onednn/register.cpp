// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "register.hpp"

namespace cldnn {
namespace onednn {

#define REGISTER_ONEDNN_IMPL(prim)                       \
    static detail::attach_##prim##_onednn attach_##prim

void register_implementations() {
    REGISTER_ONEDNN_IMPL(convolution);
}

}  // namespace onednn
}  // namespace cldnn
