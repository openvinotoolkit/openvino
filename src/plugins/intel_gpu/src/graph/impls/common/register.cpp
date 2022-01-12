// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "register.hpp"

namespace cldnn {
namespace common {

#define REGISTER_COMMON(prim)                      \
    static detail::attach_##prim##_common attach_##prim

void register_implementations() {
    REGISTER_COMMON(condition);
    REGISTER_COMMON(data);
    REGISTER_COMMON(input_layout);
    REGISTER_COMMON(loop);
    REGISTER_COMMON(prior_box);
}

}  // namespace common
}  // namespace cldnn
