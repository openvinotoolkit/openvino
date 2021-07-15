// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cldnn/primitives/condition.hpp"
#include "cldnn/primitives/loop.hpp"
#include "cldnn/primitives/data.hpp"
#include "cldnn/primitives/input_layout.hpp"
#include "cldnn/primitives/prior_box.hpp"


namespace cldnn {
namespace common {
void register_implementations();

namespace detail {

#define REGISTER_COMMON(prim)           \
    struct attach_##prim##_common {     \
        attach_##prim##_common();       \
    }

REGISTER_COMMON(condition);
REGISTER_COMMON(data);
REGISTER_COMMON(input_layout);
REGISTER_COMMON(loop);
REGISTER_COMMON(prior_box);

#undef REGISTER_COMMON

}  // namespace detail
}  // namespace common
}  // namespace cldnn
