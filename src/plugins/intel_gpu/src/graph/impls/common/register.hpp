// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/primitives/condition.hpp"
#include "intel_gpu/primitives/loop.hpp"
#include "intel_gpu/primitives/data.hpp"
#include "intel_gpu/primitives/input_layout.hpp"
#include "intel_gpu/primitives/prior_box.hpp"


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

#undef REGISTER_COMMON

}  // namespace detail
}  // namespace common
}  // namespace cldnn
