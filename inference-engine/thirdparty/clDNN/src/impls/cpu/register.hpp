// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cldnn/primitives/detection_output.hpp"
#include "cldnn/primitives/proposal.hpp"
#include "cldnn/primitives/non_max_suppression.hpp"

namespace cldnn {
namespace cpu {
void register_implementations();

namespace detail {


#define REGISTER_CPU(prim)        \
    struct attach_##prim##_impl { \
        attach_##prim##_impl();   \
    }

REGISTER_CPU(proposal);
REGISTER_CPU(non_max_suppression);
REGISTER_CPU(detection_output);

#undef REGISTER_CPU

}  // namespace detail
}  // namespace cpu
}  // namespace cldnn
