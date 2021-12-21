// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#if 0 // TODO(taylor)
#include "intel_gpu/primitives/detection_output.hpp"
#include "intel_gpu/primitives/proposal.hpp"
#include "intel_gpu/primitives/non_max_suppression.hpp"
#endif
namespace cldnn {
namespace cpu {
void register_implementations();

namespace detail {


#define REGISTER_CPU(prim)        \
    struct attach_##prim##_impl { \
        attach_##prim##_impl();   \
    }

#if 0 // TODO(taylor)
REGISTER_CPU(proposal);
REGISTER_CPU(non_max_suppression);
REGISTER_CPU(detection_output);
#endif

#undef REGISTER_CPU

}  // namespace detail
}  // namespace cpu
}  // namespace cldnn
