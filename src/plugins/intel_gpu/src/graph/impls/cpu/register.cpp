// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "register.hpp"

namespace cldnn {
namespace cpu {

#define REGISTER_CPU(prim)                                \
    static detail::attach_##prim##_impl attach_##prim

void register_implementations() {
    REGISTER_CPU(assign);
    REGISTER_CPU(detection_output);
    REGISTER_CPU(proposal);
    REGISTER_CPU(read_value);
    REGISTER_CPU(non_max_suppression);
    REGISTER_CPU(non_max_suppression_gather);
    REGISTER_CPU(shape_of);
    REGISTER_CPU(concatenation);
    REGISTER_CPU(gather);
    REGISTER_CPU(strided_slice);
    REGISTER_CPU(range);
    REGISTER_CPU(scatter_update);
    REGISTER_CPU(eltwise);
    REGISTER_CPU(crop);
    REGISTER_CPU(activation);
    REGISTER_CPU(reorder);
    REGISTER_CPU(broadcast);
    REGISTER_CPU(tile);
    REGISTER_CPU(select);
    REGISTER_CPU(reduce);
    REGISTER_CPU(fake_convert);
}

}  // namespace cpu
}  // namespace cldnn
