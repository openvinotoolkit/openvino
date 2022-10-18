// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "register.hpp"

namespace cldnn {
namespace onednn {

#define REGISTER_ONEDNN_IMPL(prim)                       \
    static detail::attach_##prim##_onednn attach_##prim

void register_implementations() {
    REGISTER_ONEDNN_IMPL(convolution);
    REGISTER_ONEDNN_IMPL(deconvolution);
    REGISTER_ONEDNN_IMPL(concatenation);
    REGISTER_ONEDNN_IMPL(eltwise);
    REGISTER_ONEDNN_IMPL(gemm);
    REGISTER_ONEDNN_IMPL(pooling);
    REGISTER_ONEDNN_IMPL(reduction);
    REGISTER_ONEDNN_IMPL(reorder);
    REGISTER_ONEDNN_IMPL(fully_connected);}

}  // namespace onednn
}  // namespace cldnn
