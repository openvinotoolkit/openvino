// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/primitives/convolution.hpp"

namespace cldnn {
namespace onednn {
void register_implementations();

namespace detail {

#define REGISTER_ONEDNN_IMPL(prim)  \
    struct attach_##prim##_onednn { \
        attach_##prim##_onednn();   \
    }

REGISTER_ONEDNN_IMPL(convolution);
REGISTER_ONEDNN_IMPL(deconvolution);
REGISTER_ONEDNN_IMPL(concatenation);
REGISTER_ONEDNN_IMPL(eltwise);
REGISTER_ONEDNN_IMPL(gemm);
REGISTER_ONEDNN_IMPL(pooling);
REGISTER_ONEDNN_IMPL(reorder);
REGISTER_ONEDNN_IMPL(fully_connected);

#undef REGISTER_ONEDNN_IMPL

}  // namespace detail
}  // namespace onednn
}  // namespace cldnn
