// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cldnn/primitives/convolution.hpp"

namespace cldnn {
namespace onednn {
void register_implementations();

namespace detail {

#define REGISTER_ONEDNN_IMPL(prim)  \
    struct attach_##prim##_onednn { \
        attach_##prim##_onednn();   \
    }

REGISTER_ONEDNN_IMPL(convolution);

#undef REGISTER_ONEDNN_IMPL

}  // namespace detail
}  // namespace onednn
}  // namespace cldnn
