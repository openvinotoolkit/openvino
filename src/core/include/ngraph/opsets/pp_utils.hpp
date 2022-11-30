// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/pp_utils.hpp"

#define _REGISTER_NGRAPH_OP(NAMESPACE, NAME, ORIG_NAMESPACE) \
    _PP_CAT(_REG_NGRAPH_OP_, _PP_IS_ENABLED(OPERATION_DEFINED_##NAME))(NAMESPACE, NAME, ORIG_NAMESPACE)

#define _REG_NGRAPH_OP_0(NAMESPACE, NAME, ORIG_NAMESPACE)
#define _REG_NGRAPH_OP_1(NAMESPACE, NAME, ORIG_NAMESPACE) \
    namespace ngraph {                                    \
    namespace NAMESPACE {                                 \
    using ORIG_NAMESPACE::NAME;                           \
    }                                                     \
    }

