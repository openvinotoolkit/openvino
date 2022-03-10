// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#define _PP_EXPAND(X) X
// Macros for names concatenation
#define _PP_CAT_(x, y) x##y
#define _PP_CAT(x, y)  _PP_CAT_(x, y)

// Placeholder for first macro argument
#define _PP_ARG_PLACEHOLDER_1 0,

// This macro returns second argument, first argument is ignored
#define _PP_SECOND_ARG(...)                   _PP_EXPAND(_PP_SECOND_ARG_(__VA_ARGS__, 0))
#define _PP_SECOND_ARG_(...)                  _PP_EXPAND(_PP_SECOND_ARG_GET(__VA_ARGS__))
#define _PP_SECOND_ARG_GET(ignored, val, ...) val

// Return macro argument value
#define _PP_IS_ENABLED(x) _PP_IS_ENABLED1(x)

// Generate junk macro or {0, } sequence if val is 1
#define _PP_IS_ENABLED1(val) _PP_IS_ENABLED2(_PP_CAT(_PP_ARG_PLACEHOLDER_, val))

// Return second argument from possible sequences {1, 0}, {0, 1, 0}
#define _PP_IS_ENABLED2(arg1_or_junk) _PP_SECOND_ARG(arg1_or_junk 1, 0)

#define _REGISTER_OV_OP(NAMESPACE, NAME, ORIG_NAMESPACE) \
    _PP_CAT(_REG_OP_, _PP_IS_ENABLED(OPERATION_DEFINED_##NAME))(NAMESPACE, NAME, ORIG_NAMESPACE)

#define _REG_OP_0(NAMESPACE, NAME, ORIG_NAMESPACE)
#define _REG_OP_1(NAMESPACE, NAME, ORIG_NAMESPACE) \
    namespace ov {                                 \
    namespace op {                                 \
    namespace NAMESPACE {                          \
    using ORIG_NAMESPACE::NAME;                    \
    }                                              \
    }                                              \
    }
