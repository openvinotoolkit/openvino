// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief Set of macro used by openvino
 * @file openvino/util/pp.hpp
 */

#pragma once

// Macros for string conversion
#define OV_PP_TOSTRING(...)  OV_PP_TOSTRING_(__VA_ARGS__)
#define OV_PP_TOSTRING_(...) #__VA_ARGS__

#define OV_PP_EXPAND(...) __VA_ARGS__

#define OV_PP_NARG(...)                         OV_PP_EXPAND(OV_PP_NARG_(__VA_ARGS__, OV_PP_RSEQ_N()))
#define OV_PP_NARG_(...)                        OV_PP_EXPAND(OV_PP_ARG_N(__VA_ARGS__))
#define OV_PP_ARG_N(_0, _1, _2, _3, _4, N, ...) N
#define OV_PP_RSEQ_N()                          0, 4, 3, 2, 1, 0
#define OV_PP_NO_ARGS(NAME)                     , , , ,

// Macros for names concatenation
#define OV_PP_CAT_(x, y)        x##y
#define OV_PP_CAT(x, y)         OV_PP_CAT_(x, y)
#define OV_PP_CAT3_(x, y, z)    x##y##z
#define OV_PP_CAT3(x, y, z)     OV_PP_CAT3_(x, y, z)
#define OV_PP_CAT4_(x, y, z, w) x##y##z##w
#define OV_PP_CAT4(x, y, z, w)  OV_PP_CAT4_(x, y, z, w)

#define OV_PP_OVERLOAD(NAME, ...) \
    OV_PP_EXPAND(OV_PP_CAT3(NAME, _, OV_PP_EXPAND(OV_PP_NARG(OV_PP_NO_ARGS __VA_ARGS__(NAME))))(__VA_ARGS__))

// Placeholder for first macro argument
#define OV_PP_ARG_PLACEHOLDER_1 0,

// This macro returns second argument, first argument is ignored
#define OV_PP_SECOND_ARG(...)                   OV_PP_EXPAND(OV_PP_SECOND_ARG_(__VA_ARGS__, 0))
#define OV_PP_SECOND_ARG_(...)                  OV_PP_EXPAND(OV_PP_SECOND_ARG_GET(__VA_ARGS__))
#define OV_PP_SECOND_ARG_GET(ignored, val, ...) val

// Return macro argument value
#define OV_PP_IS_ENABLED(x) OV_PP_IS_ENABLED1(x)

// Generate junk macro or {0, } sequence if val is 1
#define OV_PP_IS_ENABLED1(val) OV_PP_IS_ENABLED2(OV_PP_CAT(OV_PP_ARG_PLACEHOLDER_, val))

// Return second argument from possible sequences {1, 0}, {0, 1, 0}
#define OV_PP_IS_ENABLED2(arg1_or_junk) OV_PP_SECOND_ARG(arg1_or_junk 1, 0)

// Ignores inputs
#define OV_PP_IGNORE(...)

/* This macro is intended to fix C++20 [=] lambda
warning. Although C++20 identifier is 202002L,
some compilers supporting C++20, or their drafts like
C++2a, producing the warning, are using 201402L value.
Also, MSVC requires a special check due to the
__cplusplus value compatibility issues.*/
#if (defined(_MSVC_LANG) && (_MSVC_LANG >= 202002L)) || (__cplusplus >= 202002L)
#    define OV_CAPTURE_CPY_AND_THIS =, this
#else
#    define OV_CAPTURE_CPY_AND_THIS =
#endif /* C++20 */

#ifdef __linux__
#    ifndef _GNU_SOURCE
#        define _GNU_SOURCE
#        include <features.h>
#        ifndef __USE_GNU
#            define OPENVINO_MUSL_LIBC
#        endif
#        undef _GNU_SOURCE /* don't contaminate other includes unnecessarily */
#    else
#        include <features.h>
#        ifndef __USE_GNU
#            define OPENVINO_MUSL_LIBC
#        endif
#    endif

#    ifndef OPENVINO_MUSL_LIBC
#        define OPENVINO_GNU_LIBC
#    endif
#endif
