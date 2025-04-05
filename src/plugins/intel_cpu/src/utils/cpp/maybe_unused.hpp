// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
// Define a macro to silence "unused variable" warnings
#if defined(__GNUC__) || defined(__clang__)
#    define OV_CPU_MAYBE_UNUSED(x) ((void)(x))
#elif defined(_MSC_VER)
#    define OV_CPU_MAYBE_UNUSED(x) __pragma(warning(suppress : 4100)) x
#else
#    define OV_CPU_MAYBE_UNUSED(x) ((void)(x))
#endif

// Define a macro to silence "unused function" warnings
#if defined(__GNUC__) || defined(__clang__)
#    define OV_CPU_MAYBE_UNUSED_FUNCTION __attribute__((unused))
#elif defined(_MSC_VER)
#    define OV_CPU_MAYBE_UNUSED_FUNCTION __pragma(warning(suppress : 4505))
#else
#    define OV_CPU_MAYBE_UNUSED_FUNCTION
#endif