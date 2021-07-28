// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

//
// The NGRAPH_DEPRECATED macro can be used to deprecate a function declaration. For example:
//
//     NGRAPH_DEPRECATED("replace with groxify");
//     void frobnicate()
//
// The macro will expand to a deprecation attribute supported by the compiler,
// so any use of `frobnicate` will produce a compiler warning.
//

#if defined(_WIN32)
#define NGRAPH_DEPRECATED(msg) __declspec(deprecated(msg))
#elif defined(__INTEL_COMPILER)
#define NGRAPH_DEPRECATED(msg) __attribute__((deprecated(msg)))
#elif defined(__GNUC__)
#define NGRAPH_DEPRECATED(msg) __attribute__((deprecated((msg))))
#else
#define NGRAPH_DEPRECATED(msg)
#endif

// Suppress warning "-Wdeprecated-declarations" / C4996
#if defined(_MSC_VER)
#define NGRAPH_DO_PRAGMA(x) __pragma(x)
#elif defined(__GNUC__)
#define NGRAPH_DO_PRAGMA(x) _Pragma(#x)
#else
#define NGRAPH_DO_PRAGMA(x)
#endif

#if defined(_MSC_VER) && !defined(__clang__)
#define NGRAPH_SUPPRESS_DEPRECATED_START                                                           \
    NGRAPH_DO_PRAGMA(warning(push))                                                                \
    NGRAPH_DO_PRAGMA(warning(disable : 4996))
#define NGRAPH_SUPPRESS_DEPRECATED_END NGRAPH_DO_PRAGMA(warning(pop))
#elif defined(__INTEL_COMPILER)
#define NGRAPH_SUPPRESS_DEPRECATED_START                                                           \
    NGRAPH_DO_PRAGMA(warning(push))                                                                \
    NGRAPH_DO_PRAGMA(warning(disable : 1478))
NGRAPH_DO_PRAGMA(warning(disable : 1786))
#define NGRAPH_SUPPRESS_DEPRECATED_END NGRAPH_DO_PRAGMA(warning(pop))
#elif defined(__clang__) || ((__GNUC__) && (__GNUC__ * 100 + __GNUC_MINOR__ > 405))
#define NGRAPH_SUPPRESS_DEPRECATED_START                                                           \
    NGRAPH_DO_PRAGMA(GCC diagnostic push)                                                          \
    NGRAPH_DO_PRAGMA(GCC diagnostic ignored "-Wdeprecated-declarations")
#define NGRAPH_SUPPRESS_DEPRECATED_END NGRAPH_DO_PRAGMA(GCC diagnostic pop)
#else
#define NGRAPH_SUPPRESS_DEPRECATED_START
#define NGRAPH_SUPPRESS_DEPRECATED_END
#endif
