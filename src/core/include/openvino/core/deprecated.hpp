// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

//
// The OPENVINO_DEPRECATED macro can be used to deprecate a function declaration. For example:
//
//     OPENVINO_DEPRECATED("replace with groxify");
//     void frobnicate()
//
// The macro will expand to a deprecation attribute supported by the compiler,
// so any use of `frobnicate` will produce a compiler warning.
//

#if defined(_WIN32)
#    define OPENVINO_DEPRECATED(msg) __declspec(deprecated(msg))
#    if __cplusplus >= 201402L
#        define OPENVINO_ENUM_DEPRECATED(msg) [[deprecated(msg)]]
#    else
#        define OPENVINO_ENUM_DEPRECATED(msg)
#    endif
#elif defined(__INTEL_COMPILER)
#    define OPENVINO_DEPRECATED(msg)      __attribute__((deprecated(msg)))
#    define OPENVINO_ENUM_DEPRECATED(msg) OPENVINO_DEPRECATED(msg)
#elif defined(__GNUC__)
#    define OPENVINO_DEPRECATED(msg) __attribute__((deprecated(msg)))
#    if __GNUC__ < 6 && !defined(__clang__)
#        define OPENVINO_ENUM_DEPRECATED(msg)
#    else
#        define OPENVINO_ENUM_DEPRECATED(msg) OPENVINO_DEPRECATED(msg)
#    endif
#else
#    define OPENVINO_DEPRECATED(msg)
#    define OPENVINO_ENUM_DEPRECATED(msg)
#endif

// Suppress warning "-Wdeprecated-declarations" / C4996
#if defined(_MSC_VER)
#    define OPENVINO_DO_PRAGMA(x) __pragma(x)
#elif defined(__GNUC__)
#    define OPENVINO_DO_PRAGMA(x) _Pragma(#    x)
#else
#    define OPENVINO_DO_PRAGMA(x)
#endif

#if defined(_MSC_VER) && !defined(__clang__)
#    define OPENVINO_SUPPRESS_DEPRECATED_START \
        OPENVINO_DO_PRAGMA(warning(push))      \
        OPENVINO_DO_PRAGMA(warning(disable : 4996))
#    define OPENVINO_SUPPRESS_DEPRECATED_END OPENVINO_DO_PRAGMA(warning(pop))
#elif defined(__INTEL_COMPILER)
#    define OPENVINO_SUPPRESS_DEPRECATED_START \
        OPENVINO_DO_PRAGMA(warning(push))      \
        OPENVINO_DO_PRAGMA(warning(disable : 1478))
OPENVINO_DO_PRAGMA(warning(disable : 1786))
#    define OPENVINO_SUPPRESS_DEPRECATED_END OPENVINO_DO_PRAGMA(warning(pop))
#elif defined(__clang__) || ((__GNUC__) && (__GNUC__ * 100 + __GNUC_MINOR__ > 405))
#    define OPENVINO_SUPPRESS_DEPRECATED_START  \
        OPENVINO_DO_PRAGMA(GCC diagnostic push) \
        OPENVINO_DO_PRAGMA(GCC diagnostic ignored "-Wdeprecated-declarations")
#    define OPENVINO_SUPPRESS_DEPRECATED_END OPENVINO_DO_PRAGMA(GCC diagnostic pop)
#else
#    define OPENVINO_SUPPRESS_DEPRECATED_START
#    define OPENVINO_SUPPRESS_DEPRECATED_END
#endif
