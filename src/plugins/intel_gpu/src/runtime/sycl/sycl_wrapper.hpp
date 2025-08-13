// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///
/// \file This file wraps sycl/sycl.hpp and disables temporary some warnings that this header can emit.
///

#pragma once

// Check for C++.
#ifndef __cplusplus
    #error This header can be used in C++ only.
#endif

// Check for compiler and change specific diagnostics.
#if defined __INTEL_COMPILER
    #pragma warning push
    #pragma warning disable: 177
    #pragma warning disable: 411
    #pragma warning disable: 858
    #pragma warning disable: 869
    #pragma warning disable: 2557
    #pragma warning disable: 3280
    #pragma warning disable: 3346
#elif defined _MSC_VER
    #pragma warning(push)
    #pragma warning(disable: 4018 4100 4505 4201)
#elif defined __clang__
    #pragma clang diagnostic push
    #pragma clang diagnostic ignored "-Wsign-compare"
    #pragma clang diagnostic ignored "-Wunused-parameter"
    #pragma clang diagnostic ignored "-Wunused-variable"
    #pragma clang diagnostic ignored "-Wunused-function"
    #pragma clang diagnostic ignored "-Wignored-qualifiers"
    #pragma clang diagnostic ignored "-Wdeprecated-declarations"
#elif defined __GNUC__
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wsign-compare"
    #pragma GCC diagnostic ignored "-Wunused-parameter"
    #pragma GCC diagnostic ignored "-Wunused-variable"
    #pragma GCC diagnostic ignored "-Wunused-function"
    #pragma GCC diagnostic ignored "-Wignored-qualifiers"
    #pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    #if __GNUC__ >= 8
    #pragma GCC diagnostic ignored "-Wcatch-value"
    #endif
    #if __GNUC__ >= 6
    #pragma GCC diagnostic ignored "-Wignored-attributes"
    #endif
#else
    #pragma message("Unknown compiler. No changes in diagnostics will be done.")
#endif


#include "sycl_ext.hpp"

// Restore specific diagnostics.
#if defined __INTEL_COMPILER
    #pragma warning pop
#elif defined _MSC_VER
    #pragma warning(pop)
#elif defined __clang__
    #pragma clang diagnostic pop
#elif defined __GNUC__
    #pragma GCC diagnostic pop
#endif
