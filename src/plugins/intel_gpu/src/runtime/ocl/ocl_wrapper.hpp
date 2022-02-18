// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///
/// \file This file wraps cl2.hpp and disables temporary some warnings that this header can emit.
///

#pragma once

// Check for C++.
#ifndef __cplusplus
    #error This header can be used in C++ only.
#endif

// we want exceptions
#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define CL_HPP_MINIMUM_OPENCL_VERSION 200
#define CL_HPP_TARGET_OPENCL_VERSION 200

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
#elif defined __GNUC__
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wsign-compare"
    #pragma GCC diagnostic ignored "-Wunused-parameter"
    #pragma GCC diagnostic ignored "-Wunused-variable"
    #pragma GCC diagnostic ignored "-Wunused-function"
    #pragma GCC diagnostic ignored "-Wignored-qualifiers"
    #if __GNUC__ >= 8
    #pragma GCC diagnostic ignored "-Wcatch-value"
    #endif
    #if __GNUC__ >= 6
    #pragma GCC diagnostic ignored "-Wignored-attributes"
    #endif
#else
    #pragma message("Unknown compiler. No changes in diagnostics will be done.")
#endif


#include "ocl_ext.hpp"

namespace cl {
namespace detail {
CL_HPP_PARAM_NAME_INFO_1_1_DEPRECATED_IN_2_0_(CL_HPP_DECLARE_PARAM_TRAITS_)
}
}

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
