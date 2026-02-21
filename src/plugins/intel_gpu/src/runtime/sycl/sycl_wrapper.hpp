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
#elif defined _MSC_VER
    #pragma warning(push)
#elif defined __clang__
    #pragma clang diagnostic push
#elif defined __GNUC__
    #pragma GCC diagnostic push
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
