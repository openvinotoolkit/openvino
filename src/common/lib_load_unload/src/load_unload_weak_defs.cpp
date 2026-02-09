// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/lib_load_unload.hpp"

#if defined(_MSC_VER)
#define DEFINE_DEFAULT_FUNC(func_name) \
    extern "C" void func_name##_default() {}
#elif defined(__GNUC__) || defined(__clang__)
#define DEFINE_DEFAULT_FUNC(func_name) \
    extern "C" void func_name() __attribute__((weak)); \
    extern "C" void func_name() {}
#else
#    error "Compiler not supported"
#endif

#define UNLOAD_FUNC(func_name) DEFINE_DEFAULT_FUNC(func_name)

#include "openvino/unload_functions.inc"

#undef DEFINE_DEFAULT_FUNC
#undef UNLOAD_FUNC
