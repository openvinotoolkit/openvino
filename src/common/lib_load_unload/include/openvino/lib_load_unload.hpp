// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#if defined(_MSC_VER)
#define UNLOAD_FUNC(func_name) \
    extern "C" void func_name(); \
    __pragma(comment(linker, "/alternatename:" #func_name "=" #func_name "_default"))
#elif defined(__GNUC__) || defined(__clang__)
#define UNLOAD_FUNC(func_name) \
    extern "C" void func_name();
#else
#    error "Compiler not supported"
#endif

#include "openvino/unload_functions.inc"

#undef UNLOAD_FUNC
