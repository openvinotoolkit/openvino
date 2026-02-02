// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#if defined(_MSC_VER)
#    define OV_SHUTDOWN_SECTION_NAME ".shutdown_sec$u"
#    pragma section(".shutdown_sec$u", read)
#    define DECLARE_OV_SHUTDOWN_FUNC(func) __declspec(allocate(OV_SHUTDOWN_SECTION_NAME)) void (*__##func)(void) = func;
#elif defined(__GNUC__) || defined(__clang__)
#    if defined(__APPLE__)
#        define OV_SHUTDOWN_SECTION_NAME "__DATA,__shutdown_sec"
#    else
#        define OV_SHUTDOWN_SECTION_NAME "__shutdown_sec"
#    endif
#    define DECLARE_OV_SHUTDOWN_FUNC(func) \
        __attribute__((section(OV_SHUTDOWN_SECTION_NAME), used)) void (*__##func)(void) = func;
#else
#    error "Compiler not supported"
#endif
