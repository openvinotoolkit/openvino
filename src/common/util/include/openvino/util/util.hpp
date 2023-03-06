// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#ifndef OPENVINO_ENABLE_UNICODE_PATH_SUPPORT
#    ifdef _WIN32
#        if defined __INTEL_COMPILER || defined _MSC_VER
#            define OPENVINO_ENABLE_UNICODE_PATH_SUPPORT
#        endif
#    elif defined(__GNUC__) && (__GNUC__ > 5 || (__GNUC__ == 5 && __GNUC_MINOR__ > 2)) || defined(__clang__)
#        define OPENVINO_ENABLE_UNICODE_PATH_SUPPORT
#    endif
#endif

// Disabled MSVC warning
#if defined(_MSC_VER)
#    define DISABLE_WARNING_MSVC_BEGIN(id) __pragma(warning(push)) __pragma(warning(disable : id))
#    define DISABLE_WARNING_MSVC_END(id)   __pragma(warning(pop))
#else
#    define DISABLE_WARNING_MSVC_BEGIN(id)
#    define DISABLE_WARNING_MSVC_END(id)
#endif
