// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#ifndef OPENVINO_ENABLE_UNICODE_PATH_SUPPORT
#    ifdef _WIN32
#        if defined(__INTEL_COMPILER) || defined(_MSC_VER) || defined(__GNUC__)
#            define OPENVINO_ENABLE_UNICODE_PATH_SUPPORT
#        endif
#    elif defined(__clang__)
#        define OPENVINO_ENABLE_UNICODE_PATH_SUPPORT
#    elif defined(__GNUC__) && (__GNUC__ > 5 || (__GNUC__ == 5 && __GNUC_MINOR__ > 2))
#        define OPENVINO_ENABLE_UNICODE_PATH_SUPPORT
#    endif
#endif

// Disabled MSVC warning
#if defined(_MSC_VER)
#    define OPENVINO_DISABLE_WARNING_MSVC_BEGIN(id) __pragma(warning(push)) __pragma(warning(disable : id))
#    define OPENVINO_DISABLE_WARNING_MSVC_END(id)   __pragma(warning(pop))
#else
#    define OPENVINO_DISABLE_WARNING_MSVC_BEGIN(id)
#    define OPENVINO_DISABLE_WARNING_MSVC_END(id)
#endif

#if !(defined(_MSC_VER) && __cplusplus == 199711L)
#    if __cplusplus >= 201103L
#        define OPENVINO_CPP_VER_11
#        if __cplusplus >= 201402L
#            define OPENVINO_CPP_VER_14
#            if __cplusplus >= 201703L
#                define OPENVINO_CPP_VER_17
#                if __cplusplus >= 202002L
#                    define OPENVINO_CPP_VER_20
#                endif
#            endif
#        endif
#    endif
#elif defined(_MSC_VER) && __cplusplus == 199711L
#    if _MSVC_LANG >= 201103L
#        define OPENVINO_CPP_VER_11
#        if _MSVC_LANG >= 201402L
#            define OPENVINO_CPP_VER_14
#            if _MSVC_LANG >= 201703L
#                define OPENVINO_CPP_VER_17
#                if _MSVC_LANG >= 202002L
#                    define OPENVINO_CPP_VER_20
#                endif
#            endif
#        endif
#    endif
#endif
