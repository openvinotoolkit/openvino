// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
/**
 * @brief Define a separate value for every version of C++ standard upto currently supported by build setup.
 */
#if !(defined(_MSC_VER) && __cplusplus == 199711L)
#    if __cplusplus >= 201103L
#        define OPENVINO_CPP_VER_AT_LEAST_11
#        if __cplusplus >= 201402L
#            define OPENVINO_CPP_VER_AT_LEAST_14
#            if __cplusplus >= 201703L
#                define OPENVINO_CPP_VER_AT_LEAST_17
#                if __cplusplus >= 202002L
#                    define OPENVINO_CPP_VER_AT_LEAST_20
#                endif
#            endif
#        endif
#    endif
#elif defined(_MSC_VER) && __cplusplus == 199711L
#    if _MSVC_LANG >= 201103L
#        define OPENVINO_CPP_VER_AT_LEAST_11
#        if _MSVC_LANG >= 201402L
#            define OPENVINO_CPP_VER_AT_LEAST_14
#            if _MSVC_LANG >= 201703L
#                define OPENVINO_CPP_VER_AT_LEAST_17
#                if _MSVC_LANG >= 202002L
#                    define OPENVINO_CPP_VER_AT_LEAST_20
#                endif
#            endif
#        endif
#    endif
#endif

#if !defined(__GNUC__) || (__GNUC__ > 12 || __GNUC__ == 12 && __GNUC_MINOR__ >= 3)
#    define GCC_NOT_USED_OR_VER_AT_LEAST_12_3
#endif

#if !defined(__clang__) || defined(__clang__) && __clang_major__ >= 17
#    define CLANG_NOT_USED_OR_VER_AT_LEAST_17
#endif

#if defined(__GNUC__) && (__GNUC__ < 12 || __GNUC__ == 12 && __GNUC_MINOR__ < 3)
#    define GCC_VER_LESS_THEN_12_3
#endif

#if defined(__clang__) && __clang_major__ < 17
#    define CLANG_VER_LESS_THEN_17
#endif
