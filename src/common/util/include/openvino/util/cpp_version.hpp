// Copyright (C) 2018-2025 Intel Corporation
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
