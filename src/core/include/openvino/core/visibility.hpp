// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// https://gcc.gnu.org/wiki/Visibility
// Generic helper definitions for shared library support

#ifndef OPENVINO_EXTERN_C
#    ifdef __cplusplus
#        define OPENVINO_EXTERN_C extern "C"
#    else
#        define OPENVINO_EXTERN_C
#    endif
#endif

#if defined _WIN32
#    define OPENVINO_CDECL   __cdecl
#    define OPENVINO_STDCALL __stdcall
#else
#    define OPENVINO_CDECL
#    define OPENVINO_STDCALL
#endif

#ifndef OPENVINO_ENABLE_UNICODE_PATH_SUPPORT
#    ifdef _WIN32
#        if defined __INTEL_COMPILER || defined _MSC_VER
#            define OPENVINO_ENABLE_UNICODE_PATH_SUPPORT
#        endif
#    elif defined(__GNUC__) && (__GNUC__ > 5 || (__GNUC__ == 5 && __GNUC_MINOR__ > 2)) || defined(__clang__)
#        define OPENVINO_ENABLE_UNICODE_PATH_SUPPORT
#    endif
#endif

#if defined _WIN32 || defined __CYGWIN__
#    define OPENVINO_CORE_IMPORTS __declspec(dllimport)
#    define OPENVINO_CORE_EXPORTS __declspec(dllexport)
#    define _OPENVINO_HIDDEN_METHOD
#elif defined(__GNUC__) && __GNUC__ >= 4
#    define OPENVINO_CORE_IMPORTS   __attribute__((visibility("default")))
#    define OPENVINO_CORE_EXPORTS   __attribute__((visibility("default")))
#    define _OPENVINO_HIDDEN_METHOD __attribute__((visibility("hidden")))
#else
#    define OPENVINO_CORE_IMPORTS
#    define OPENVINO_CORE_EXPORTS
#    define _OPENVINO_HIDDEN_METHOD
#endif
