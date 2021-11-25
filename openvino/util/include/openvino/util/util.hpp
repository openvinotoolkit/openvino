// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#if defined _WIN32 || defined __CYGWIN__
#    define OPENVINO_UTIL_CORE_IMPORTS __declspec(dllimport)
#    define OPENVINO_UTIL_CORE_EXPORTS __declspec(dllexport)
#elif defined(__GNUC__) && __GNUC__ >= 4
#    define OPENVINO_UTIL_CORE_IMPORTS __attribute__((visibility("default")))
#    define OPENVINO_UTIL_CORE_EXPORTS __attribute__((visibility("default")))
#else
#    define OPENVINO_UTIL_CORE_IMPORTS
#    define OPENVINO_UTIL_CORE_EXPORTS
#endif

// Now we use the generic helper definitions above to define OPENVINO_API
// OPENVINO_API is used for the public API symbols. It either DLL imports or DLL exports
//    (or does nothing for static build)

#ifdef _WIN32
#    pragma warning(disable : 4251)
#    pragma warning(disable : 4275)
#endif

#ifdef OPENVINO_STATIC_LIBRARY  // defined if we are building or calling ov_runtime as a static library
#    define OPENVINO_UTIL_API
#else
#    ifdef ov_runtime_EXPORTS  // defined if we are building the ov_runtime library
#        define OPENVINO_UTIL_API OPENVINO_UTIL_CORE_EXPORTS
#    else
#        define OPENVINO_UTIL_API OPENVINO_UTIL_CORE_IMPORTS
#    endif  // ov_runtime_EXPORTS
#endif      // OPENVINO_STATIC_LIBRARY

#ifndef OPENVINO_ENABLE_UNICODE_PATH_SUPPORT
#    ifdef _WIN32
#        if defined __INTEL_COMPILER || defined _MSC_VER
#            define OPENVINO_ENABLE_UNICODE_PATH_SUPPORT
#        endif
#    elif defined(__GNUC__) && (__GNUC__ > 5 || (__GNUC__ == 5 && __GNUC_MINOR__ > 2)) || defined(__clang__)
#        define OPENVINO_ENABLE_UNICODE_PATH_SUPPORT
#    endif
#endif
