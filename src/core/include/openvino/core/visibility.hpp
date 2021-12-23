// Copyright (C) 2018-2021 Intel Corporation
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
#elif defined(__GNUC__) && __GNUC__ >= 4
#    define OPENVINO_CORE_IMPORTS __attribute__((visibility("default")))
#    define OPENVINO_CORE_EXPORTS __attribute__((visibility("default")))
#else
#    define OPENVINO_CORE_IMPORTS
#    define OPENVINO_CORE_EXPORTS
#endif

#define OV_NEW_API 1
// Now we use the generic helper definitions above to define OPENVINO_API
// OPENVINO_API is used for the public API symbols. It either DLL imports or DLL exports
//    (or does nothing for static build)

#ifdef _WIN32
#    pragma warning(disable : 4251)
#    pragma warning(disable : 4275)
#endif

#ifdef OPENVINO_STATIC_LIBRARY  // defined if we are building or calling ov_runtime as a static library
#    define OPENVINO_API
#    define OPENVINO_API_C(...) __VA_ARGS__
#else
#    ifdef IMPLEMENT_OPENVINO_API  // defined if we are building the ov_runtime DLL (instead of using it)
#        define OPENVINO_API        OPENVINO_CORE_EXPORTS
#        define OPENVINO_API_C(...) OPENVINO_EXTERN_C OPENVINO_CORE_EXPORTS __VA_ARGS__ OPENVINO_CDECL
#    else
#        define OPENVINO_API        OPENVINO_CORE_IMPORTS
#        define OPENVINO_API_C(...) OPENVINO_EXTERN_C OPENVINO_CORE_IMPORTS __VA_ARGS__ OPENVINO_CDECL
#    endif  // IMPLEMENT_OPENVINO_API
#endif      // OPENVINO_STATIC_LIBRARY
