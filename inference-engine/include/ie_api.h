// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief The macro defines a symbol import/export mechanism essential for Microsoft Windows(R) OS.
 * @file ie_api.h
 */
#pragma once

#include "details/ie_no_copy.hpp"

#if defined(USE_STATIC_IE) || ( defined(__GNUC__) && (__GNUC__ < 4) )
    #define INFERENCE_ENGINE_API(TYPE) extern "C" TYPE
    #define INFERENCE_ENGINE_API_CPP(type) type
    #define INFERENCE_ENGINE_API_CLASS(type)    type
    #define INFERENCE_ENGINE_CDECL __attribute__((cdecl))
#else
    #if defined(_WIN32)
        #define INFERENCE_ENGINE_CDECL

        #ifdef IMPLEMENT_INFERENCE_ENGINE_API
            #define INFERENCE_ENGINE_API(type) extern "C"   __declspec(dllexport) type __cdecl
            #define INFERENCE_ENGINE_API_CPP(type)  __declspec(dllexport) type __cdecl
            #define INFERENCE_ENGINE_API_CLASS(type)        __declspec(dllexport) type
        #else
            #define INFERENCE_ENGINE_API(type) extern "C"  __declspec(dllimport) type __cdecl
            #define INFERENCE_ENGINE_API_CPP(type)  __declspec(dllimport) type __cdecl
            #define INFERENCE_ENGINE_API_CLASS(type)   __declspec(dllimport) type
        #endif
    #else
        #define INFERENCE_ENGINE_CDECL __attribute__((cdecl))
        #ifdef IMPLEMENT_INFERENCE_ENGINE_API
            #define INFERENCE_ENGINE_API(type) extern "C" __attribute__((visibility("default"))) type
            #define INFERENCE_ENGINE_API_CPP(type) __attribute__((visibility("default"))) type
            #define INFERENCE_ENGINE_API_CLASS(type) __attribute__((visibility("default"))) type
        #else
            #define INFERENCE_ENGINE_API(type)   extern "C"   type
            #define INFERENCE_ENGINE_API_CPP(type)   type
            #define INFERENCE_ENGINE_API_CLASS(type)   type
        #endif
    #endif
#endif

#if defined(_WIN32)
    #define INFERENCE_ENGINE_DEPRECATED  __declspec(deprecated)
#else
    #define INFERENCE_ENGINE_DEPRECATED __attribute__((deprecated))
#endif

// Suppress warning "-Wdeprecated-declarations" / C4996
#if defined(_MSC_VER)
    #define IE_DO_PRAGMA(x) __pragma(x)
#elif defined(__GNUC__)
    #define IE_DO_PRAGMA(x) _Pragma (#x)
#else
    #define IE_DO_PRAGMA(x)
#endif

#if defined (_MSC_VER) && !defined (__clang__)
#define IE_SUPPRESS_DEPRECATED_START \
    IE_DO_PRAGMA(warning(push)) \
    IE_DO_PRAGMA(warning(disable: 4996))
#define IE_SUPPRESS_DEPRECATED_END IE_DO_PRAGMA(warning(pop))
#elif defined(__INTEL_COMPILER)
#define IE_SUPPRESS_DEPRECATED_START \
    IE_DO_PRAGMA(warning(push)) \
    IE_DO_PRAGMA(warning(disable:1478))
#define IE_SUPPRESS_DEPRECATED_END IE_DO_PRAGMA(warning(pop))
#elif defined(__clang__) || ((__GNUC__)  && (__GNUC__*100 + __GNUC_MINOR__ > 405))
#define IE_SUPPRESS_DEPRECATED_START \
    IE_DO_PRAGMA(GCC diagnostic push) \
    IE_DO_PRAGMA(GCC diagnostic ignored "-Wdeprecated-declarations")
#define IE_SUPPRESS_DEPRECATED_END IE_DO_PRAGMA(GCC diagnostic pop)
#else
#define IE_SUPPRESS_DEPRECATED_START
#define IE_SUPPRESS_DEPRECATED_END
#endif

#ifndef ENABLE_UNICODE_PATH_SUPPORT
    #if defined(_WIN32)
        #define ENABLE_UNICODE_PATH_SUPPORT
    #elif defined(__GNUC__) && (__GNUC__ > 5 || (__GNUC__ == 5 && __GNUC_MINOR__ > 2))
        #define ENABLE_UNICODE_PATH_SUPPORT
    #endif
#endif
