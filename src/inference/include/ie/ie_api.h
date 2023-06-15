// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief The macro defines a symbol import/export mechanism essential for Microsoft Windows(R) OS.
 * @file ie_api.h
 */

#pragma once

#if !defined(IN_OV_COMPONENT) && !defined(IE_LEGACY_HEADER_INCLUDED)
#    define IE_LEGACY_HEADER_INCLUDED
#    ifdef _MSC_VER
#        pragma message( \
            "The Inference Engine API is deprecated and will be removed in the 2024.0 release. For instructions on transitioning to the new API, please refer to https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html")
#    else
#        warning("The Inference Engine API is deprecated and will be removed in the 2024.0 release. For instructions on transitioning to the new API, please refer to https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html")
#    endif
#endif

#if defined(OPENVINO_STATIC_LIBRARY) || defined(USE_STATIC_IE) || (defined(__GNUC__) && (__GNUC__ < 4))
#    define INFERENCE_ENGINE_API(...)       extern "C" __VA_ARGS__
#    define INFERENCE_ENGINE_API_CPP(...)   __VA_ARGS__
#    define INFERENCE_ENGINE_API_CLASS(...) __VA_ARGS__
#else
#    if defined(_WIN32) || defined(__CYGWIN__)
#        ifdef IMPLEMENT_INFERENCE_ENGINE_API
#            define INFERENCE_ENGINE_API(...)       extern "C" __declspec(dllexport) __VA_ARGS__ __cdecl
#            define INFERENCE_ENGINE_API_CPP(...)   __declspec(dllexport) __VA_ARGS__
#            define INFERENCE_ENGINE_API_CLASS(...) __declspec(dllexport) __VA_ARGS__
#        else
#            define INFERENCE_ENGINE_API(...)       extern "C" __declspec(dllimport) __VA_ARGS__ __cdecl
#            define INFERENCE_ENGINE_API_CPP(...)   __declspec(dllimport) __VA_ARGS__
#            define INFERENCE_ENGINE_API_CLASS(...) __declspec(dllimport) __VA_ARGS__
#        endif
#    else
#        define INFERENCE_ENGINE_API(...)       extern "C" __attribute__((visibility("default"))) __VA_ARGS__
#        define INFERENCE_ENGINE_API_CPP(...)   __attribute__((visibility("default"))) __VA_ARGS__
#        define INFERENCE_ENGINE_API_CLASS(...) __attribute__((visibility("default"))) __VA_ARGS__
#    endif
#endif

#if defined(__GNUC__)
#    define INFERENCE_ENGINE_DEPRECATED(msg) __attribute__((deprecated(msg)))
#    if __GNUC__ >= 6 || defined(__clang__)
#        define INFERENCE_ENGINE_ENUM_DEPRECATED(msg) INFERENCE_ENGINE_DEPRECATED(msg)
#    else
#        define INFERENCE_ENGINE_ENUM_DEPRECATED(msg)
#    endif
#elif defined(_MSC_VER)
#    define INFERENCE_ENGINE_DEPRECATED(msg) __declspec(deprecated(msg))
#    if _MSC_VER >= 1900 /* VS2015 */
#        define INFERENCE_ENGINE_ENUM_DEPRECATED(msg) [[deprecated(msg)]]
#    else
#        define INFERENCE_ENGINE_ENUM_DEPRECATED(msg)
#    endif
#elif defined(__INTEL_COMPILER)
#    define INFERENCE_ENGINE_DEPRECATED(msg)      __attribute__((deprecated(msg)))
#    define INFERENCE_ENGINE_ENUM_DEPRECATED(msg) INFERENCE_ENGINE_DEPRECATED(msg)
#else
#    define INFERENCE_ENGINE_DEPRECATED(msg)
#    define INFERENCE_ENGINE_ENUM_DEPRECATED(msg)
#endif

#define INFERENCE_ENGINE_1_0_DEPRECATED                                                                              \
    INFERENCE_ENGINE_DEPRECATED("The Inference Engine API is deprecated and will be removed in the 2024.0 release. " \
                                "For instructions on transitioning to the new API, please refer to "                 \
                                "https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html")

// Suppress warning "-Wdeprecated-declarations" / C4996
#if defined(__GNUC__)
#    define IE_DO_PRAGMA(x) _Pragma(#    x)
#elif defined(_MSC_VER)
#    define IE_DO_PRAGMA(x) __pragma(x)
#else
#    define IE_DO_PRAGMA(x)
#endif

#if (defined(__GNUC__) && (__GNUC__ * 100 + __GNUC_MINOR__ > 405)) || defined(__clang__)
#    define IE_SUPPRESS_DEPRECATED_START  \
        IE_DO_PRAGMA(GCC diagnostic push) \
        IE_DO_PRAGMA(GCC diagnostic ignored "-Wdeprecated-declarations")
#    define IE_SUPPRESS_DEPRECATED_END IE_DO_PRAGMA(GCC diagnostic pop)
#elif defined(_MSC_VER)
#    define IE_SUPPRESS_DEPRECATED_START \
        IE_DO_PRAGMA(warning(push))      \
        IE_DO_PRAGMA(warning(disable : 4996))
#    define IE_SUPPRESS_DEPRECATED_END IE_DO_PRAGMA(warning(pop))
#elif defined(__INTEL_COMPILER)
#    define IE_SUPPRESS_DEPRECATED_START      \
        IE_DO_PRAGMA(warning(push))           \
        IE_DO_PRAGMA(warning(disable : 1478)) \
        IE_DO_PRAGMA(warning(disable : 1786))
#    define IE_SUPPRESS_DEPRECATED_END IE_DO_PRAGMA(warning(pop))
#else
#    define IE_SUPPRESS_DEPRECATED_START
#    define IE_SUPPRESS_DEPRECATED_END
#endif

#if defined __GNUC__ && (__GNUC__ <= 4 || (__GNUC__ == 5 && __GNUC_MINOR__ <= 5) || \
                         (defined __i386__ || defined __arm__ || defined __aarch64__))
#    define _IE_SUPPRESS_DEPRECATED_START_GCC IE_SUPPRESS_DEPRECATED_START
#    define _IE_SUPPRESS_DEPRECATED_END_GCC   IE_SUPPRESS_DEPRECATED_END
#else
#    define _IE_SUPPRESS_DEPRECATED_START_GCC
#    define _IE_SUPPRESS_DEPRECATED_END_GCC
#endif

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

#ifdef OPENVINO_ENABLE_UNICODE_PATH_SUPPORT
#    ifndef ENABLE_UNICODE_PATH_SUPPORT
#        define ENABLE_UNICODE_PATH_SUPPORT
#    endif
#endif

/**
 * @def INFERENCE_PLUGIN_API(type)
 * @brief Defines Inference Engine Plugin API method
 * @param type A plugin type
 */

#if (defined(_WIN32) || defined(__CYGWIN__)) && defined(IMPLEMENT_INFERENCE_ENGINE_PLUGIN)
#    define INFERENCE_PLUGIN_API(type) extern "C" __declspec(dllexport) type
#else
#    define INFERENCE_PLUGIN_API(type) INFERENCE_ENGINE_API(type)
#endif
