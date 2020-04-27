// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief The macro defines a symbol import/export mechanism essential for Microsoft Windows(R) OS.
 *
 * @file ie_api.h
 */
#pragma once

#include "details/ie_no_copy.hpp"

#if defined(USE_STATIC_IE) || (defined(__GNUC__) && (__GNUC__ < 4))
#define INFERENCE_ENGINE_API(...) extern "C" __VA_ARGS__
#define INFERENCE_ENGINE_API_CPP(...) __VA_ARGS__
#define INFERENCE_ENGINE_API_CLASS(...) __VA_ARGS__
#define INFERENCE_ENGINE_CDECL __attribute__((cdecl))
#else
#if defined(_WIN32)
#define INFERENCE_ENGINE_CDECL

#ifdef IMPLEMENT_INFERENCE_ENGINE_API
#define INFERENCE_ENGINE_API(...) extern "C" __declspec(dllexport) __VA_ARGS__ __cdecl
#define INFERENCE_ENGINE_API_CPP(...) __declspec(dllexport) __VA_ARGS__ __cdecl
#define INFERENCE_ENGINE_API_CLASS(...) __declspec(dllexport) __VA_ARGS__
#else
#define INFERENCE_ENGINE_API(...) extern "C" __declspec(dllimport) __VA_ARGS__ __cdecl
#define INFERENCE_ENGINE_API_CPP(...) __declspec(dllimport) __VA_ARGS__ __cdecl
#define INFERENCE_ENGINE_API_CLASS(...) __declspec(dllimport) __VA_ARGS__
#endif
#else
#define INFERENCE_ENGINE_CDECL __attribute__((cdecl))
#define INFERENCE_ENGINE_API(...) extern "C" __attribute__((visibility("default"))) __VA_ARGS__
#define INFERENCE_ENGINE_API_CPP(...) __attribute__((visibility("default"))) __VA_ARGS__
#define INFERENCE_ENGINE_API_CLASS(...) __attribute__((visibility("default"))) __VA_ARGS__
#endif
#endif

#if defined(_WIN32)
#define INFERENCE_ENGINE_DEPRECATED(msg) __declspec(deprecated(msg))
#elif defined __INTEL_COMPILER
#define INFERENCE_ENGINE_DEPRECATED(msg) __attribute__((deprecated(msg)))
#elif defined(__GNUC__)
#define INFERENCE_ENGINE_DEPRECATED(msg) __attribute__((deprecated((msg))))
#else
#define INFERENCE_ENGINE_DEPRECATED(msg)
#endif

#if defined IMPLEMENT_INFERENCE_ENGINE_API || defined IMPLEMENT_INFERENCE_ENGINE_PLUGIN
# define INFERENCE_ENGINE_INTERNAL(msg)
#else
# define INFERENCE_ENGINE_INTERNAL(msg) INFERENCE_ENGINE_DEPRECATED(msg)
#endif

#if defined IMPLEMENT_INFERENCE_ENGINE_API || defined IMPLEMENT_INFERENCE_ENGINE_PLUGIN
# define INFERENCE_ENGINE_INTERNAL_CNNLAYER_CLASS(...) INFERENCE_ENGINE_API_CLASS(__VA_ARGS__)
#else
# define INFERENCE_ENGINE_INTERNAL_CNNLAYER_CLASS(...)                                                                           \
    INFERENCE_ENGINE_INTERNAL("Migrate to IR v10 and work with ngraph::Function directly. The method will be removed in 2020.3") \
    INFERENCE_ENGINE_API_CLASS(__VA_ARGS__)
#endif

// Suppress warning "-Wdeprecated-declarations" / C4996
#if defined(_MSC_VER)
#define IE_DO_PRAGMA(x) __pragma(x)
#elif defined(__GNUC__)
#define IE_DO_PRAGMA(x) _Pragma(#x)
#else
#define IE_DO_PRAGMA(x)
#endif

#if defined(_MSC_VER) && !defined(__clang__)
#define IE_SUPPRESS_DEPRECATED_START \
    IE_DO_PRAGMA(warning(push))      \
    IE_DO_PRAGMA(warning(disable : 4996))
#define IE_SUPPRESS_DEPRECATED_END IE_DO_PRAGMA(warning(pop))
#elif defined(__INTEL_COMPILER)
#define IE_SUPPRESS_DEPRECATED_START \
    IE_DO_PRAGMA(warning(push))      \
    IE_DO_PRAGMA(warning(disable : 1478))
    IE_DO_PRAGMA(warning(disable : 1786))
#define IE_SUPPRESS_DEPRECATED_END IE_DO_PRAGMA(warning(pop))
#elif defined(__clang__) || ((__GNUC__) && (__GNUC__ * 100 + __GNUC_MINOR__ > 405))
#define IE_SUPPRESS_DEPRECATED_START  \
    IE_DO_PRAGMA(GCC diagnostic push) \
    IE_DO_PRAGMA(GCC diagnostic ignored "-Wdeprecated-declarations")
#define IE_SUPPRESS_DEPRECATED_END IE_DO_PRAGMA(GCC diagnostic pop)
#else
#define IE_SUPPRESS_DEPRECATED_START
#define IE_SUPPRESS_DEPRECATED_END
#endif

#ifdef _WIN32
# define IE_SUPPRESS_DEPRECATED_START_WIN IE_SUPPRESS_DEPRECATED_START
# define IE_SUPPRESS_DEPRECATED_END_WIN IE_SUPPRESS_DEPRECATED_END
#else
# define IE_SUPPRESS_DEPRECATED_START_WIN
# define IE_SUPPRESS_DEPRECATED_END_WIN
#endif

#ifndef ENABLE_UNICODE_PATH_SUPPORT
# ifdef _WIN32
#  ifdef __INTEL_COMPILER
#   define ENABLE_UNICODE_PATH_SUPPORT
#  endif
#  if defined _MSC_VER && defined _MSVC_LANG && _MSVC_LANG < 201703L
#   define ENABLE_UNICODE_PATH_SUPPORT
#  endif
# elif defined(__GNUC__) && (__GNUC__ > 5 || (__GNUC__ == 5 && __GNUC_MINOR__ > 2)) || defined(__clang__)
#  define ENABLE_UNICODE_PATH_SUPPORT
# endif
#endif
