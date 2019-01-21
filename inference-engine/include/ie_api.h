// Copyright (C) 2018 Intel Corporation
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
