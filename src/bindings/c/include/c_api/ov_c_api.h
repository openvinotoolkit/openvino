// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @file ov_c_api.h
 * C API of OpenVINO 2.0 bridge unlocks using of OpenVINO 2.0
 * library and all its plugins in native applications disabling usage
 * of C++ API. The scope of API covers significant part of C++ API and includes
 * an ability to read model from the disk, modify input and output information
 * to correspond their runtime representation like data types or memory layout,
 * load in-memory model to different devices including
 * heterogeneous and multi-device modes, manage memory where input and output
 * is allocated and manage inference flow.
**/

/**
 *  @defgroup ov_c_api OpenVINO 2.0 C API
 * OpenVINO 2.0 C API
 */

#ifndef IE_C_API_H
#define IE_C_API_H

#include <stdint.h>
#include <stdio.h>

#ifdef __cplusplus
    #define INFERENCE_ENGINE_C_API_EXTERN extern "C"
#else
    #define INFERENCE_ENGINE_C_API_EXTERN
#endif

#if defined(OPENVINO_STATIC_LIBRARY) || defined(__GNUC__) && (__GNUC__ < 4)
    #define INFERENCE_ENGINE_C_API(...) INFERENCE_ENGINE_C_API_EXTERN __VA_ARGS__
    #define IE_NODISCARD
#else
    #if defined(_WIN32)
        #define INFERENCE_ENGINE_C_API_CALLBACK __cdecl
        #ifdef openvino_c_EXPORTS
            #define INFERENCE_ENGINE_C_API(...) INFERENCE_ENGINE_C_API_EXTERN   __declspec(dllexport) __VA_ARGS__ __cdecl
        #else
            #define INFERENCE_ENGINE_C_API(...) INFERENCE_ENGINE_C_API_EXTERN  __declspec(dllimport) __VA_ARGS__ __cdecl
        #endif
        #define IE_NODISCARD
    #else
        #define INFERENCE_ENGINE_C_API(...) INFERENCE_ENGINE_C_API_EXTERN __attribute__((visibility("default"))) __VA_ARGS__
        #define IE_NODISCARD __attribute__((warn_unused_result))
    #endif
#endif

#ifndef INFERENCE_ENGINE_C_API_CALLBACK
    #define INFERENCE_ENGINE_C_API_CALLBACK
#endif


typedef struct ov_core ov_core_t;

#endif  // IE_C_API_H
