//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef NPU_MLIR_RUNTIME_H
#define NPU_MLIR_RUNTIME_H

#if defined(__cplusplus)
#    pragma once
#endif

#include "ze_api.h"
#include "ze_graph_ext.h"

#if defined(__cplusplus)
#    include <cstdint>
#    include <cstdlib>
#else
#    include <stdint.h>
#    include <stdlib.h>
#endif

#if defined(__cplusplus)
extern "C" {
#endif

///////////////////////////////////////////////////////////////////////////////
/// @brief Supported versions
///
/// @details
///     - Graph extension versions contain major and minor attributes, use
///       ::NMR_MAJOR_VERSION and ::NMR_MINOR_VERSION
typedef enum _npu_mlir_runtime_version_t {
    NPU_MLIR_RUNTIME_VERSION_1_0 = ZE_MAKE_VERSION(1, 0),             ///< version 1.0
    NPU_MLIR_RUNTIME_VERSION_CURRENT = NPU_MLIR_RUNTIME_VERSION_1_0,  ///< latest known version
    NPU_MLIR_RUNTIME_VERSION_FORCE_UINT32 = 0x7fffffff,
} npu_mlir_runtime_version_t;

///////////////////////////////////////////////////////////////////////////////
#ifndef NMR_APICALL
#    if defined(_WIN32)
/// @brief Calling convention for all API functions
#        define NMR_APICALL __cdecl
#    else
#        define NMR_APICALL
#    endif  // defined(_WIN32)
#endif      // NMR_APICALL

///////////////////////////////////////////////////////////////////////////////
#ifndef NMR_APIEXPORT
#    if defined(_WIN32)
/// @brief Windows-specific dllexport storage-class attribute
#        define NMR_APIEXPORT __declspec(dllexport)
#    else
#        define NMR_APIEXPORT
#    endif  // defined(_WIN32)
#endif      // NMR_APIEXPORT

///////////////////////////////////////////////////////////////////////////////
/// @brief NPU MLIR runtime handle
typedef struct _npu_mlir_runtime_handle_t* npu_mlir_runtime_handle_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Defined Return/Error codes
typedef enum _nmr_result_t {
    NMR_RESULT_SUCCESS = 0,
    NMR_RESULT_ERROR_INVALID_NULL_POINTER = 0x80000001,
    NMR_RESULT_ERROR_UNKNOWN = 0x8ffffffe,
    NMR_RESULT_FORCE_UINT32 = 0x8fffffff,
} nmr_result_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Blob descriptor
typedef struct _nmr_blob_desc_t {
    size_t inputSize;       ///< [in] Size of input buffer in bytes
    const uint8_t* pInput;  ///< [in] Pointer to input buffer
} nmr_blob_desc_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Runtime properties
typedef struct _npu_mlir_runtime_properties_t {
    uint32_t numOfSubGraphs;
    uint32_t numOfGraphArgs;
} npu_mlir_runtime_properties_t;

typedef struct _npu_mlir_runtime_mem_ref_t {
    const void* basePtr;
    const void* data;
    int64_t offset;
    int64_t sizes[4];
    int64_t strides[4];
} npu_mlir_runtime_mem_ref_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Execute params
typedef struct _npu_mlir_runtime_execute_params_t {
    npu_mlir_runtime_mem_ref_t* pInputs;
    uint32_t numOfInputs;
    npu_mlir_runtime_mem_ref_t* pOutputs;
    uint32_t numOfOutputs;
    ze_context_handle_t ctx;
    ze_device_handle_t device;
    ze_graph_dditable_ext_t* graphDdiTableExt;
    ze_command_list_handle_t* commandLists;
    uint64_t numCommandLists;
    ze_command_queue_handle_t commandQueue;
    ze_fence_handle_t inferenceFence;
    ze_event_handle_t event;
} npu_mlir_runtime_execute_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Init MLIR runtime instance and return handle
NMR_APIEXPORT nmr_result_t NMR_APICALL nmrRuntimeCreate(
    const nmr_blob_desc_t* desc,                ///< [in] pointer to graph descriptor
    npu_mlir_runtime_handle_t* phRuntime,       ///< [out] pointer to handle of mlir runtime object created
    npu_mlir_runtime_properties_t* pProperties  ///< [in] pointer to properties of the runtime
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Destroy MLIR runtime instance
NMR_APIEXPORT nmr_result_t NMR_APICALL nmrRuntimeDestroy(
    npu_mlir_runtime_handle_t hRuntime  ///< [in][release] handle of mlir runtime object to destroy
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Get metadata from MLIR runtime instance
NMR_APIEXPORT nmr_result_t NMR_APICALL nmrRuntimeGetMetadata(
    npu_mlir_runtime_handle_t hRuntime,  ///< [in][release] handle of mlir runtime object
    uint32_t argIndex,
    ze_graph_argument_properties_3_t*
        pGraphArgumentProperties  ///< [in,out] query result for graph argument properties.
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Execute MLIR runtime with params
NMR_APIEXPORT nmr_result_t NMR_APICALL nmrRuntimeExecute(
    npu_mlir_runtime_handle_t hRuntime,         ///< [in][release] handle of mlir runtime object
    npu_mlir_runtime_execute_params_t* pParams  ///< [in] pointer to execution parameters
);

#if defined(__cplusplus)
}  // extern "C"
#endif

#endif  // NPU_MLIR_RUNTIME_H
