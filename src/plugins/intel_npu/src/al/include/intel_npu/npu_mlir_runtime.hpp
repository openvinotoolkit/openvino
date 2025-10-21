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
///       ::NPU_MLIR_RUNTIME_MAJOR_VERSION and ::NPU_MLIR_RUNTIME_MINOR_VERSION
typedef enum _npu_mlir_runtime_version_t {
    NPU_MLIR_RUNTIME_VERSION_1_0 = ZE_MAKE_VERSION(1, 0),             ///< version 1.0
    NPU_MLIR_RUNTIME_VERSION_CURRENT = NPU_MLIR_RUNTIME_VERSION_1_0,  ///< latest known version
    NPU_MLIR_RUNTIME_VERSION_FORCE_UINT32 = 0x7fffffff,
} npu_mlir_runtime_version_t;

///////////////////////////////////////////////////////////////////////////////
#ifndef NPU_MLIR_RUNTIME_APICALL
#    if defined(_WIN32)
/// @brief Calling convention for all API functions
#        define NPU_MLIR_RUNTIME_APICALL __cdecl
#    else
#        define NPU_MLIR_RUNTIME_APICALL
#    endif  // defined(_WIN32)
#endif      // NPU_MLIR_RUNTIME_APICALL

///////////////////////////////////////////////////////////////////////////////
#ifndef NPU_MLIR_RUNTIME_APIEXPORT
#    if defined(_WIN32)
/// @brief Windows-specific dllexport storage-class attribute
#        define NPU_MLIR_RUNTIME_APIEXPORT __declspec(dllexport)
#    else
#        define NPU_MLIR_RUNTIME_APIEXPORT
#    endif  // defined(_WIN32)
#endif      // NPU_MLIR_RUNTIME_APIEXPORT

///////////////////////////////////////////////////////////////////////////////
/// @brief NPU MLIR runtime handle
typedef struct _npu_mlir_runtime_handle_t* npu_mlir_runtime_handle_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Defined Return/Error codes
typedef enum _npu_mlir_runtime_result_t {
    NPU_MLIR_RUNTIME_RESULT_SUCCESS = 0,
    NPU_MLIR_RUNTIME_RESULT_ERROR_INVALID_NULL_POINTER = 0x80000001,
    NPU_MLIR_RUNTIME_RESULT_ERROR_UNKNOWN = 0x8ffffffe,
    NPU_MLIR_RUNTIME_RESULT_FORCE_UINT32 = 0x8fffffff,
} npu_mlir_runtime_result_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Blob descriptor
typedef struct _npu_mlir_runtime_blob_desc_t {
    size_t inputSize;       ///< [in] Size of input buffer in bytes
    const uint8_t* pInput;  ///< [in] Pointer to input buffer
} npu_mlir_runtime_blob_desc_t;

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
    npu_mlir_runtime_mem_ref_t** pInputs;
    uint32_t numOfInputs;
    npu_mlir_runtime_mem_ref_t** pOutputs;
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
NPU_MLIR_RUNTIME_APIEXPORT npu_mlir_runtime_result_t NPU_MLIR_RUNTIME_APICALL npuMLIRRuntimeCreate(
    const npu_mlir_runtime_blob_desc_t* desc,   ///< [in] pointer to graph descriptor
    npu_mlir_runtime_handle_t* phRuntime,       ///< [out] pointer to handle of mlir runtime object created
    npu_mlir_runtime_properties_t* pProperties  ///< [in] pointer to properties of the runtime
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Destroy MLIR runtime instance
NPU_MLIR_RUNTIME_APIEXPORT npu_mlir_runtime_result_t NPU_MLIR_RUNTIME_APICALL npuMLIRRuntimeDestroy(
    npu_mlir_runtime_handle_t hRuntime  ///< [in][release] handle of mlir runtime object to destroy
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Get metadata from MLIR runtime instance
NPU_MLIR_RUNTIME_APIEXPORT npu_mlir_runtime_result_t NPU_MLIR_RUNTIME_APICALL
npuMLIRRuntimeGetMetadata(npu_mlir_runtime_handle_t hRuntime,  ///< [in][release] handle of mlir runtime object
                          uint32_t argIndex,
                          ze_graph_argument_properties_3_t*
                              pGraphArgumentProperties,  ///< [in,out] query result for graph argument properties.
                          _ze_graph_argument_metadata_t* pGraphArgumentMetadata);

///////////////////////////////////////////////////////////////////////////////
/// @brief Execute MLIR runtime with params
NPU_MLIR_RUNTIME_APIEXPORT npu_mlir_runtime_result_t NPU_MLIR_RUNTIME_APICALL npuMLIRRuntimeExecute(
    npu_mlir_runtime_handle_t hRuntime,         ///< [in][release] handle of mlir runtime object
    npu_mlir_runtime_execute_params_t* pParams  ///< [in] pointer to execution parameters
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Predit output shape based on input shape
NPU_MLIR_RUNTIME_APIEXPORT npu_mlir_runtime_result_t NPU_MLIR_RUNTIME_APICALL npuMLIRRuntimePredictOutputShape(
    npu_mlir_runtime_handle_t hRuntime,              ///< [in][release] handle of mlir runtime object
    ze_graph_argument_properties_3_t** pInputArgs,   ///< [in] pointer to input argument properties
    uint32_t numOfInputArgs,                         ///< [in] number of input arguments
    ze_graph_argument_properties_3_t** pOutputArgs,  ///< [out] pointer to output argument properties
    uint32_t numOfOutputArgs                         ///< [in] number of
);

#if defined(__cplusplus)
}  // extern "C"
#endif

#endif  // NPU_MLIR_RUNTIME_H
