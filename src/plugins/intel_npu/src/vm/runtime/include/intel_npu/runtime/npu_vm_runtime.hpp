// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef NPU_VM_RUNTIME_H
#define NPU_VM_RUNTIME_H

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
#ifndef NPU_VM_RUNTIME_APICALL
#    if defined(_WIN32)
/// @brief Calling convention for all API functions
#        define NPU_VM_RUNTIME_APICALL __cdecl
#    else
#        define NPU_VM_RUNTIME_APICALL
#    endif  // defined(_WIN32)
#endif      // NPU_VM_RUNTIME_APICALL

///////////////////////////////////////////////////////////////////////////////
#ifndef NPU_VM_RUNTIME_APIEXPORT
#    if defined(_WIN32)
/// @brief Windows-specific dllexport storage-class attribute
#        define NPU_VM_RUNTIME_APIEXPORT __declspec(dllexport)
#    else
#        define NPU_VM_RUNTIME_APIEXPORT
#    endif  // defined(_WIN32)
#endif      // NPU_VM_RUNTIME_APIEXPORT

///////////////////////////////////////////////////////////////////////////////
#ifndef NPU_VM_RUNTIME_MAKE_VERSION
/// @brief Generates generic API versions
#    define NPU_VM_RUNTIME_MAKE_VERSION(_major, _minor) ((_major << 16) | (_minor & 0x0000ffff))
#endif  // NPU_VM_RUNTIME_MAKE_VERSION

///////////////////////////////////////////////////////////////////////////////
#ifndef NPU_VM_RUNTIME_MAJOR_VERSION
/// @brief Extracts API major version
#    define NPU_VM_RUNTIME_MAJOR_VERSION(_ver) (_ver >> 16)
#endif  // NPU_VM_RUNTIME_MAJOR_VERSION

///////////////////////////////////////////////////////////////////////////////
#ifndef NPU_VM_RUNTIME_MINOR_VERSION
/// @brief Extracts API minor version
#    define NPU_VM_RUNTIME_MINOR_VERSION(_ver) (_ver & 0x0000ffff)
#endif  // NPU_VM_RUNTIME_MINOR_VERSION

///////////////////////////////////////////////////////////////////////////////
/// @brief Supported versions
///
/// @details
///     - VM runtime versions contain major and minor attributes, use
///       ::NPU_VM_RUNTIME_MAJOR_VERSION and ::NPU_VM_RUNTIME_MINOR_VERSION
typedef enum _npu_vm_runtime_version_t {
    NPU_VM_RUNTIME_VERSION_1_0 = ZE_MAKE_VERSION(1, 0),             ///< version 1.0
    NPU_VM_RUNTIME_VERSION_1_1 = ZE_MAKE_VERSION(1, 1),             ///< version 1.1
    NPU_VM_RUNTIME_VERSION_CURRENT = NPU_VM_RUNTIME_VERSION_1_1,  ///< latest known version
    NPU_VM_RUNTIME_VERSION_FORCE_UINT32 = 0x7fffffff,
} npu_vm_runtime_version_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief NPU VM runtime handle
typedef struct _npu_vm_runtime_handle_t* npu_vm_runtime_handle_t;

//////////////////////////////////////////////////////////////////////////////
/// @brief NPU VM runtime MemRef handle
typedef struct _npu_vm_runtime_mem_ref_handle_t* npu_vm_runtime_mem_ref_handle_t;

//////////////////////////////////////////////////////////////////////////////
/// @brief NPU VM runtime execution context handle
typedef struct _npu_vm_runtime_execution_context_handle_t* npu_vm_runtime_execution_context_handle_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Defined Return/Error codes
typedef enum _npu_vm_runtime_result_t {
    NPU_VM_RUNTIME_RESULT_SUCCESS = 0,
    NPU_VM_RUNTIME_RESULT_ERROR_INVALID_NULL_POINTER = 0x80000001,
    NPU_VM_RUNTIME_RESULT_ERROR_UNSUPPORTED_DIM_COUNT = 0x80000002,
    NPU_VM_RUNTIME_RESULT_ERROR_UNSUPPORTED_VERSION = 0x80000003,
    NPU_VM_RUNTIME_RESULT_ERROR_UNKNOWN = 0x8ffffffe,
    NPU_VM_RUNTIME_RESULT_FORCE_UINT32 = 0x8fffffff,
} npu_vm_runtime_result_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Blob descriptor
typedef struct _npu_vm_runtime_blob_desc_t {
    size_t inputSize;       ///< [in] Size of input buffer in bytes
    const uint8_t* pInput;  ///< [in] Pointer to input buffer
} npu_vm_runtime_blob_desc_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Runtime properties
typedef struct _npu_vm_runtime_properties_t {
    uint32_t numOfSubGraphs;
    uint32_t numOfGraphArgs;
} npu_vm_runtime_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Execute params
typedef struct _npu_vm_runtime_execute_params_t {
    npu_vm_runtime_mem_ref_handle_t* pInputs;
    uint32_t numOfInputs;
    npu_vm_runtime_mem_ref_handle_t* pOutputs;
    uint32_t numOfOutputs;
    ze_context_handle_t ctx;
    ze_device_handle_t device;
    ze_graph_dditable_ext_t* graphDdiTableExt;
    ze_command_list_handle_t* commandLists;
    uint64_t numCommandLists;
    ze_command_queue_handle_t commandQueue;
    ze_fence_handle_t inferenceFence;
    ze_event_handle_t event;
    npu_vm_runtime_execution_context_handle_t executionContext;
} npu_vm_runtime_execute_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Predict output shape params
typedef struct _npu_vm_runtime_predict_output_shape_params_t {
    npu_vm_runtime_mem_ref_handle_t* pInputs;
    uint32_t numOfInputs;
    npu_vm_runtime_mem_ref_handle_t* pOutputs;
    uint32_t numOfOutputs;
} npu_vm_runtime_predict_output_shape_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Returns the API version
NPU_VM_RUNTIME_APIEXPORT npu_vm_runtime_result_t NPU_VM_RUNTIME_APICALL npuVMRuntimeGetAPIVersion(
    npu_vm_runtime_version_t* pVersion  ///< [out] pointer to version information
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Init VM runtime instance and return handle
NPU_VM_RUNTIME_APIEXPORT npu_vm_runtime_result_t NPU_VM_RUNTIME_APICALL npuVMRuntimeCreate(
    const npu_vm_runtime_blob_desc_t* desc,   ///< [in] pointer to graph descriptor
    npu_vm_runtime_handle_t* phRuntime,       ///< [out] pointer to handle of VM runtime object created
    npu_vm_runtime_properties_t* pProperties  ///< [out] pointer to properties of the runtime
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Destroy VM runtime instance
NPU_VM_RUNTIME_APIEXPORT npu_vm_runtime_result_t NPU_VM_RUNTIME_APICALL npuVMRuntimeDestroy(
    npu_vm_runtime_handle_t hRuntime  ///< [in][release] handle of VM runtime object to destroy
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Get metadata from VM runtime instance
NPU_VM_RUNTIME_APIEXPORT npu_vm_runtime_result_t NPU_VM_RUNTIME_APICALL npuVMRuntimeGetMetadata(
    npu_vm_runtime_handle_t hRuntime,                          ///< [in] handle of VM runtime object
    uint32_t argIndex,                                           ///< [in] index of the argument
    ze_graph_argument_properties_3_t* pGraphArgumentProperties,  ///< [out] query result for graph argument properties
    ze_graph_argument_metadata_t* pGraphArgumentMetadata,        ///< [out] query result for graph argument metadata
    int64_t* upperBound);

///////////////////////////////////////////////////////////////////////////////
/// @brief Execute VM runtime with params
NPU_VM_RUNTIME_APIEXPORT npu_vm_runtime_result_t NPU_VM_RUNTIME_APICALL npuVMRuntimeExecute(
    npu_vm_runtime_handle_t hRuntime,         ///< [in] handle of VM runtime object
    npu_vm_runtime_execute_params_t* pParams  ///< [in] pointer to execution parameters
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Predit output shape based on input shape
NPU_VM_RUNTIME_APIEXPORT npu_vm_runtime_result_t NPU_VM_RUNTIME_APICALL npuVMRuntimePredictOutputShape(
    npu_vm_runtime_handle_t hRuntime,                      ///< [in] handle of VM runtime object
    npu_vm_runtime_predict_output_shape_params_t* pParams  ///< [in] pointer to predict output shape parameters
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Create MemRef handle
NPU_VM_RUNTIME_APIEXPORT npu_vm_runtime_result_t NPU_VM_RUNTIME_APICALL npuVMRuntimeCreateMemRef(
    int64_t dimsCount,                             ///< [in] value of tensor rank
    npu_vm_runtime_mem_ref_handle_t* phMemRef);  ///< [out] handle of VM runtime MemRef object

///////////////////////////////////////////////////////////////////////////////
/// @brief Destroy MemRef handle
NPU_VM_RUNTIME_APIEXPORT npu_vm_runtime_result_t NPU_VM_RUNTIME_APICALL
npuVMRuntimeDestroyMemRef(npu_vm_runtime_mem_ref_handle_t hMemRef);  ///< [out] handle of VM runtime MemRef object

///////////////////////////////////////////////////////////////////////////////
/// @brief Set new value to MemRef
NPU_VM_RUNTIME_APIEXPORT npu_vm_runtime_result_t NPU_VM_RUNTIME_APICALL
npuVMRuntimeSetMemRef(npu_vm_runtime_mem_ref_handle_t hMemRef,  ///< [in] handle of VM runtime MemRef object
                        const void* basePtr,                        ///< [in] pointer to basePtr
                        const void* data,                           ///< [in] pointer to data
                        int64_t offset,                             ///< [in] offset in MemRef
                        int64_t* pSizes,                            ///< [in] pointer to tensor sizes
                        int64_t* pStrides,                          ///< [in] pointer to tensor strides
                        int64_t dimsCount);                         ///< [in] value of tensor rank

///////////////////////////////////////////////////////////////////////////////
/// @brief Set new value to MemRef
NPU_VM_RUNTIME_APIEXPORT npu_vm_runtime_result_t NPU_VM_RUNTIME_APICALL
npuVMRuntimeParseMemRef(npu_vm_runtime_mem_ref_handle_t hMemRef,  ///< [in] handle of VM runtime MemRef object
                          const void** pBasePtr,                      ///< [out] pointer to basePtr
                          const void** pData,                         ///< [out] pointer to data
                          int64_t* pOffset,                           ///< [out] offset in MemRef
                          int64_t* pSizes,                            ///< [out] pointer to tensor sizes
                          int64_t* pStrides,                          ///< [out] pointer to tensor strides
                          int64_t* pDimsCount);                       ///< [out] value of tensor rank

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @brief Extension version 1.1

///////////////////////////////////////////////////////////////////////////////
/// @brief Init VM runtime instance and return handle
NPU_VM_RUNTIME_APIEXPORT npu_vm_runtime_result_t NPU_VM_RUNTIME_APICALL npuVMRuntimeCreateExecutionContext(
    npu_vm_runtime_handle_t hRuntime,  ///< [in] handle of VM runtime object
    npu_vm_runtime_execution_context_handle_t*
        phExecutionHandle  ///< [out] pointer to handle of VM runtime execution context created
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Destroy VM runtime instance
NPU_VM_RUNTIME_APIEXPORT npu_vm_runtime_result_t NPU_VM_RUNTIME_APICALL npuVMRuntimeDestroyExecutionContext(
    npu_vm_runtime_execution_context_handle_t
        phExecutionHandle  ///< [in][release] handle of execution context object to destroy
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Update mutable command list used in execution and execute
NPU_VM_RUNTIME_APIEXPORT npu_vm_runtime_result_t NPU_VM_RUNTIME_APICALL npuVMRuntimeUpdateMutableCommandList(
    npu_vm_runtime_handle_t hRuntime,          ///< [in] handle of VM runtime object
    npu_vm_runtime_execute_params_t* pParams,  ///< [in] pointer to execution parameters
    uint64_t* argIndexArray,                     ///< [in] pointer to argument index list
    uint64_t argIndexArraySize);                 ///< [in] size of argument index list

#if defined(__cplusplus)
}  // extern "C"
#endif

#endif  // NPU_VM_RUNTIME_H
