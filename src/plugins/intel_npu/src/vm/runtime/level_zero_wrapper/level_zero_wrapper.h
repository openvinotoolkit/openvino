//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#if defined(_WIN32)
#define NPU_I_CONV __cdecl
#ifdef NPU_I_API_DLL
#define NPU_I_API __declspec(dllexport)
#else
#define NPU_I_API __declspec(dllimport)
#endif
#define NPU_API(ReturnType) extern "C" NPU_I_API ReturnType NPU_I_CONV
#else
#define NPU_I_API __attribute__((visibility("default")))
#define NPU_API(ReturnType) extern "C" NPU_I_API ReturnType
#endif

#include <stdint.h>
#include <ze_graph_ext.h>

/**
 *@brief Structure to hold argument descriptor information
 *@note the struct is a copy of intel_npu::ArgumentDescriptor to avoid dependency on intel_npu lib for now
 */
struct ArgumentDescriptor {
    ze_graph_argument_properties_3_t info;
    ze_graph_argument_property_strides_t infoStrides;
    uint32_t idx;
};

/* @brief Allocate memory available to the device
 * @param bytes Number of bytes to allocate
 * @param context Context handle
 * @return Pointer to the allocated memory
 */
NPU_API(void*) npu_level_zero_alloc(int64_t bytes, void* context);

/* @brief Append a memory copy operation to the command list
 * @param src Source buffer
 * @param dst Destination buffer
 * @param size Number of bytes to copy
 * @param commandList Command list handle
 */
NPU_API(int32_t) npu_level_zero_append_memory_copy(void* src, void* dst, int64_t size, void** commandList);

/* @brief Append a barrier operation to the command list
 * @param commandList Command list handle
 */
NPU_API(int32_t) npu_level_zero_append_barrier(void* commandList);

/* brief Load graph (elf binary) and initialize graph
 * @param kernel Kernel buffer
 * @param kernelSize Size of the kernel buffer
 * @param context Context handle
 * @param device Device handle
 * @param ddiTable DDI table handle
 * @param commandList Command list handle
 * @param commandQueue Command queue handle
 */
NPU_API(int32_t)
npu_level_zero_create_graph(void* kernel, int64_t kernelSize, void* context, void* device, void* ddiTable,
                            void* commandList, void* commandQueue);

/* brief Load graph (elf binary) and initialize graph
 * @param kernels Kernel buffer
 * @param kernelSizes Size of the kernel buffer
 * @param numKernels Number of kernels
 * @param context Context handle
 * @param device Device handle
 * @param ddiTable DDI table handle
 * @param commandList Command list handle
 * @param commandQueue Command queue handle
 */
NPU_API(int32_t)
npu_level_zero_create_graphs(void** kernels, int64_t* kernelSizes, int32_t numKernels, void* context, void* device,
                             void* ddiTable, void* commandList, void* commandQueue);

/* brief Reset command list
 * @param commandList Command list handle
 */
NPU_API(int32_t)
npu_level_zero_reset_commandlist(void** commandList);

/* brief Reset command lists
 * @param commandList Command list handle
 * @param numCommandLists the number of command lists
 */
NPU_API(int32_t)
npu_level_zero_reset_commandlists(void** commandList, int32_t numCommandLists);

/* @brief Record inference execution in a command list
 * @param inputs input tensor descs
 * @param outputs output tensor descs
 * @param kernel Kernel buffer
 * @param context Context handle
 * @param device Device handle
 * @param ddiTable DDI table handle
 * @param commandList Command list handle
 * @note A graph handle will be stored in a map
 */
NPU_API(int32_t)
npu_level_zero_execute_graph(void** inputDescs, int32_t numInputs, void** outputDescs, int32_t numOutputs,
                             void* kernelName, void* kernel, int64_t kernelSize, void* context, void* device,
                             void* ddiTable, void** commandList, int64_t commandListIndex, void* execCtx);

/* @brief Record inference execution in a command list
 * @param commandList Command list handle
 * @param commandQueue Command queue handle
 * @param fence fence handle for sync
 * @param event event handle for sync
 * @note A graph handle will be obtained from a map using a given kernel pointer
 */
NPU_API(int32_t)
npu_level_zero_submit_commandlist(void** commandList, void* commandQueue, void* fence, void* event);

/* @brief Deserializes and returns network metadata
 * @param metadata serialized metadata
 * @param metadataSize size of serialized metadata
 * @param networkMetadata a pointer of vpux::NetworkMetadata
 * @param inputDescs a pointer of std::vector<ArgumentDescriptor> for inputs
 * @param outputDescs a pointer of std::vector<ArgumentDescriptor> for outputs
 */
NPU_API(int32_t)
npu_level_zero_get_network_metadata(void* metadata, uint64_t metadataSize, void* networkMetadata, void* inputDescs,
                                    void* outputDescs);

/* @brief Return an error message
 */
NPU_API(void)
npu_level_zero_get_last_error(char** pError);

/* @brief Creates an executionContext
 * @param handle MLIRRuntime handle
 * @param numArgs the number of arguments (inputs and outputs)
 * @param return handle of created execution context
 */
NPU_API(int32_t)
npu_level_zero_create_execution_context(void* handle, int64_t numSubGraphs, int64_t numNetworkArgs, void** ret);

/* @brief reset an executionContext
 * @param handle execution context handle
 */
NPU_API(int32_t)
npu_level_zero_reset_execution_context(void* handle, void** commandList, int64_t numCommandLists);

/* @brief destroy an executionContext
 * @param handle execution context handle
 */
NPU_API(int32_t)
npu_level_zero_destroy_execution_context(void* handle);

/* @brief Update mutable command list used in execution
 * @param handle MLIRRuntime handle
 * @param networkArgArr array of network arguments (inputs and outputs)
 * @param networkArgArraySize size of network arguments array
 * @param argIndexArr array of argument indices to update
 * @param argIndexSize size of argument indices array
 * @param commandQueue command queue handle
 */
NPU_API(int32_t)
npu_level_zero_update_mutable_command_list(void* handle, void* networkArgArr, uint64_t networkArgArraySize,
                                           void* argIndexArr, uint64_t argIndexSize);

NPU_API(void) npu_level_zero_init();
NPU_API(void) npu_level_zero_destroy();
