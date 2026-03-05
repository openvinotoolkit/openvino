//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#define NOMINMAX

#include "level_zero_wrapper.h"

#include <stdio.h>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <type_traits>
#include <vector>
#include "intel_npu/network_metadata.hpp"
#include "intel_npu/utils/zero/zero_utils.hpp"
// #include "vpux/compiler/core/developer_build_utils.hpp"
// #include "vpux/compiler/dialect/HostExec/params.hpp"
// #include "vpux/utils/IE/network_metadata.hpp"
// #include "vpux/utils/logger/logger.hpp"
#include "vpux_headers/serial_metadata.hpp"
#include "ze_graph_ext.h"

/* #define TEST */

struct MemRefDesc {
    void* data;
    uint64_t offset;
    uint64_t dimCount;
    uint64_t networkArgIndex;
    uint64_t sizes[5];
    uint64_t strides[5];
    uint64_t elementByteSize;
};

// Workaround for win specific save funtions
#ifndef _WIN32
#include <errno.h>
#include <string.h>

// Basic strcpy_s implementation for Linux
inline error_t strcpy_s(char* dest, size_t destsz, const char* src) {
    if (dest == NULL || src == NULL || destsz == 0) {
        return EINVAL;
    }

    size_t srcsz = strlen(src);
    if (srcsz >= destsz) {
        dest[0] = '\0';  // Null terminate even in case of failure
        return ERANGE;
    }

    strcpy(dest, src);
    return 0;
}

template <size_t size>
inline error_t strcpy_s(char (&dest)[size], const char* src) {
    return strcpy_s(dest, size, src);
}

// Basic memcpy_s implementation for Linux
inline error_t memcpy_s(void* dest, size_t destsz, const void* src, size_t count) {
    if (dest == NULL || src == NULL) {
        return EINVAL;
    }

    if (destsz < count) {
        return ERANGE;
    }

    memcpy(dest, src, count);
    return 0;
}
#endif

// End of workaround for win specific save funtions

#define RETURN_SUCCESS() return static_cast<uint32_t>(ZE_RESULT_SUCCESS);

#ifdef TEST
NPU_API(void*) npu_level_zero_alloc(int64_t size, void*) {
    printf("npu_level_zero_alloc was called %lld\n", size);
#if !defined(WIN32)
    void* result = aligned_alloc(64, size);
#else
    void* result = _aligned_malloc(size, 64);
#endif
    return result;
}

NPU_API(int32_t) npu_level_zero_append_memory_copy(void* src, void* dst, int64_t size, void* commandList) {
    printf("npu_level_zero_append_memory_copy was called %ld\n", size);
    RETURN_SUCCESS();
}

NPU_API(int32_t) npu_level_zero_append_barrier(void* commandList) {
    printf("npu_level_zero_append_barrier was called\n");
    RETURN_SUCCESS();
}

NPU_API(int32_t)
npu_level_zero_create_graph(void* kernel, int64_t kernelSize, void* context, void* device, void* ddiTable,
                            void* commandList, void* commandQueue) {
    printf("npu_level_zero_create_graph was called\n");
    RETURN_SUCCESS();
}

NPU_API(int32_t)
npu_level_zero_create_graphs(void** kernels, int64_t* kernelSizes, int32_t numKernels, void* context, void* device,
                             void* ddiTable, void* commandList, void* commandQueue) {
    printf("npu_level_zero_create_graphs was called\n");
    RETURN_SUCCESS();
}

NPU_API(int32_t)
npu_level_zero_execute_graph(void** input, int32_t numInputs, void** output, int32_t numOutputs, void* kernel,
                             int64_t kernelSize, void* context, void* device, void* ddiTable, void* commandList) {
    printf("npu_level_zero_execute_graph was called\n");
    RETURN_SUCCESS();
}

NPU_API(int32_t)
npu_level_zero_submit_commandlist(void* commandList, void* commandQueue, void* fence, void* event) {
    printf("npu_level_zero_submit_commandlist was called\n");
    RETURN_SUCCESS();
}

NPU_API(void)
npu_level_zero_get_last_error(char** pError) {
    printf("npu_level_zero_get_last_error was called\n");
}

NPU_API(int32_t)
npu_level_zero_reset_commandlist(void* commandList) {
    printf("npu_level_zero_reset_commandlist was called\n");
    RETURN_SUCCESS();
}

NPU_API(int32_t)
npu_level_zero_reset_commandlists(void** commandList, int32_t numCommandLists) {
    printf("npu_level_zero_reset_commandlists was called\n");
    RETURN_SUCCESS();
}

NPU_API(int32_t)
npu_level_zero_get_network_metadata(void* metadata, uint64_t metadataSize, void* levelZeroMetadata, void* inputDescs,
                                    void* outputDescs) {
    printf("npu_level_zero_metadata was called\n");
    RETURN_SUCCESS();
}

#else

struct graph_info {
    ze_graph_handle_t graphHandle;
    uint32_t numArgs;
    uint32_t numInputArgs;
    graph_info(): graphHandle(nullptr), numArgs(0), numInputArgs(0) {
    }
    graph_info(ze_graph_handle_t handle, uint32_t numArgs, uint32_t numInputArgs)
            : graphHandle(handle), numArgs(numArgs), numInputArgs(numInputArgs) {
    }
};

constexpr size_t max_message_length = 256;
// std::unique_ptr<vpux::Logger> logger = nullptr;
static char lastErrorMessage[max_message_length];
#define ERROR_HANDLE(result, msg)                                                         \
    if (result != ZE_RESULT_SUCCESS) {                                                    \
        size_t size = std::min(static_cast<size_t>(max_message_length - 1), strlen(msg)); \
        std::strncpy(lastErrorMessage, msg, size);                                        \
        lastErrorMessage[size] = '\0';                                                    \
        return static_cast<int32_t>(result);                                              \
    }

ze_graph_dditable_ext_t* ddiTableHandle = nullptr;
std::map<void*, graph_info>* graphMap = nullptr;

struct scratch_buffer {
    ze_context_handle_t contextHandle;
    void* data;
    int64_t size;

    bool inRange(uint64_t address) const {
        return ((address >= reinterpret_cast<uint64_t>(data)) && (address < (reinterpret_cast<uint64_t>(data) + size)));
    }
};

struct ze_memory_deleter {
    void operator()(scratch_buffer* buffer) const {
        if (buffer != nullptr) {
            if (buffer->data != nullptr) {
                zeMemFree(buffer->contextHandle, buffer->data);
            }
        }
    }
};

struct graph_argument_binding {
    uint64_t cmdId;
    ze_command_list_handle_t commandListHandle;
    uint64_t networkArgIndex;
    uint64_t argIndex;

    uint64_t bufferOffset;
};

std::ostream& operator<<(std::ostream& o, const graph_argument_binding& binding) {
    o << "cmdId: " << binding.cmdId;
    o << ", commandListHandle: 0x" << std::hex << reinterpret_cast<uint64_t>(binding.commandListHandle) << std::dec;
    o << ", networkArgIndex: " << binding.networkArgIndex;
    o << ", argIndex: " << binding.argIndex;
    o << ", bufferOffset: " << binding.bufferOffset;
    return o;
}

struct execution_context {
    // commandListIndex, networkArgIndex, list of bindings for the same network argument in the same command list
    std::vector<std::vector<std::vector<graph_argument_binding>>> argumentBindings;
    std::vector<uint64_t> mutableCommandListIds;
    execution_context(size_t numSubGraphs, size_t numNetworkArgs)
            : argumentBindings(numSubGraphs), mutableCommandListIds(numSubGraphs) {
        for (auto& bindings : argumentBindings) {
            bindings.resize(numNetworkArgs);
        }
    }
    void add_binding(size_t graphIndex, const graph_argument_binding& binding) {
        argumentBindings[graphIndex][binding.networkArgIndex].emplace_back(binding);

        // if (logger) {
        //     std::ostringstream oss;
        //     oss << "Added a binding[" << graphIndex << ", " << binding.networkArgIndex << "] " << binding;
        //     logger->info("{0}", oss.str());
        // }
    }

    std::tuple<uint32_t, std::string> queryDriverExtensionVersion(
            const char* extName, uint32_t extCurrentVersion, std::vector<ze_driver_extension_properties_t>& extProps,
            uint32_t count) {
        const char* functionExtName = nullptr;
        uint32_t targetVersion = 0;

        for (uint32_t i = 0; i < count; ++i) {
            auto& property = extProps[i];

            if (strncmp(property.name, extName, strlen(extName)) != 0) {
                continue;
            }

            if (property.version >= extCurrentVersion) {
                functionExtName = property.name;
                targetVersion = extCurrentVersion;
                break;
            }

            // Use the latest version supported by the driver - We need to go through all the properties for older
            // drivers that use specific names for different graph ext versions, e.g.: ZE_extension_graph_1_1,
            // ZE_extension_graph_1_2
            if (property.version > targetVersion) {
                functionExtName = property.name;
                targetVersion = property.version;
            }
        }

        return std::make_tuple(targetVersion, functionExtName ? functionExtName : "");
    }

    void reset(void** commandList, uint64_t numCommandLists) {
        for (auto& bindingsPerCommandList : argumentBindings) {
            for (auto& bindings : bindingsPerCommandList) {
                bindings.clear();
            }
        }
        if (commandList == nullptr || numCommandLists == 0) {
            for (uint64_t i = 0; i < numCommandLists; ++i) {
                mutableCommandListIds[i] = 0;
            }
        } else {
            ze_command_list_handle_t* commandListHandles = reinterpret_cast<ze_command_list_handle_t*>(commandList);
            if (commandListHandles[0] != nullptr) {
                // the first command list may have some commands recorded in npu plugin
                auto commandListHandle = commandListHandles[0];
                uint64_t cmdId = 0;
                if (commandListHandle != nullptr) {
                    ze_mutable_command_id_exp_desc_t mutable_cmd_id_desc = {};
                    mutable_cmd_id_desc.stype = ZE_STRUCTURE_TYPE_MUTABLE_COMMAND_ID_EXP_DESC;
                    mutable_cmd_id_desc.flags = ZE_MUTABLE_COMMAND_EXP_FLAG_GRAPH_ARGUMENTS;
                    auto result = zeCommandListGetNextCommandIdExp(commandListHandle, &mutable_cmd_id_desc, &cmdId);
                    if (result == ZE_RESULT_ERROR_UNINITIALIZED) {
                        // If the command list is closed, set cmdId to 0
                        cmdId = 0;
                    } else {
                        if (result == ZE_RESULT_ERROR_INVALID_ENUMERATION) {
                            // If ZE_MUTABLE_COMMAND_EXP_FLAG_GRAPH_ARGUMENTS is not supported by the driver, try again
                            // with ZE_MUTABLE_COMMAND_EXP_FLAG_GRAPH_ARGUMENT_DEPRECATED
                            mutable_cmd_id_desc.flags = ZE_MUTABLE_COMMAND_EXP_FLAG_GRAPH_ARGUMENT_DEPRECATED;
                            zeCommandListGetNextCommandIdExp(commandListHandle, &mutable_cmd_id_desc, &cmdId);
                        }
                    }
                }
                mutableCommandListIds[0] = cmdId;
            }

            for (uint64_t i = 1; i < numCommandLists; ++i) {
                // For other command lists, we can just set the mutable command id to 1
                // as no inference execution command will be recorded in those command lists in npu plugin
                mutableCommandListIds[i] = 1;
            }
        }
    }
};

std::unique_ptr<scratch_buffer, ze_memory_deleter> scratchBuffer = nullptr;

// shared library initialization function for ExecutionEngine
NPU_API(void) __mlir_execution_engine_init() {
    graphMap = new std::map<void*, graph_info>();

    // #if defined(VPUX_DEVELOPER_BUILD) || !defined(NDEBUG)
    //     std::string logLevelFlag;
    //     vpux::parseEnv("OV_NPU_LOG_LEVEL", logLevelFlag);
    //     if (logLevelFlag.size() > 0 && std::string(logLevelFlag).find("LOG_") == 0) {
    //         logger = std::make_unique<vpux::Logger>(llvm::StringLiteral("LEVEL_ZERO_WRAPPER"),
    //                                                 vpux::Logger::global().level());
    //     }
    // #endif
}

// shared library destroy function for ExecutionEngine
NPU_API(void) __mlir_execution_engine_destroy() {
    if (graphMap) {
        if (ddiTableHandle) {
            for (auto& g : *graphMap) {
                ddiTableHandle->pfnDestroy(g.second.graphHandle);
                g.second.graphHandle = nullptr;
            }
        }
        delete graphMap;
        graphMap = nullptr;
    }

    if (scratchBuffer) {
        scratchBuffer.reset();
    }

    // if (logger) {
    //     logger.reset();
    // }
}

NPU_API(void) npu_level_zero_init() {
    __mlir_execution_engine_init();
}

NPU_API(void) npu_level_zero_destroy() {
    __mlir_execution_engine_destroy();
}

NPU_API(void*) npu_level_zero_alloc(int64_t bytes, void* context) {
    if (scratchBuffer != nullptr && scratchBuffer->size >= bytes) {
        return scratchBuffer->data;
    }

    if (scratchBuffer == nullptr || scratchBuffer->size < bytes) {
        // if (logger) {
        //     logger->info("Allocating scratch buffer of size {0} bytes", bytes);
        // }
        ze_host_mem_alloc_flag_t flag = {};
        ze_host_mem_alloc_desc_t desc = {ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC, nullptr,
                                         static_cast<ze_host_mem_alloc_flags_t>(flag)};
        auto contextHandle = static_cast<ze_context_handle_t>(context);
        void* data = nullptr;
        // user is responsible for alignment, so we pass 0 for alignment
        int32_t res = zeMemAllocHost(contextHandle, &desc, bytes, /*no alignment*/ 0, &data);

        if (res == ZE_RESULT_SUCCESS) {
            scratchBuffer.reset(new scratch_buffer{contextHandle, data, bytes});

            return data;
        }
    }

    return nullptr;
}

NPU_API(int32_t) npu_level_zero_append_memory_copy(void* src, void* dst, int64_t size, void** commandList) {
    auto commandListHandle =
            (commandList == nullptr) ? nullptr : *reinterpret_cast<ze_command_list_handle_t*>(commandList);
    if (commandListHandle == nullptr) {
        ERROR_HANDLE(ZE_RESULT_ERROR_INVALID_NULL_HANDLE, "Invalid nullpointer");
    }
    auto result = zeCommandListAppendMemoryCopy(commandListHandle, dst, src, size, nullptr, 0, nullptr);
    ERROR_HANDLE(result, "Failed to append memory copy");

    RETURN_SUCCESS();
}

NPU_API(int32_t) npu_level_zero_append_barrier(void* commandList) {
    auto commandListHandle = static_cast<ze_command_list_handle_t>(commandList);
    auto result = zeCommandListAppendBarrier(commandListHandle, nullptr, 0, nullptr);
    if (result != ZE_RESULT_SUCCESS) {
        ERROR_HANDLE(result, "Failed to append barrier");
    }

    RETURN_SUCCESS();
}

NPU_API(int32_t)
npu_level_zero_create_graph(void* kernel, int64_t kernelSize, void* context, void* device, void* ddiTable,
                            void* commandList, void* commandQueue) {
    // if (logger) {
    //     logger->info("Creating graph for kernel at address {0} of size {1}", kernel, kernelSize);
    // }

    auto* ddiTableHandle = static_cast<ze_graph_dditable_ext_t*>(ddiTable);
    if (::ddiTableHandle == nullptr) {
        ::ddiTableHandle = ddiTableHandle;
    }

    ze_graph_desc_t desc = {
            ZE_STRUCTURE_TYPE_GRAPH_DESC,       nullptr, ZE_GRAPH_FORMAT_NATIVE, static_cast<size_t>(kernelSize),
            reinterpret_cast<uint8_t*>(kernel), nullptr};

    auto contextHandle = static_cast<ze_context_handle_t>(context);
    auto deviceHandle = static_cast<ze_device_handle_t>(device);

    ze_pfnGraphCreate_ext_t pfnCreate = ddiTableHandle->pfnCreate;
    ze_graph_handle_t graphHandle = nullptr;
    auto result = pfnCreate(contextHandle, deviceHandle, &desc, &graphHandle);
    ERROR_HANDLE(result, "Failed to create graph");

    ze_graph_properties_t props{};
    props.stype = ZE_STRUCTURE_TYPE_GRAPH_PROPERTIES;
    result = ddiTableHandle->pfnGetProperties(graphHandle, &props);
    auto numInputArguments = 0;
    for (uint32_t index = 0; index < props.numGraphArgs; ++index) {
        ze_graph_argument_properties_3_t arg3{};
        arg3.stype = ZE_STRUCTURE_TYPE_GRAPH_ARGUMENT_PROPERTIES_3;

        ze_graph_argument_property_strides_t strides{ZE_STRUCTURE_TYPE_GRAPH_ARGUMENT_PROPERTY_STRIDES, nullptr, false};
        arg3.pNext = reinterpret_cast<void*>(&strides);
        result = ddiTableHandle->pfnGetArgumentProperties3(graphHandle, index, &arg3);
        ERROR_HANDLE(result, "Failed to get argument properties");

        if (arg3.type == ZE_GRAPH_ARGUMENT_TYPE_INPUT) {
            numInputArguments++;
        }
    }

    auto commandListHandle = static_cast<ze_command_list_handle_t>(commandList);
    auto commandQueueHandle = static_cast<ze_command_queue_handle_t>(commandQueue);
    ze_pfnAppendGraphInitialize_ext_t pfnAppendGraphInitialize = ddiTableHandle->pfnAppendGraphInitialize;

    if (commandListHandle != nullptr) {
        result = pfnAppendGraphInitialize(commandListHandle, graphHandle, /*profiling_query_handle*/ nullptr, 0,
                                          nullptr);
        ERROR_HANDLE(result, "Failed to append graph initialize");
    }

    (*graphMap)[kernel] = graph_info(graphHandle, props.numGraphArgs, numInputArguments);

    // if (logger) {
    //     logger->info("Created graph for kernel");
    // }

    RETURN_SUCCESS();
}

NPU_API(int32_t)
npu_level_zero_create_graphs(void** kernels, int64_t* kernelSizes, int32_t numKernels, void* context, void* device,
                             void* ddiTable, void* commandList, void* commandQueue) {
    auto* ddiTableHandle = static_cast<ze_graph_dditable_ext_t*>(ddiTable);
    auto commandListHandle = static_cast<ze_command_list_handle_t>(commandList);
    auto commandQueueHandle = static_cast<ze_command_queue_handle_t>(commandQueue);
    ze_pfnAppendGraphInitialize_ext_t pfnAppendGraphInitialize = ddiTableHandle->pfnAppendGraphInitialize;

    for (int32_t kernelIndex = 0; kernelIndex < numKernels; ++kernelIndex) {
        ze_graph_desc_t desc = {ZE_STRUCTURE_TYPE_GRAPH_PROPERTIES,
                                nullptr,
                                ZE_GRAPH_FORMAT_NATIVE,
                                static_cast<size_t>(kernelSizes[kernelIndex]),
                                reinterpret_cast<uint8_t*>(kernels[kernelIndex]),
                                nullptr};

        auto contextHandle = static_cast<ze_context_handle_t>(context);
        auto deviceHandle = static_cast<ze_device_handle_t>(device);

        ze_pfnGraphCreate_ext_t pfnCreate = ddiTableHandle->pfnCreate;
        ze_graph_handle_t graphHandle = nullptr;
        auto result = pfnCreate(contextHandle, deviceHandle, &desc, &graphHandle);
        ERROR_HANDLE(result, "Failed to create graph");

        ze_graph_properties_t props{};
        props.stype = ZE_STRUCTURE_TYPE_GRAPH_PROPERTIES;
        result = ddiTableHandle->pfnGetProperties(graphHandle, &props);
        auto numInputArguments = 0;
        for (uint32_t index = 0; index < props.numGraphArgs; ++index) {
            ze_graph_argument_properties_3_t arg3{};
            arg3.stype = ZE_STRUCTURE_TYPE_GRAPH_ARGUMENT_PROPERTIES;
            result = ddiTableHandle->pfnGetArgumentProperties3(graphHandle, index, &arg3);
            ERROR_HANDLE(result, "Failed to get argument properties\n");

            if (arg3.type == ZE_GRAPH_ARGUMENT_TYPE_INPUT) {
                numInputArguments++;
            } else {
                break;
            }
        }

        result = pfnAppendGraphInitialize(commandListHandle, graphHandle, /*profiling_query_handle*/ nullptr, 0,
                                          nullptr);
        ERROR_HANDLE(result, "Failed to append graph initialize");

        (*graphMap)[kernels[kernelIndex]] = graph_info(graphHandle, props.numGraphArgs, numInputArguments);
    }

    RETURN_SUCCESS();
}

int32_t set_arguments(uint64_t index, const MemRefDesc& desc, ze_graph_handle_t graphHandle,
                      ze_graph_dditable_ext_t* ddiTableHandle) {
    ze_result_t result = ZE_RESULT_SUCCESS;

    if (ZE_GRAPH_EXT_VERSION_CURRENT >= ZE_GRAPH_EXT_VERSION_1_15) {
        // Below is an example implementation.
        // When a new graph ext is available in master, this will be finalized.
        //
        ze_graph_argument_value_tensor_t tensor_value{
                ZE_STRUCTURE_TYPE_GRAPH_ARGUMENT_TENSOR, nullptr,
                reinterpret_cast<void*>(reinterpret_cast<uint64_t>(desc.data) + desc.elementByteSize * desc.offset)};

        // Strides information
        ze_graph_argument_value_strides_t tensor_strides = {};
        tensor_strides.stype = ZE_STRUCTURE_TYPE_GRAPH_ARGUMENT_STRIDES;
        tensor_strides.pNext = nullptr;
        for (auto dim = 0; dim < desc.dimCount; dim++) {
            // store strides in reverse order
            tensor_strides.userStrides[dim] = static_cast<uint32_t>(desc.strides[(desc.dimCount - 1) - dim]);
        }
        tensor_value.pNext = reinterpret_cast<void*>(&tensor_strides);
        result = ddiTableHandle->pfnSetArgumentValue2(graphHandle, index, &tensor_value);
    } else {
        result = ddiTableHandle->pfnSetArgumentValue(graphHandle, index, desc.data);
    }

    // if (logger) {
    //     logger->info("{0}", desc);
    // }
    return result;
}

NPU_API(int32_t)
npu_level_zero_execute_graph(void** inputDescs, int32_t numInputs, void** outputDescs, int32_t numOutputs,
                             void* kernelName, void* kernel, int64_t kernelSize, void* context, void* device,
                             void* ddiTable, void** commandList, int64_t commandListIndex, void* execCtx) {
    // if (logger) {
    //     const char* kernelNameStr = static_cast<const char*>(kernelName);
    //     logger->info("Executing graph for kernel {0} at address {1} of size {2} in cmdListIndex {3}", kernelNameStr,
    //                  kernel, kernelSize, commandListIndex);
    // }
    auto inputs = reinterpret_cast<MemRefDesc*>(inputDescs);
    auto outputs = reinterpret_cast<MemRefDesc*>(outputDescs);

    if (inputs == nullptr || outputs == nullptr) {
        ERROR_HANDLE(ZE_RESULT_ERROR_INVALID_NULL_POINTER, "Invalid null pointer");
    }

    if (numInputs <= 0 || numOutputs <= 0) {
        ERROR_HANDLE(ZE_RESULT_ERROR_INVALID_SIZE, "Invalid size");
    }

    graph_info graphInfo = (*graphMap)[kernel];
    if (graphInfo.graphHandle == nullptr) {
        // this is required until graph_init function is generated
        auto result = npu_level_zero_create_graph(kernel, kernelSize, context, device, ddiTable, nullptr, nullptr);

        ERROR_HANDLE(result, "Failed to compile a graph");

        graphInfo = (*graphMap)[kernel];
        if (graphInfo.graphHandle == nullptr) {
            ERROR_HANDLE(ZE_RESULT_ERROR_INVALID_ARGUMENT, "Invalid arguments");
        }
    }
    if (graphInfo.numArgs != (numInputs + numOutputs)) {
        ERROR_HANDLE(ZE_RESULT_ERROR_INVALID_ARGUMENT, "Invalid arguments");
    }
    auto* ddiTableHandle = static_cast<ze_graph_dditable_ext_t*>(ddiTable);

    const auto graphHandle = graphInfo.graphHandle;
    for (uint32_t index = 0; index < graphInfo.numArgs; ++index) {
        // Process inputs
        if (index < graphInfo.numInputArgs) {
            ERROR_HANDLE(set_arguments(index, inputs[index], graphHandle, ddiTableHandle),
                         "Failed to set input arguments");

        } else {
            ERROR_HANDLE(set_arguments(index, outputs[index - graphInfo.numInputArgs], graphHandle, ddiTableHandle),
                         "Failed to set output arguments");
        }
    }

    ze_pfnAppendGraphExecute_ext_t pfnAppendGraphExecute = ddiTableHandle->pfnAppendGraphExecute;
    auto commandListHandle =
            (commandList == nullptr) ? nullptr : *reinterpret_cast<ze_command_list_handle_t*>(commandList);

    if (commandListHandle == nullptr) {
        ERROR_HANDLE(ZE_RESULT_ERROR_INVALID_NULL_HANDLE, "Invalid nullpointer");
    }

    auto result = pfnAppendGraphExecute(commandListHandle, graphHandle, nullptr, nullptr, 0, nullptr);
    if (result == ZE_RESULT_ERROR_UNINITIALIZED) {
        result = zeCommandListReset(commandListHandle);
        ERROR_HANDLE(result, "Failed to reset command list");
        result = pfnAppendGraphExecute(commandListHandle, graphHandle, nullptr, nullptr, 0, nullptr);
    }

    // #if defined(VPUX_DEVELOPER_BUILD) || !defined(NDEBUG)
    //     if ((result != ZE_RESULT_SUCCESS) && logger) {
    //         for (uint32_t index = 0; index < graphInfo.numArgs; ++index) {
    //             vpux::HostExec::MemRefDesc desc;
    //             if (index < graphInfo.numInputArgs) {
    //                 desc = inputs[index];
    //             } else {
    //                 desc = outputs[index - graphInfo.numInputArgs];
    //             }
    //             logger->error("Set argument index({0}) with {1}", index, desc);
    //         }
    //     }
    // #endif

    ERROR_HANDLE(result, "Failed to append graph execute");

    if (execCtx != nullptr) {
        auto* execContext = reinterpret_cast<execution_context*>(execCtx);
        if (commandListIndex >= execContext->mutableCommandListIds.size()) {
            ERROR_HANDLE(ZE_RESULT_ERROR_INVALID_NULL_POINTER, "Invalid commandList Index");
        }

        // Need to increase cmdId for each execute graph call
        // as it is used to index inferences stored in a command list
        uint64_t cmdId = execContext->mutableCommandListIds[commandListIndex]++;

        for (uint32_t index = 0; index < graphInfo.numArgs; ++index) {
            // Process inputs
            if (index < graphInfo.numInputArgs) {
                auto& input = inputs[index];
                // For repeating block use case, inputs from the second iteration will be scratch buffer.
                // so skip those too.
                if (input.networkArgIndex >= graphInfo.numArgs ||
                    (scratchBuffer != nullptr && scratchBuffer->inRange(reinterpret_cast<uint64_t>(input.data)))) {
                    // no need to track this argument as it is not mapped to network argument of main module
                    continue;
                }
                execContext->add_binding(commandListIndex, {cmdId, commandListHandle, input.networkArgIndex, index,
                                                            input.elementByteSize * input.offset});

            } else {
                auto& output = outputs[index - graphInfo.numInputArgs];
                // For repeating block use case, outputs can be from the first iteration to N-1 th iteration.
                // so skip those too.
                if (output.networkArgIndex >= graphInfo.numArgs ||
                    (scratchBuffer != nullptr && scratchBuffer->inRange(reinterpret_cast<uint64_t>(output.data)))) {
                    // no need to track this argument as it is not mapped to network argument of main module
                    continue;
                }
                execContext->add_binding(commandListIndex, {cmdId, commandListHandle, output.networkArgIndex, index,
                                                            output.elementByteSize * output.offset});
            }
        }
    }

    // if (logger) {
    //     logger->info("Executed graph for kernel");
    // }

    RETURN_SUCCESS();
}

NPU_API(int32_t)
npu_level_zero_submit_commandlist(void** commandLists, void* commandQueue, void* fence, void* event) {
    // if (logger) {
    //     logger->info("Submitting command list: fence{0}, event {1}", fence, event);
    // }

    auto commandListHandle =
            (commandLists == nullptr) ? nullptr : *reinterpret_cast<ze_command_list_handle_t*>(commandLists);
    auto commandQueueHandle = static_cast<ze_command_queue_handle_t>(commandQueue);
    auto fenceHandle = static_cast<ze_fence_handle_t>(fence);
    auto eventHandle = static_cast<ze_event_handle_t>(event);
    if (commandListHandle == nullptr) {
        ERROR_HANDLE(ZE_RESULT_ERROR_INVALID_NULL_HANDLE, "Invalid nullpointer");
    }

    // note commnad queue is null when immediate command list is used
    if (commandQueueHandle != nullptr) {
        if (eventHandle != nullptr) {
            zeCommandListAppendBarrier(commandListHandle, nullptr, 0, nullptr);
            zeCommandListAppendSignalEvent(commandListHandle, eventHandle);
            zeCommandListClose(commandListHandle);
            auto result = zeCommandQueueExecuteCommandLists(commandQueueHandle, 1, &commandListHandle, nullptr);
            ERROR_HANDLE(result, "Failed to zeCommandQueueExecuteCommandList");

        } else {
            if (fence == nullptr) {
                // add a barrier at the end of command list to ensure all commands are finished
                zeCommandListAppendBarrier(commandListHandle, nullptr, 0, nullptr);
            }
            zeCommandListClose(commandListHandle);
            auto result = zeCommandQueueExecuteCommandLists(commandQueueHandle, 1, &commandListHandle, fenceHandle);
            ERROR_HANDLE(result, "Failed to zeCommandQueueExecuteCommandList");
        }
    }
    // if (logger) {
    //     logger->info("Submitted command list");
    // }
    RETURN_SUCCESS();
}

NPU_API(void)
npu_level_zero_get_last_error(char** pError) {
    *pError = lastErrorMessage;
}

NPU_API(int32_t)
npu_level_zero_reset_commandlist(void** commandLists) {
    if (commandLists == nullptr) {
        ERROR_HANDLE(ZE_RESULT_ERROR_INVALID_ARGUMENT, "Invalid argument");
    }

    auto commandListHandle = *reinterpret_cast<ze_command_list_handle_t*>(commandLists);

    if (commandListHandle == nullptr) {
        ERROR_HANDLE(ZE_RESULT_ERROR_INVALID_ARGUMENT, "Invalid argument");
    }

    auto result = zeCommandListReset(commandListHandle);
    ERROR_HANDLE(result, "Failed to reset a commandlist");
    RETURN_SUCCESS();
}

NPU_API(int32_t)
npu_level_zero_reset_commandlists(void** commandLists, int32_t numCommandLists) {
    for (int32_t index = 0; index < numCommandLists; ++index) {
        auto commandListHandle = static_cast<ze_command_list_handle_t>(commandLists[index]);

        if (commandListHandle == nullptr) {
            ERROR_HANDLE(ZE_RESULT_ERROR_INVALID_ARGUMENT, "Invalid argument");
        }

        auto result = zeCommandListReset(commandListHandle);
        ERROR_HANDLE(result, "Failed to reset a commandlist");
    }

    RETURN_SUCCESS();
}

NPU_API(int32_t)
npu_level_zero_get_network_metadata(void* metadata, uint64_t metadataSize, void* networkMetadata, void* inputDescs,
                                    void* outputDescs) {
    if (metadata == nullptr || networkMetadata == nullptr || inputDescs == nullptr || outputDescs == nullptr) {
        ERROR_HANDLE(ZE_RESULT_ERROR_INVALID_ARGUMENT, "Invalid argument");
    }

    auto blob = reinterpret_cast<uint8_t*>(metadata);
    auto deserializedMetadata = elf::MetadataSerialization::deserialize(blob, metadataSize);

    auto input_descriptors = reinterpret_cast<std::vector<ArgumentDescriptor>*>(inputDescs);
    auto output_descriptors = reinterpret_cast<std::vector<ArgumentDescriptor>*>(outputDescs);

    intel_npu::NetworkMetadata* network = reinterpret_cast<intel_npu::NetworkMetadata*>(networkMetadata);

    // Populate network metadata from deserialized metadata
    network->name = deserializedMetadata->mIdentification.blob_name;
    network->inputs.resize(deserializedMetadata->mNetInputs.size());
    network->outputs.resize(deserializedMetadata->mNetOutputs.size());
    network->profilingOutputs.resize(deserializedMetadata->mProfilingOutputs.size());
    network->numStreams = deserializedMetadata->mResourceRequirements.nn_slice_count_;

    /////////////////////////////////////////////////////////////////////////////////////////////////////
    // Note: The blow coders are from L0 to initialize as L0 driver does not support IRGraph(LLVM w/ ELFs for subgraphs)
    // This will be remove when dynamic model compilation is supported by L0 (CID)
    static std::map<elf::DType, ze_graph_argument_precision_t> precisions = {
            {elf::DType::DType_NOT_SET, ZE_GRAPH_ARGUMENT_PRECISION_UNKNOWN},
            {elf::DType::DType_FP64, ZE_GRAPH_ARGUMENT_PRECISION_FP64},
            {elf::DType::DType_FP32, ZE_GRAPH_ARGUMENT_PRECISION_FP32},
            {elf::DType::DType_FP16, ZE_GRAPH_ARGUMENT_PRECISION_FP16},
            {elf::DType::DType_U64, ZE_GRAPH_ARGUMENT_PRECISION_UINT64},
            {elf::DType::DType_U32, ZE_GRAPH_ARGUMENT_PRECISION_UINT32},
            {elf::DType::DType_U16, ZE_GRAPH_ARGUMENT_PRECISION_UINT16},
            {elf::DType::DType_U8, ZE_GRAPH_ARGUMENT_PRECISION_UINT8},
            {elf::DType::DType_U4, ZE_GRAPH_ARGUMENT_PRECISION_UINT4},
            {elf::DType::DType_I64, ZE_GRAPH_ARGUMENT_PRECISION_INT64},
            {elf::DType::DType_I32, ZE_GRAPH_ARGUMENT_PRECISION_INT32},
            {elf::DType::DType_I16, ZE_GRAPH_ARGUMENT_PRECISION_INT16},
            {elf::DType::DType_I8, ZE_GRAPH_ARGUMENT_PRECISION_INT8},
            {elf::DType::DType_I4, ZE_GRAPH_ARGUMENT_PRECISION_INT4},
            {elf::DType::DType_BIN, ZE_GRAPH_ARGUMENT_PRECISION_BIN},
            {elf::DType::DType_I4X, ZE_GRAPH_ARGUMENT_PRECISION_NF4},
            {elf::DType::DType_BFP16, ZE_GRAPH_ARGUMENT_PRECISION_BF16}};

    static std::map<size_t, ze_graph_argument_layout_t> layouts = {
            {0x1, ZE_GRAPH_ARGUMENT_LAYOUT_C},         {0x12, ZE_GRAPH_ARGUMENT_LAYOUT_NC},
            {0x21, ZE_GRAPH_ARGUMENT_LAYOUT_CN},       {0x123, ZE_GRAPH_ARGUMENT_LAYOUT_CHW},
            {0x1234, ZE_GRAPH_ARGUMENT_LAYOUT_NCHW},   {0x1342, ZE_GRAPH_ARGUMENT_LAYOUT_NHWC},
            {0x12345, ZE_GRAPH_ARGUMENT_LAYOUT_NCDHW}, {0x13452, ZE_GRAPH_ARGUMENT_LAYOUT_NDHWC}};

    auto set_properties = [](ze_graph_argument_properties_3_t& properties, elf::TensorRef& tensor_desc,
                             elf::TensorRef& network_desc, elf::OVNode& node, intel_npu::IODescriptor& io_desc) {
        strcpy_s<ZE_MAX_GRAPH_ARGUMENT_NAME>(properties.name, tensor_desc.name);
        strcpy_s<ZE_MAX_GRAPH_ARGUMENT_NAME>(properties.debug_friendly_name, node.friendly_name);

        if (node.tensor_names_count == 0) {
            strcpy_s<ZE_MAX_GRAPH_ARGUMENT_NAME>(node.tensor_names[node.tensor_names_count++], tensor_desc.name);
            strcpy_s<ZE_MAX_GRAPH_ARGUMENT_NAME>(node.tensor_names[node.tensor_names_count++], node.friendly_name);
        }

        for (uint32_t i = 0; i < node.tensor_names_count; i++) {
            strcpy_s<ZE_MAX_GRAPH_ARGUMENT_NAME>(properties.associated_tensor_names[i], node.tensor_names[i]);

            io_desc.outputTensorNames.insert(node.tensor_names[i]);
        }
        properties.associated_tensor_names_count = node.tensor_names_count;

        memcpy_s(properties.dims, sizeof(properties.dims), tensor_desc.dimensions,
                 tensor_desc.dimensions_size * sizeof(uint32_t));

        properties.dims_count = tensor_desc.dimensions_size;
        properties.networkPrecision = precisions[network_desc.data_type];
        properties.devicePrecision = precisions[tensor_desc.data_type];
        properties.networkLayout = layouts[network_desc.order];
        properties.deviceLayout = layouts[tensor_desc.order];
    };

    auto set_property_strides = [](ze_graph_argument_property_strides_t& strides, elf::OVNode& node) {
        strides = {ZE_STRUCTURE_TYPE_GRAPH_ARGUMENT_PROPERTY_STRIDES, nullptr, false};
        const auto dynamicDim = std::numeric_limits<uint64_t>::max();
        for (uint32_t index = 0; index < node.shape_size; index++) {
            // store strides in reverse order
            if (node.shape[index] == dynamicDim) {
                strides.supportsDynamicStrides = true;
                break;
            }
        }
    };

    input_descriptors->resize(network->inputs.size());
    for (size_t index = 0; index < input_descriptors->size(); ++index) {
        auto& descriptor = input_descriptors->at(index);
        descriptor.idx = index;
        auto& properties = descriptor.info;
        properties.type = ZE_GRAPH_ARGUMENT_TYPE_INPUT;

        auto tensor = index < deserializedMetadata->mInTensorDescriptors.size()
                              ? deserializedMetadata->mInTensorDescriptors[index]
                              : elf::TensorRef{};
        auto net = index < deserializedMetadata->mNetInputs.size() ? deserializedMetadata->mNetInputs[index]
                                                                   : elf::TensorRef{};
        auto node = index < deserializedMetadata->mOVParameters.size() ? deserializedMetadata->mOVParameters[index]
                                                                       : elf::OVNode{};

        set_properties(properties, tensor, net, node, network->inputs.at(index));
        set_property_strides(descriptor.infoStrides, node);

        properties.quantReverseScale = 1.0f;
        properties.quantZeroPoint = 0;
    }

    output_descriptors->resize(network->outputs.size());
    const auto inputCount = network->inputs.size();
    for (size_t index = 0; index < output_descriptors->size(); ++index) {
        auto& descriptor = output_descriptors->at(index);
        descriptor.idx = index + inputCount;
        auto& properties = descriptor.info;
        properties.type = ZE_GRAPH_ARGUMENT_TYPE_OUTPUT;

        auto tensor = index < deserializedMetadata->mOutTensorDescriptors.size()
                              ? deserializedMetadata->mOutTensorDescriptors[index]
                              : elf::TensorRef{};
        auto net = index < deserializedMetadata->mNetOutputs.size() ? deserializedMetadata->mNetOutputs[index]
                                                                    : elf::TensorRef{};
        auto node = index < deserializedMetadata->mOVResults.size() ? deserializedMetadata->mOVResults[index]
                                                                    : elf::OVNode{};

        set_properties(properties, tensor, net, node, network->outputs.at(index));
        set_property_strides(descriptor.infoStrides, node);

        properties.quantReverseScale = 1.0f;
        properties.quantZeroPoint = 0;
    }

    RETURN_SUCCESS();
}

NPU_API(int32_t)
npu_level_zero_create_execution_context(void* handle, int64_t numSubGraphs, int64_t numNetworkArgs, void** ret) {
    // if (logger) {
    //     logger->info("npu_level_zero_create_execution_context: {0} {1}", numSubGraphs, numNetworkArgs);
    // }
    execution_context* context = new execution_context(numSubGraphs, numNetworkArgs);
    *ret = static_cast<void*>(context);

    RETURN_SUCCESS();
}

NPU_API(int32_t)
npu_level_zero_reset_execution_context(void* handle, void** commandList, int64_t numCommandLists) {
    // if (logger) {
    //     logger->info("npu_level_zero_reset_execution_context");
    // }

    execution_context* context = reinterpret_cast<execution_context*>(handle);
    if (context != nullptr) {
        context->reset(commandList, numCommandLists);
    }

    RETURN_SUCCESS();
}

NPU_API(int32_t)
npu_level_zero_destroy_execution_context(void* handle) {
    // if (logger) {
    //     logger->info("npu_level_zero_destroy_execution_context");
    // }

    execution_context* context = reinterpret_cast<execution_context*>(handle);
    if (context != nullptr) {
        delete context;
    }

    RETURN_SUCCESS();
}

NPU_API(int32_t)
npu_level_zero_update_mutable_command_list(void* handle, void* networkArgArr, uint64_t networkArgArraySize,
                                           void* argIndexArr, uint64_t argIndexSize) {
    // if (logger) {
    //     logger->info("npu_level_zero_update_mutable_command_list");
    // }

    execution_context* context = reinterpret_cast<execution_context*>(handle);
    uint64_t* networkArgArray = reinterpret_cast<uint64_t*>(networkArgArr);
    uint64_t* argIndexArray = reinterpret_cast<uint64_t*>(argIndexArr);
    if (context != nullptr && argIndexArr != nullptr && networkArgArray != nullptr) {
        for (auto& bindingsPerCmdList : context->argumentBindings) {
            for (uint64_t index = 0; index < argIndexSize; ++index) {
                uint64_t argIndex = argIndexArray[index];

                if (argIndex >= networkArgArraySize) {
                    ERROR_HANDLE(ZE_RESULT_ERROR_INVALID_ARGUMENT, "Invalid argument index");
                }

                // Process mutable arguments
                for (auto& binding : bindingsPerCmdList[index]) {
                    const void* bufferPtr = reinterpret_cast<void*>((networkArgArray)[argIndex] + binding.bufferOffset);
                    ze_mutable_graph_argument_exp_desc_t desc = {ZE_STRUCTURE_TYPE_MUTABLE_GRAPH_ARGUMENT_EXP_DESC,
                                                                 nullptr, binding.cmdId,
                                                                 static_cast<uint32_t>(binding.argIndex), bufferPtr};
                    ze_mutable_commands_exp_desc_t mutable_commands_exp_desc_t = {
                            ZE_STRUCTURE_TYPE_MUTABLE_COMMANDS_EXP_DESC, &desc, 0};
                    auto result = zeCommandListUpdateMutableCommandsExp(binding.commandListHandle,
                                                                        &mutable_commands_exp_desc_t);
                    // if (logger) {
                    //     std::ostringstream oss;
                    //     oss << "Updating mutable argument:" << binding << " with buffer pointer " << std::hex
                    //         << bufferPtr << std::dec << ", result: " << result;
                    //     logger->info("{0}", oss.str());
                    // }

                    if (result != ZE_RESULT_SUCCESS) {
                        std::ostringstream oss;
                        oss << "Failed to set mutable argument:" << binding << " with buffer pointer " << std::hex
                            << bufferPtr << std::dec;

                        ERROR_HANDLE(result, oss.str().c_str());
                    }
                }
            }
        }
    } else {
        ERROR_HANDLE(ZE_RESULT_ERROR_INVALID_NULL_POINTER, "Invalid nullpointer");
    }

    RETURN_SUCCESS();
}
#endif
