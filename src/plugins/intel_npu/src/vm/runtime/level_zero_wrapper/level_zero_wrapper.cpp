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
#include <map>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <vector>


#define RETURN_SUCCESS() return static_cast<uint32_t>(ZE_RESULT_SUCCESS);

NPU_API(void*) npu_level_zero_alloc(int64_t size, void*) {
    printf("npu_level_zero_alloc was called %ld\n", size);
#if !defined(WIN32)
    void* result = aligned_alloc(64, size);
#else
    void* result = _aligned_malloc(size, 64);
#endif
    return result;
}

NPU_API(int32_t) npu_level_zero_append_memory_copy(void* src, void* dst, int64_t size, void** commandList) {
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
npu_level_zero_execute_graph(void** inputDescs, int32_t numInputs, void** outputDescs, int32_t numOutputs,
                             void* kernelName, void* kernel, int64_t kernelSize, void* context, void* device,
                             void* ddiTable, void** commandList) {
    printf("npu_level_zero_submit_commandlist was called\n");
    RETURN_SUCCESS();
}

NPU_API(void)
npu_level_zero_get_last_error(char** pError) {
    printf("npu_level_zero_get_last_error was called\n");
}

NPU_API(int32_t)
npu_level_zero_reset_commandlist(void** commandList) {
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
