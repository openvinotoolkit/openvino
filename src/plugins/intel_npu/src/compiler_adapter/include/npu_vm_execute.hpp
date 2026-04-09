// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>

#include "intel_npu/common/idynamic_graph.hpp"
#include "npu_vm_runtime_api.hpp"
#include "intel_npu/utils/zero/zero_init.hpp"
#include "npu_vm_runtime_api.hpp"
#include "openvino/core/except.hpp"

namespace intel_npu {

using MemRefType = IDynamicGraph::MemRefType;
using GraphArguments = IDynamicGraph::GraphArguments;

/**
 * @brief VM-level implementation for a single MemRef handle.
 *
 * Stored as the @c _impl field (std::shared_ptr<void>) of IDynamicGraph::MemRefType
 * so that higher-level pipeline code can manipulate MemRefs without depending
 * on DynamicGraph internals.
 */
struct MemRefTypeImpl {
    npu_vm_runtime_mem_ref_handle_t _memRef;

    MemRefTypeImpl() : _memRef(nullptr) {}

    ~MemRefTypeImpl() {
        destroyMemRef();
    }

    void UpdateMemRefHandleStatus(MemRefType& memref) {
        // Update current MemRef handle to use latest metadata
        if (_memRef == nullptr) {
            createMemRef(memref._dimsCount);
        }
        auto result = npuVMRuntimeSetMemRef(_memRef,
                                            memref._basePtr,
                                            memref._data,
                                            memref._offset,
                                            memref._sizes.data(),
                                            memref._strides.data(),
                                            memref._dimsCount);
        if (result != NPU_VM_RUNTIME_RESULT_SUCCESS) {
            throw std::runtime_error("Failed to update MemRef handle");
        }
    }

    void alignWithHandle(MemRefType& memref) {
        if (_memRef == nullptr) {
            return;
        }

        if (npuVMRuntimeParseMemRef(_memRef,
                                    &memref._basePtr,
                                    &memref._data,
                                    &memref._offset,
                                    memref._sizes.data(),
                                    memref._strides.data(),
                                    &memref._dimsCount) != NPU_VM_RUNTIME_RESULT_SUCCESS) {
            throw std::runtime_error("Failed to parse MemRef handle");
        }
    }

private:
    void createMemRef(int64_t dimsCount) {
        if (_memRef == nullptr) {
            auto result = npuVMRuntimeCreateMemRef(dimsCount, &_memRef);
            if (result != NPU_VM_RUNTIME_RESULT_SUCCESS) {
                OPENVINO_THROW("Failed to create MemRef handle");
            }
        }
    }

    void destroyMemRef() {
        if (_memRef != nullptr) {
            npuVMRuntimeDestroyMemRef(_memRef);
            _memRef = nullptr;
        }
    }
};

/**
 * @brief VM-level execute-parameter cache for a GraphArguments binding.
 *
 * Stored as the @c _impl field (std::shared_ptr<void>) of IDynamicGraph::GraphArguments.
 * Caches the flattened MemRef handle vectors so they are not reallocated on every
 * inference.
 */
struct GraphArgumentsImpl : public GraphArguments {
    std::vector<npu_vm_runtime_mem_ref_handle_t> _inputMemRefs;
    std::vector<npu_vm_runtime_mem_ref_handle_t> _outputMemRefs;
    npu_vm_runtime_execute_params_t _executeParams = {};
};

/**
 * @brief Execute an VM graph directly, using the per-pipeline GraphArguments binding.
 *
 * This function is the authoritative implementation for DynamicPipeline::push().
 * It is intentionally decoupled from DynamicGraph so that the graph object
 * does not hold inference-request-specific state.
 *
 * @param engine    VM runtime engine handle obtained via DynamicGraph::get_vm_engine().
 * @param zeroInitStruct  Level Zero device/context handles.
 * @param args            Per-inference-request I/O binding (owned by the pipeline).
 * @param commandLists    Level Zero command list handles (owned by PipelinedCommandLists).
 * @param commandQueue    Level Zero command queue handle.
 * @param fence           Fence for synchronisation, or nullptr when using events.
 * @param event           Event for synchronisation, or nullptr when using fences.
 */
void vm_execute_graph(npu_vm_runtime_handle_t engine,
                      const std::shared_ptr<ZeroInitStructsHolder>& zeroInitStruct,
                      IDynamicGraph::GraphArguments& args,
                      std::vector<ze_command_list_handle_t>& commandLists,
                      ze_command_queue_handle_t commandQueue,
                      ze_fence_handle_t fence,
                      ze_event_handle_t event);

}  // namespace intel_npu
