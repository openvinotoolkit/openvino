// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>

#include "intel_npu/common/idynamic_graph.hpp"
#include "intel_npu/utils/vm/npu_vm_runtime_api.hpp"
#include "openvino/core/except.hpp"

namespace intel_npu {

/**
 * @brief Implementation of the opaque @c _impl carried by @ref MemRefType.
 * @details Wraps a VM-runtime MemRef handle and tracks whether the host-side description
 * changed since the last sync to the device.
 *
 * The type is shared between the compiler adapter (which seeds the initial bindings) and
 * the level-zero backend pipeline (which drives the VM execution). It MUST stay header-only
 * so that both static libraries can construct/destruct it consistently.
 */
struct DynamicGraphMemRefImpl {
    npu_vm_runtime_mem_ref_handle_t _memRef = nullptr;
    bool _ptrUpdated = false;
    bool _shapeUpdated = false;
    bool _strideUpdated = false;

    DynamicGraphMemRefImpl() = default;

    ~DynamicGraphMemRefImpl() {
        destroyMemRef();
    }

    /// Push the latest host-side description into the underlying VM MemRef and record what changed.
    void updateMemRefHandleStatus(MemRefType& memref) {
        if (_memRef == nullptr) {
            createMemRef(memref._dimsCount);
        } else {
            MemRefType tempMemRef(memref._basePtr,
                                  memref._data,
                                  memref._offset,
                                  memref._sizes,
                                  memref._strides,
                                  memref._dimsCount);
            alignWithHandle(tempMemRef);

            _ptrUpdated = (memref._basePtr != tempMemRef._basePtr || memref._data != tempMemRef._data ||
                           memref._offset != tempMemRef._offset);
            _shapeUpdated = (memref._sizes != tempMemRef._sizes);
            _strideUpdated = (memref._strides != tempMemRef._strides);
        }
        auto result = npuVMRuntimeSetMemRef(_memRef,
                                            memref._basePtr,
                                            memref._data,
                                            memref._offset,
                                            memref._sizes.data(),
                                            memref._strides.data(),
                                            memref._dimsCount);
        if (result != NPU_VM_RUNTIME_RESULT_SUCCESS) {
            OPENVINO_THROW("Failed to update MemRef handle");
        }
    }

    /// Read the description currently stored in the VM MemRef back into @p memref.
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
            OPENVINO_THROW("Failed to parse MemRef handle");
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
 * @brief Implementation of the opaque @c _impl carried by @ref GraphArguments.
 * @details Owns the VM execution context and the flattened MemRef-handle arrays used by
 * @c npuVMRuntimeExecute. The execution context is created lazily by the pipeline on the
 * first execute call and destroyed when this struct is destroyed.
 */
struct DynamicGraphArgumentsImpl {
    std::vector<npu_vm_runtime_mem_ref_handle_t> _inputMemRefs;
    std::vector<npu_vm_runtime_mem_ref_handle_t> _outputMemRefs;
    npu_vm_runtime_execute_params_t _executeParams = {};

    ~DynamicGraphArgumentsImpl() {
        if (_executeParams.executionContext != nullptr) {
            npuVMRuntimeDestroyExecutionContext(_executeParams.executionContext);
            _executeParams.executionContext = nullptr;
        }
    }
};

}  // namespace intel_npu
