// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/common/dynamic_arguments_impl.hpp"

#include "intel_npu/utils/vm/npu_vm_runtime_api.hpp"
#include "openvino/core/except.hpp"

namespace intel_npu {

DynamicMemRefImpl::~DynamicMemRefImpl() {
    destroyMemRef();
}

void DynamicMemRefImpl::updateMemRefHandleStatus(DynamicMemRefType& memref) {
    if (_memRef == nullptr) {
        createMemRef(memref._dimsCount);
    } else {
        DynamicMemRefType tempMemRef(memref._basePtr,
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

void DynamicMemRefImpl::alignWithHandle(DynamicMemRefType& memref) {
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

void DynamicMemRefImpl::createMemRef(int64_t dimsCount) {
    if (_memRef == nullptr) {
        auto result = npuVMRuntimeCreateMemRef(dimsCount, &_memRef);
        if (result != NPU_VM_RUNTIME_RESULT_SUCCESS) {
            OPENVINO_THROW("Failed to create MemRef handle");
        }
    }
}

void DynamicMemRefImpl::destroyMemRef() {
    if (_memRef != nullptr) {
        npuVMRuntimeDestroyMemRef(_memRef);
        _memRef = nullptr;
    }
}

DynamicArgumentsImpl::~DynamicArgumentsImpl() {
    if (_executeParams.executionContext != nullptr) {
        npuVMRuntimeDestroyExecutionContext(_executeParams.executionContext);
        _executeParams.executionContext = nullptr;
    }
}

void DynamicArgumentsImpl::ensureExecutionContext(npu_vm_runtime_handle_t vmRuntime) {
    if (_executeParams.executionContext != nullptr) {
        return;
    }
    if (npuVMRuntimeCreateExecutionContext(vmRuntime, &_executeParams.executionContext) !=
        NPU_VM_RUNTIME_RESULT_SUCCESS) {
        OPENVINO_THROW("Failed to create a VM execution context");
    }
}

}  // namespace intel_npu
