// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/common/dynamic_arguments.hpp"

#include <sstream>

#include "intel_npu/utils/vm/npu_vm_runtime_api.hpp"
#include "openvino/core/except.hpp"

namespace intel_npu {

void DynamicMemRefType::setArg(const void* arg) {
    _basePtr = _data = arg;
}

void DynamicMemRefType::setSize(const ov::Shape& shape) {
    // Note: check difference between shape from compiler and shape from IR.
    if (_dimsCount == 0) {
        _dimsCount = static_cast<uint32_t>(shape.size());
        _sizes.resize(shape.size());
        _strides.resize(shape.size());
    } else if (_dimsCount != static_cast<int64_t>(shape.size())) {
        OPENVINO_THROW("Dimension count mismatch. Current dimension count: ",
                       _dimsCount,
                       ", new dimension count: ",
                       shape.size());
    }

    for (int64_t i = 0; i < _dimsCount; ++i) {
        _sizes[i] = static_cast<int64_t>(shape[i]);
    }
}

void DynamicMemRefType::setStrides(const ov::Strides& strides, int32_t elementSize) {
    if (_dimsCount == 0) {
        OPENVINO_THROW("Dimension count is zero, shall call setSize before setStrides");
    } else if (_dimsCount != static_cast<int64_t>(strides.size())) {
        OPENVINO_THROW("Dimension count mismatch. Current dimension count: ",
                       _dimsCount,
                       ", new dimension count: ",
                       strides.size());
    }

    for (int64_t i = 0; i < _dimsCount; ++i) {
        _strides[i] = static_cast<int64_t>(strides[i] / elementSize);
    }
}

void DynamicMemRefType::set(const void* arg, int64_t offset, std::shared_ptr<ov::ITensor> tensor) {
    _basePtr = _data = arg;
    _offset = offset;
    if (_dimsCount == 0) {
        _dimsCount = static_cast<uint32_t>(tensor->get_shape().size());
        _sizes.resize(_dimsCount);
        _strides.resize(_dimsCount);
    } else if (_dimsCount != static_cast<int64_t>(tensor->get_shape().size())) {
        OPENVINO_THROW("Dimension count mismatch. Current dimension count: ",
                       _dimsCount,
                       ", new dimension count: ",
                       tensor->get_shape().size());
    }

    auto& shape = tensor->get_shape();
    for (int64_t j = 0; j < _dimsCount; j++) {
        _sizes[j] = static_cast<int64_t>(shape[j]);
    }
    auto& strides = tensor->get_strides();
    size_t elementSize = tensor->get_element_type().bitwidth() < 8 ? 1 : tensor->get_element_type().size();
    for (int64_t j = 0; j < _dimsCount; j++) {
        _strides[j] = static_cast<int64_t>(strides[j] / elementSize);
    }
}

void DynamicMemRefType::updateStride() {
    // Note: NCHW layout style
    uint64_t stride = 1;
    for (int64_t i = _dimsCount - 1; i >= 0; --i) {
        _strides[i] = stride;
        stride *= _sizes[i];
    }
}

// The comparision only checks shape and strides now
bool DynamicMemRefType::compare(const DynamicMemRefType& memref) {
    if (memref._dimsCount != _dimsCount || _sizes.size() != memref._sizes.size() ||
        _strides.size() != memref._strides.size())
        return false;
    size_t dimsCount = static_cast<size_t>(_dimsCount);
    if (memref._sizes.size() != dimsCount || memref._strides.size() != dimsCount)
        return false;
    for (size_t i = 0; i < dimsCount; i++) {
        if (_sizes[i] != memref._sizes[i] || _strides[i] != memref._strides[i]) {
            return false;
        }
    }
    return true;
}

std::ostream& operator<<(std::ostream& os, const DynamicMemRefType& memRef) {
    os << "BasePtr: " << memRef._basePtr << ", Data: " << memRef._data << ", Offset: " << memRef._offset
       << ", Sizes: [";
    for (int64_t size : memRef._sizes) {
        os << size << " ";
    }
    os << "], Strides: [";
    for (int64_t stride : memRef._strides) {
        os << stride << " ";
    }
    os << "]";

    return os;
}

std::string DynamicMemRefType::toString() {
    std::stringstream stream;
    stream << *this;
    return stream.str();
}

void DynamicArguments::setArgumentProperties(uint32_t argi,
                                           const void* argv,
                                           const ov::Shape& sizes,
                                           const std::vector<size_t>& strides) {
    auto assign_slot = [&](DynamicMemRefType& slot) {
        slot._basePtr = slot._data = const_cast<void*>(argv);
        if (slot._dimsCount == 0) {
            slot._dimsCount = static_cast<int64_t>(sizes.size());
            slot._sizes.resize(sizes.size());
            slot._strides.resize(strides.size());
        } else if (slot._dimsCount != static_cast<int64_t>(sizes.size())) {
            OPENVINO_THROW("Dimension count mismatch. Current dimension count: ",
                           slot._dimsCount,
                           ", new dimension count: ",
                           sizes.size());
        }
        for (int64_t i = 0; i < slot._dimsCount; i++) {
            slot._sizes[i] = static_cast<int64_t>(sizes[i]);
            slot._strides[i] = static_cast<int64_t>(strides[i]);
        }
    };

    if (argi < _inputs.size()) {
        assign_slot(_inputs[argi]);
    } else {
        auto idx = argi - _inputs.size();
        if (idx < _outputs.size()) {
            assign_slot(_outputs[idx]);
        }
    }
}

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
