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

DynamicMemRefType::DynamicMemRefType(DynamicMemRefType&& other) noexcept
    : _basePtr(other._basePtr),
      _data(other._data),
      _offset(other._offset),
      _sizes(std::move(other._sizes)),
      _strides(std::move(other._strides)),
      _dimsCount(other._dimsCount),
      _memRef(other._memRef),
      _ptrUpdated(other._ptrUpdated),
      _shapeUpdated(other._shapeUpdated),
      _strideUpdated(other._strideUpdated) {
    other._memRef = nullptr;
}

DynamicMemRefType& DynamicMemRefType::operator=(DynamicMemRefType&& other) noexcept {
    if (this != &other) {
        destroyMemRef();
        _basePtr = other._basePtr;
        _data = other._data;
        _offset = other._offset;
        _sizes = std::move(other._sizes);
        _strides = std::move(other._strides);
        _dimsCount = other._dimsCount;
        _memRef = other._memRef;
        _ptrUpdated = other._ptrUpdated;
        _shapeUpdated = other._shapeUpdated;
        _strideUpdated = other._strideUpdated;
        other._memRef = nullptr;
    }
    return *this;
}

DynamicMemRefType::~DynamicMemRefType() {
    destroyMemRef();
}

void DynamicMemRefType::updateMemRefHandleStatus() {
    if (_memRef == nullptr) {
        createMemRef();
    } else {
        // Read back the device-side description and diff against our host-side state.
        const void* deviceBasePtr = nullptr;
        const void* deviceData = nullptr;
        int64_t deviceOffset = 0;
        std::vector<int64_t> deviceSizes(_sizes.size());
        std::vector<int64_t> deviceStrides(_strides.size());
        int64_t deviceDimsCount = 0;
        if (npuVMRuntimeParseMemRef(_memRef,
                                    &deviceBasePtr,
                                    &deviceData,
                                    &deviceOffset,
                                    deviceSizes.data(),
                                    deviceStrides.data(),
                                    &deviceDimsCount) != NPU_VM_RUNTIME_RESULT_SUCCESS) {
            OPENVINO_THROW("Failed to parse MemRef handle");
        }
        _ptrUpdated = (_basePtr != deviceBasePtr || _data != deviceData || _offset != deviceOffset);
        _shapeUpdated = (_sizes != deviceSizes);
        _strideUpdated = (_strides != deviceStrides);
    }
    auto result = npuVMRuntimeSetMemRef(_memRef, _basePtr, _data, _offset, _sizes.data(), _strides.data(), _dimsCount);
    if (result != NPU_VM_RUNTIME_RESULT_SUCCESS) {
        OPENVINO_THROW("Failed to update MemRef handle");
    }
}

void DynamicMemRefType::alignWithHandle() {
    if (_memRef == nullptr) {
        return;
    }
    if (npuVMRuntimeParseMemRef(_memRef, &_basePtr, &_data, &_offset, _sizes.data(), _strides.data(), &_dimsCount) !=
        NPU_VM_RUNTIME_RESULT_SUCCESS) {
        OPENVINO_THROW("Failed to parse MemRef handle");
    }
}

void DynamicMemRefType::createMemRef() {
    if (_memRef == nullptr) {
        auto result = npuVMRuntimeCreateMemRef(_dimsCount, &_memRef);
        if (result != NPU_VM_RUNTIME_RESULT_SUCCESS) {
            OPENVINO_THROW("Failed to create MemRef handle");
        }
    }
}

void DynamicMemRefType::destroyMemRef() {
    if (_memRef != nullptr) {
        npuVMRuntimeDestroyMemRef(_memRef);
        _memRef = nullptr;
    }
}

DynamicArguments::~DynamicArguments() {
    if (_executionContext != nullptr) {
        npuVMRuntimeDestroyExecutionContext(_executionContext);
        _executionContext = nullptr;
    }
}

void DynamicArguments::ensureExecutionContext(npu_vm_runtime_handle_t vmRuntime) {
    if (_executionContext != nullptr) {
        return;
    }
    if (npuVMRuntimeCreateExecutionContext(vmRuntime, &_executionContext) != NPU_VM_RUNTIME_RESULT_SUCCESS) {
        OPENVINO_THROW("Failed to create a VM execution context");
    }
}

}  // namespace intel_npu
