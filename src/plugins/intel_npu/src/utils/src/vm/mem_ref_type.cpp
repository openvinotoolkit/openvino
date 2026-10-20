// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/utils/vm/mem_ref_type.hpp"

#include <sstream>

#include "intel_npu/utils/vm/npu_vm_runtime_api.hpp"
#include "openvino/core/except.hpp"

namespace intel_npu {

void MemRefType::setArg(const void* arg) {
    if (_basePtr != arg || _data != arg) {
        markDirty(PTR_DIRTY);
    }
    _basePtr = _data = arg;
}

void MemRefType::setOffset(int64_t offset) {
    if (_offset != offset) {
        markDirty(PTR_DIRTY);
    }
    _offset = offset;
}

void MemRefType::setSize(const ov::Shape& shape) {
    // Note: check difference between shape from compiler and shape from IR.
    if (_dimsCount == 0) {
        if (!_sizes.empty() || !_strides.empty() || _dimsCount != static_cast<int64_t>(shape.size())) {
            markDirty(SHAPE_DIRTY);
        }
        _dimsCount = static_cast<int64_t>(shape.size());
        _sizes.resize(shape.size());
        _strides.resize(shape.size());
    } else if (_dimsCount != static_cast<int64_t>(shape.size())) {
        OPENVINO_THROW("Dimension count mismatch. Current dimension count: ",
                       _dimsCount,
                       ", new dimension count: ",
                       shape.size());
    }

    for (int64_t dimIndex = 0; dimIndex < _dimsCount; ++dimIndex) {
        const auto size = static_cast<int64_t>(shape[dimIndex]);
        if (_sizes[dimIndex] != size) {
            markDirty(SHAPE_DIRTY);
        }
        _sizes[dimIndex] = size;
    }
}

void MemRefType::setStrides(const ov::Strides& strides, int32_t elementSize) {
    if (_dimsCount == 0 && !strides.empty()) {
        OPENVINO_THROW("Dimension count is zero, shall call setSize before setStrides");
    } else if (_dimsCount != static_cast<int64_t>(strides.size())) {
        OPENVINO_THROW("Dimension count mismatch. Current dimension count: ",
                       _dimsCount,
                       ", new dimension count: ",
                       strides.size());
    }

    for (int64_t dimIndex = 0; dimIndex < _dimsCount; ++dimIndex) {
        const auto stride = static_cast<int64_t>(strides[dimIndex] / elementSize);
        if (_strides[dimIndex] != stride) {
            markDirty(STRIDE_DIRTY);
        }
        _strides[dimIndex] = stride;
    }
}

void MemRefType::set(const void* arg, int64_t offset, std::shared_ptr<ov::ITensor> tensor) {
    setArg(arg);
    setOffset(offset);
    setSize(tensor->get_shape());
    size_t elementSize = tensor->get_element_type().bitwidth() < 8 ? 1 : tensor->get_element_type().size();
    setStrides(tensor->get_strides(), static_cast<int32_t>(elementSize));
}

void MemRefType::updateStride() {
    // Note: NCHW layout style
    uint64_t stride = 1;
    for (int64_t dimIndex = _dimsCount - 1; dimIndex >= 0; --dimIndex) {
        if (_strides[dimIndex] != static_cast<int64_t>(stride)) {
            markDirty(STRIDE_DIRTY);
        }
        _strides[dimIndex] = stride;
        stride *= _sizes[dimIndex];
    }
}

bool MemRefType::isDirty() const {
    return _dirtyFlag != 0;
}

uint32_t MemRefType::getDirtyFlag() const {
    return _dirtyFlag;
}

void MemRefType::markDirty(uint32_t dirtyFlag) {
    _dirtyFlag |= dirtyFlag;
}

void MemRefType::clearDirty() {
    _dirtyFlag = 0;
}

// The comparision only checks shape and strides now
bool MemRefType::compare(const MemRefType& memref) {
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

std::ostream& operator<<(std::ostream& os, const MemRefType& memRef) {
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

std::string MemRefType::toString() {
    std::stringstream stream;
    stream << *this;
    return stream.str();
}

}  // namespace intel_npu
