// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <iosfwd>
#include <memory>
#include <string>
#include <vector>

#include "intel_npu/runtime/npu_vm_runtime.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/runtime/itensor.hpp"

namespace intel_npu {

struct MemRefType {
    const void* _basePtr;
    const void* _data;
    int64_t _offset;
    std::vector<int64_t> _sizes;
    std::vector<int64_t> _strides;
    int64_t _dimsCount;
    std::shared_ptr<void> _impl;

    MemRefType() : _basePtr(nullptr), _data(nullptr), _offset(0), _sizes(), _strides(), _dimsCount(0) {}

    MemRefType(const void* basePtr,
               const void* data,
               int64_t offset,
               const std::vector<int64_t>& sizes,
               const std::vector<int64_t>& strides,
               int64_t dimsCount)
        : _basePtr(basePtr),
          _data(data),
          _offset(offset),
          _sizes(sizes),
          _strides(strides),
          _dimsCount(dimsCount) {}

    // Copy intentionally drops the runtime impl: VM MemRef handles must not be aliased
    // across copies (would cause double-destroy / shared device state).
    MemRefType(const MemRefType& other)
        : _basePtr(other._basePtr),
          _data(other._data),
          _offset(other._offset),
          _sizes(other._sizes),
          _strides(other._strides),
          _dimsCount(other._dimsCount) {}
    MemRefType& operator=(const MemRefType& other) {
        if (this != &other) {
            _basePtr = other._basePtr;
            _data = other._data;
            _offset = other._offset;
            _sizes = other._sizes;
            _strides = other._strides;
            _dimsCount = other._dimsCount;
            _impl.reset();
        }
        return *this;
    }
    MemRefType(MemRefType&&) noexcept = default;
    MemRefType& operator=(MemRefType&&) noexcept = default;
    ~MemRefType() = default;

    void setArg(const void* arg);
    void setSize(const ov::Shape& shape);
    void setStrides(const ov::Strides& strides, int32_t elementSize = 1);
    void set(const void* basePtr, int64_t offset, std::shared_ptr<ov::ITensor> tensor);
    void updateStride();
    bool compare(const MemRefType& memref);
    friend std::ostream& operator<<(std::ostream& os, const MemRefType& memRef);
    std::string toString();
};

}  // namespace intel_npu
