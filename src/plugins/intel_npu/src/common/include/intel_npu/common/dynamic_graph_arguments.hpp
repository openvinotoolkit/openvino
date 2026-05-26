// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_npu/common/igraph.hpp"
#include "intel_npu/utils/zero/zero_init.hpp"

namespace intel_npu {

/// Host-side description of a single VM-runtime MemRef. Used as the per-IO entry inside
/// @ref GraphArguments. The opaque @c _impl pointer carries a @c DynamicGraphMemRefImpl
/// (defined in dynamic_graph_vm_impl.hpp) which owns the underlying VM MemRef handle.
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

    void setArg(const void* arg);
    void setSize(const ov::Shape& shape);
    void setStrides(const ov::Strides& strides, int32_t elementSize = 1);
    void set(const void* basePtr, int64_t offset, std::shared_ptr<ov::ITensor> tensor);
    void updateStride();
    bool compare(const MemRefType& memref);
    friend std::ostream& operator<<(std::ostream& os, const MemRefType& memRef);
    std::string toString();
};

/// Runtime carrier for VM-pipeline inputs/outputs. Owned by @c PipelinedCommandLists.
/// The opaque @c _impl holds a @c DynamicGraphArgumentsImpl (VM execution context + cached
/// mem-ref vectors), created lazily on first VM execution.
struct GraphArguments {
    std::vector<MemRefType> _inputs;
    std::vector<MemRefType> _outputs;
    std::shared_ptr<void> _impl;

    void setArgumentProperties(uint32_t argi,
                               const void* argv,
                               const ov::Shape& shapes,
                               const std::vector<size_t>& strides);
};

}  // namespace intel_npu
