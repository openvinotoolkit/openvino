// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cpu_shape.h>
#include <cpu_memory.h>
#include <openvino/core/node.hpp>

namespace ov {
namespace intel_cpu {

class IShapeInfer {
public:
    using port_mask_t = uint32_t;
public:
    ~IShapeInfer() = default;

    virtual std::vector<VectorDims> infer(
        const std::vector<std::reference_wrapper<const VectorDims>>& input_shapes,
        const std::unordered_map<size_t, MemoryPtr>& data_dependency) = 0;

    // infer may generate padding as by-product, these APIs is designed to retrieve them back
    virtual const ov::CoordinateDiff& get_pads_begin() = 0;
    virtual const ov::CoordinateDiff& get_pads_end() = 0;
    virtual port_mask_t get_port_mask() const = 0;
};

class ShapeInferEmptyPads : public IShapeInfer {
public:
    const ov::CoordinateDiff& get_pads_begin() final {
        return m_emptyVec;
    }
    const ov::CoordinateDiff& get_pads_end() final {
        return m_emptyVec;
    }
private:
    static const ov::CoordinateDiff m_emptyVec;
};

using ShapeInferPtr = std::shared_ptr<IShapeInfer>;
using ShapeInferCPtr = std::shared_ptr<const IShapeInfer>;

class ShapeInferFactory {
public:
    ~ShapeInferFactory() = default;
    virtual ShapeInferPtr makeShapeInfer() const = 0;
};

class DefaultShapeInferFactory final : public ShapeInferFactory {
public:
    DefaultShapeInferFactory(std::shared_ptr<ov::Node> op, IShapeInfer::port_mask_t port_mask) : m_op(op), m_port_mask(port_mask) {}
    ShapeInferPtr makeShapeInfer() const override;
private:
    std::shared_ptr<ov::Node> m_op;
    IShapeInfer::port_mask_t m_port_mask;
};
}   // namespace intel_cpu
}   // namespace ov
