// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <node.h>

#include <utility>

#include "shape_inference/shape_inference_cpu.hpp"

#pragma once

namespace ov::intel_cpu::node {
using Result = IShapeInfer::Result;
class ReshapeShapeInfer : public ShapeInferEmptyPads {
public:
    ReshapeShapeInfer(bool specialZero) : m_specialZero(specialZero) {}
    Result infer(const std::vector<std::reference_wrapper<const VectorDims>>& input_shapes,
                 const std::unordered_map<size_t, MemoryPtr>& data_dependency) override;
    [[nodiscard]] port_mask_t get_port_mask() const override {
        return PortMask(1);
    }

private:
    bool m_specialZero;
};

class SqueezeShapeInfer : public ShapeInferEmptyPads {
public:
    SqueezeShapeInfer() = default;
    Result infer(const std::vector<std::reference_wrapper<const VectorDims>>& input_shapes,
                 const std::unordered_map<size_t, MemoryPtr>& data_dependency) override;
    [[nodiscard]] port_mask_t get_port_mask() const override {
        return PortMask(1);
    }
};

class UnsqueezeShapeInfer : public ShapeInferEmptyPads {
public:
    UnsqueezeShapeInfer() = default;
    Result infer(const std::vector<std::reference_wrapper<const VectorDims>>& input_shapes,
                 const std::unordered_map<size_t, MemoryPtr>& data_dependency) override;
    [[nodiscard]] port_mask_t get_port_mask() const override {
        return PortMask(1);
    }
};

class ReshapeShapeInferFactory : public ShapeInferFactory {
public:
    ReshapeShapeInferFactory(std::shared_ptr<ov::Node> op) : m_op(std::move(op)) {}
    [[nodiscard]] ShapeInferPtr makeShapeInfer() const override;

private:
    std::shared_ptr<ov::Node> m_op;
};
}  // namespace ov::intel_cpu::node
