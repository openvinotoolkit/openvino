// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <node.h>
#include "shape_inference/shape_inference_cpu.hpp"

#pragma once
namespace ov {
namespace intel_cpu {
namespace node {
using Result = IShapeInfer::Result;
class ReshapeShapeInfer : public ShapeInferEmptyPads {
public:
    ReshapeShapeInfer(bool specialZero) : m_specialZero(specialZero) {}
    Result infer(const std::vector<std::reference_wrapper<const VectorDims>>& input_shapes,
                                  const std::unordered_map<size_t, MemoryPtr>& data_dependency) override;
    port_mask_t get_port_mask() const override {
        return PortMask(1);
    }

private:
    bool m_specialZero;
};

class SqueezeShapeInfer : public ShapeInferEmptyPads {
public:
    SqueezeShapeInfer() {}
    Result infer(const std::vector<std::reference_wrapper<const VectorDims>>& input_shapes,
                                  const std::unordered_map<size_t, MemoryPtr>& data_dependency) override;
    port_mask_t get_port_mask() const override {
        return PortMask(1);
    }
};

class UnsqueezeShapeInfer : public ShapeInferEmptyPads {
public:
    UnsqueezeShapeInfer() {}
    Result infer(const std::vector<std::reference_wrapper<const VectorDims>>& input_shapes,
                                  const std::unordered_map<size_t, MemoryPtr>& data_dependency) override;
    port_mask_t get_port_mask() const override {
        return PortMask(1);
    }
};

class ReshapeShapeInferFactory : public ShapeInferFactory {
public:
    ReshapeShapeInferFactory(std::shared_ptr<ov::Node> op) : m_op(op) {}
    ShapeInferPtr makeShapeInfer() const override;

private:
    std::shared_ptr<ov::Node> m_op;
};
} // namespace node
} // namespace intel_cpu
} // namespace ov

