// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shape_inference/shape_inference_cpu.hpp"
#include <node.h>

#pragma once

namespace ov {
namespace intel_cpu {
namespace node {

class RandomUniformShapeInfer : public ShapeInferEmptyPads {
public:
    explicit RandomUniformShapeInfer() {}
    IShapeInfer::Result infer(
        const std::vector<std::reference_wrapper<const VectorDims>>& input_shapes,
        const std::unordered_map<size_t, MemoryPtr>& data_dependency) override;

    port_mask_t get_port_mask() const override {
        return PortMask(0);
    }
};

class RandomUniformShapeInferFactory : public ShapeInferFactory {
public:
    explicit RandomUniformShapeInferFactory(const std::shared_ptr<ov::Node>& op);
    ShapeInferPtr makeShapeInfer() const override;

private:
    std::shared_ptr<ov::Node> m_op;
};

} // namespace node
} // namespace intel_cpu
} // namespace ov
