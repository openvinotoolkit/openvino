// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shape_inference_cpu.hpp"

namespace ov {
namespace intel_cpu {

/**
 * Specific shape inference implementation designed to cover cases where there are no actual output shape calculation
 * and all the output shapes are equal to the input tensor shapes.
 * 
 */
class ShapeInferPassThrough final : public ShapeInferEmptyPads {
public:
    ShapeInferPassThrough() = default;
    std::vector<VectorDims> infer(
        const std::vector<std::reference_wrapper<const VectorDims>>& input_shapes,
        const std::unordered_map<size_t, MemoryPtr>& data_dependency) override {
        IE_ASSERT(!input_shapes.empty());
        return {input_shapes.front()};
    }
    port_mask_t get_port_mask() const override {
        return EMPTY_PORT_MASK;
    }
};

class PassThroughShapeInferFactory final : public ShapeInferFactory {
public:
    ShapeInferPtr makeShapeInfer() const override {
        return std::make_shared<ShapeInferPassThrough>();
    }
};

} // namespace intel_cpu
} // namespace ov