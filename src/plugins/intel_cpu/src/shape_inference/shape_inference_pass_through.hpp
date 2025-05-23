// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shape_inference_cpu.hpp"

namespace ov::intel_cpu {

/**
 * Specific shape inference implementation designed to cover cases where there are no actual output shape calculation
 * and all the output shapes are equal to the input tensor shapes.
 *
 */
class ShapeInferPassThrough final : public ShapeInferEmptyPads {
public:
    ShapeInferPassThrough() = default;
    Result infer(const std::vector<std::reference_wrapper<const VectorDims>>& input_shapes,
                 const std::unordered_map<size_t, MemoryPtr>& data_dependency) override {
        OPENVINO_ASSERT(!input_shapes.empty());
        return {{input_shapes.front()}, ShapeInferStatus::success};
    }
    [[nodiscard]] port_mask_t get_port_mask() const override {
        return EMPTY_PORT_MASK;
    }
};

class PassThroughShapeInferFactory final : public ShapeInferFactory {
public:
    [[nodiscard]] ShapeInferPtr makeShapeInfer() const override {
        return std::make_shared<ShapeInferPassThrough>();
    }
};

}  // namespace ov::intel_cpu
