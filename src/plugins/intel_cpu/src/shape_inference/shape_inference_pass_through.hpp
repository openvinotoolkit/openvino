// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shape_inference_cpu.hpp"
#include "transformations/cpu_opset/common/op/read_value_with_subgraph.hpp"

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
    Result infer(
        const std::vector<std::reference_wrapper<const VectorDims>>& input_shapes,
        const std::unordered_map<size_t, MemoryPtr>& data_dependency) override {
        OPENVINO_ASSERT(!input_shapes.empty());
        return {{input_shapes.front()}, ShapeInferStatus::success};
    }
    port_mask_t get_port_mask() const override {
        return EMPTY_PORT_MASK;
    }
};

class PassThroughShapeInferFactory final : public ShapeInferFactory {
public:
    explicit PassThroughShapeInferFactory() {}

    ShapeInferPtr makeShapeInfer() const override {
        return std::make_shared<ShapeInferPassThrough>();
    }

private:
    std::shared_ptr<ov::Node> m_op = nullptr;
    std::shared_ptr<ov::Model> m_body = nullptr;
};

} // namespace intel_cpu
} // namespace ov
