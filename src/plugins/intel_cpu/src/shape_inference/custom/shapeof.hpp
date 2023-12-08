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

/**
 * Implements Shape Of shape inference algorithm. The output shape is simply a 1D tensor with the size of the input tensor
 * rank.
 *
 */
class ShapeOfShapeInfer : public ShapeInferEmptyPads {
public:
    ShapeOfShapeInfer() = default;
    Result infer(
        const std::vector<std::reference_wrapper<const VectorDims>>& input_shapes,
        const std::unordered_map<size_t, MemoryPtr>& data_dependency) override {
        OPENVINO_ASSERT(!input_shapes.empty());
        return {{VectorDims{input_shapes.front().get().size()}}, ShapeInferStatus::success};
    }

    port_mask_t get_port_mask() const override {
        return EMPTY_PORT_MASK;
    }
};

class ShapeOfShapeInferFactory : public ShapeInferFactory {
public:
    ShapeInferPtr makeShapeInfer() const override {
        return std::make_shared<ShapeOfShapeInfer>();
    }
};

} // namespace node
} // namespace intel_cpu
} // namespace ov

