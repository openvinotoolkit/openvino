// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <node.h>

#include "shape_inference/shape_inference_cpu.hpp"

#pragma once

namespace ov::intel_cpu::node {
using Result = IShapeInfer::Result;

/**
 * Implements Eltwise shape inference algorithm. The algorithm is based on broadcasting all the input shapes
 * according to the NUMPY broadcast rule. This implementation is more lightweight than the ngraph one.
 *
 */
class EltwiseShapeInfer : public ShapeInferEmptyPads {
public:
    Result infer(const std::vector<std::reference_wrapper<const VectorDims>>& input_shapes,
                 const std::unordered_map<size_t, MemoryPtr>& data_dependency) override;
    [[nodiscard]] port_mask_t get_port_mask() const override {
        return EMPTY_PORT_MASK;
    }
};

class EltwiseShapeInferFactory : public ShapeInferFactory {
public:
    [[nodiscard]] ShapeInferPtr makeShapeInfer() const override {
        return std::make_shared<EltwiseShapeInfer>();
    }
};
}  // namespace ov::intel_cpu::node
