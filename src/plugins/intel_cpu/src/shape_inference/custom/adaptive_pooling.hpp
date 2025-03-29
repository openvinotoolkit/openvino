// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <node.h>

#include <utility>

#include "shape_inference/shape_inference_cpu.hpp"

#pragma once

namespace ov::intel_cpu::node {
using Result = IShapeInfer::Result;

/**
 * Implements Adaptive Pooling shape inference algorithm. The output tensor shape consists of the input [N, C]
 * dimensions and the [D_out, H_out, W_out] dimensions, which are placed in the second input parameter.
 *
 */
class AdaptivePoolingShapeInfer : public ShapeInferEmptyPads {
public:
    explicit AdaptivePoolingShapeInfer(size_t outputs_count) : m_outputs_count(outputs_count) {}
    Result infer(const std::vector<std::reference_wrapper<const VectorDims>>& input_shapes,
                 const std::unordered_map<size_t, MemoryPtr>& data_dependency) override;

    [[nodiscard]] port_mask_t get_port_mask() const override {
        return PortMask(1);
    }

private:
    size_t m_outputs_count;
};

class AdaptivePoolingShapeInferFactory : public ShapeInferFactory {
public:
    AdaptivePoolingShapeInferFactory(std::shared_ptr<ov::Node> op) : m_op(std::move(op)) {}
    [[nodiscard]] ShapeInferPtr makeShapeInfer() const override;

private:
    std::shared_ptr<ov::Node> m_op;
};

}  // namespace ov::intel_cpu::node
