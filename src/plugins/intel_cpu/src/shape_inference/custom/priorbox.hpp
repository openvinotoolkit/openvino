// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <node.h>

#include <utility>

#include "shape_inference/shape_inference_cpu.hpp"

#pragma once

namespace ov::intel_cpu::node {
using Result = IShapeInfer::Result;

class PriorBoxShapeInfer : public ShapeInferEmptyPads {
    /**
     * Implements Prior Box Clustered shape inference algorithm. The output shape is [2,  4 * height * width *
     * number_of_priors]. `number_of_priors` is an attribute of the operation. heigh and width are in the the first
     * input parameter.
     *
     */
public:
    explicit PriorBoxShapeInfer(int64_t number_of_priors) : m_number_of_priors(number_of_priors) {}
    Result infer(const std::vector<std::reference_wrapper<const VectorDims>>& input_shapes,
                 const std::unordered_map<size_t, MemoryPtr>& data_dependency) override;

    [[nodiscard]] port_mask_t get_port_mask() const override {
        return PortMask(0);
    }

private:
    int64_t m_number_of_priors = 0;
};

class PriorBoxShapeInferFactory : public ShapeInferFactory {
public:
    explicit PriorBoxShapeInferFactory(std::shared_ptr<ov::Node> op) : m_op(std::move(op)) {}
    [[nodiscard]] ShapeInferPtr makeShapeInfer() const override;

private:
    std::shared_ptr<ov::Node> m_op;
};
}  // namespace ov::intel_cpu::node
