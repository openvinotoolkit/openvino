// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <node.h>

#include "shape_inference/shape_inference_cpu.hpp"

#pragma once

namespace ov::intel_cpu::node {
using Result = IShapeInfer::Result;

class FCShapeInfer : public ShapeInferEmptyPads {
public:
    FCShapeInfer(size_t outPut_rank) : out_rank(outPut_rank) {}
    Result infer(const std::vector<std::reference_wrapper<const VectorDims>>& input_shapes,
                 const std::unordered_map<size_t, MemoryPtr>& data_dependency) override;

    [[nodiscard]] port_mask_t get_port_mask() const override {
        return EMPTY_PORT_MASK;
    }

private:
    size_t out_rank = 0;
};

class FCShapeInferFactory : public ShapeInferFactory {
public:
    FCShapeInferFactory(const std::shared_ptr<ov::Node>& op) : m_op(op) {}
    [[nodiscard]] ShapeInferPtr makeShapeInfer() const override {
        return std::make_shared<FCShapeInfer>(m_op->get_output_partial_shape(0).rank().get_length());
    }

private:
    std::shared_ptr<const ov::Node> m_op;
};
}  // namespace ov::intel_cpu::node
