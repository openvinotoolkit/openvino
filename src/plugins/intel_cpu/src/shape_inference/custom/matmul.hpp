// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <node.h>
#include "shape_inference/shape_inference_cpu.hpp"
#include "shape_inference/shape_inference_ngraph.hpp"

#pragma once
namespace ov {
namespace intel_cpu {
namespace node {
using Result = IShapeInfer::Result;
class MMShapeInfer : public ShapeInferEmptyPads {
public:
    MMShapeInfer(const size_t& out_rank, const bool& transpose_a, const bool& transpose_b) :
        m_out_rank(out_rank), m_transpose_a(transpose_a), m_transpose_b(transpose_b) {
        m_shapeY = VectorDims(m_out_rank, 1); // for output and cache
    }
    Result infer(
        const std::vector<std::reference_wrapper<const VectorDims>>& input_shapes,
        const std::unordered_map<size_t, MemoryPtr>& data_dependency) override;

    port_mask_t get_port_mask() const override {
        return EMPTY_PORT_MASK;
    }

private:
    VectorDims m_shapeY;
    const size_t m_out_rank;
    const bool m_transpose_a;
    const bool m_transpose_b;
};

class MMShapeInferFactory : public ShapeInferFactory {
public:
    MMShapeInferFactory(const std::shared_ptr<ov::Node>& op) : m_op(op) {}
    ShapeInferPtr makeShapeInfer() const override;
private:
    std::shared_ptr<ov::Node> m_op;
};
} // namespace node
} // namespace intel_cpu
} // namespace ov

