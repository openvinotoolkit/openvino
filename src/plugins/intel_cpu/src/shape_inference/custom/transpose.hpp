// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <node.h>

#include <utility>

#include "shape_inference/shape_inference_cpu.hpp"

#pragma once

namespace ov::intel_cpu::node {
using Result = IShapeInfer::Result;

class TransposeDynShapeInfer : public ShapeInferEmptyPads {
public:
    TransposeDynShapeInfer() = default;
    Result infer(const std::vector<std::reference_wrapper<const VectorDims>>& input_shapes,
                 const std::unordered_map<size_t, MemoryPtr>& data_dependency) override {
        OPENVINO_THROW("TODO: Support parameterized Order input for dynamic shapes.");
    }
    [[nodiscard]] port_mask_t get_port_mask() const override {
        return EMPTY_PORT_MASK;
    }

private:
};

class TransposeShapeInfer : public ShapeInferEmptyPads {
public:
    TransposeShapeInfer(const size_t& out_rank, const std::vector<size_t>& axes_vec);

    Result infer(const std::vector<std::reference_wrapper<const VectorDims>>& input_shapes,
                 const std::unordered_map<size_t, MemoryPtr>& data_dependency) override;

    [[nodiscard]] port_mask_t get_port_mask() const override {
        return EMPTY_PORT_MASK;
    }

private:
    const size_t m_out_rank;
    const std::vector<size_t> m_axes_vec;
    VectorDims m_outputShape;
    const bool m_needReverse;
};

class TransposeShapeInferFactory : public ShapeInferFactory {
public:
    TransposeShapeInferFactory(std::shared_ptr<ov::Node> op) : m_op(std::move(op)) {}
    [[nodiscard]] ShapeInferPtr makeShapeInfer() const override;

private:
    const std::shared_ptr<ov::Node> m_op;
};

}  // namespace ov::intel_cpu::node
