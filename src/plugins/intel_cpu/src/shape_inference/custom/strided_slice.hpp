// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <node.h>

#include <utility>

#include "shape_inference/shape_inference_cpu.hpp"

#pragma once

namespace ov::intel_cpu::node {
using Result = IShapeInfer::Result;

constexpr IShapeInfer::port_mask_t port_mask = PortMask(/*BEGIN_ID*/ 1, /*END_ID*/ 2, /*STRIDE_ID*/ 3, /*AXES_ID*/ 4);

class StridedSliceShapeInfer : public ShapeInferEmptyPads {
public:
    StridedSliceShapeInfer(size_t output_size,
                           std::unordered_set<int64_t> begin_mask,
                           std::unordered_set<int64_t> end_mask,
                           std::unordered_set<int64_t> new_axis_mask,
                           std::unordered_set<int64_t> shrink_axis_mask);

    Result infer(const std::vector<std::reference_wrapper<const VectorDims>>& input_shapes,
                 const std::unordered_map<size_t, MemoryPtr>& data_dependency) override;

    port_mask_t get_port_mask() const override {
        return port_mask;
    }

private:
    VectorDims m_outputShape;
    const std::unordered_set<int64_t> m_begin_mask_set;
    const std::unordered_set<int64_t> m_end_mask_set;
    const std::unordered_set<int64_t> m_new_axis_mask_set;
    const std::unordered_set<int64_t> m_shrink_axis_mask_set;
};

class StridedSliceShapeInferFactory : public ShapeInferFactory {
public:
    StridedSliceShapeInferFactory(std::shared_ptr<ov::Node> op) : m_op(std::move(op)) {}
    [[nodiscard]] ShapeInferPtr makeShapeInfer() const override;

private:
    const std::shared_ptr<ov::Node> m_op;
};

}  // namespace ov::intel_cpu::node
