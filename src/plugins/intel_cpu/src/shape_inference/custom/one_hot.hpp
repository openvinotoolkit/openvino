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
 * Implements One Hot shape inference algorithm. The output shape is the input `indices` tensor shape, where a new axis
 * of size `depth` is inserted at the dimension defined by the `axis` parameter.
 *
 */
class OneHotShapeInfer : public ShapeInferEmptyPads {
public:
    explicit OneHotShapeInfer(int64_t axis) : m_axis(axis) {}
    Result infer(const std::vector<std::reference_wrapper<const VectorDims>>& input_shapes,
                 const std::unordered_map<size_t, MemoryPtr>& data_dependency) override;

    [[nodiscard]] port_mask_t get_port_mask() const override {
        return PortMask(1);
    }

private:
    int64_t m_axis = 0;
};

class OneHotShapeInferFactory : public ShapeInferFactory {
public:
    OneHotShapeInferFactory(std::shared_ptr<ov::Node> op) : m_op(std::move(op)) {}
    [[nodiscard]] ShapeInferPtr makeShapeInfer() const override;

private:
    std::shared_ptr<ov::Node> m_op;
};

}  // namespace ov::intel_cpu::node
