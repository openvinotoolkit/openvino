// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <node.h>

#include <utility>

#include "shape_inference/shape_inference_cpu.hpp"

#pragma once

namespace ov::intel_cpu::node {
using Result = IShapeInfer::Result;

class GatherShapeInfer : public ShapeInferEmptyPads {
public:
    GatherShapeInfer(bool isAxisInputConst, bool isIndicesScalar, int axis, int batchDims)
        : m_isAxisInputConst(isAxisInputConst),
          m_isIndicesScalar(isIndicesScalar),
          m_axis(axis),
          m_batchDims(batchDims) {}

    Result infer(const std::vector<std::reference_wrapper<const VectorDims>>& input_shapes,
                 const std::unordered_map<size_t, MemoryPtr>& data_dependency) override;
    [[nodiscard]] port_mask_t get_port_mask() const override {
        return PortMask(2);
    }

private:
    bool m_isAxisInputConst = false;
    bool m_isIndicesScalar = false;
    int m_axis = 0;
    int m_batchDims = 0;
};

class GatherShapeInferFactory : public ShapeInferFactory {
public:
    GatherShapeInferFactory(std::shared_ptr<ov::Node> op) : m_op(std::move(op)) {}
    [[nodiscard]] ShapeInferPtr makeShapeInfer() const override;

private:
    std::shared_ptr<ov::Node> m_op;
};
}  // namespace ov::intel_cpu::node
