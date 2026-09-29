// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <functional>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

#include "cpu_memory.h"
#include "cpu_types.h"
#include "openvino/core/node.hpp"
#include "shape_inference/shape_inference_cpu.hpp"

namespace ov::intel_cpu::node {
using Result = IShapeInfer::Result;

/**
 * Implements Color Convert shape inference algorithm. Depending on whether it has only single plain H dimension is
 * passed through or recalculated as 2/3 of the initial size.
 *
 */
class ColorConvertShapeInfer : public ShapeInferEmptyPads {
public:
    explicit ColorConvertShapeInfer(bool singlePlane) : m_singlePlane(singlePlane) {}
    Result infer(const std::vector<std::reference_wrapper<const VectorDims>>& input_shapes,
                 const std::unordered_map<size_t, MemoryPtr>& data_dependency) override;
    [[nodiscard]] port_mask_t get_port_mask() const override {
        return EMPTY_PORT_MASK;
    }

private:
    bool m_singlePlane = false;
};

class ColorConvertToNV12ShapeInfer : public ShapeInferEmptyPads {
public:
    explicit ColorConvertToNV12ShapeInfer(bool singlePlane) : m_singlePlane(singlePlane) {}
    Result infer(const std::vector<std::reference_wrapper<const VectorDims>>& input_shapes,
                 const std::unordered_map<size_t, MemoryPtr>& data_dependency) override;
    [[nodiscard]] port_mask_t get_port_mask() const override {
        return EMPTY_PORT_MASK;
    }

private:
    bool m_singlePlane = true;
};

class ColorConvertShapeInferFactory : public ShapeInferFactory {
public:
    explicit ColorConvertShapeInferFactory(std::shared_ptr<ov::Node> op) : m_op(std::move(op)) {}
    [[nodiscard]] ShapeInferPtr makeShapeInfer() const override;

private:
    std::shared_ptr<ov::Node> m_op;
};

}  // namespace ov::intel_cpu::node
