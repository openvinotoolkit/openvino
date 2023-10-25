// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <node.h>
#include "shape_inference/shape_inference_cpu.hpp"

#pragma once
namespace ov {
namespace intel_cpu {
namespace node {
using Result = IShapeInfer::Result;

/**
 * Implements Color Convert shape inference algorithm. Depending on wether it has only single plain H dimension is
 * passed through or recalculated as 2/3 of the initial size.
 *
 */
class ColorConvertShapeInfer : public ShapeInferEmptyPads {
public:
    ColorConvertShapeInfer(bool singlePlain) : m_singlePlain(singlePlain) {}
    Result infer(const std::vector<std::reference_wrapper<const VectorDims>>& input_shapes,
                           const std::unordered_map<size_t, MemoryPtr>& data_dependency) override;
    port_mask_t get_port_mask() const override {
        return EMPTY_PORT_MASK;
    }

private:
    bool m_singlePlain = false;
};

class ColorConvertShapeInferFactory : public ShapeInferFactory {
public:
    ColorConvertShapeInferFactory(std::shared_ptr<ov::Node> op) : m_op(op) {}
    ShapeInferPtr makeShapeInfer() const override;

private:
    std::shared_ptr<ov::Node> m_op;
};

} // namespace node
} // namespace intel_cpu
} // namespace ov

