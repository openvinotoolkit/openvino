// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <node.h>

#include <openvino/op/multinomial.hpp>

#include "shape_inference/shape_inference_cpu.hpp"

#pragma once

namespace ov {
namespace intel_cpu {
namespace node {
using Result = IShapeInfer::Result;
class MultinomialShapeInfer : public ShapeInferEmptyPads {
public:
    MultinomialShapeInfer(std::shared_ptr<const ov::op::v13::Multinomial> op) : m_op(op){};

    Result infer(const std::vector<std::reference_wrapper<const VectorDims>>& input_shapes,
                 const std::unordered_map<size_t, MemoryPtr>& data_dependency) override;

    port_mask_t get_port_mask() const override {
        return PortMask(1);
    }

private:
    std::shared_ptr<const ov::op::v13::Multinomial> m_op;
};

class MultinomialShapeInferFactory : public ShapeInferFactory {
public:
    explicit MultinomialShapeInferFactory(const std::shared_ptr<ov::Node>& op) : m_op(op){};
    ShapeInferPtr makeShapeInfer() const override;

private:
    const std::shared_ptr<ov::Node> m_op;
};

}  // namespace node
}  // namespace intel_cpu
}  // namespace ov
