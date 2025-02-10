// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <node.h>

#include <utility>

#include "shape_inference/shape_inference_cpu.hpp"

#pragma once
namespace ov {
namespace intel_cpu {
namespace node {

class SDPAShapeInferFactory : public ShapeInferFactory {
public:
    SDPAShapeInferFactory(std::shared_ptr<ov::Node> op) : m_op(std::move(op)) {}
    ShapeInferPtr makeShapeInfer() const override;

private:
    std::shared_ptr<ov::Node> m_op;
};
}  // namespace node
}  // namespace intel_cpu
}  // namespace ov
