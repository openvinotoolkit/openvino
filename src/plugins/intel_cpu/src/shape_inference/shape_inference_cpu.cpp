// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shape_inference/shape_inference_cpu.hpp"

#include "shape_inference/shape_inference.hpp"

namespace ov::intel_cpu {
NgraphShapeInferFactory::NgraphShapeInferFactory(std::shared_ptr<ov::Node> op) : m_op(std::move(op)) {}

ShapeInferPtr NgraphShapeInferFactory::makeShapeInfer() const {
    return make_shape_inference(m_op);
}

const ov::CoordinateDiff ShapeInferEmptyPads::m_emptyVec = {};

}  // namespace ov::intel_cpu
