// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shape_inference_ngraph.hpp"

namespace ov {
namespace intel_cpu {
NgraphShapeInferFactory::NgraphShapeInferFactory(std::shared_ptr<ov::Node> op) : m_op(std::move(op)) {}

ShapeInferPtr NgraphShapeInferFactory::makeShapeInfer() const {
    return std::make_shared<NgraphShapeInfer>(make_shape_inference(m_op));
}

const ov::CoordinateDiff ShapeInferEmptyPads::m_emptyVec = {};

}  // namespace intel_cpu
}  // namespace ov
