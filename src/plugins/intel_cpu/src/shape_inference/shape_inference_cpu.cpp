// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shape_inference_ngraph.hpp"

using namespace ov::intel_cpu;

ShapeInferPtr NgraphShapeInferFactory::makeShapeInfer() const {
    return std::make_shared<NgraphShapeInfer>(make_shape_inference(m_op), m_port_mask);
}

const ov::CoordinateDiff ShapeInferEmptyPads::m_emptyVec = {};
