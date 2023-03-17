// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/pass/lowered/linear_IR_transformation.hpp"
#include "snippets/snippets_isa.hpp"
#include "snippets/itt.hpp"


namespace ngraph {
namespace snippets {
namespace pass {
namespace lowered {

void LinearIRTransformationPipeline::register_transformation(const std::shared_ptr<pass::lowered::LinearIRTransformation>& transformation) {
    m_transformations.push_back(transformation);
}

void LinearIRTransformationPipeline::run(LoweredExprIR& linear_ir) {
    for (const auto& transformation : m_transformations) {
        transformation->run(linear_ir);
    }
}

} // namespace lowered
} // namespace pass
} // namespace snippets
} // namespace ngraph
