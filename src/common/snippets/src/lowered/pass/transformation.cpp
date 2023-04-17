// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/pass/transformation.hpp"


namespace ngraph {
namespace snippets {
namespace lowered {
namespace pass {

void TransformationPipeline::register_transformation(const std::shared_ptr<Transformation>& transformation) {
    m_transformations.push_back(transformation);
}

void TransformationPipeline::run(LinearIR& linear_ir) {
    for (const auto& transformation : m_transformations) {
        transformation->run(linear_ir);
    }
}

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ngraph
