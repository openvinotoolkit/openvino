// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ngraph/validation_util.hpp>
#include "vpu/ngraph/operations/static_shape_loop.hpp"

namespace ngraph { namespace vpu { namespace op {

StaticShapeLoop::StaticShapeLoop(const Loop& loop) : Loop(loop), m_evaluatedIterationsCount{ngraph::PartialShape::dynamic()} {}

void StaticShapeLoop::validate_and_infer_types() {
    auto& iterationsCount = m_evaluatedIterationsCount;
    if (iterationsCount.is_dynamic()) {
        Loop::validate_and_infer_types();

        NODE_VALIDATION_CHECK(this, ngraph::evaluate_as_partial_shape(input_value(0), iterationsCount),
                              "Encountered a loop for which upper-bound estimation for iterations count ", input_value(0), " failed");
    }

    const auto& maxIterationsCount = iterationsCount[0].get_max_length();
    NODE_VALIDATION_CHECK(this, maxIterationsCount > 0,
                          "Encountered a loop with non-positive upper-bound estimation for iterations count ",
                          maxIterationsCount);

    const auto& body = get_function();
    for (const auto& outputDescription : get_output_descriptions()) {
        if (const auto& concatOutputDescription = ngraph::as_type_ptr<ngraph::op::util::SubGraphOp::ConcatOutputDescription>(outputDescription)) {
            const auto& bodyOutput = body->output(concatOutputDescription->m_body_value_index);
            const auto& axis = concatOutputDescription->m_axis;
            auto partialShape = bodyOutput.get_partial_shape();
            partialShape[axis] *= maxIterationsCount;

            const auto& concatOutput = output(concatOutputDescription->m_output_index);
            set_output_type(concatOutputDescription->m_output_index, concatOutput.get_element_type(), partialShape);
        }
    }
}

bool StaticShapeLoop::visit_attributes(AttributeVisitor& visitor) {
    return Loop::visit_attributes(visitor);
}

}  // namespace op
}  // namespace vpu
}  // namespace ngraph
