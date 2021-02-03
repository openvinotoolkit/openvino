// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ngraph/validation_util.hpp>
#include "vpu/ngraph/operations/static_shape_loop.hpp"

namespace ngraph { namespace vpu { namespace op {

NGRAPH_RTTI_DEFINITION(ngraph::vpu::op::StaticShapeLoop, "StaticShapeLoop", 0);

StaticShapeLoop::StaticShapeLoop(const Loop& loop) : Loop(loop) {}

void StaticShapeLoop::validate_and_infer_types() {
    const auto isLoopStatic = [this]() {
        const auto& outs = outputs();
        return !outs.empty() && std::all_of(outs.cbegin(), outs.cend(), [](const Output<Node>& output) { return output.get_partial_shape().is_static(); });
    };

    if (isLoopStatic()) {
        return;
    }

    Loop::validate_and_infer_types();

    const auto maxIterationsCountEstimation = ngraph::maximum_value(input_value(0).get_node_shared_ptr());
    NODE_VALIDATION_CHECK(this, maxIterationsCountEstimation.first,
        "Encountered a loop for which upper-bound estimation for iterations count ", input_value(0), " failed");
    const auto& maxIterationsCount = maxIterationsCountEstimation.second;
    NODE_VALIDATION_CHECK(this, maxIterationsCount > 0, "Encountered a loop with non-positive upper-bound estimation for iterations count ",
        maxIterationsCountEstimation.second);

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

}  // namespace op
}  // namespace vpu
}  // namespace ngraph
