// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu/ngraph/operations/static_shape_broadcast.hpp"
#include "vpu/ngraph/utilities.hpp"

#include "vpu/utils/error.hpp"

#include "ngraph/opsets/opset3.hpp"
#include "ngraph/evaluator.hpp"

namespace ngraph { namespace vpu { namespace op {

constexpr NodeTypeInfo StaticShapeBroadcast::type_info;

StaticShapeBroadcast::StaticShapeBroadcast(const Output<Node>& arg,
                                           const Output<Node>& targetShape,
                                           const Output<Node>& axesMapping,
                                           const ngraph::op::BroadcastModeSpec& broadcastSpec)
        : ::ngraph::op::util::BroadcastBase{arg, targetShape, axesMapping, broadcastSpec},
          m_evaluatedOutputShape{PartialShape::dynamic()} {
    constructor_validate_and_infer_types();
}

StaticShapeBroadcast::StaticShapeBroadcast(const Output<Node>& arg,
                                           const Output<Node>& targetShape,
                                           const ngraph::op::BroadcastModeSpec& broadcastSpec)
        : ::ngraph::op::util::BroadcastBase{arg, targetShape, broadcastSpec},
          m_evaluatedOutputShape{PartialShape::dynamic()} {
    constructor_validate_and_infer_types();
}

void StaticShapeBroadcast::validate_and_infer_types() {
    if (m_mode.m_type == ngraph::op::BroadcastType::EXPLICIT) {
        NODE_VALIDATION_CHECK(this, get_input_size() == 3,
                              "StaticShapeBroadcast (", get_friendly_name(), ") ",
                              "with explicit mode must have 3 inputs, provided: ",
                              get_input_size());
    } else if (m_mode.m_type == ngraph::op::BroadcastType::NUMPY) {
        NODE_VALIDATION_CHECK(this, get_input_size() == 2,
                              "StaticShapeBroadcast (", get_friendly_name(), ") ",
                              "with numpy mode must have 2 inputs, provided: ",
                              get_input_size());
    } else {
        NODE_VALIDATION_CHECK(this, false,
                              "StaticShapeBroadcast (", get_friendly_name(), ") ",
                              "doesn't support ", m_mode.m_type, " mode");
    }

    if (get_output_partial_shape(0).is_dynamic()) {
        ::ngraph::op::util::BroadcastBase::validate_and_infer_types();
        // Try to evaluate output shape. After some transformations further, we may not be able
        // to evaluate the target shape again, then we will leave the evaluated shape unchanged.
        // For example, DynamicToStaticShapeShapeOf remove ShapeOf and pass the second input of DSR.
        const auto evaluatedDimensionValues = evaluateTargetShape(input_value(1));
        NODE_VALIDATION_CHECK(this, !evaluatedDimensionValues.empty(), "StaticShapeBroadcast (", get_friendly_name(), ") can't evaluate output shape");

        const auto evaluatedTargetShape = ngraph::PartialShape(evaluatedDimensionValues);
        if (evaluatedTargetShape.is_static()) {
            m_evaluatedOutputShape = evaluatedTargetShape;
        }
        NODE_VALIDATION_CHECK(this, m_evaluatedOutputShape.is_static(),
                              "StaticShapeBroadcast (", get_friendly_name(), ") ",
                              "can't evaluate output shape, got: ", m_evaluatedOutputShape);
        NODE_VALIDATION_CHECK(this, m_evaluatedOutputShape.all_non_negative(),
                              "StaticShapeBroadcast (", get_friendly_name(), ") ",
                              "expects non-negative shape, got: ", m_evaluatedOutputShape);
        set_output_type(0, get_input_element_type(0), m_evaluatedOutputShape);
    }
}

std::shared_ptr<Node> StaticShapeBroadcast::clone_with_new_inputs(const OutputVector& newInputs) const {
    check_new_args_count(this, newInputs);
    if (newInputs.size() == 2) {
        return std::make_shared<StaticShapeBroadcast>(
                newInputs.at(0), newInputs.at(1), m_mode);
    } else {
        return std::make_shared<StaticShapeBroadcast>(
                newInputs.at(0), newInputs.at(1), newInputs.at(2), m_mode);
    }
}

bool StaticShapeBroadcast::visit_attributes(ngraph::AttributeVisitor& visitor) {
    std::string mode;
    if (m_mode.m_type == ngraph::op::BroadcastType::EXPLICIT) {
        mode = "explicit";
    } else if (m_mode.m_type == ngraph::op::BroadcastType::NUMPY) {
        mode = "numpy";
    }
    visitor.on_attribute("mode", mode);

    return true;
}

bool StaticShapeBroadcast::evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) {
    return ::ngraph::op::util::BroadcastBase::evaluate(outputs, inputs);
}

}  // namespace op
}  // namespace vpu
}  // namespace ngraph
