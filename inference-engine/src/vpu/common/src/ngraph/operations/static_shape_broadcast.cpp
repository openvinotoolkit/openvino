// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu/ngraph/operations/static_shape_broadcast.hpp"
#include "vpu/ngraph/utilities.hpp"

#include "vpu/utils/error.hpp"

#include "ngraph/opsets/opset3.hpp"
#include "ngraph/evaluator.hpp"
#include <ngraph/validation_util.hpp>

namespace ngraph { namespace vpu { namespace op {

constexpr NodeTypeInfo StaticShapeBroadcast::type_info;

StaticShapeBroadcast::StaticShapeBroadcast(const Output<Node>& arg,
                                           const Output<Node>& targetShape,
                                           const Output<Node>& axesMapping,
                                           const ngraph::op::BroadcastModeSpec& broadcastSpec)
        : ::ngraph::op::v3::Broadcast{arg, targetShape, axesMapping, broadcastSpec} {
    constructor_validate_and_infer_types();
}

StaticShapeBroadcast::StaticShapeBroadcast(const Output<Node>& arg,
                                           const Output<Node>& targetShape,
                                           const ngraph::op::BroadcastModeSpec& broadcastSpec)
        : ::ngraph::op::v3::Broadcast{arg, targetShape, broadcastSpec} {
    constructor_validate_and_infer_types();
}

void StaticShapeBroadcast::validate_and_infer_types() {
    if (m_mode.m_type == ngraph::op::BroadcastType::EXPLICIT) {
        NODE_VALIDATION_CHECK(this, get_input_size() == 3,
                              "StaticShapeBroadcast (", get_friendly_name(), ") ",
                              "with explicit mode must have 3 inputs, provided: ",
                              get_input_size());
    } else if (m_mode.m_type == ngraph::op::BroadcastType::NUMPY || m_mode.m_type == ngraph::op::BroadcastType::BIDIRECTIONAL) {
        NODE_VALIDATION_CHECK(this, get_input_size() == 2,
                              "StaticShapeBroadcast (", get_friendly_name(), ") ",
                              "with ", m_mode.m_type, " mode must have 2 inputs, provided: ",
                              get_input_size());
    } else {
        NODE_VALIDATION_CHECK(this, false,
                              "StaticShapeBroadcast (", get_friendly_name(), ") ",
                              "doesn't support ", m_mode.m_type, " mode");
    }

    if (get_output_partial_shape(0).is_static()) {
        return;
    }

    ::ngraph::op::v3::Broadcast::validate_and_infer_types();
    // Try to evaluate output shape. After some transformations further, we may not be able
    // to evaluate the target shape again, then we will leave the evaluated shape unchanged.
    // For example, EliminateShapeOfAfterDSR remove ShapeOf and pass the second input of DSR.
    ngraph::PartialShape evaluatedTargetShape;
    if (!ngraph::evaluate_as_partial_shape(input_value(1), evaluatedTargetShape) ||
        evaluatedTargetShape.is_dynamic()) {
        NODE_VALIDATION_CHECK(this, false,
                              "StaticShapeBroadcast (", get_friendly_name(), ") can't evaluate output shape");
    }

    auto targetShape = evaluatedTargetShape.get_shape();

    if (m_mode.m_type == ngraph::op::BroadcastType::BIDIRECTIONAL) {
        auto inputShape = get_input_partial_shape(0).get_shape();

        auto& lowRankShape = targetShape.size() < inputShape.size() ? targetShape : inputShape;
        auto& highRankShape = lowRankShape == targetShape ? inputShape : targetShape;

        while (lowRankShape.size() < highRankShape.size()) {
            lowRankShape.insert(lowRankShape.begin(), 1);
        }

        for (size_t i = 0; i < targetShape.size(); i++) {
            targetShape[i] = std::max(targetShape[i], inputShape[i]);
        }
    }
    set_output_type(0, get_input_element_type(0), targetShape);
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
    } else if (m_mode.m_type == ngraph::op::BroadcastType::BIDIRECTIONAL) {
        mode = "bidirectional";
    } else {
        NODE_VALIDATION_CHECK(this, false,
            "StaticShapeBroadcast (", get_friendly_name(), ") ",
            "has ", m_mode.m_type, " mode which isn't supported");
    }
    visitor.on_attribute("mode", mode);

    return true;
}

bool StaticShapeBroadcast::evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const {
    return ::ngraph::op::v3::Broadcast::evaluate(outputs, inputs);
}

}  // namespace op
}  // namespace vpu
}  // namespace ngraph
