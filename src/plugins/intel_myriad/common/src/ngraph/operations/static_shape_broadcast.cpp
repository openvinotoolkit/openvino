// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu/ngraph/operations/static_shape_broadcast.hpp"
#include "vpu/ngraph/utilities.hpp"

#include "vpu/utils/error.hpp"

#include "ngraph/opsets/opset3.hpp"
#include <ngraph/validation_util.hpp>

namespace ngraph { namespace vpu { namespace op {

StaticShapeBroadcast::StaticShapeBroadcast(const Output<Node>& arg,
                                           const Output<Node>& targetShape,
                                           const Output<Node>& axesMapping,
                                           const ngraph::op::BroadcastModeSpec& broadcastSpec)
        : ::ngraph::op::v3::Broadcast{arg, targetShape, axesMapping, broadcastSpec},
          m_evaluatedOutputShape{PartialShape::dynamic()} {
    constructor_validate_and_infer_types();
}

StaticShapeBroadcast::StaticShapeBroadcast(const Output<Node>& arg,
                                           const Output<Node>& targetShape,
                                           const ngraph::op::BroadcastModeSpec& broadcastSpec)
        : ::ngraph::op::v3::Broadcast{arg, targetShape, broadcastSpec},
          m_evaluatedOutputShape{PartialShape::dynamic()} {
    constructor_validate_and_infer_types();
}

void StaticShapeBroadcast::validate_and_infer_types() {
    auto& outputShape = m_evaluatedOutputShape;
    if (outputShape.is_dynamic()) {
        ::ngraph::op::v3::Broadcast::validate_and_infer_types();

        outputShape = get_output_partial_shape(0);
        NODE_VALIDATION_CHECK(this, outputShape.rank().is_static(), "StaticShapeBroadcast (", get_friendly_name(), ") ",
                              "output is expected to be of static rank");
        for (size_t i = 0; i < outputShape.rank().get_length(); i++) {
            outputShape[i] = outputShape[i].get_max_length();
        }
    }

    NODE_VALIDATION_CHECK(this, outputShape.is_static(),
                          "StaticShapeBroadcast (", get_friendly_name(), ") can't evaluate output shape");

    set_output_type(0, get_input_element_type(0), outputShape);
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
