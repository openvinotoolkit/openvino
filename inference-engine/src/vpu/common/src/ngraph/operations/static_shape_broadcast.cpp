// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu/ngraph/operations/static_shape_broadcast.hpp"

#include "vpu/utils/error.hpp"

#include "ngraph/opsets/opset3.hpp"
#include "ngraph/evaluator.hpp"

namespace ngraph { namespace vpu { namespace op {

namespace {

HostTensorVector evaluateShapeOf(Node* node, const HostTensorVector&) {
    auto shapeOf = as_type<opset3::ShapeOf>(node);
    const auto inputValue = shapeOf->input_value(0);
    const auto outputValue = shapeOf->output(0);
    const auto inputTensors =
            HostTensorVector{std::make_shared<runtime::HostTensor>(inputValue)};
    const auto outputTensors =
            HostTensorVector{std::make_shared<runtime::HostTensor>(outputValue)};

    shapeOf->evaluate(outputTensors, inputTensors);
    return outputTensors;
}

HostTensorVector evaluateConstant(Node* node, const HostTensorVector&) {
    const auto constantNode = as_type<opset3::Constant>(node);
    const auto constant = std::make_shared<opset3::Constant>(*constantNode);

    const auto outputTensor = std::make_shared<runtime::HostTensor>(constant);

    return {outputTensor};
}

HostTensorVector evaluateOp(Node* node, const HostTensorVector& inputTensors) {
    HostTensorVector outputTensors;
    for (const auto& output : node->outputs()) {
        outputTensors.push_back(std::make_shared<HostTensor>(output));
    }

    node->evaluate(outputTensors, inputTensors);
    return outputTensors;
}

PartialShape evaluateTargetShape(const Output<Node>& value) {
    static Evaluator<HostTensorPtr>::op_handler_map handlers = {
            {opset3::ShapeOf::type_info,  evaluateShapeOf},
            {opset3::Constant::type_info, evaluateConstant},
            {opset3::Gather::type_info,   evaluateOp},
            {opset3::Concat::type_info,   evaluateOp}};
    Evaluator<HostTensorPtr>::value_map value_map;
    Evaluator<HostTensorPtr> evaluator(handlers, value_map);

    const auto shapeTensor = evaluator.evaluate(value);
    if (!shapeTensor || !shapeTensor->get_is_allocated()) {
        return PartialShape::dynamic();
    }
    const auto shapeConstNode = std::make_shared<opset3::Constant>(shapeTensor);
    const auto resultShape = Shape{shapeConstNode->cast_vector<size_t>()};

    return resultShape;
}

}  // namespace

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
        const auto evaluatedTargetShape = evaluateTargetShape(input_value(1));
        if (evaluatedTargetShape.is_static()) {
            m_evaluatedOutputShape = evaluatedTargetShape;
        }
        NODE_VALIDATION_CHECK(this, m_evaluatedOutputShape.is_static(),
                              "StaticShapeBroadcast (", get_friendly_name(), ") ",
                              "can't evaluate output shape, got: ", m_evaluatedOutputShape);
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

}  // namespace op
}  // namespace vpu
}  // namespace ngraph
