// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu/ngraph/utilities.hpp"

#include "ngraph/opsets/opset3.hpp"
#include "ngraph/evaluator.hpp"

ngraph::HostTensorVector evaluateShapeOf(ngraph::Node* node, const ngraph::HostTensorVector&) {
    auto shapeOf = ngraph::as_type<ngraph::opset3::ShapeOf>(node);
    const auto inputValue = shapeOf->input_value(0);
    const auto outputValue = shapeOf->output(0);
    const auto inputTensors =
        ngraph::HostTensorVector{std::make_shared<ngraph::runtime::HostTensor>(inputValue)};
    const auto outputTensors =
        ngraph::HostTensorVector{std::make_shared<ngraph::runtime::HostTensor>(outputValue)};

    shapeOf->evaluate(outputTensors, inputTensors);
    return outputTensors;
}

ngraph::HostTensorVector evaluateConstant(ngraph::Node* node, const ngraph::HostTensorVector&) {
    const auto constantNode = ngraph::as_type<ngraph::opset3::Constant>(node);
    const auto constant = std::make_shared<ngraph::opset3::Constant>(*constantNode);

    const auto outputTensor = std::make_shared<ngraph::runtime::HostTensor>(constant);

    return {outputTensor};
}

ngraph::HostTensorVector evaluateOp(ngraph::Node* node, const ngraph::HostTensorVector& inputTensors) {
    ngraph::HostTensorVector outputTensors;
    for (const auto& output : node->outputs()) {
        outputTensors.push_back(std::make_shared<ngraph::HostTensor>(output));
    }

    node->evaluate(outputTensors, inputTensors);
    return outputTensors;
}

std::vector<std::int64_t> evaluateTargetShape(const ngraph::Output<ngraph::Node>& value) {
    static ngraph::Evaluator<ngraph::HostTensorPtr>::op_handler_map handlers = {
        {ngraph::opset3::ShapeOf::type_info,  evaluateShapeOf},
        {ngraph::opset3::Constant::type_info, evaluateConstant},
        {ngraph::opset3::Gather::type_info,   evaluateOp},
        {ngraph::opset3::Concat::type_info,   evaluateOp},
        {ngraph::opset3::Reshape::type_info,  evaluateOp},
        {ngraph::opset3::Multiply::type_info, evaluateOp},
        {ngraph::opset3::Squeeze::type_info,  evaluateOp},
    };
    ngraph::Evaluator<ngraph::HostTensorPtr>::value_map value_map;
    ngraph::Evaluator<ngraph::HostTensorPtr> evaluator(handlers, value_map);

    const auto shapeTensor = evaluator.evaluate(value);
    if (!shapeTensor || !shapeTensor->get_is_allocated()) {
        return {};
    }

    const auto shapeConstNode = std::make_shared<ngraph::opset3::Constant>(shapeTensor);
    return {shapeConstNode->cast_vector<std::int64_t>()};
}

namespace vpu {

void printTo(std::ostream& stream, const ngraph::NodeTypeInfo& object) {
    stream << object.name << " ver. " << object.version;
}

}  // namespace vpu
