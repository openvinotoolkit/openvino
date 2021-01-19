// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu/ngraph/utilities.hpp"

#include "ngraph/opsets/opset3.hpp"
#include "ngraph/opsets/opset5.hpp"
#include "ngraph/evaluator.hpp"

#include <numeric>

namespace vpu {
namespace {

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

} // namespace

std::vector<std::int64_t> evaluateTargetShape(const ngraph::Output<ngraph::Node>& value) {
    static ngraph::Evaluator<ngraph::HostTensorPtr>::op_handler_map handlers = {
        {ngraph::opset3::ShapeOf::type_info,   evaluateShapeOf},
        {ngraph::opset3::Constant::type_info,  evaluateConstant},
        {ngraph::opset3::Gather::type_info,    evaluateOp},
        {ngraph::opset3::Concat::type_info,    evaluateOp},
        {ngraph::opset3::Reshape::type_info,   evaluateOp},
        {ngraph::opset3::Multiply::type_info,  evaluateOp},
        {ngraph::opset3::Squeeze::type_info,   evaluateOp},
        {ngraph::opset5::Unsqueeze::type_info, evaluateOp},
        {ngraph::opset5::Equal::type_info,     evaluateOp},
        {ngraph::opset5::Select::type_info,    evaluateOp},
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

std::shared_ptr<ngraph::Node> shapeToConstant(const ngraph::element::Type& type, const ngraph::Shape& shape) {
    return ngraph::opset5::Constant::create(type, {shape.size()}, shape);
}

std::shared_ptr<ngraph::Node> gatherShapeElements(const ngraph::Output<ngraph::Node>& shape, int startIndex, size_t elemCount) {
    std::vector<int64_t> shapePart(elemCount);
    std::iota(shapePart.begin(), shapePart.end(), startIndex);

    return std::make_shared<ngraph::opset5::Gather>(
        shape,
        ngraph::opset5::Constant::create(ngraph::element::i64, {elemCount}, shapePart),
        ngraph::opset5::Constant::create(ngraph::element::i64, {}, {0}));
}

}  // namespace vpu
