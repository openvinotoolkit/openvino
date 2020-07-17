// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_functions/low_precision_transformations/multiply_function.hpp"

#include <ngraph/opsets/opset1.hpp>
#include <ngraph_ops/type_relaxed.hpp>
#include "ngraph_functions/subgraph_builders.hpp"

namespace ngraph {
namespace builder {
namespace subgraph {

std::shared_ptr<ngraph::Function> MultiplyFunction::getOriginal(
    const ngraph::element::Type precision,
    const ngraph::Shape& inputShape,
    const bool& broadcast,
    const ngraph::pass::low_precision::LayerTransformation::Params& params,
    const MultiplyActualValues& actualValues,
    const bool& constInput) {
    const auto input1 = std::make_shared<ngraph::opset1::Parameter>(
        actualValues.precision1,
        broadcast ? ngraph::Shape({ inputShape[0], inputShape[1], 1, 1 }) : ngraph::Shape(inputShape));
    std::shared_ptr<ngraph::Node> parent1 = input1;

    const std::shared_ptr<ngraph::Node> convert1 = std::make_shared<ngraph::opset1::Convert>(parent1, precision);
    parent1 = convert1;

    if (!actualValues.subtractValues1.empty()) {
        const std::shared_ptr<ngraph::Node> subtract1 = std::make_shared< ngraph::opset1::Subtract >(
            parent1,
            std::make_shared<ngraph::opset1::Constant>(
                precision, Shape({ actualValues.subtractValues1.size() }), actualValues.subtractValues1));
        parent1 = subtract1;
    }

    if (!actualValues.mutliplyValues1.empty()) {
        const std::shared_ptr<ngraph::Node> multiply1 = std::make_shared< ngraph::opset1::Multiply >(
            parent1,
            std::make_shared<ngraph::opset1::Constant>(
                precision, Shape({ actualValues.mutliplyValues1.size() }), actualValues.mutliplyValues1));
        parent1 = multiply1;
    }

    std::shared_ptr<ngraph::Node> parent2;
    std::shared_ptr<ngraph::opset1::Parameter> input2;
    if (constInput) {
        const auto const2 = std::make_shared<ngraph::opset1::Constant>(
            actualValues.precision2, Shape({ actualValues.mutliplyValues2.size() }), actualValues.mutliplyValues2);
        parent2 = const2;
    } else {
        input2 = std::make_shared<ngraph::opset1::Parameter>(
            actualValues.precision2,
            ngraph::Shape(inputShape));
        parent2 = input2;

        const std::shared_ptr<ngraph::Node> convert2 = std::make_shared<ngraph::opset1::Convert>(parent2, precision);
        parent2 = convert2;

        if (!actualValues.subtractValues2.empty()) {
            const std::shared_ptr<ngraph::Node> subtract2 = std::make_shared< ngraph::opset1::Subtract >(
                parent2,
                std::make_shared<ngraph::opset1::Constant>(
                    precision, Shape({ actualValues.subtractValues2.size() }), actualValues.subtractValues2));
            parent2 = subtract2;
        }

        if (!actualValues.mutliplyValues2.empty()) {
            const std::shared_ptr<ngraph::Node> multiply2 = std::make_shared< ngraph::opset1::Multiply >(
                parent2,
                std::make_shared<ngraph::opset1::Constant>(
                    precision, Shape({ actualValues.mutliplyValues2.size() }), actualValues.mutliplyValues2));
            parent2 = multiply2;
        }
    }

    const auto multiply = std::make_shared<ngraph::opset1::Multiply>(parent1, parent2);

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(multiply) };
    const auto inputs = constInput ? ngraph::ParameterVector{ input1 }: ngraph::ParameterVector{ input1, input2 };

    return std::make_shared<ngraph::Function>(results, inputs, "MultiplyTransformation");
}

std::shared_ptr<ngraph::Function> MultiplyFunction::getReference(
    const ngraph::element::Type precision,
    const ngraph::Shape& inputShape,
    const bool& broadcast,
    const ngraph::pass::low_precision::LayerTransformation::Params& params,
    const MultiplyExpectedValues& expectedValues,
    const bool& constInput) {
    const auto input1 = std::make_shared<ngraph::opset1::Parameter>(
        expectedValues.precision1,
        broadcast ? ngraph::Shape({ inputShape[0], inputShape[1], 1, 1 }) : ngraph::Shape(inputShape));

    std::shared_ptr<ngraph::Node> parent1 = input1;
    //if (!(expectedValues.subtractValues1.empty() && expectedValues.mutliplyValues1.empty())) {
    const std::shared_ptr<ngraph::Node> convert1 = std::make_shared<ngraph::opset1::Convert>(parent1, precision);
    parent1 = convert1;
    //}

    if (!expectedValues.subtractValues1.empty()) {
        const std::shared_ptr<ngraph::Node> subtract1 = std::make_shared<ngraph::opset1::Subtract>(
            parent1,
            std::make_shared<ngraph::opset1::Constant>(
                precision, Shape({ expectedValues.subtractValues1.size() }), expectedValues.subtractValues1));
        parent1 = subtract1;
    }

    if (!expectedValues.mutliplyValues1.empty()) {
        const std::shared_ptr<ngraph::Node> multiply1 = std::make_shared<ngraph::opset1::Multiply>(
            parent1,
            std::make_shared<ngraph::opset1::Constant>(
                precision, Shape({ expectedValues.mutliplyValues1.size() }), expectedValues.mutliplyValues1));
        parent1 = multiply1;
    }

    std::shared_ptr<ngraph::Node> parent2;
    std::shared_ptr<ngraph::opset1::Parameter> input2;
    if (constInput) {
        const auto const2 = std::make_shared<ngraph::opset1::Constant>(
            expectedValues.precision2, Shape({ expectedValues.mutliplyValues2.size() }), expectedValues.mutliplyValues2);
        parent2 = const2;
    } else {
        input2 = std::make_shared<ngraph::opset1::Parameter>(
            expectedValues.precision2,
            ngraph::Shape(inputShape));
        parent2 = input2;

        //if (!(expectedValues.subtractValues2.empty() && expectedValues.mutliplyValues2.empty())) {
        const std::shared_ptr<ngraph::Node> convert2 = std::make_shared<ngraph::opset1::Convert>(parent2, precision);
        parent2 = convert2;
        //}

        if (!expectedValues.subtractValues2.empty()) {
            const std::shared_ptr<ngraph::Node> subtract2 = std::make_shared<ngraph::opset1::Subtract>(
                parent2,
                std::make_shared<ngraph::opset1::Constant>(
                    precision, Shape({ expectedValues.subtractValues2.size() }), expectedValues.subtractValues2));
            parent2 = subtract2;
        }

        if (!expectedValues.mutliplyValues2.empty()) {
            const std::shared_ptr<ngraph::Node> multiply2 = std::make_shared<ngraph::opset1::Multiply>(
                parent2,
                std::make_shared<ngraph::opset1::Constant>(
                    precision, Shape({ expectedValues.mutliplyValues2.size() }), expectedValues.mutliplyValues2));
            parent2 = multiply2;
        }
    }
    const auto multiply = std::make_shared< ngraph::opset1::Multiply >(parent1, parent2);

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(multiply) };
    const auto inputs = constInput ? ngraph::ParameterVector{ input1 } : ngraph::ParameterVector{ input1, input2 };

    return std::make_shared<ngraph::Function>(results, inputs, "MultiplyTransformation");
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
