// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_functions/low_precision_transformations/multiply_function.hpp"

#include <ngraph/opsets/opset1.hpp>
#include <ngraph_ops/type_relaxed.hpp>
#include "ngraph_functions/subgraph_builders.hpp"
#include "transformations/low_precision/common/dequantization_op.hpp"
#include "transformations/low_precision/network_helper.hpp"

using namespace ngraph::pass::low_precision;

namespace ngraph {
namespace builder {
namespace subgraph {

std::shared_ptr<ngraph::Function> MultiplyFunction::getOriginal(
    const ngraph::element::Type precision,
    const ngraph::Shape& inputShape,
    const bool& broadcast,
    const ngraph::pass::low_precision::LayerTransformation::Params& params,
    const MultiplyActualValues& actualValues,
    const bool& constInput,
    const bool& constantFolding) {
    std::shared_ptr<ngraph::Node> parent1;
    std::shared_ptr<ngraph::opset1::Parameter> input1;
    if (constantFolding) {
        const auto const1 = std::make_shared<ngraph::opset1::Constant>(
                actualValues.precision1, Shape({ 1, actualValues.mutliplyValues1.size(), 1, 1 }), actualValues.mutliplyValues1);
        parent1 = const1;
    } else {
        input1 = std::make_shared<ngraph::opset1::Parameter>(
                actualValues.precision1,
                broadcast ? ngraph::Shape({inputShape[0], inputShape[1], 1, 1}) : ngraph::Shape(inputShape));
        parent1 = input1;
    }
    const std::shared_ptr<ngraph::Node> convert1 = std::make_shared<DequantizationConvert>(parent1, precision);
    parent1 = convert1;

    if (!actualValues.subtractValues1.empty()) {
        const std::shared_ptr<ngraph::Node> subtract1 = std::make_shared<DequantizationSubtract >(
            parent1,
            std::make_shared<ngraph::opset1::Constant>(
                precision, Shape({ actualValues.subtractValues1.size() }), actualValues.subtractValues1));
        parent1 = subtract1;
    }

    if (!actualValues.mutliplyValues1.empty() && !constantFolding) {
        const std::shared_ptr<ngraph::Node> multiply1 = std::make_shared<DequantizationMultiply >(
            parent1,
            std::make_shared<ngraph::opset1::Constant>(
                precision, Shape({ actualValues.mutliplyValues1.size() }), actualValues.mutliplyValues1));
        parent1 = multiply1;
    }

    std::shared_ptr<ngraph::Node> parent2;
    std::shared_ptr<ngraph::opset1::Parameter> input2;
    if (constInput || constantFolding) {
        const auto const2 = std::make_shared<ngraph::opset1::Constant>(
            actualValues.precision2, Shape({ actualValues.mutliplyValues2.size() }), actualValues.mutliplyValues2);
        parent2 = const2;
    } else {
        input2 = std::make_shared<ngraph::opset1::Parameter>(
            actualValues.precision2,
            ngraph::Shape(inputShape));
        parent2 = input2;

        const std::shared_ptr<ngraph::Node> convert2 = std::make_shared<DequantizationConvert>(parent2, precision);
        parent2 = convert2;

        if (!actualValues.subtractValues2.empty()) {
            const std::shared_ptr<ngraph::Node> subtract2 = std::make_shared<DequantizationSubtract >(
                parent2,
                std::make_shared<ngraph::opset1::Constant>(
                    precision, Shape({ actualValues.subtractValues2.size() }), actualValues.subtractValues2));
            parent2 = subtract2;
        }

        if (!actualValues.mutliplyValues2.empty()) {
            const std::shared_ptr<ngraph::Node> multiply2 = std::make_shared<DequantizationMultiply >(
                parent2,
                std::make_shared<ngraph::opset1::Constant>(
                    precision, Shape({ actualValues.mutliplyValues2.size() }), actualValues.mutliplyValues2));
            parent2 = multiply2;
        }
    }

    const auto multiply = std::make_shared<ngraph::opset1::Multiply>(parent1, parent2);
    multiply->set_friendly_name("output");

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(multiply) };
    const auto inputs = constantFolding ?
            ngraph::ParameterVector{ } :
            constInput ?
                ngraph::ParameterVector{ input1 } :
                ngraph::ParameterVector{ input1, input2 };

    return std::make_shared<ngraph::Function>(results, inputs, "MultiplyTransformation");
}

std::shared_ptr<ngraph::Function> MultiplyFunction::getReference(
    const ngraph::element::Type precision,
    const ngraph::Shape& inputShape,
    const bool& broadcast,
    const ngraph::pass::low_precision::LayerTransformation::Params& params,
    const MultiplyExpectedValues& expectedValues,
    const bool& constInput,
    const bool& constantFolding) {
    std::shared_ptr<ngraph::Node> parent1;
    std::shared_ptr<ngraph::opset1::Parameter> input1;
    if (constantFolding) {
        const auto const1 = std::make_shared<ngraph::opset1::Constant>(
                expectedValues.precision1, Shape({1, expectedValues.mutliplyValues1.size(), 1, 1}),
                expectedValues.mutliplyValues1);
        parent1 = const1;

        const1->set_friendly_name("output");
    } else {
        input1 = std::make_shared<ngraph::opset1::Parameter>(
                expectedValues.precision1,
                broadcast ? ngraph::Shape({inputShape[0], inputShape[1], 1, 1}) : ngraph::Shape(inputShape));
        parent1 = input1;

        //if (!(expectedValues.subtractValues1.empty() && expectedValues.mutliplyValues1.empty())) {
        const std::shared_ptr<ngraph::Node> convert1 = std::make_shared<DequantizationConvert>(parent1, precision);
        parent1 = convert1;
        //}
    }

    if (!expectedValues.subtractValues1.empty()) {
        const std::shared_ptr<ngraph::Node> subtract1 = std::make_shared<DequantizationSubtract>(
                parent1,
                std::make_shared<ngraph::opset1::Constant>(
                        precision, Shape({expectedValues.subtractValues1.size()}), expectedValues.subtractValues1));
        parent1 = subtract1;
    }

    if (!expectedValues.mutliplyValues1.empty() && !constantFolding) {
        const std::shared_ptr<ngraph::Node> multiply1 = std::make_shared<DequantizationMultiply>(
                parent1,
                std::make_shared<ngraph::opset1::Constant>(
                        precision, Shape({expectedValues.mutliplyValues1.size()}), expectedValues.mutliplyValues1));
        parent1 = multiply1;
    }

    std::shared_ptr<ngraph::Node> parent2;
    std::shared_ptr<ngraph::opset1::Parameter> input2;
    if (constInput && !constantFolding) {
        const auto const2 = std::make_shared<ngraph::opset1::Constant>(
                expectedValues.precision2, Shape({expectedValues.mutliplyValues2.size()}),
                expectedValues.mutliplyValues2);
        parent2 = const2;
    } else {
        input2 = std::make_shared<ngraph::opset1::Parameter>(
                expectedValues.precision2,
                ngraph::Shape(inputShape));
        parent2 = input2;

        if (!(expectedValues.subtractValues2.empty() && expectedValues.mutliplyValues2.empty())) {
            const std::shared_ptr<ngraph::Node> convert2 = std::make_shared<DequantizationConvert>(parent2, precision);
            parent2 = convert2;
        }

        if (!expectedValues.subtractValues2.empty()) {
            const std::shared_ptr<ngraph::Node> subtract2 = std::make_shared<DequantizationSubtract>(
                    parent2,
                    std::make_shared<ngraph::opset1::Constant>(
                            precision, Shape({expectedValues.subtractValues2.size()}), expectedValues.subtractValues2));
            parent2 = subtract2;
        }

        if (!expectedValues.mutliplyValues2.empty()) {
            const std::shared_ptr<ngraph::Node> multiply2 = std::make_shared<DequantizationMultiply>(
                    parent2,
                    std::make_shared<ngraph::opset1::Constant>(
                            precision, Shape({expectedValues.mutliplyValues2.size()}), expectedValues.mutliplyValues2));
            parent2 = multiply2;
        }
    }
    std::shared_ptr<ngraph::Node> multiply;
    if (!constantFolding) {
        auto multiplyOriginal = ngraph::opset1::Multiply(
                ngraph::op::TemporaryReplaceOutputType(parent1, element::f32).get(),
                ngraph::op::TemporaryReplaceOutputType(parent2, element::f32).get());

        multiply = std::make_shared<ngraph::op::TypeRelaxed<ngraph::opset1::Multiply>>(
                multiplyOriginal,
                std::vector<element::Type>{element::f32, element::f32},
                std::vector<element::Type>{});
        multiply->set_friendly_name("output");
    }

    ngraph::ResultVector results{ constantFolding ? std::make_shared<ngraph::opset1::Result>(parent1) : std::make_shared<ngraph::opset1::Result>(multiply) };
    const auto inputs = constantFolding ?
                        ngraph::ParameterVector{ } :
                        constInput ?
                        ngraph::ParameterVector{ input1 } :
                        ngraph::ParameterVector{ input1, input2 };

    return std::make_shared<ngraph::Function>(results, inputs, "MultiplyTransformation");
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
