// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_functions/low_precision_transformations/concat_function.hpp"

#include <ngraph/opsets/opset1.hpp>
#include "ngraph_ops/type_relaxed.hpp"
#include "transformations/low_precision/network_helper.hpp"

#include "ngraph_functions/subgraph_builders.hpp"
#include "ngraph_functions/low_precision_transformations/common/fake_quantize_on_data.hpp"
#include "ngraph_functions/low_precision_transformations/common/dequantization_operations.hpp"

namespace ngraph {
namespace builder {
namespace subgraph {

using namespace ngraph::pass;

std::pair<float, float> getQuantizationInterval(const ngraph::element::Type precision) {
    const bool unsignedInterval = precision == ngraph::element::u8;
    const float low = unsignedInterval ? 0.f : -128.f;
    const float hight = unsignedInterval ? 255.f : 127.f;
    return std::make_pair(low, hight);
}

std::shared_ptr<Node> createDequantization(
    const std::shared_ptr<ngraph::Node> data,
    const DequantizationOperations& dequantizationOperations) {
    std::shared_ptr<ngraph::Node> parent = data;

    if (dequantizationOperations.convertOutputPrecision != ngraph::element::undefined) {
        std::shared_ptr<ngraph::opset1::Convert> convert = std::make_shared<ngraph::opset1::Convert>(
            parent,
            dequantizationOperations.convertOutputPrecision);
        parent = convert;
    }

    if (!dequantizationOperations.subtractValues.empty()) {
        std::shared_ptr<ngraph::opset1::Subtract> subtract = std::make_shared<ngraph::opset1::Subtract>(
            parent,
            std::make_shared<ngraph::opset1::Constant>(
                parent->get_output_element_type(0),
                dequantizationOperations.subtractValues.size() == 1ul ?
                    Shape{} :
                    Shape{dequantizationOperations.subtractValues.size(), 1, 1},
                dequantizationOperations.subtractValues));
        parent = subtract;
    }

    if (!dequantizationOperations.multiplyValues.empty()) {
        std::shared_ptr<ngraph::opset1::Multiply> multiply = std::make_shared<ngraph::opset1::Multiply>(
            parent,
            std::make_shared<ngraph::opset1::Constant>(
                parent->get_output_element_type(0),
                dequantizationOperations.multiplyValues.size() == 1ul ?
                    Shape{} :
                    Shape{ dequantizationOperations.multiplyValues.size(), 1, 1 },
                dequantizationOperations.multiplyValues));
        parent = multiply;
    }

    return parent;
}

std::shared_ptr<ngraph::Function> ConcatFunction::getOriginal(
    const ngraph::element::Type precision,
    const ngraph::Shape& inputShape,
    const FakeQuantizeOnData& fqOnData1,
    const FakeQuantizeOnData& fqOnData2) {
    const auto input1 = std::make_shared<ngraph::opset1::Parameter>(precision, inputShape);
    input1->set_friendly_name("input1");
    const auto fakeQuantize1 = ngraph::builder::makeFakeQuantize(
        input1,
        precision,
        fqOnData1.quantizationLevel,
        fqOnData1.constantShape,
        fqOnData1.inputLowValues,
        fqOnData1.inputHighValues,
        fqOnData1.outputLowValues,
        fqOnData1.outputHighValues);

    const std::vector<size_t> inputShape2 = inputShape;
    const auto input2 = std::make_shared<ngraph::opset1::Parameter>(precision, ngraph::Shape(inputShape2));
    input2->set_friendly_name("input2");
    const auto fakeQuantize2 = ngraph::builder::makeFakeQuantize(
        input2,
        precision,
        fqOnData2.quantizationLevel,
        fqOnData2.constantShape,
        fqOnData2.inputLowValues,
        fqOnData2.inputHighValues,
        fqOnData2.outputLowValues,
        fqOnData2.outputHighValues);

    const std::shared_ptr<ngraph::opset1::Concat> concat = std::make_shared<ngraph::opset1::Concat>(
        ngraph::OutputVector{ fakeQuantize1->output(0), fakeQuantize2->output(0) }, 1);

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(concat) };
    std::shared_ptr<ngraph::Function> function = std::make_shared<ngraph::Function>(
        results,
        ngraph::ParameterVector{ input1, input2 },
        "ConcatTransformation");

    return function;
}

//std::shared_ptr<ngraph::Function> ConcatFunction::getOriginal(
//    const ngraph::element::Type ngPrecision,
//    const ngraph::Shape& inputShape,
//    const DequantizationOperations& dequantizationOperations1,
//    const DequantizationOperations& dequantizationOperations2) {
//    const auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngPrecision, inputShape);
//    input1->set_friendly_name("input1");
//    const std::shared_ptr<Node> branch1 = createBranch(input1, dequantizationOperations1);
//
//    const auto input2 = std::make_shared<ngraph::opset1::Parameter>(ngPrecision, inputShape);
//    input1->set_friendly_name("input2");
//    const std::shared_ptr<Node> branch2 = createBranch(input1, dequantizationOperations2);
//
//    const std::shared_ptr<ngraph::opset1::Concat> concat = std::make_shared<ngraph::opset1::Concat>(
//        ngraph::OutputVector{ branch1->output(0), branch2->output(0) }, 1);
//
//    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(concat) };
//    std::shared_ptr<ngraph::Function> function = std::make_shared<ngraph::Function>(
//        results,
//        ngraph::ParameterVector{ input1, input2 },
//        "ConcatTransformation");
//
//    return function;
//}

std::shared_ptr<ngraph::Function> ConcatFunction::getReference(
    const ngraph::element::Type precision,
    const ngraph::Shape& inputShape,
    const FakeQuantizeOnData& fqOnData1,
    const FakeQuantizeOnData& fqOnData2,
    const DequantizationOperations& dequantizationOperations) {
    const auto input1 = std::make_shared<ngraph::opset1::Parameter>(precision, inputShape);
    input1->set_friendly_name("input1");
    const auto fakeQuantize1 = ngraph::builder::makeFakeQuantizeTypeRelaxed(
        input1,
        precision,
        fqOnData1.quantizationLevel,
        fqOnData1.constantShape,
        fqOnData1.inputLowValues,
        fqOnData1.inputHighValues,
        fqOnData1.outputLowValues,
        fqOnData1.outputHighValues);

    const std::vector<size_t> inputShape2 = inputShape;
    const auto input2 = std::make_shared<ngraph::opset1::Parameter>(precision, ngraph::Shape(inputShape2));
    input2->set_friendly_name("input2");
    const auto fakeQuantize2 = ngraph::builder::makeFakeQuantizeTypeRelaxed(
        input2,
        precision,
        fqOnData2.quantizationLevel,
        fqOnData2.constantShape,
        fqOnData2.inputLowValues,
        fqOnData2.inputHighValues,
        fqOnData2.outputLowValues,
        fqOnData2.outputHighValues);

    const std::shared_ptr<ngraph::opset1::Concat> concat = std::make_shared<ngraph::op::TypeRelaxed<ngraph::opset1::Concat>>(
        ngraph::OutputVector{ fakeQuantize1->output(0), fakeQuantize2->output(0) }, 1);

    const std::shared_ptr<ngraph::Node> lastDequantization = createDequantization(concat, dequantizationOperations);

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(lastDequantization) };
    std::shared_ptr<ngraph::Function> function = std::make_shared<ngraph::Function>(
        results,
        ngraph::ParameterVector{ input1, input2 },
        "ConcatTransformation");

    if (fqOnData1.outputPrecision != fqOnData2.outputPrecision) {
        THROW_IE_EXCEPTION << "FakeQuantize expected precisions are different";
    }
    const ngraph::element::Type fqOnDataPrecision = fqOnData1.outputPrecision;
    if (fqOnDataPrecision != ngraph::element::undefined) {
        if (fakeQuantize1->get_output_element_type(0) != fakeQuantize2->get_output_element_type(0)) {
            THROW_IE_EXCEPTION << "FakeQuantize operation precisions are different";
        }
        const ngraph::element::Type fakeQuantizePrecision = fakeQuantize1->get_output_element_type(0);

        if (fqOnDataPrecision != fakeQuantizePrecision) {
            ngraph::pass::low_precision::NetworkHelper::setOutDataPrecision(fakeQuantize1, fqOnDataPrecision);
            ngraph::pass::low_precision::NetworkHelper::setOutDataPrecision(fakeQuantize2, fqOnDataPrecision);
            ngraph::pass::low_precision::NetworkHelper::setOutDataPrecision(concat, fqOnDataPrecision);
        }
    }

    return function;
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
