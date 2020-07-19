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

std::shared_ptr<Node> makeDequantization(
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
                    Shape{ 1, dequantizationOperations.subtractValues.size(), 1, 1 },
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
                    Shape{ 1, dequantizationOperations.multiplyValues.size(), 1, 1 },
                dequantizationOperations.multiplyValues));
        parent = multiply;
    }

    return parent;
}

std::shared_ptr<Node> makeFakeQuantize(
    const std::shared_ptr<ngraph::op::v0::Parameter>& input,
    const ngraph::element::Type precision,
    const FakeQuantizeOnData& fqOnData) {
    return ngraph::builder::makeFakeQuantize(
        input,
        precision,
        fqOnData.quantizationLevel,
        fqOnData.constantShape,
        fqOnData.inputLowValues,
        fqOnData.inputHighValues,
        fqOnData.outputLowValues,
        fqOnData.outputHighValues);
}

std::shared_ptr<Node> makeFakeQuantizeTypeRelaxed(
    const std::shared_ptr<ngraph::op::v0::Parameter>& input,
    const ngraph::element::Type precision,
    const FakeQuantizeOnData& fqOnData) {
    return ngraph::builder::makeFakeQuantizeTypeRelaxed(
        input,
        precision,
        fqOnData.quantizationLevel,
        fqOnData.constantShape,
        fqOnData.inputLowValues,
        fqOnData.inputHighValues,
        fqOnData.outputLowValues,
        fqOnData.outputHighValues);
}

std::shared_ptr<ngraph::Function> ConcatFunction::getOriginal(
    const ngraph::element::Type precision,
    const ngraph::Shape& inputShape,
    const FakeQuantizeOnData& fqOnData1,
    const FakeQuantizeOnData& fqOnData2) {
    const auto input1 = std::make_shared<ngraph::opset1::Parameter>(precision, inputShape);
    input1->set_friendly_name("input1");
    const auto fakeQuantize1 = makeFakeQuantize(input1, precision, fqOnData1);

    const std::vector<size_t> inputShape2 = inputShape;
    const auto input2 = std::make_shared<ngraph::opset1::Parameter>(precision, ngraph::Shape(inputShape2));
    input2->set_friendly_name("input2");
    const auto fakeQuantize2 = makeFakeQuantize(input2, precision, fqOnData2);

    const std::shared_ptr<ngraph::opset1::Concat> concat = std::make_shared<ngraph::opset1::Concat>(
        ngraph::OutputVector{ fakeQuantize1->output(0), fakeQuantize2->output(0) }, 1);

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(concat) };
    std::shared_ptr<ngraph::Function> function = std::make_shared<ngraph::Function>(
        results,
        ngraph::ParameterVector{ input1, input2 },
        "ConcatTransformation");

    return function;
}

std::shared_ptr<ngraph::Function> ConcatFunction::getOriginalWithNeighbors(
    const ngraph::element::Type precision,
    const ngraph::Shape& inputShape,
    const FakeQuantizeOnData& fqOnData1,
    const FakeQuantizeOnData& fqOnData2,
    const FakeQuantizeOnData& fqOnData3) {
    const auto interval = getQuantizationInterval(ngraph::element::u8);
    const float low = interval.first;
    const float hight = interval.second;

    const auto input1 = std::make_shared<ngraph::opset1::Parameter>(precision, ngraph::Shape(inputShape));
    input1->set_friendly_name("input1");
    const auto fakeQuantize1 = makeFakeQuantize(input1, precision, fqOnData1);
    fakeQuantize1->set_friendly_name("fakeQuantize1");

    const auto input2 = std::make_shared<ngraph::opset1::Parameter>(precision, ngraph::Shape(inputShape));
    input2->set_friendly_name("input2");
    const auto fakeQuantize2 = makeFakeQuantize(input2, precision, fqOnData2);
    fakeQuantize2->set_friendly_name("fakeQuantize2");

    const auto input3 = std::make_shared<ngraph::opset1::Parameter>(precision, ngraph::Shape(inputShape));
    input3->set_friendly_name("input3");
    const auto fakeQuantize3 = makeFakeQuantize(input3, precision, fqOnData3);
    fakeQuantize3->set_friendly_name("fakeQuantize3");

    const auto concat1 = std::make_shared<ngraph::opset1::Concat>(ngraph::OutputVector{
        fakeQuantize1->output(0),
        fakeQuantize2->output(0) },
        1ull);
    concat1->set_friendly_name("concat1");

    const auto concat2 = std::make_shared<ngraph::opset1::Concat>(ngraph::OutputVector{
        fakeQuantize2->output(0),
        fakeQuantize3->output(0) },
        1ull);
    concat2->set_friendly_name("concat2");

    const ngraph::ResultVector results{
        std::make_shared<ngraph::opset1::Result>(concat1),
        std::make_shared<ngraph::opset1::Result>(concat2)
    };

    std::shared_ptr<ngraph::Function> function = std::make_shared<ngraph::Function>(
        results,
        ngraph::ParameterVector{ input1, input2, input3 },
        "ConcatWithNeighborsTransformation");

    return function;
}

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

    const std::shared_ptr<ngraph::Node> lastDequantization = makeDequantization(concat, dequantizationOperations);

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

std::shared_ptr<ngraph::Function> ConcatFunction::getReferenceWithNeighbors(
    const ngraph::element::Type precision,
    const ngraph::Shape& inputShape,
    const FakeQuantizeOnData& fqOnData1,
    const FakeQuantizeOnData& fqOnData2,
    const FakeQuantizeOnData& fqOnData3,
    const DequantizationOperations& dequantizationOperations1,
    const DequantizationOperations& dequantizationOperations2) {
    const auto interval = getQuantizationInterval(ngraph::element::u8);
    const float low = interval.first;
    const float hight = interval.second;

    const auto input1 = std::make_shared<ngraph::opset1::Parameter>(precision, ngraph::Shape(inputShape));
    input1->set_friendly_name("input1");
    const auto fakeQuantize1 = makeFakeQuantizeTypeRelaxed(input1, precision, fqOnData1);
    fakeQuantize1->set_friendly_name("fakeQuantize1");

    const auto input2 = std::make_shared<ngraph::opset1::Parameter>(precision, ngraph::Shape(inputShape));
    input2->set_friendly_name("input2");
    const auto fakeQuantize2 = makeFakeQuantizeTypeRelaxed(input2, precision, fqOnData2);
    fakeQuantize2->set_friendly_name("fakeQuantize2");

    const auto input3 = std::make_shared<ngraph::opset1::Parameter>(precision, ngraph::Shape(inputShape));
    input3->set_friendly_name("input3");
    const auto fakeQuantize3 = makeFakeQuantizeTypeRelaxed(input3, precision, fqOnData3);
    fakeQuantize3->set_friendly_name("fakeQuantize3");

    const auto concat1 = std::make_shared<ngraph::opset1::Concat>(ngraph::OutputVector{
        fakeQuantize1->output(0),
        fakeQuantize2->output(0) },
        1ull);
    concat1->set_friendly_name("concat1");

    const auto concat2 = std::make_shared<ngraph::opset1::Concat>(ngraph::OutputVector{
        fakeQuantize2->output(0),
        fakeQuantize3->output(0) },
        1ull);
    concat2->set_friendly_name("concat2");

    const std::shared_ptr<ngraph::Node> lastDequantization1 = makeDequantization(concat1, dequantizationOperations1);

    const std::shared_ptr<ngraph::Node> lastDequantization2 = makeDequantization(concat2, dequantizationOperations2);

    const ngraph::ResultVector results{
        std::make_shared<ngraph::opset1::Result>(lastDequantization1),
        std::make_shared<ngraph::opset1::Result>(lastDequantization2)
    };

    std::shared_ptr<ngraph::Function> function = std::make_shared<ngraph::Function>(
        results,
        ngraph::ParameterVector{ input1, input2, input3 },
        "ConcatWithNeighborsTransformation");

    if ((fqOnData1.outputPrecision != fqOnData2.outputPrecision) || (fqOnData2.outputPrecision != fqOnData3.outputPrecision)) {
        THROW_IE_EXCEPTION << "FakeQuantize expected precisions are different";
    }
    const ngraph::element::Type fqOnDataPrecision = fqOnData1.outputPrecision;
    if (fqOnDataPrecision != ngraph::element::undefined) {
        if ((fakeQuantize1->get_output_element_type(0) != fakeQuantize2->get_output_element_type(0)) ||
            (fakeQuantize2->get_output_element_type(0) != fakeQuantize3->get_output_element_type(0))){
            THROW_IE_EXCEPTION << "FakeQuantize operation precisions are different";
        }
        const ngraph::element::Type fakeQuantizePrecision = fakeQuantize1->get_output_element_type(0);

        if (fqOnDataPrecision != fakeQuantizePrecision) {
            ngraph::pass::low_precision::NetworkHelper::setOutDataPrecision(fakeQuantize1, fqOnDataPrecision);
            ngraph::pass::low_precision::NetworkHelper::setOutDataPrecision(fakeQuantize2, fqOnDataPrecision);
            ngraph::pass::low_precision::NetworkHelper::setOutDataPrecision(fakeQuantize3, fqOnDataPrecision);
            ngraph::pass::low_precision::NetworkHelper::setOutDataPrecision(concat1, fqOnDataPrecision);
            ngraph::pass::low_precision::NetworkHelper::setOutDataPrecision(concat2, fqOnDataPrecision);
        }
    }

    return function;
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
