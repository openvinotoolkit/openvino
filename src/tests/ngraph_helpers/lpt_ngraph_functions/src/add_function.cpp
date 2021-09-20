// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "lpt_ngraph_functions/add_function.hpp"

#include "low_precision/network_helper.hpp"
#include "low_precision/layer_transformation.hpp"

#include "ngraph/opsets/opset1.hpp"

#include "lpt_ngraph_functions/common/dequantization_operations.hpp"
#include "ngraph_functions/subgraph_builders.hpp"

using namespace ngraph::pass::low_precision;

namespace ngraph {
namespace builder {
namespace subgraph {

std::shared_ptr<ngraph::Function> AddFunction::getOriginal(
    const ngraph::element::Type precision,
    const ngraph::PartialShape& inputShape1,
    const ngraph::PartialShape& inputShape2,
    const bool broadcast,
    const ngraph::pass::low_precision::LayerTransformation::Params& params,
    const ngraph::element::Type& precision1,
    const ngraph::builder::subgraph::DequantizationOperations& dequantization1,
    const ngraph::element::Type& precision2,
    const ngraph::builder::subgraph::DequantizationOperations& dequantization2,
    const int constInput,
    const std::vector<float>& constValues,
    const std::string& additionalLayer) {
    std::shared_ptr<ngraph::Node> input1;
    std::shared_ptr<ngraph::Node> parent1;
    if (constInput == 0) {
        parent1 = std::make_shared<ngraph::opset1::Constant>(
            precision,
            inputShape1.to_shape(),
            constValues);
    } else {
        input1 = std::make_shared<ngraph::opset1::Parameter>(
            additionalLayer != "" ? precision : (precision1.is_real() ? precision : precision1),
            broadcast ? ngraph::PartialShape({inputShape1[0], inputShape1[1], 1, 1}) : inputShape1);
        if (additionalLayer != "") {
            parent1 = ngraph::builder::subgraph::makeFakeQuantizeTypeRelaxed(
                input1,
                precision,
                {256, Shape{}, {0}, {255}, {0}, {255}, precision1});
        } else {
            parent1 = input1;
        }
    }

    auto dequantizationStructure1 = dequantization1;
    dequantizationStructure1.multiply.outPrecision = precision;
    if (dequantizationStructure1.multiply.empty()) {
        dequantizationStructure1.subtract.outPrecision = precision;
    }

    const auto dequantizationOp1 = dequantization1.empty() ? parent1 : makeDequantization(parent1, dequantizationStructure1);

    std::shared_ptr<ngraph::Node> input2;
    if (constInput == 1) {
        input2 = std::make_shared<ngraph::opset1::Constant>(
            precision,
            inputShape2.to_shape(),
            constValues);
    } else {
        input2 = std::make_shared<ngraph::opset1::Parameter>(
            precision2.is_real() ? precision : precision2, inputShape2);
    }
    auto parent = input2;
    if (additionalLayer == "convolution") {
        parent = std::make_shared<ngraph::op::TypeRelaxed<ngraph::opset1::Convolution>>(
            std::vector<element::Type>{ element::f32, element::f32 },
            std::vector<element::Type>{ precision },
            ngraph::op::TemporaryReplaceOutputType(parent, element::f32).get(),
            ngraph::op::TemporaryReplaceOutputType(
                std::make_shared<ngraph::opset1::Constant>(element::i8, Shape{ 1, 4, 1, 1 }, std::vector<float>{0.8f, 0.8f, 0.8f, 0.8f}),
                element::f32).get(),
            ngraph::Strides{ 1, 1 },
            ngraph::CoordinateDiff{ 0, 0 },
            ngraph::CoordinateDiff{ 0, 0 },
            ngraph::Strides{ 1, 1 });
    }
    std::shared_ptr<Node> additional_output = nullptr;
    if (additionalLayer == "convolution_multiconsumers") {
        parent = std::make_shared<ngraph::op::TypeRelaxed<ngraph::opset1::Convolution>>(
                std::vector<element::Type>{ element::f32, element::f32 },
                std::vector<element::Type>{ precision },
                ngraph::op::TemporaryReplaceOutputType(parent, element::f32).get(),
                ngraph::op::TemporaryReplaceOutputType(
                        std::make_shared<ngraph::opset1::Constant>(element::i8, Shape{ 1, 4, 1, 1 }, std::vector<float>{0.8f, 0.8f, 0.8f, 0.8f}),
                        element::f32).get(),
                ngraph::Strides{ 1, 1 },
                ngraph::CoordinateDiff{ 0, 0 },
                ngraph::CoordinateDiff{ 0, 0 },
                ngraph::Strides{ 1, 1 });
        additional_output = parent;
    }
    if (additionalLayer == "group_convolution") {
        parent = std::make_shared< ngraph::op::TypeRelaxed<ngraph::opset1::GroupConvolution>>(
            std::vector<element::Type>{ element::f32, element::f32 },
            std::vector<element::Type>{ precision },
            ngraph::op::TemporaryReplaceOutputType(parent, element::f32).get(),
            ngraph::op::TemporaryReplaceOutputType(
                std::make_shared<ngraph::opset1::Constant>(element::i8, Shape{ 4, 1, 1, 1, 1 }, std::vector<float>{0.8f, 0.8f, 0.8f, 0.8f}),
                element::f32).get(),
            ngraph::Strides{ 1, 1 },
            ngraph::CoordinateDiff{ 0, 0 },
            ngraph::CoordinateDiff{ 0, 0 },
            ngraph::Strides{ 1, 1 });
    }
    if (additionalLayer != "") {
        parent = std::make_shared<ngraph::opset1::Add>(
            parent,
            std::make_shared<ngraph::opset1::Constant>(precision, Shape{ 1, 1, 1, 1 }, std::vector<float>{1.f}));
        parent = ngraph::builder::subgraph::makeFakeQuantizeTypeRelaxed(
            parent,
            precision,
            {256, Shape{}, { 0 }, { 255 }, { 0 }, { 255 }, element::u8});
    }

    auto dequantizationStructure2 = dequantization2;
    dequantizationStructure2.multiply.outPrecision = precision;
    const auto dequantizationOp2 = dequantization2.empty() ? parent : makeDequantization(parent, dequantizationStructure2);

    const auto add = std::make_shared<ngraph::opset1::Add>(dequantizationOp1, dequantizationOp2);
    add->set_friendly_name("output");
    auto& rtInfo = add->get_rt_info();
    rtInfo["Variant::std::string"] = "add";

    std::shared_ptr<Node> output = add;
    if (additional_output != nullptr) {
        output = std::make_shared<opset1::Multiply>(add, additional_output);
        output->set_friendly_name("output_multiply");
    }

    ngraph::ResultVector results {std::make_shared<ngraph::opset1::Result>(output)};
    ngraph::ParameterVector parameters;
    if (constInput == -1) {
        parameters = { ov::as_type_ptr<ngraph::opset1::Parameter>(input1), ov::as_type_ptr<ngraph::opset1::Parameter>(input2) };
    } else if (constInput == 0) {
        parameters = { ov::as_type_ptr<ngraph::opset1::Parameter>(input2) };
    } else if (constInput == 1) {
        parameters = { ov::as_type_ptr<ngraph::opset1::Parameter>(input1) };
    } else {
        throw std::runtime_error("Unexpected constant input index");
    }
    return std::make_shared<ngraph::Function>(results, parameters, "AddTransformation");
}

std::shared_ptr<ngraph::Function> AddFunction::getOriginal(
    const ngraph::element::Type precision,
    const ngraph::PartialShape& inputShape,
    const bool broadcast,
    const ngraph::builder::subgraph::FakeQuantizeOnData& fqOnData1,
    const ngraph::builder::subgraph::FakeQuantizeOnData& fqOnData2) {
    ngraph::PartialShape inputShape2 = inputShape;

    if (broadcast) {
        inputShape2[2] = 1;
        inputShape2[3] = 1;
    }

    auto fq1 = fqOnData1;
    auto fq2 = fqOnData2;

    const auto input1 = std::make_shared<ngraph::opset1::Parameter>(precision, inputShape);
    const auto fakeQuantize1 = fq1.empty() ?
        nullptr :
        ngraph::builder::makeFakeQuantize(
            input1, precision, fq1.quantizationLevel, fq1.constantShape,
            fq1.inputLowValues, fq1.inputHighValues, fq1.outputLowValues, fq1.outputHighValues);

    const auto input2 = std::make_shared<ngraph::opset1::Parameter>(precision, inputShape2);
    const auto fakeQuantize2 = fq2.empty() ?
        nullptr :
        ngraph::builder::makeFakeQuantize(
            input2, precision, fq2.quantizationLevel, fq2.constantShape,
            fq2.inputLowValues, fq2.inputHighValues, fq2.outputLowValues, fq2.outputHighValues);

    const auto add = std::make_shared<ngraph::opset1::Add>(
        fq1.empty() ? input1 : fakeQuantize1,
        fq2.empty() ? input2 : fakeQuantize2);

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(add) };
    return std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input1, input2 }, "AddTransformation");
}

namespace {

std::shared_ptr<ngraph::opset1::FakeQuantize> makeFakeQuantizeWithNames(
        const Output<Node>& parent,
        const ngraph::element::Type precision,
        const ngraph::builder::subgraph::FakeQuantizeOnData& fqOnData,
        const std::string name) {
    auto fq = ngraph::builder::subgraph::makeFakeQuantize(parent, precision, fqOnData);
    fq->set_friendly_name(name);
    fq->get_input_node_ptr(1)->set_friendly_name(name + "/inputLow");
    fq->get_input_node_ptr(2)->set_friendly_name(name + "/inputHigh");
    fq->get_input_node_ptr(3)->set_friendly_name(name + "/outputLow");
    fq->get_input_node_ptr(4)->set_friendly_name(name + "/outputHigh");
    return fq;
}

} // namespace

std::shared_ptr<ngraph::Function> AddFunction::getOriginalSubgraphWithConvolutions(
        const ngraph::element::Type precision,
        const ngraph::PartialShape& inputShape,
        const bool broadcast,
        const ngraph::builder::subgraph::FakeQuantizeOnData& fqOnDataBefore1,
        const ngraph::builder::subgraph::Convolution& convolution1,
        const ngraph::builder::subgraph::FakeQuantizeOnData& fqOnDataAfter1,
        const ngraph::builder::subgraph::FakeQuantizeOnData& fqOnDataBefore2,
        const ngraph::builder::subgraph::Convolution& convolution2,
        const ngraph::builder::subgraph::FakeQuantizeOnData& fqOnDataAfter2,
        const ngraph::builder::subgraph::FakeQuantizeOnData& fqOnDataAfter) {
    ngraph::PartialShape inputShape2 = inputShape;

    if (broadcast) {
        inputShape2[2] = 1;
        inputShape2[3] = 1;
    }

    auto makeBranch = [&](
        const ngraph::element::Type precision,
        const ngraph::PartialShape& inputShape,
        const size_t index,
        const ngraph::builder::subgraph::FakeQuantizeOnData& fqOnDataBefore,
        const ngraph::builder::subgraph::Convolution& convolution,
        const ngraph::builder::subgraph::FakeQuantizeOnData& fqOnDataAfter) ->
            std::pair<std::shared_ptr<ngraph::opset1::Parameter>, std::shared_ptr<ngraph::Node>> {
        const auto input = std::make_shared<ngraph::opset1::Parameter>(precision, inputShape);
        input->set_friendly_name("input" + std::to_string(index));
        std::shared_ptr<ngraph::Node> parent = input;

        if (!fqOnDataBefore.empty()) {
            parent = makeFakeQuantizeWithNames(parent, precision, fqOnDataBefore, "fakeQuantizeBefore" + std::to_string(index));
        }

        if (!convolution.empty()) {
            parent = makeConvolution(parent, convolution);
            parent->set_friendly_name("convolution" + std::to_string(index));
        }

        if (!fqOnDataAfter.empty()) {
            parent = makeFakeQuantizeWithNames(parent, precision, fqOnDataAfter, "fakeQuantizeAfter" + std::to_string(index));
        }

        return std::make_pair(input, parent);
    };

    const auto branch1 = makeBranch(precision, inputShape, 1, fqOnDataBefore1, convolution1, fqOnDataAfter1);
    const auto branch2 = makeBranch(precision, inputShape, 2, fqOnDataBefore2, convolution2, fqOnDataAfter2);

    std::shared_ptr<ngraph::Node> result = std::make_shared<ngraph::opset1::Add>(branch1.second, branch2.second);
    result->set_friendly_name("add");

    if (!fqOnDataAfter.empty()) {
        result = makeFakeQuantizeWithNames(result, precision, fqOnDataAfter, "fakeQuantizeAfter");

        // we need a some operation to move dequantization operations away from FakeQuantize to avoid cleanup fuse
        result = std::make_shared<ngraph::opset1::MaxPool>(
            result,
            Strides{ 1, 1 },
            Shape{ 1, 1 },
            Shape{ 0, 0 },
            Shape{ 2, 2 },
            op::RoundingType::FLOOR);
        result->set_friendly_name("maxPool");
    }

    result = std::make_shared<ngraph::opset1::Result>(result);
    result->set_friendly_name("result");

    ngraph::ResultVector results{ std::dynamic_pointer_cast<ngraph::opset1::Result>(result) };
    return std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ branch1.first, branch2.first }, "AddTransformation");
}

std::shared_ptr<ngraph::Function> AddFunction::getReference(
    const ngraph::element::Type precision,
    const ngraph::PartialShape& inputShape1,
    const ngraph::PartialShape& inputShape2,
    const bool broadcast,
    const ngraph::pass::low_precision::LayerTransformation::Params& params,
    const ngraph::element::Type& precision1,
    const ngraph::builder::subgraph::DequantizationOperations& dequantization1,
    const ngraph::element::Type& precision2,
    const ngraph::builder::subgraph::DequantizationOperations& dequantization2,
    const ngraph::builder::subgraph::DequantizationOperations& dequantizationAfter,
    const int constInputIndex,
    const std::vector<float>& constValues,
    const std::string& additionalLayer,
    const std::string& operationType) {
    std::shared_ptr<ngraph::Node> input1;
    std::shared_ptr<ngraph::Node> parent1;
    if (constInputIndex == 0) {
        parent1 = std::make_shared<ngraph::opset1::Constant>(
            dequantizationAfter.empty() ? precision : element::f32,
            inputShape1.to_shape(),
            constValues);
    } else {
        input1 = std::make_shared<ngraph::opset1::Parameter>(
            additionalLayer != "" ? precision : (precision1.is_real() ? precision : precision1),
            broadcast ? ngraph::PartialShape({inputShape1[0], inputShape1[1], 1, 1}) : inputShape1);
        if (additionalLayer != "") {
            parent1 = ngraph::builder::subgraph::makeFakeQuantizeTypeRelaxed(
                    input1,
                    precision,
                    {256, Shape{}, {0}, {255}, {0}, {255}, precision1});
        } else {
            parent1 = input1;
        }
    }

    auto dequantizationStructure1 = dequantization1;
    dequantizationStructure1.multiply.outPrecision = dequantizationAfter.empty() ? precision : element::f32;
    const auto dequantizationOp1 = ov::is_type<ngraph::opset1::Constant>(parent1) ? parent1 : makeDequantization(parent1, dequantizationStructure1);

    std::shared_ptr<ngraph::Node> input2;
    if (constInputIndex == 1) {
        input2 = std::make_shared<ngraph::opset1::Constant>(
            dequantizationAfter.empty() ? precision : element::f32,
            inputShape2.to_shape(),
            constValues);
    } else {
        input2 = std::make_shared<ngraph::opset1::Parameter>(
            precision2.is_real() ? precision : precision2, inputShape2);
    }
    auto parent = input2;
    if (additionalLayer == "convolution") {
        parent = std::make_shared<ngraph::op::TypeRelaxed<ngraph::opset1::Convolution>>(
            std::vector<element::Type>{ element::f32, element::f32 },
            std::vector<element::Type>{ precision },
            ngraph::op::TemporaryReplaceOutputType(parent, element::f32).get(),
            ngraph::op::TemporaryReplaceOutputType(
                std::make_shared<ngraph::opset1::Constant>(element::i8, Shape{ 1, 4, 1, 1 }, std::vector<float>{0.8f, 0.8f, 0.8f, 0.8f}),
                element::f32).get(),
            ngraph::Strides{ 1, 1 },
            ngraph::CoordinateDiff{ 0, 0 },
            ngraph::CoordinateDiff{ 0, 0 },
            ngraph::Strides{ 1, 1 });
    }
    std::shared_ptr<Node> additional_output = nullptr;
    if (additionalLayer == "convolution_multiconsumers") {
        parent = std::make_shared<ngraph::op::TypeRelaxed<ngraph::opset1::Convolution>>(
                std::vector<element::Type>{ element::f32, element::f32 },
                std::vector<element::Type>{ precision },
                ngraph::op::TemporaryReplaceOutputType(parent, element::f32).get(),
                ngraph::op::TemporaryReplaceOutputType(
                        std::make_shared<ngraph::opset1::Constant>(element::i8, Shape{ 1, 4, 1, 1 }, std::vector<float>{0.8f, 0.8f, 0.8f, 0.8f}),
                        element::f32).get(),
                ngraph::Strides{ 1, 1 },
                ngraph::CoordinateDiff{ 0, 0 },
                ngraph::CoordinateDiff{ 0, 0 },
                ngraph::Strides{ 1, 1 });
        additional_output = parent;
    }
    if (additionalLayer == "group_convolution") {
        parent = std::make_shared< ngraph::op::TypeRelaxed<ngraph::opset1::GroupConvolution>>(
            std::vector<element::Type>{ element::f32, element::f32 },
            std::vector<element::Type>{ precision },
            ngraph::op::TemporaryReplaceOutputType(parent, element::f32).get(),
            ngraph::op::TemporaryReplaceOutputType(
                std::make_shared<ngraph::opset1::Constant>(element::i8, Shape{ 4, 1, 1, 1, 1 }, std::vector<float>{0.8f, 0.8f, 0.8f, 0.8f}),
                element::f32).get(),
            ngraph::Strides{ 1, 1 },
            ngraph::CoordinateDiff{ 0, 0 },
            ngraph::CoordinateDiff{ 0, 0 },
            ngraph::Strides{ 1, 1 });
    }
    if (additionalLayer != "") {
        parent = std::make_shared<ngraph::opset1::Add>(
            parent,
            std::make_shared<ngraph::opset1::Constant>(precision, Shape{ 1, 1, 1, 1 }, std::vector<float>{1.f}));
        parent = ngraph::builder::subgraph::makeFakeQuantizeTypeRelaxed(
            parent,
            precision,
            {256, Shape{}, { 0 }, { 255 }, { 0 }, { 255 }, element::u8});
    }

    auto dequantizationStructure2 = dequantization2;
    dequantizationStructure2.multiply.outPrecision = dequantizationAfter.empty() ? precision : element::f32;
    const auto dequantizationOp2 = ov::is_type<ngraph::opset1::Constant>(parent) ? parent : makeDequantization(parent, dequantizationStructure2);

    const std::shared_ptr<Node> add = operationType == "Add" ?
        std::dynamic_pointer_cast<Node>(std::make_shared<ngraph::op::TypeRelaxed<ngraph::opset1::Add>>(
            std::vector<element::Type>{ element::f32, element::f32 },
            std::vector<element::Type>{ element::f32 },
            ngraph::op::TemporaryReplaceOutputType(dequantizationOp1, element::f32).get(),
            ngraph::op::TemporaryReplaceOutputType(dequantizationOp2, element::f32).get())) :
        std::make_shared<ngraph::op::TypeRelaxed<opset1::Subtract>>(
            std::vector<element::Type>{ element::f32, element::f32 },
            std::vector<element::Type>{ element::f32 },
            ngraph::op::TemporaryReplaceOutputType(dequantizationOp1, element::f32).get(),
            ngraph::op::TemporaryReplaceOutputType(dequantizationOp2, element::f32).get());

    NetworkHelper::setOutDataPrecisionForTypeRelaxed(add, dequantizationAfter.empty() ? precision : element::f32);
    auto& rtInfo = add->get_rt_info();
    rtInfo["Variant::std::string"] = "add";

    auto dequantizationStructureAfter = dequantizationAfter;
    dequantizationStructureAfter.multiply.outPrecision = precision;
    const auto dequantizationOpAfter = makeDequantization(add, dequantizationStructureAfter);

    dequantizationOpAfter->set_friendly_name("output");
    std::shared_ptr<Node> output = dequantizationOpAfter;
    if (additional_output != nullptr) {
        output = std::make_shared<opset1::Multiply>(dequantizationOpAfter, additional_output);
        output->set_friendly_name("output_multiply");
    }

    ngraph::ResultVector results {std::make_shared<ngraph::opset1::Result>(output)};

    ngraph::ParameterVector parameters;
    if (constInputIndex == -1) {
        parameters = { ov::as_type_ptr<ngraph::opset1::Parameter>(input1), ov::as_type_ptr<ngraph::opset1::Parameter>(input2) };
    } else if (constInputIndex == 0) {
        parameters = { ov::as_type_ptr<ngraph::opset1::Parameter>(input2) };
    } else if (constInputIndex == 1) {
        parameters = { ov::as_type_ptr<ngraph::opset1::Parameter>(input1) };
    } else {
        throw std::runtime_error("Unexpected constant input index");
    }
    return std::make_shared<ngraph::Function>(results, parameters, "AddTransformation");
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
