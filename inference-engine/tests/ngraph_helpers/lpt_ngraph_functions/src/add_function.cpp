// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/opsets/opset1.hpp"

#include "lpt_ngraph_functions/common/dequantization_operations.hpp"
#include "lpt_ngraph_functions/common/builders.hpp"
#include "lpt_ngraph_functions/add_function.hpp"
#include "ngraph_functions/subgraph_builders.hpp"
#include "low_precision/network_helper.hpp"
#include "low_precision/layer_transformation.hpp"

using namespace ngraph::pass::low_precision;

namespace ngraph {
namespace builder {
namespace subgraph {

std::shared_ptr<ngraph::Function> AddFunction::getOriginal(
    const ngraph::element::Type precision,
    const ngraph::Shape& inputShape,
    const bool broadcast,
    const ngraph::pass::low_precision::LayerTransformation::Params& params,
    const ngraph::element::Type& precision1,
    const ngraph::builder::subgraph::DequantizationOperations& dequantization1,
    const ngraph::element::Type& precision2,
    const ngraph::builder::subgraph::DequantizationOperations& dequantization2,
    const int constInput,
    const ngraph::builder::subgraph::Constant constant,
    const std::string& additionalLayer) {
    std::shared_ptr<ngraph::Node> input1;
    if (constInput == 0) {
        input1 = std::make_shared<ngraph::opset1::Constant>(
            constant.outPrecision == element::undefined ? precision : constant.outPrecision,
            constant.shapeIsDefined ? constant.shape : inputShape,
            constant.values);
    } else {
        input1 = std::make_shared<ngraph::opset1::Parameter>(
            precision1.is_real() ? precision : precision1,
            broadcast ? ngraph::Shape({ inputShape[0], inputShape[1], 1, 1 }) : ngraph::Shape(inputShape));
    }

    auto dequantizationStructure1 = dequantization1;
    dequantizationStructure1.multiply.outPrecision = precision;
    if (dequantizationStructure1.multiply.empty()) {
        dequantizationStructure1.subtract.outPrecision = precision;
    }

    const auto dequantizationOp1 = dequantization1.empty() ? input1 : makeDequantization(input1, dequantizationStructure1);

    std::shared_ptr<ngraph::Node> input2;
    if (constInput == 1) {
        input2 = std::make_shared<ngraph::opset1::Constant>(
            constant.outPrecision == element::undefined ? precision : constant.outPrecision,
            constant.shapeIsDefined ? constant.shape : inputShape,
            constant.values);
    } else {
        input2 = std::make_shared<ngraph::opset1::Parameter>(
            precision2.is_real() ? precision : precision2, ngraph::Shape(inputShape));
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
    rtInfo["Variant::std::string"] = std::make_shared<VariantWrapper<std::string>>("add");

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(add) };
    ngraph::ParameterVector parameters;
    if (constInput == -1) {
        parameters = { as_type_ptr<ngraph::opset1::Parameter>(input1), as_type_ptr<ngraph::opset1::Parameter>(input2) };
    } else if (constInput == 0) {
        parameters = { as_type_ptr<ngraph::opset1::Parameter>(input2) };
    } else if (constInput == 1) {
        parameters = { as_type_ptr<ngraph::opset1::Parameter>(input1) };
    } else {
        throw std::runtime_error("Unexpected constant input index");
    }
    return std::make_shared<ngraph::Function>(results, parameters, "AddTransformation");
}

std::shared_ptr<ngraph::Function> AddFunction::getOriginal(
    const ngraph::element::Type precision,
    const ngraph::Shape& inputShape,
    const bool broadcast,
    const ngraph::builder::subgraph::FakeQuantizeOnData& fqOnData1,
    const AddOperation& operation1,
    const ngraph::builder::subgraph::FakeQuantizeOnData& fqOnData2,
    const AddOperation& operation2,
    const int constInput) {
    ngraph::Shape inputShape2 = inputShape;

    ngraph::Shape constantShape = inputShape;
    constantShape[0] = 1ul;

    if (broadcast) {
        inputShape2[2] = 1ul;
        inputShape2[3] = 1ul;

        constantShape[2] = 1ul;
        constantShape[3] = 1ul;
    }

    auto fq1 = fqOnData1;
    auto fq2 = fqOnData2;

    auto createBranch = [](
        const ngraph::element::Type precision,
        const std::shared_ptr<ngraph::Node>& parent,
        const ngraph::builder::subgraph::FakeQuantizeOnData& fqOnData,
        const AddOperation& operation,
        const size_t branchIndex) -> std::shared_ptr<ngraph::Node> {
        auto result = fqOnData.empty() ?
            parent :
            std::dynamic_pointer_cast<ngraph::Node>(ngraph::builder::makeFakeQuantize(
                parent,
                precision,
                fqOnData.quantizationLevel,
                fqOnData.constantShape,
                fqOnData.inputLowValues,
                fqOnData.inputHighValues,
                fqOnData.outputLowValues,
                fqOnData.outputHighValues));

        if (!operation.empty()) {
            std::shared_ptr<Node> tmpOnWeights = std::make_shared<ngraph::opset1::Constant>(
                operation.constantOnWeights.outPrecision,
                operation.constantOnWeights.shape,
                operation.constantOnWeights.values);
            tmpOnWeights->set_friendly_name("constantOnWeights" + std::to_string(branchIndex));

            if (!operation.fakeQuantizeOnWeights.empty()) {
                tmpOnWeights = ngraph::builder::subgraph::makeFakeQuantize(
                    tmpOnWeights,
                    operation.constantOnWeights.outPrecision,
                    operation.fakeQuantizeOnWeights);
                tmpOnWeights->set_friendly_name("fakeQuantizeOnWeights" + std::to_string(branchIndex));
            }

            if (!operation.dequantizationOperations.empty()) {
                tmpOnWeights = makeDequantization(tmpOnWeights, operation.dequantizationOperations);
            }

            result = std::make_shared<ngraph::op::TypeRelaxed<ngraph::opset1::Convolution>>(
                std::vector<element::Type>{ element::f32, element::f32 },
                std::vector<element::Type>{ element::f32 },
                ngraph::op::TemporaryReplaceOutputType(result, element::f32).get(),
                ngraph::op::TemporaryReplaceOutputType(tmpOnWeights, element::f32).get(),
                ngraph::Strides{ 1, 1 },
                ngraph::CoordinateDiff{ 0, 0 },
                ngraph::CoordinateDiff{ 0, 0 },
                ngraph::Strides{ 1, 1 });
            result->set_friendly_name("convolution" + std::to_string(branchIndex));
        }

        return result;
    };

    ngraph::ParameterVector inputs;

    const std::shared_ptr<ngraph::Node> input1 = constInput == 0 ?
        std::make_shared<ngraph::opset1::Constant>(element::f32, constantShape, std::vector<float>(1.f, ngraph::shape_size(constantShape))) :
        std::dynamic_pointer_cast<ngraph::Node>(std::make_shared<ngraph::opset1::Parameter>(precision, inputShape));
    if (is_type<op::Parameter>(input1)) {
        inputs.push_back(as_type_ptr<op::Parameter>(input1));
    }
    const auto parent1 = createBranch(precision, input1, fq1, operation1, 0ul);

    const std::shared_ptr<ngraph::Node>& input2 = constInput == 1 ?
        std::make_shared<ngraph::opset1::Constant>(element::f32, constantShape, std::vector<float>(1.f, ngraph::shape_size(constantShape))) :
        std::dynamic_pointer_cast<ngraph::Node>(std::make_shared<ngraph::opset1::Parameter>(precision, inputShape2));
    if (is_type<op::Parameter>(input2)) {
        inputs.push_back(as_type_ptr<op::Parameter>(input2));
    }
    const auto parent2 = createBranch(precision, input2, fq2, operation2, 1ul);

    const auto add = std::make_shared<ngraph::opset1::Add>(parent1, parent2);
    add->set_friendly_name("add");

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(add) };
    return std::make_shared<ngraph::Function>(results, inputs, "AddTransformation");
}

std::shared_ptr<ngraph::Function> AddFunction::getReference(
    const ngraph::element::Type precision,
    const ngraph::Shape& inputShape,
    const bool broadcast,
    const ngraph::pass::low_precision::LayerTransformation::Params& params,
    const ngraph::element::Type& precision1,
    const ngraph::builder::subgraph::DequantizationOperations& dequantization1,
    const ngraph::element::Type& precision2,
    const ngraph::builder::subgraph::DequantizationOperations& dequantization2,
    const ngraph::builder::subgraph::DequantizationOperations& dequantizationAfter,
    const int constInputIndex,
    const ngraph::builder::subgraph::Constant constant,
    const std::string& additionalLayer,
    const std::string& operationType) {
    std::shared_ptr<ngraph::Node> input1;
    if (constInputIndex == 0) {
        input1 = std::make_shared<ngraph::opset1::Constant>(
            constant.outPrecision == element::undefined ? (dequantizationAfter.empty() ? precision : element::f32) : constant.outPrecision,
            constant.shapeIsDefined ? constant.shape : inputShape,
            constant.values);
    } else {
        input1 = std::make_shared<ngraph::opset1::Parameter>(
            precision1.is_real() ? precision : precision1,
            broadcast ? ngraph::Shape({ inputShape[0], inputShape[1], 1, 1 }) : ngraph::Shape(inputShape));
    }

    auto dequantizationStructure1 = dequantization1;
    dequantizationStructure1.multiply.outPrecision = dequantizationAfter.empty() ? precision : element::f32;
    const auto dequantizationOp1 = is_type<ngraph::opset1::Constant>(input1) ? input1 : makeDequantization(input1, dequantizationStructure1);

    std::shared_ptr<ngraph::Node> input2;
    if (constInputIndex == 1) {
        input2 = std::make_shared<ngraph::opset1::Constant>(
            constant.outPrecision == element::undefined ? (dequantizationAfter.empty() ? precision : element::f32) : constant.outPrecision,
            constant.shapeIsDefined ? constant.shape : inputShape,
            constant.values);
    } else {
        input2 = std::make_shared<ngraph::opset1::Parameter>(
            precision2.is_real() ? precision : precision2, ngraph::Shape(inputShape));
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
    const auto dequantizationOp2 = is_type<ngraph::opset1::Constant>(parent) ? parent : makeDequantization(parent, dequantizationStructure2);

    const std::shared_ptr<Node> add = operationType == "Add" ?
        std::dynamic_pointer_cast<Node>(std::make_shared<ngraph::op::TypeRelaxed<ngraph::opset1::Add>>(
            std::vector<element::Type>{ element::f32, element::f32 },
            std::vector<element::Type>{ element::f32 },
            ngraph::op::TemporaryReplaceOutputType(dequantizationOp1, element::f32).get(),
            ngraph::op::TemporaryReplaceOutputType(dequantizationOp2, element::f32).get())) :
        std::make_shared<ngraph::op::TypeRelaxed<DequantizationSubtract>>(
            std::vector<element::Type>{ element::f32, element::f32 },
            std::vector<element::Type>{ element::f32 },
            ngraph::op::TemporaryReplaceOutputType(dequantizationOp1, element::f32).get(),
            ngraph::op::TemporaryReplaceOutputType(dequantizationOp2, element::f32).get());

    NetworkHelper::setOutDataPrecisionForTypeRelaxed(add, dequantizationAfter.empty() ? precision : element::f32);
    auto& rtInfo = add->get_rt_info();
    rtInfo["Variant::std::string"] = std::make_shared<VariantWrapper<std::string>>("add");

    auto dequantizationStructureAfter = dequantizationAfter;
    dequantizationStructureAfter.multiply.outPrecision = precision;
    const auto dequantizationOpAfter = makeDequantization(add, dequantizationStructureAfter);

    dequantizationOpAfter->set_friendly_name("output");

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(dequantizationOpAfter) };
    ngraph::ParameterVector parameters;
    if (constInputIndex == -1) {
        parameters = { as_type_ptr<ngraph::opset1::Parameter>(input1), as_type_ptr<ngraph::opset1::Parameter>(input2) };
    } else if (constInputIndex == 0) {
        parameters = { as_type_ptr<ngraph::opset1::Parameter>(input2) };
    } else if (constInputIndex == 1) {
        parameters = { as_type_ptr<ngraph::opset1::Parameter>(input1) };
    } else {
        throw std::runtime_error("Unexpected constant input index");
    }
    return std::make_shared<ngraph::Function>(results, parameters, "AddTransformation");
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
