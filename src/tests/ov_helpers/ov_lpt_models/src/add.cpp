// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_lpt_models/add.hpp"

#include "low_precision/network_helper.hpp"
#include "low_precision/layer_transformation.hpp"

#include "openvino/opsets/opset1.hpp"

#include "ov_lpt_models/common/dequantization_operations.hpp"
#include "ov_models/subgraph_builders.hpp"

using namespace ov::pass::low_precision;

namespace ngraph {
namespace builder {
namespace subgraph {

namespace {
std::shared_ptr<Node> configure_postops(const std::shared_ptr<Node>& parent,
                                        const ov::element::Type& precision,
                                        const std::string& postops_configuration) {
    std::shared_ptr<Node> res = parent;
    if (postops_configuration.empty() || postops_configuration == "bias") {
        auto bias = ov::opset1::Constant::create(precision, { 1, 1, 1, 1 }, {1.f});
        res = std::make_shared<ov::opset1::Add>(res, bias);
    } else if (postops_configuration == "bias_on_zero_input") {
        auto bias = ov::opset1::Constant::create(precision, { 1, 1, 1, 1 }, {1.f});
        res = std::make_shared<ov::opset1::Add>(bias, res);
    } else {
        return parent;
    }

    return ngraph::builder::subgraph::makeFakeQuantizeTypeRelaxed(
           res,
           precision,
           {256, Shape{}, { 0 }, { 255 }, { 0 }, { 255 }, element::u8});
}
}  // namespace

std::shared_ptr<ov::Model> AddFunction::getOriginal(
    const ov::element::Type precision,
    const ov::PartialShape& inputShape1,
    const ov::PartialShape& inputShape2,
    const bool broadcast,
    const ov::pass::low_precision::LayerTransformation::Params& params,
    const ov::element::Type& precision1,
    const ngraph::builder::subgraph::DequantizationOperations& dequantization1,
    const ov::element::Type& precision2,
    const ngraph::builder::subgraph::DequantizationOperations& dequantization2,
    const int constInput,
    const std::vector<float>& constValues,
    const std::string& additionalLayer,
    const std::string& postops_configuration) {
    std::shared_ptr<ov::Node> input1;
    std::shared_ptr<ov::Node> parent1;
    if (constInput == 0) {
        parent1 = std::make_shared<ov::opset1::Constant>(
            precision,
            inputShape1.to_shape(),
            constValues);
    } else {
        input1 = std::make_shared<ov::opset1::Parameter>(
            additionalLayer != "" ? precision : (precision1.is_real() ? precision : precision1),
            broadcast ? ov::PartialShape({inputShape1[0], inputShape1[1], 1, 1}) : inputShape1);
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

    std::shared_ptr<ov::Node> input2;
    if (constInput == 1) {
        input2 = std::make_shared<ov::opset1::Constant>(
            precision,
            inputShape2.to_shape(),
            constValues);
    } else {
        input2 = std::make_shared<ov::opset1::Parameter>(
            precision2.is_real() ? precision : precision2, inputShape2);
    }
    auto parent = input2;
    if (additionalLayer == "convolution") {
        parent = std::make_shared<ov::op::TypeRelaxed<ov::opset1::Convolution>>(
            std::vector<element::Type>{ element::f32, element::f32 },
            std::vector<element::Type>{ precision },
            ov::op::TemporaryReplaceOutputType(parent, element::f32).get(),
            ov::op::TemporaryReplaceOutputType(
                std::make_shared<ov::opset1::Constant>(element::i8, Shape{ 1, 4, 1, 1 }, std::vector<float>{0.8f, 0.8f, 0.8f, 0.8f}),
                element::f32).get(),
            ov::Strides{ 1, 1 },
            ov::CoordinateDiff{ 0, 0 },
            ov::CoordinateDiff{ 0, 0 },
            ov::Strides{ 1, 1 });
    }
    std::shared_ptr<Node> additional_output = nullptr;
    if (additionalLayer == "convolution_multiconsumers") {
        parent = std::make_shared<ov::op::TypeRelaxed<ov::opset1::Convolution>>(
                std::vector<element::Type>{ element::f32, element::f32 },
                std::vector<element::Type>{ precision },
                ov::op::TemporaryReplaceOutputType(parent, element::f32).get(),
                ov::op::TemporaryReplaceOutputType(
                        std::make_shared<ov::opset1::Constant>(element::i8, Shape{ 1, 4, 1, 1 }, std::vector<float>{0.8f, 0.8f, 0.8f, 0.8f}),
                        element::f32).get(),
                ov::Strides{ 1, 1 },
                ov::CoordinateDiff{ 0, 0 },
                ov::CoordinateDiff{ 0, 0 },
                ov::Strides{ 1, 1 });
        additional_output = parent;
    }
    if (additionalLayer == "group_convolution") {
        parent = std::make_shared< ov::op::TypeRelaxed<ov::opset1::GroupConvolution>>(
            std::vector<element::Type>{ element::f32, element::f32 },
            std::vector<element::Type>{ precision },
            ov::op::TemporaryReplaceOutputType(parent, element::f32).get(),
            ov::op::TemporaryReplaceOutputType(
                std::make_shared<ov::opset1::Constant>(element::i8, Shape{ 4, 1, 1, 1, 1 }, std::vector<float>{0.8f, 0.8f, 0.8f, 0.8f}),
                element::f32).get(),
            ov::Strides{ 1, 1 },
            ov::CoordinateDiff{ 0, 0 },
            ov::CoordinateDiff{ 0, 0 },
            ov::Strides{ 1, 1 });
    }
    if (additionalLayer != "") {
        parent = configure_postops(parent, precision, postops_configuration);
    }

    auto dequantizationStructure2 = dequantization2;
    dequantizationStructure2.multiply.outPrecision = precision;
    const auto dequantizationOp2 = dequantization2.empty() ? parent : makeDequantization(parent, dequantizationStructure2);

    const auto add = std::make_shared<ov::opset1::Add>(dequantizationOp1, dequantizationOp2);
    add->set_friendly_name("output");
    auto& rtInfo = add->get_rt_info();
    rtInfo["Variant::std::string"] = "add";

    std::shared_ptr<Node> output = add;
    if (additional_output != nullptr) {
        output = std::make_shared<ov::opset1::Multiply>(add, additional_output);
        output->set_friendly_name("output_multiply");
    }

    ov::ResultVector results {std::make_shared<ov::opset1::Result>(output)};
    ov::ParameterVector parameters;
    if (constInput == -1) {
        parameters = { ov::as_type_ptr<ov::opset1::Parameter>(input1), ov::as_type_ptr<ov::opset1::Parameter>(input2) };
    } else if (constInput == 0) {
        parameters = { ov::as_type_ptr<ov::opset1::Parameter>(input2) };
    } else if (constInput == 1) {
        parameters = { ov::as_type_ptr<ov::opset1::Parameter>(input1) };
    } else {
        throw std::runtime_error("Unexpected constant input index");
    }
    return std::make_shared<ov::Model>(results, parameters, "AddTransformation");
}

std::shared_ptr<ov::Model> AddFunction::getOriginal(
    const ov::element::Type precision,
    const ov::PartialShape& inputShape,
    const bool broadcast,
    const ngraph::builder::subgraph::FakeQuantizeOnData& fqOnData1,
    const ngraph::builder::subgraph::FakeQuantizeOnData& fqOnData2) {
    ov::PartialShape inputShape2 = inputShape;

    if (broadcast) {
        inputShape2[2] = 1;
        inputShape2[3] = 1;
    }

    auto fq1 = fqOnData1;
    auto fq2 = fqOnData2;

    const auto input1 = std::make_shared<ov::opset1::Parameter>(precision, inputShape);
    const auto fakeQuantize1 = fq1.empty() ?
        nullptr :
        ngraph::builder::makeFakeQuantize(
            input1, precision, fq1.quantizationLevel, fq1.constantShape,
            fq1.inputLowValues, fq1.inputHighValues, fq1.outputLowValues, fq1.outputHighValues);

    const auto input2 = std::make_shared<ov::opset1::Parameter>(precision, inputShape2);
    const auto fakeQuantize2 = fq2.empty() ?
        nullptr :
        ngraph::builder::makeFakeQuantize(
            input2, precision, fq2.quantizationLevel, fq2.constantShape,
            fq2.inputLowValues, fq2.inputHighValues, fq2.outputLowValues, fq2.outputHighValues);

    const auto add = std::make_shared<ov::opset1::Add>(
        fq1.empty() ? input1 : fakeQuantize1,
        fq2.empty() ? input2 : fakeQuantize2);

    ov::ResultVector results{ std::make_shared<ov::opset1::Result>(add) };
    return std::make_shared<ov::Model>(results, ov::ParameterVector{ input1, input2 }, "AddTransformation");
}

std::shared_ptr<ov::Model> AddFunction::getReference(
    const ov::element::Type precision,
    const ov::PartialShape& inputShape1,
    const ov::PartialShape& inputShape2,
    const bool broadcast,
    const ov::pass::low_precision::LayerTransformation::Params& params,
    const ov::element::Type& precision1,
    const ngraph::builder::subgraph::DequantizationOperations& dequantization1,
    const ov::element::Type& precision2,
    const ngraph::builder::subgraph::DequantizationOperations& dequantization2,
    const ngraph::builder::subgraph::DequantizationOperations& dequantizationAfter,
    const int constInputIndex,
    const std::vector<float>& constValues,
    const std::string& additionalLayer,
    const std::string& operationType,
    const std::string& postops_configuration) {
    std::shared_ptr<ov::Node> input1;
    std::shared_ptr<ov::Node> parent1;
    if (constInputIndex == 0) {
        parent1 = std::make_shared<ov::opset1::Constant>(
            dequantizationAfter.empty() ? precision : element::f32,
            inputShape1.to_shape(),
            constValues);
    } else {
        input1 = std::make_shared<ov::opset1::Parameter>(
            additionalLayer != "" ? precision : (precision1.is_real() ? precision : precision1),
            broadcast ? ov::PartialShape({inputShape1[0], inputShape1[1], 1, 1}) : inputShape1);
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
    const auto dequantizationOp1 = ov::is_type<ov::opset1::Constant>(parent1) ? parent1 : makeDequantization(parent1, dequantizationStructure1);

    std::shared_ptr<ov::Node> input2;
    if (constInputIndex == 1) {
        input2 = std::make_shared<ov::opset1::Constant>(
            dequantizationAfter.empty() ? precision : element::f32,
            inputShape2.to_shape(),
            constValues);
    } else {
        input2 = std::make_shared<ov::opset1::Parameter>(
            precision2.is_real() ? precision : precision2, inputShape2);
    }
    auto parent = input2;
    if (additionalLayer == "convolution") {
        parent = std::make_shared<ov::op::TypeRelaxed<ov::opset1::Convolution>>(
            std::vector<element::Type>{ element::f32, element::f32 },
            std::vector<element::Type>{ precision },
            ov::op::TemporaryReplaceOutputType(parent, element::f32).get(),
            ov::op::TemporaryReplaceOutputType(
                std::make_shared<ov::opset1::Constant>(element::i8, Shape{ 1, 4, 1, 1 }, std::vector<float>{0.8f, 0.8f, 0.8f, 0.8f}),
                element::f32).get(),
            ov::Strides{ 1, 1 },
            ov::CoordinateDiff{ 0, 0 },
            ov::CoordinateDiff{ 0, 0 },
            ov::Strides{ 1, 1 });
    }
    std::shared_ptr<Node> additional_output = nullptr;
    if (additionalLayer == "convolution_multiconsumers") {
        parent = std::make_shared<ov::op::TypeRelaxed<ov::opset1::Convolution>>(
                std::vector<element::Type>{ element::f32, element::f32 },
                std::vector<element::Type>{ precision },
                ov::op::TemporaryReplaceOutputType(parent, element::f32).get(),
                ov::op::TemporaryReplaceOutputType(
                        std::make_shared<ov::opset1::Constant>(element::i8, Shape{ 1, 4, 1, 1 }, std::vector<float>{0.8f, 0.8f, 0.8f, 0.8f}),
                        element::f32).get(),
                ov::Strides{ 1, 1 },
                ov::CoordinateDiff{ 0, 0 },
                ov::CoordinateDiff{ 0, 0 },
                ov::Strides{ 1, 1 });
        additional_output = parent;
    }
    if (additionalLayer == "group_convolution") {
        parent = std::make_shared< ov::op::TypeRelaxed<ov::opset1::GroupConvolution>>(
            std::vector<element::Type>{ element::f32, element::f32 },
            std::vector<element::Type>{ precision },
            ov::op::TemporaryReplaceOutputType(parent, element::f32).get(),
            ov::op::TemporaryReplaceOutputType(
                std::make_shared<ov::opset1::Constant>(element::i8, Shape{ 4, 1, 1, 1, 1 }, std::vector<float>{0.8f, 0.8f, 0.8f, 0.8f}),
                element::f32).get(),
            ov::Strides{ 1, 1 },
            ov::CoordinateDiff{ 0, 0 },
            ov::CoordinateDiff{ 0, 0 },
            ov::Strides{ 1, 1 });
    }
    if (additionalLayer != "") {
        parent = configure_postops(parent, precision, postops_configuration);
    }

    auto dequantizationStructure2 = dequantization2;
    dequantizationStructure2.multiply.outPrecision = dequantizationAfter.empty() ? precision : element::f32;
    const auto dequantizationOp2 = ov::is_type<ov::opset1::Constant>(parent) ? parent : makeDequantization(parent, dequantizationStructure2);

    const std::shared_ptr<Node> add = operationType == "Add" ?
        std::dynamic_pointer_cast<Node>(std::make_shared<ov::op::TypeRelaxed<ov::opset1::Add>>(
            std::vector<element::Type>{ element::f32, element::f32 },
            std::vector<element::Type>{ element::f32 },
            ov::op::TemporaryReplaceOutputType(dequantizationOp1, element::f32).get(),
            ov::op::TemporaryReplaceOutputType(dequantizationOp2, element::f32).get())) :
        std::make_shared<ov::op::TypeRelaxed<ov::opset1::Subtract>>(
            std::vector<element::Type>{ element::f32, element::f32 },
            std::vector<element::Type>{ element::f32 },
            ov::op::TemporaryReplaceOutputType(dequantizationOp1, element::f32).get(),
            ov::op::TemporaryReplaceOutputType(dequantizationOp2, element::f32).get());

    NetworkHelper::setOutDataPrecisionForTypeRelaxed(add, dequantizationAfter.empty() ? precision : element::f32);
    auto& rtInfo = add->get_rt_info();
    rtInfo["Variant::std::string"] = "add";

    auto dequantizationStructureAfter = dequantizationAfter;
    dequantizationStructureAfter.multiply.outPrecision = precision;
    const auto dequantizationOpAfter = makeDequantization(add, dequantizationStructureAfter);

    dequantizationOpAfter->set_friendly_name("output");
    std::shared_ptr<Node> output = dequantizationOpAfter;
    if (additional_output != nullptr) {
        output = std::make_shared<ov::opset1::Multiply>(dequantizationOpAfter, additional_output);
        output->set_friendly_name("output_multiply");
    }

    ov::ResultVector results {std::make_shared<ov::opset1::Result>(output)};

    ov::ParameterVector parameters;
    if (constInputIndex == -1) {
        parameters = { ov::as_type_ptr<ov::opset1::Parameter>(input1), ov::as_type_ptr<ov::opset1::Parameter>(input2) };
    } else if (constInputIndex == 0) {
        parameters = { ov::as_type_ptr<ov::opset1::Parameter>(input2) };
    } else if (constInputIndex == 1) {
        parameters = { ov::as_type_ptr<ov::opset1::Parameter>(input1) };
    } else {
        throw std::runtime_error("Unexpected constant input index");
    }
    return std::make_shared<ov::Model>(results, parameters, "AddTransformation");
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
