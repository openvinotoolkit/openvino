// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_functions/low_precision_transformations/add_function.hpp"
#include "low_precision/network_helper.hpp"

#include <ngraph/opsets/opset1.hpp>
#include "ngraph_functions/builders.hpp"
#include "ngraph_functions/subgraph_builders.hpp"

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
    const std::vector<float>& constValues,
    const std::string& additionalLayer) {
    std::shared_ptr<ngraph::Node> input1;
    if (constInput == 0) {
        input1 = std::make_shared<ngraph::opset1::Constant>(
            precision,
            inputShape,
            constValues);
    } else {
        input1 = std::make_shared<ngraph::opset1::Parameter>(
            precision1,
            broadcast ? ngraph::Shape({ inputShape[0], inputShape[1], 1, 1 }) : ngraph::Shape(inputShape));
    }
    input1->set_friendly_name("input1");

    const auto dequantizationOp1 = is_type<ngraph::opset1::Constant>(input1) ? input1 : makeDequantization(input1, dequantization1);

    std::shared_ptr<ngraph::Node> input2;
    if (constInput == 1) {
        input2 = std::make_shared<ngraph::opset1::Constant>(
            precision,
            inputShape,
            constValues);
    } else {
        input2 = std::make_shared<ngraph::opset1::Parameter>(
            precision2, ngraph::Shape(inputShape));
    }
    input2->set_friendly_name("input2");

    auto parent = input2;
    if (!additionalLayer.empty()) {
        if (additionalLayer == "convolution") {
            parent = std::make_shared<ngraph::op::TypeRelaxed<ngraph::opset1::Convolution>>(
                std::vector<element::Type>{ element::f32, element::f32 },
                std::vector<element::Type>{ element::f32 },
                ngraph::op::TemporaryReplaceOutputType(parent, element::f32).get(),
                ngraph::op::TemporaryReplaceOutputType(
                    std::make_shared<ngraph::opset1::Constant>(element::i8, Shape{ 1, 4, 1, 1 }, std::vector<float>{0.8f, 0.8f, 0.8f, 0.8f}),
                    element::f32).get(),
                ngraph::Strides{ 1, 1 },
                ngraph::CoordinateDiff{ 0, 0 },
                ngraph::CoordinateDiff{ 0, 0 },
                ngraph::Strides{ 1, 1 });
            parent->set_friendly_name("AdditionalLayer/Convolution");
        } else if (additionalLayer == "group_convolution") {
            parent = std::make_shared< ngraph::op::TypeRelaxed<ngraph::opset1::GroupConvolution>>(
                std::vector<element::Type>{ element::f32, element::f32 },
                std::vector<element::Type>{ element::f32 },
                ngraph::op::TemporaryReplaceOutputType(parent, element::f32).get(),
                ngraph::op::TemporaryReplaceOutputType(
                    std::make_shared<ngraph::opset1::Constant>(element::i8, Shape{ 4, 1, 1, 1, 1 }, std::vector<float>{0.8f, 0.8f, 0.8f, 0.8f}),
                    element::f32).get(),
                ngraph::Strides{ 1, 1 },
                ngraph::CoordinateDiff{ 0, 0 },
                ngraph::CoordinateDiff{ 0, 0 },
                ngraph::Strides{ 1, 1 });
            parent->set_friendly_name("AdditionalLayer/GroupConvolution");
        }

        parent = std::make_shared<ngraph::opset1::Add>(
            parent,
            std::make_shared<ngraph::opset1::Constant>(element::f32, Shape{ 1, 1, 1, 1 }, std::vector<float>{1.f}));
        parent->set_friendly_name("AdditionalLayer/Add");

        parent = ngraph::builder::subgraph::makeFakeQuantizeTypeRelaxed(
            parent,
            ngraph::element::f32,
            { 256, Shape{}, { 0 }, { 255 }, { 0 }, { 255 }, element::u8 });
        parent->set_friendly_name("AdditionalLayer/FakeQuantize");
    }
    const auto dequantizationOp2 = is_type<ngraph::opset1::Constant>(parent) ? parent : makeDequantization(parent, dequantization2);

    const auto add = std::make_shared<ngraph::opset1::Add>(dequantizationOp1, dequantizationOp2);
    add->set_friendly_name("output");
    auto& rtInfo = add->get_rt_info();
    rtInfo["Variant::std::string"] = std::make_shared<VariantWrapper<std::string>>("add");

    const auto result = std::make_shared<ngraph::opset1::Result>(add);
    result->set_friendly_name("result");
    ngraph::ResultVector results{ result };

    ngraph::ParameterVector parameters;
    if (constInput == -1) {
        parameters = { as_type_ptr<ngraph::opset1::Parameter>(input1), as_type_ptr<ngraph::opset1::Parameter>(input2) };
    } else if (constInput == 0) {
        parameters = { as_type_ptr<ngraph::opset1::Parameter>(input2) };
    } else if (constInput == 1) {
        parameters = { as_type_ptr<ngraph::opset1::Parameter>(input1) };
    } else {
        THROW_IE_EXCEPTION << "Unexpected constant input index";
    }
    return std::make_shared<ngraph::Function>(results, parameters, "AddTransformation");
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
    const int constInput,
    const std::vector<float>& constValues,
    const std::string& additionalLayer,
    const std::string& operationType) {
    std::shared_ptr<ngraph::Node> input1;
    // Constant operations after transformations are on 1 input only
    const auto constInputIndex = constInput == 0 ? 1 : -1;
    if (constInputIndex == 0) {
        input1 = std::make_shared<ngraph::opset1::Constant>(
            precision,
            inputShape,
            constValues);
    } else {
        input1 = std::make_shared<ngraph::opset1::Parameter>(
            precision1,
            broadcast ? ngraph::Shape({ inputShape[0], inputShape[1], 1, 1 }) : ngraph::Shape(inputShape));
    }
    input1->set_friendly_name(constInput == 0 ? "input2" : "input1");

    const auto dequantizationOp1 = is_type<ngraph::opset1::Constant>(input1) ? input1 : makeDequantization(input1, dequantization1);

    std::shared_ptr<ngraph::Node> input2;
    if (constInputIndex == 1) {
        input2 = std::make_shared<ngraph::opset1::Constant>(
            precision,
            inputShape,
            constValues);
    } else {
        input2 = std::make_shared<ngraph::opset1::Parameter>(
            precision2, ngraph::Shape(inputShape));
    }
    input2->set_friendly_name(constInput == 0 ? "input1" : "input2");

    auto parent = input2;
    if (!additionalLayer.empty()) {
        if (additionalLayer == "convolution") {
            parent = std::make_shared<ngraph::op::TypeRelaxed<ngraph::opset1::Convolution>>(
                std::vector<element::Type>{ element::f32, element::f32 },
                std::vector<element::Type>{ element::f32 },
                ngraph::op::TemporaryReplaceOutputType(parent, element::f32).get(),
                ngraph::op::TemporaryReplaceOutputType(
                    std::make_shared<ngraph::opset1::Constant>(element::i8, Shape{ 1, 4, 1, 1 }, std::vector<float>{0.8f, 0.8f, 0.8f, 0.8f}),
                    element::f32).get(),
                ngraph::Strides{ 1, 1 },
                ngraph::CoordinateDiff{ 0, 0 },
                ngraph::CoordinateDiff{ 0, 0 },
                ngraph::Strides{ 1, 1 });
            parent->set_friendly_name("AdditionalLayer/Convolution");
        } else if (additionalLayer == "group_convolution") {
            parent = std::make_shared< ngraph::op::TypeRelaxed<ngraph::opset1::GroupConvolution>>(
                std::vector<element::Type>{ element::f32, element::f32 },
                std::vector<element::Type>{ element::f32 },
                ngraph::op::TemporaryReplaceOutputType(parent, element::f32).get(),
                ngraph::op::TemporaryReplaceOutputType(
                    std::make_shared<ngraph::opset1::Constant>(element::i8, Shape{ 4, 1, 1, 1, 1 }, std::vector<float>{0.8f, 0.8f, 0.8f, 0.8f}),
                    element::f32).get(),
                ngraph::Strides{ 1, 1 },
                ngraph::CoordinateDiff{ 0, 0 },
                ngraph::CoordinateDiff{ 0, 0 },
                ngraph::Strides{ 1, 1 });
            parent->set_friendly_name("AdditionalLayer/GroupConvolution");
        }

        parent = std::make_shared<ngraph::opset1::Add>(
            parent,
            std::make_shared<ngraph::opset1::Constant>(element::f32, Shape{ 1, 1, 1, 1 }, std::vector<float>{1.f}));
        parent->set_friendly_name("AdditionalLayer/Add");

        parent = ngraph::builder::subgraph::makeFakeQuantizeTypeRelaxed(
            parent,
            ngraph::element::f32,
            { 256, Shape{}, { 0 }, { 255 }, { 0 }, { 255 }, element::u8 });
        parent->set_friendly_name("AdditionalLayer/FakeQuantize");
    }
    const auto dequantizationOp2 = is_type<ngraph::opset1::Constant>(parent) ? parent : makeDequantization(parent, dequantization2);

    const std::shared_ptr<Node> add = operationType == "Add" ?
        std::dynamic_pointer_cast<Node>(std::make_shared<ngraph::op::TypeRelaxed<ngraph::opset1::Add>>(
            std::vector<element::Type>{ element::f32, element::f32 },
            std::vector<element::Type>{},
            ngraph::op::TemporaryReplaceOutputType(dequantizationOp1, element::f32).get(),
            ngraph::op::TemporaryReplaceOutputType(dequantizationOp2, element::f32).get())) :
        std::make_shared<ngraph::op::TypeRelaxed<DequantizationSubtract>>(
            std::vector<element::Type>{ element::f32, element::f32 },
            std::vector<element::Type>{},
            ngraph::op::TemporaryReplaceOutputType(dequantizationOp1, element::f32).get(),
            ngraph::op::TemporaryReplaceOutputType(dequantizationOp2, element::f32).get());

    NetworkHelper::setOutDataPrecisionForTypeRelaxed(add, precision);
    add->set_friendly_name("output_original");
    auto& rtInfo = add->get_rt_info();
    rtInfo["Variant::std::string"] = std::make_shared<VariantWrapper<std::string>>("add");

    const auto dequantizationOpAfter = makeDequantization(add, dequantizationAfter);
    dequantizationOpAfter->get_input_node_shared_ptr(1)->set_friendly_name(
        input1->get_friendly_name() + "/DequantizationMultiply/Constant");
    dequantizationOpAfter->set_friendly_name("output");

    const auto result = std::make_shared<ngraph::opset1::Result>(dequantizationOpAfter);
    result->set_friendly_name("result");
    ngraph::ResultVector results{ result };

    ngraph::ParameterVector parameters;
    if (constInputIndex == -1) {
        parameters = { as_type_ptr<ngraph::opset1::Parameter>(input1), as_type_ptr<ngraph::opset1::Parameter>(input2) };
    } else if (constInputIndex == 0) {
        parameters = { as_type_ptr<ngraph::opset1::Parameter>(input2) };
    } else if (constInputIndex == 1) {
        parameters = { as_type_ptr<ngraph::opset1::Parameter>(input1) };
    } else {
        THROW_IE_EXCEPTION << "Unexpected constant input index";
    }
    return std::make_shared<ngraph::Function>(results, parameters, "AddTransformation");
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
