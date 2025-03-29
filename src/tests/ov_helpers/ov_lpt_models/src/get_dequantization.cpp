// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_lpt_models/get_dequantization.hpp"

#include <vector>
#include <memory>
#include <vector>

#include <low_precision/common/fake_quantize_dequantization.hpp>
#include <low_precision/network_helper.hpp>

#include "ov_lpt_models/common/builders.hpp"

namespace ov {
namespace builder {
namespace subgraph {

std::shared_ptr<ov::Model> GetDequantizationFunction::get(
    const ov::element::Type& precision,
    const Shape& shape,
    const FakeQuantizeOnData& fakeQuantize,
    const ov::builder::subgraph::DequantizationOperations& dequantization) {
    const std::shared_ptr<ov::Node> input = std::make_shared<ov::opset1::Parameter>(
        ov::element::f32,
        shape);

    std::shared_ptr<ov::Node> parent = input;
    if (!fakeQuantize.empty()) {
        parent = ov::builder::subgraph::makeFakeQuantizeTypeRelaxed(parent, precision, fakeQuantize);
    }

    if (!dequantization.empty()) {
        parent = makeDequantization(parent, dequantization);
        parent->set_friendly_name("output");
    }

    return std::make_shared<ov::Model>(
        ov::ResultVector{ std::make_shared<ov::opset1::Result>(parent) },
        ov::ParameterVector{ ov::as_type_ptr<op::v0::Parameter>(input) },
        "DequantizationFunction");
}

std::shared_ptr<ov::Model> GetDequantizationFunction::get(
    const ov::element::Type& precision,
    const Shape& shape,
    const FakeQuantizeOnData& fakeQuantize,
    const ov::pass::low_precision::FakeQuantizeDequantization& dequantization) {
    const std::shared_ptr<ov::Node> input = std::make_shared<ov::opset1::Parameter>(
        ov::element::f32,
        shape);

    std::shared_ptr<ov::Node> parent = input;
    if (!fakeQuantize.empty()) {
        parent = ov::builder::subgraph::makeFakeQuantizeTypeRelaxed(parent, precision, fakeQuantize);
    }

    if (dequantization.convert != nullptr) {
        parent = dequantization.convert->clone_with_new_inputs({ parent });
        parent->set_friendly_name(dequantization.convert->get_friendly_name());
    }

    if (dequantization.subtract != nullptr) {
        const auto parent2 = dequantization.subtractConvert == nullptr ?
            std::dynamic_pointer_cast<ov::Node>(dequantization.subtractConstant) :
            dequantization.subtractConvert;
        const auto index = ov::pass::low_precision::NetworkHelper::getChildInputIndex(parent2, dequantization.subtract);
        parent = dequantization.subtract->clone_with_new_inputs(index == 1ul ?
            OutputVector{ parent, parent2 } :
            OutputVector{ parent2, parent });
        parent->set_friendly_name(dequantization.subtract->get_friendly_name());
    }

    if (dequantization.multiply != nullptr) {
        const auto index = ov::pass::low_precision::NetworkHelper::getChildInputIndex(dequantization.multiplyConstant, dequantization.multiply);
        parent = dequantization.multiply->clone_with_new_inputs(index == 1ul ?
            OutputVector{ parent, dequantization.multiplyConstant } :
            OutputVector{ dequantization.multiplyConstant, parent });
        parent->set_friendly_name(dequantization.multiply->get_friendly_name());
    }

    return std::make_shared<ov::Model>(
        ov::ResultVector{ std::make_shared<ov::opset1::Result>(parent) },
        ov::ParameterVector{ ov::as_type_ptr<op::v0::Parameter>(input) },
        "DequantizationFunction");
}

std::shared_ptr<ov::Model> GetDequantizationFunction::getOriginal(
    bool isConvert, bool isSubtract, size_t subDataInput, size_t mulDataInput) {
    const std::shared_ptr<ov::Node> input = std::make_shared<ov::opset1::Parameter>(
        ov::element::f32,
        ov::Shape{ 1, 3, 10, 10 });

    const auto convert = isConvert ? std::make_shared<ov::opset1::Convert>(input, ov::element::f32) : nullptr;
    std::shared_ptr<ov::Node> parent = isConvert ? convert : input;

    auto subConst = std::make_shared<ov::opset1::Constant>(ov::element::f32, Shape{}, 1.f);
    const auto& subArg0 = subDataInput == 0 ? parent : subConst;
    const auto& subArg1 = subDataInput == 0 ? subConst : parent;
    const auto subtract = isSubtract ? std::make_shared<ov::opset1::Subtract>(subArg0, subArg1) : nullptr;

    if (subtract != nullptr) {
        parent = subtract;
    }

    auto mulConst = std::make_shared<ov::opset1::Constant>(ov::element::f32, Shape{}, 1.f);
    const auto& mulArg0 = mulDataInput == 0 ? parent : mulConst;
    const auto& mulArg1 = mulDataInput == 0 ? mulConst : parent;
    const auto multiply = std::make_shared<ov::opset1::Multiply>(mulArg0, mulArg1);

    return std::make_shared<ov::Model>(
        ov::ResultVector{ std::make_shared<ov::opset1::Result>(multiply) },
        ov::ParameterVector{ ov::as_type_ptr<op::v0::Parameter>(input) },
        "Dequantization");
}

std::shared_ptr<ov::Model> GetDequantizationFunction::getReference(
    ov::pass::low_precision::FakeQuantizeDequantization dequantization) {
    return std::make_shared<ov::Model>(
        ov::ResultVector{ std::make_shared<ov::opset1::Result>(dequantization.multiply) },
        ov::ParameterVector{ ov::as_type_ptr<op::v0::Parameter>(dequantization.data.get_node_shared_ptr()) },
        "Dequantization");
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ov
