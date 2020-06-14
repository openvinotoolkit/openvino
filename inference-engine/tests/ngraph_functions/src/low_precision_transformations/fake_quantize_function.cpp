// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_functions/low_precision_transformations/fake_quantize_function.hpp"

#include <ngraph/opsets/opset1.hpp>
#include "ngraph_ops/multiply_add.hpp"
// #include "ngraph_ops/type_relaxed.hpp"
#include "ngraph_functions/subgraph_builders.hpp"

namespace ngraph {
namespace builder {
namespace subgraph {

std::shared_ptr<ngraph::Function> FakeQuantizeFunction::getOriginal(
    const ngraph::element::Type precision,
    const ngraph::Shape& inputShape) {
    const float k = 50.f;

    const auto input = std::make_shared<ngraph::opset1::Parameter>(precision, ngraph::Shape(inputShape));
    input->set_friendly_name("input");

    const auto fakeQuantize = ngraph::builder::makeFakeQuantize(
        input, precision, 256ul, { },
        { 0.f }, { 255.f / k }, { 0.f }, { 255.f / k });
    fakeQuantize->set_friendly_name("fakeQuantize");

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(fakeQuantize) };
    return std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input }, "FakeQuantizeFunction");
}

//template <typename T, typename... Args>
//std::shared_ptr<Node> fold(Args&&... args) {
//    auto node = std::make_shared<T>(std::forward<Args>(args)...);
//    if (node->get_output_size() == 1) {
//        OutputVector folded;
//        if (node->constant_fold(folded)) {
//            return folded[0].get_node_shared_ptr();
//        }
//    }
//    return node;
//}

//template <typename OperationType>
//void setOutDataPrecision(std::shared_ptr<OperationType> layer, const element::Type& precision) {
//    // check if it already exteded operation node
//    if (auto relaxed_layer = std::dynamic_pointer_cast<ngraph::op::TypeRelaxedBase>(layer)) {
//        relaxed_layer->set_overriden_output_type(precision);
//        std::dynamic_pointer_cast<ngraph::Node>(layer)->validate_and_infer_types();
//    } else {
//        // TODO: Make such replacements in advance for all supported polymorphic layer types
//        // extend a node with new semantics: overriden output data_type
//        // FIXME: OperationType should be a real type of an object, otherwise it will lead to undefined behavior
//        auto replacement = std::make_shared<ngraph::op::TypeRelaxed<OperationType>>(*layer, precision);
//        ngraph::copy_runtime_info(layer, replacement);
//        ngraph::replace_node(layer, replacement);
//    }
//}

std::shared_ptr<ngraph::Function> FakeQuantizeFunction::getReference(
    const ngraph::element::Type precision,
    const ngraph::Shape& inputShape) {
    // TODO: precision is hardcoded
    const auto expectedPrecision = ngraph::element::i8;

    const float k = 50.f;

    const auto input = std::make_shared<ngraph::opset1::Parameter>(precision, ngraph::Shape(inputShape));
    input->set_friendly_name("input");

    const auto fakeQuantize = ngraph::builder::makeFakeQuantize(
        input, precision, 256ul, { },
        { 0.f }, { 255.f }, { 0.f }, { 255.f });
    fakeQuantize->set_friendly_name("fakeQuantize");
    fakeQuantize->set_output_type(0, expectedPrecision, inputShape);
    //setOutDataPrecision(fakeQuantize, expectedPrecision);

    const auto convert = std::make_shared<ngraph::opset1::Convert>(fakeQuantize, precision);

    auto dequantize = std::make_shared<ngraph::op::MultiplyAdd>(
        convert,
        ngraph::opset1::Constant::create(precision, ngraph::Shape{ }, { k }),
        ngraph::opset1::Constant::create(precision, ngraph::Shape{ }, { 0.f }));

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(dequantize) };
    return std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input }, "FakeQuantizeFunction");
}

}
}
}
