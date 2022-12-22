//// Copyright (C) 2023 Intel Corporation
//// SPDX-License-Identifier: Apache-2.0
////
//
//#include "lpt_ngraph_functions/snippets_precision_propagation_function.hpp"
//
//#include "low_precision/network_helper.hpp"
//#include "low_precision/layer_transformation.hpp"
//
//#include "ngraph/opsets/opset1.hpp"
//
//#include "lpt_ngraph_functions/common/dequantization_operations.hpp"
//#include "ngraph_functions/subgraph_builders.hpp"
//
//using namespace ngraph::pass::low_precision;
//
//namespace ngraph {
//namespace builder {
//namespace subgraph {
//
//namespace {
//std::pair<std::shared_ptr<ngraph::opset1::Parameter>, std::shared_ptr<ov::Node>> make_branch(
//    const ngraph::element::Type precision,
//    const ngraph::PartialShape& inputShape,
//    const size_t index) {
//    const auto parameter = std::make_shared<ngraph::opset1::Parameter>(precision, inputShape);
//    parameter->set_friendly_name("parameter" + std::to_string(index));
//
//    std::shared_ptr<Node> max_pool = std::make_shared<ngraph::opset1::MaxPool>(
//        parameter,
//        Strides{ 1, 1 },
//        Shape{ 1, 1 },
//        Shape{ 0, 0 },
//        Shape{ 2, 2 },
//        op::RoundingType::FLOOR);
//    max_pool->set_friendly_name("maxPool" + std::to_string(index));
//
//    return { parameter, max_pool };
//}
//} // namespace
//
//std::shared_ptr<ngraph::Function> SnippetsPrecisionPropagationFunction::get(
//    const ngraph::element::Type precision1,
//    const ngraph::PartialShape& inputShape1,
//    const ngraph::element::Type precision2,
//    const ngraph::PartialShape& inputShape2) {
//    const auto branch1 = make_branch(precision1, inputShape1, 1);
//    const auto branch2 = make_branch(precision2, inputShape2, 2);
//
//    std::shared_ptr<Node> parent = std::make_shared<ngraph::opset1::Add>(branch1.second, branch2.second);
//
//    parent = std::make_shared<ngraph::opset1::Maximum>(
//        parent,
//        std::make_shared<ngraph::opset1::Constant>(precision1, Shape{}, std::vector<float>{0.f}));
//
//    parent = std::make_shared<ngraph::opset1::Minimum>(
//        parent,
//        std::make_shared<ngraph::opset1::Constant>(precision1, Shape{}, std::vector<float>{10.f}));
//
//    const ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(parent) };
//    const ngraph::ParameterVector parameters{ branch1.first, branch2.first };
//    return std::make_shared<ngraph::Function>(results, parameters, "SnippetsPrecisionPropagation");
//}
//
//}  // namespace subgraph
//}  // namespace builder
//}  // namespace ngraph
