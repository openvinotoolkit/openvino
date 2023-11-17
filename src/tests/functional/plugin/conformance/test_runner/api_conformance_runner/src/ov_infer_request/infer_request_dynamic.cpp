// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "behavior/ov_infer_request/infer_request_dynamic.hpp"
#include "ov_api_conformance_helpers.hpp"


namespace {
using namespace ov::test::behavior;
using namespace ov::test::conformance;

std::shared_ptr<ngraph::Function> ovGetFunction1() {
    const std::vector<size_t> inputShape = {1, 4, 20, 20};
    const ngraph::element::Type_t ngPrc = ngraph::element::Type_t::f32;

    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(inputShape))};
    params.front()->set_friendly_name("Param_1");
    params.front()->get_output_tensor(0).set_names({"input_tensor"});

    auto in2add = ngraph::builder::makeConstant(ngPrc, {1, 4, 1, 1}, std::vector<float>{}, true);
    auto add = ngraph::builder::makeEltwise(params[0], in2add, ngraph::helpers::EltwiseTypes::ADD);
    auto relu1 = std::make_shared<ngraph::opset1::Relu>(add->output(0));
    relu1->get_output_tensor(0).set_names({"relu1"});
    auto relu2 = std::make_shared<ngraph::opset1::Relu>(add->output(0));
    relu2->get_output_tensor(0).set_names({"relu2"});

    ngraph::NodeVector results{relu1, relu2};
    return std::make_shared<ngraph::Function>(results, params, "AddTwoOutputEdges");
}

std::shared_ptr<ngraph::Function> ovGetFunction2() {
    const std::vector<size_t> inputShape = {1, 4, 20, 20};
    const ngraph::element::Type_t ngPrc = ngraph::element::Type_t::f32;

    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(inputShape))};
    params.front()->set_friendly_name("Param_1");
    params.front()->get_output_tensor(0).set_names({"input_tensor"});
    auto splitAxisOp = std::make_shared<ov::op::v0::Constant>(ov::element::Type_t::i64, ov::Shape{}, std::vector<int64_t>{1});
    auto split = std::make_shared<ov::op::v1::Split>(params[0], splitAxisOp, 2);

    auto in2add = ngraph::builder::makeConstant(ngPrc, {1, 2, 1, 1}, std::vector<float>{}, true);
    auto add = ngraph::builder::makeEltwise(split->output(0), in2add, ngraph::helpers::EltwiseTypes::ADD);
    auto relu1 = std::make_shared<ngraph::opset1::Relu>(add);

    auto in2mult = ngraph::builder::makeConstant(ngPrc, {1, 2, 1, 1}, std::vector<float>{}, true);
    auto mult = ngraph::builder::makeEltwise(split->output(1), in2mult, ngraph::helpers::EltwiseTypes::MULTIPLY);
    auto relu2 = std::make_shared<ngraph::opset1::Relu>(mult);

    auto concat = std::make_shared<ngraph::opset1::Concat>(ngraph::OutputVector{relu1->output(0), relu2->output(0)}, 3);
    concat->get_output_tensor(0).set_names({"concat"});

    return std::make_shared<ngraph::Function>(concat, params, "SplitAddConcat");
}

INSTANTIATE_TEST_SUITE_P(ov_infer_request_1, OVInferRequestDynamicTests,
                        ::testing::Combine(
                                ::testing::Values(ovGetFunction1()),
                                ::testing::Values(std::vector<std::pair<std::vector<size_t>, std::vector<size_t>>>{
                                    {{1, 4, 20, 20}, {1, 4, 20, 20}},
                                    {{2, 4, 20, 20}, {2, 4, 20, 20}}}),
                                ::testing::Values(targetDevice),
                                ::testing::Values(pluginConfig)),
                        OVInferRequestDynamicTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(ov_infer_request_2, OVInferRequestDynamicTests,
                        ::testing::Combine(
                                ::testing::Values(ovGetFunction2()),
                                ::testing::Values(std::vector<std::pair<std::vector<size_t>, std::vector<size_t>>>{
                                    {{1, 4, 20, 20}, {1, 2, 20, 40}},
                                    {{2, 4, 20, 20}, {2, 2, 20, 40}}}),
                                ::testing::Values(targetDevice),
                                ::testing::Values(pluginConfig)),
                        OVInferRequestDynamicTests::getTestCaseName);
}  // namespace
