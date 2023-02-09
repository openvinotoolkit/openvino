// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "behavior/ov_infer_request/infer_request_dynamic.hpp"
#include "behavior/ov_infer_request/inference_chaining.hpp"
#include "openvino/op/util/multiclass_nms_base.hpp"

using namespace ov::test::behavior;

namespace {

auto configs = []() {
    return std::vector<ov::AnyMap>{{}};
};

auto AutoConfigs = []() {
    return std::vector<ov::AnyMap>{
                                #ifdef ENABLE_INTEL_CPU
                                   {ov::device::priorities(CommonTestUtils::DEVICE_GPU, CommonTestUtils::DEVICE_CPU)},
                                   {ov::device::priorities(CommonTestUtils::DEVICE_CPU, CommonTestUtils::DEVICE_GPU)},
                                #endif
                                   {}};
};

auto MultiConfigs = []() {
    return std::vector<ov::AnyMap>{{ov::device::priorities(CommonTestUtils::DEVICE_GPU)},
                                #ifdef ENABLE_INTEL_CPU
                                   {ov::device::priorities(CommonTestUtils::DEVICE_GPU, CommonTestUtils::DEVICE_CPU)},
                                   {ov::device::priorities(CommonTestUtils::DEVICE_CPU, CommonTestUtils::DEVICE_GPU)}
                                #endif
                                   };
};

auto AutoNotSupportConfigs = []() {
    return std::vector<ov::AnyMap>{};
};

std::shared_ptr<ngraph::Function> getFunction1() {
    const std::vector<size_t> inputShape = {1, 4, 20, 20};
    const ngraph::element::Type_t ngPrc = ngraph::element::Type_t::f32;

    auto params = ngraph::builder::makeParams(ngPrc, {inputShape});
    params.front()->set_friendly_name("Param_1");
    params.front()->get_output_tensor(0).set_names({"input_tensor"});

    auto relu = std::make_shared<ngraph::opset1::Relu>(params[0]);
    relu->get_output_tensor(0).set_names({"relu"});

    return std::make_shared<ngraph::Function>(relu, params, "SimpleActivation");
}

std::shared_ptr<ngraph::Function> getFunction2() {
    const std::vector<size_t> inputShape = {1, 4, 20, 20};
    const ngraph::element::Type_t ngPrc = ngraph::element::Type_t::f32;

    auto params = ngraph::builder::makeParams(ngPrc, {inputShape});
    params.front()->set_friendly_name("Param_1");
    params.front()->get_output_tensor(0).set_names({"input_tensor"});
    auto split = ngraph::builder::makeSplit(params[0], ngPrc, 2, 1);

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

std::shared_ptr<ngraph::Function> getFunction_DynamicOutput() {
    auto boxes = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 2, 4});
    boxes->set_friendly_name("param_1");
    boxes->get_output_tensor(0).set_names({"input_tensor_1"});
    auto scores = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 1, 2});
    scores->set_friendly_name("param_2");
    scores->get_output_tensor(0).set_names({"input_tensor_2"});
    auto max_output_boxes_per_class = ov::op::v0::Constant::create(ov::element::i64,  ov::Shape{}, {10});
    auto iou_threshold = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {0.75});
    auto score_threshold = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {0.7});
    auto nms = std::make_shared<ov::op::v9::NonMaxSuppression>(boxes, scores, max_output_boxes_per_class,
                                                                                  iou_threshold, score_threshold);
    auto res = std::make_shared<ov::op::v0::Result>(nms);
    res->get_output_tensor(0).set_names({"output_dynamic"});
    auto func = std::make_shared<ngraph::Function>(ov::NodeVector{nms}, ngraph::ParameterVector{boxes, scores});
    return func;
}

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests_1, OVInferRequestDynamicTests,
                        ::testing::Combine(
                                ::testing::Values(getFunction1()),
                                ::testing::Values(std::vector<std::pair<std::vector<size_t>, std::vector<size_t>>>{
                                    {{1, 4, 20, 20}, {1, 4, 20, 20}},
                                    {{2, 4, 20, 20}, {2, 4, 20, 20}}}),
                                ::testing::Values(CommonTestUtils::DEVICE_GPU),
                                ::testing::ValuesIn(configs())),
                        OVInferRequestDynamicTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests, OVInferRequestDynamicTests,
                        ::testing::Combine(
                                ::testing::Values(getFunction2()),
                                ::testing::Values(std::vector<std::pair<std::vector<size_t>, std::vector<size_t>>>{
                                    {{1, 4, 20, 20}, {1, 2, 20, 40}},
                                    {{2, 4, 20, 20}, {2, 2, 20, 40}}}),
                                ::testing::Values(CommonTestUtils::DEVICE_AUTO),
                                ::testing::ValuesIn(AutoConfigs())),
                        OVInferRequestDynamicTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests, OVInferenceChaining,
                        ::testing::Combine(
                                ::testing::Values(CommonTestUtils::DEVICE_AUTO),
                                ::testing::ValuesIn(AutoConfigs())),
                        OVInferenceChaining::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests, OVNotSupportRequestDynamicTests,
                        ::testing::Combine(
                                ::testing::Values(getFunction2()),
                                ::testing::Values(std::vector<std::pair<std::vector<size_t>, std::vector<size_t>>>{
                                    {{1, 4, 20, 20}, {1, 2, 20, 40}},
                                    {{2, 4, 20, 20}, {2, 2, 20, 40}}}),
                                ::testing::Values(CommonTestUtils::DEVICE_AUTO),
                                ::testing::ValuesIn(AutoNotSupportConfigs())),
                        OVInferRequestDynamicTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Multi_BehaviorTests, OVInferRequestDynamicTests,
                        ::testing::Combine(
                                ::testing::Values(getFunction2()),
                                ::testing::Values(std::vector<std::pair<std::vector<size_t>, std::vector<size_t>>>{
                                    {{1, 4, 20, 20}, {1, 2, 20, 40}},
                                    {{2, 4, 20, 20}, {2, 2, 20, 40}}}),
                                ::testing::Values(CommonTestUtils::DEVICE_MULTI),
                                ::testing::ValuesIn(MultiConfigs())),
                        OVInferRequestDynamicTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Multi_BehaviorTests, OVInferRequestDynamicOutputTests,
                        ::testing::Combine(
                                ::testing::Values(getFunction_DynamicOutput()),
                                ::testing::Values(std::vector<std::pair<std::vector<size_t>, std::vector<size_t>>>{
                                    {}}),
                                ::testing::Values(CommonTestUtils::DEVICE_MULTI),
                                ::testing::ValuesIn(MultiConfigs())),
                        OVInferRequestDynamicTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests, OVInferRequestDynamicOutputTests,
                        ::testing::Combine(
                                ::testing::Values(getFunction_DynamicOutput()),
                                ::testing::Values(std::vector<std::pair<std::vector<size_t>, std::vector<size_t>>>{
                                    {}}),
                                ::testing::Values(CommonTestUtils::DEVICE_AUTO),
                                ::testing::ValuesIn(AutoConfigs())),
                        OVInferRequestDynamicTests::getTestCaseName);
}  // namespace
