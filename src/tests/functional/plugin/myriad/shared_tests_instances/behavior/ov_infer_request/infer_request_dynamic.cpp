// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "behavior/ov_infer_request/infer_request_dynamic.hpp"
#include <vpu/private_plugin_config.hpp>
#include <vpu/configuration/options/dump_internal_graph_file_name.hpp>
#include <vpu/configuration/options/dump_all_passes_directory.hpp>
#include <vpu/configuration/options/dump_all_passes.hpp>

using namespace ov::test::behavior;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP16
};

const std::vector<ov::AnyMap> configs = {
    {}
};

const std::vector<ov::AnyMap> HeteroConfigs = {
    {ov::device::priorities(CommonTestUtils::DEVICE_MYRIAD, CommonTestUtils::DEVICE_CPU)}
};

std::shared_ptr<ngraph::Function> getFunction1() {
    const std::vector<size_t> inputShape = {1, 4, 20, 20};
    const ngraph::element::Type_t ngPrc = ngraph::element::Type_t::f32;

    auto params = ngraph::builder::makeParams(ngPrc, {inputShape});
    params.front()->set_friendly_name("Param_1");
    params.front()->get_output_tensor(0).set_names({"input_tensor"});

    auto in2add = ngraph::builder::makeConstant(ngPrc, {1, 4, 1, 1}, std::vector<float>{}, true);
    auto add = ngraph::builder::makeEltwise(params[0], in2add, ngraph::helpers::EltwiseTypes::ADD);
    auto relu1 = std::make_shared<ngraph::opset1::Relu>(add->output(0));
    relu1->get_output_tensor(0).set_names({"relu1"});

    ngraph::NodeVector results{relu1};
    return std::make_shared<ngraph::Function>(results, params, "AddOneOutputEdge");
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

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests_1, OVInferRequestDynamicTests,
                        ::testing::Combine(
                                ::testing::Values(getFunction1()),
                                ::testing::Values(std::vector<std::pair<std::vector<size_t>, std::vector<size_t>>>{
                                    {{1, 4, 20, 20}, {1, 4, 20, 20}},
                                    {{2, 4, 20, 20}, {2, 4, 20, 20}}}),
                                ::testing::Values(CommonTestUtils::DEVICE_MYRIAD),
                                ::testing::ValuesIn(configs)),
                        OVInferRequestDynamicTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Hetero_BehaviorTests, OVInferRequestDynamicTests,
                        ::testing::Combine(
                                ::testing::Values(getFunction2()),
                                ::testing::Values(std::vector<std::pair<std::vector<size_t>, std::vector<size_t>>>{
                                    {{1, 4, 20, 20}, {1, 2, 20, 40}},
                                    {{2, 4, 20, 20}, {2, 2, 20, 40}}}),
                                ::testing::Values(CommonTestUtils::DEVICE_HETERO),
                                ::testing::ValuesIn(HeteroConfigs)),
                        OVInferRequestDynamicTests::getTestCaseName);
}  // namespace
