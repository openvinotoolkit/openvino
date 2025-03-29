// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "behavior/ov_infer_request/infer_request_dynamic.hpp"
#include "behavior/ov_infer_request/inference_chaining.hpp"

using namespace ov::test::behavior;

namespace {

auto configs = []() {
    return std::vector<ov::AnyMap>{{}};
};

std::shared_ptr<ov::Model> getFunction1() {
    const std::vector<size_t> inputShape = {1, 4, 20, 20};
    const ov::element::Type_t ngPrc = ov::element::Type_t::f32;

    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(inputShape))};
    params.front()->set_friendly_name("Param_1");
    params.front()->get_output_tensor(0).set_names({"input_tensor"});

    auto relu = std::make_shared<ov::op::v0::Relu>(params[0]);
    relu->get_output_tensor(0).set_names({"relu"});

    return std::make_shared<ov::Model>(relu, params, "SimpleActivation");
}

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests_1, OVInferRequestDynamicTests,
                        ::testing::Combine(
                                ::testing::Values(getFunction1()),
                                ::testing::Values(std::vector<std::pair<std::vector<size_t>, std::vector<size_t>>>{
                                    {{1, 4, 20, 20}, {1, 4, 20, 20}},
                                    {{2, 4, 20, 20}, {2, 4, 20, 20}}}),
                                ::testing::Values(ov::test::utils::DEVICE_GPU),
                                ::testing::ValuesIn(configs())),
                        OVInferRequestDynamicTests::getTestCaseName);
}  // namespace
