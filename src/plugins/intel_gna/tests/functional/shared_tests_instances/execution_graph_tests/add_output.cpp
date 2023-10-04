// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "execution_graph_tests/add_output.hpp"

#include <common_test_utils/test_constants.hpp>

#include "functional_test_utils/plugin_cache.hpp"
#include "ov_models/builders.hpp"

InferenceEngine::CNNNetwork getTargetNetwork() {
    ngraph::Shape shape = {1, 200};
    ngraph::element::Type type = ngraph::element::f32;

    auto input = std::make_shared<ngraph::op::v0::Parameter>(type, shape);
    auto mem_i = std::make_shared<ngraph::op::v0::Constant>(type, shape, 0);
    auto mem_r = std::make_shared<ngraph::op::v3::ReadValue>(mem_i, "r_1-3");
    auto mul = std::make_shared<ngraph::op::v1::Multiply>(mem_r, input);
    auto mem_w = std::make_shared<ngraph::op::v3::Assign>(mul, "r_1-3");
    auto sigm = std::make_shared<ngraph::op::Sigmoid>(mul);
    mem_r->set_friendly_name("Memory_1");
    mem_w->add_control_dependency(mem_r);
    sigm->add_control_dependency(mem_w);

    auto function =
        std::make_shared<ngraph::Function>(ngraph::NodeVector{sigm}, ngraph::ParameterVector{input}, "addOutput");
    return InferenceEngine::CNNNetwork{function};
}

std::vector<addOutputsParams> testCases = {
    addOutputsParams(getTargetNetwork(), {"Memory_1"}, ov::test::utils::DEVICE_GNA)};

INSTANTIATE_TEST_SUITE_P(smoke_AddOutputBasic,
                         AddOutputsTest,
                         ::testing::ValuesIn(testCases),
                         AddOutputsTest::getTestCaseName);
