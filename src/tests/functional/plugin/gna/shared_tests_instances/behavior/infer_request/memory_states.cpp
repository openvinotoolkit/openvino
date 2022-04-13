// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <common_test_utils/test_constants.hpp>
#include "behavior/infer_request/memory_states.hpp"
#include "functional_test_utils/plugin_cache.hpp"
#include "ngraph_functions/builders.hpp"

using namespace BehaviorTestsDefinitions;

namespace {
InferenceEngine::CNNNetwork getNetwork() {
    ngraph::Shape shape = {1, 200};
    ngraph::element::Type type = ngraph::element::f32;

    auto input = std::make_shared<ngraph::op::v0::Parameter>(type, shape);
    auto mem_i1 = std::make_shared<ngraph::op::v0::Constant>(type, shape, 0);
    auto mem_r1 = std::make_shared<ngraph::op::v3::ReadValue>(mem_i1, "r_1-3");
    auto mul1 = std::make_shared<ngraph::op::v1::Multiply>(mem_r1, input);

    auto mem_i2 = std::make_shared<ngraph::op::v0::Constant>(type, shape, 0);
    auto mem_r2 = std::make_shared<ngraph::op::v3::ReadValue>(mem_i2, "c_1-3");
    auto mul2 = std::make_shared<ngraph::op::v1::Multiply>(mem_r2, mul1);
    auto mem_w2 = std::make_shared<ngraph::op::v3::Assign>(mul2, "c_1-3");

    auto mem_w1 = std::make_shared<ngraph::op::v3::Assign>(mul2, "r_1-3");
    auto sigm = std::make_shared<ngraph::op::Sigmoid>(mul2);

    mem_r1->set_friendly_name("Memory_1");
    mem_w1->add_control_dependency(mem_r1);
    sigm->add_control_dependency(mem_w1);

    mem_r2->set_friendly_name("Memory_2");
    mem_w2->add_control_dependency(mem_r2);
    sigm->add_control_dependency(mem_w2);

    auto function = std::make_shared<ngraph::Function>(ngraph::NodeVector{sigm}, ngraph::ParameterVector{input}, "addOutput");
    return InferenceEngine::CNNNetwork{function};
}

std::vector<memoryStateParams> memoryStateTestCases = {
        memoryStateParams(getNetwork(), {"c_1-3", "r_1-3"}, CommonTestUtils::DEVICE_GNA)
};

INSTANTIATE_TEST_SUITE_P(smoke_VariableStateBasic, InferRequestVariableStateTest,
                         ::testing::ValuesIn(memoryStateTestCases),
                         InferRequestVariableStateTest::getTestCaseName);
} // namespace
