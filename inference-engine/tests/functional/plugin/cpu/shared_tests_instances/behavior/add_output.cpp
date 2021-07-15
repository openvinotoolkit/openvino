// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <common_test_utils/test_constants.hpp>
#include "behavior/add_output.hpp"
#include "functional_test_utils/plugin_cache.hpp"
#include "ngraph_functions/builders.hpp"

using namespace ngraph;

InferenceEngine::CNNNetwork getTargetNetwork() {
    ngraph::Shape shape = {1, 200};
    ngraph::element::Type type = ngraph::element::f32;

    auto input = std::make_shared<op::v0::Parameter>(type, shape);
    auto mem_i = std::make_shared<op::v0::Constant>(type, shape, 0);
    auto mem_r = std::make_shared<op::v3::ReadValue>(mem_i, "id");
    auto mul   = std::make_shared<ngraph::op::v1::Multiply>(mem_r, input);
    auto mem_w = std::make_shared<op::v3::Assign>(mul, "id");
    auto sigm = std::make_shared<ngraph::op::Sigmoid>(mul);
    mem_r->set_friendly_name("Memory");
    mem_w->add_control_dependency(mem_r);
    sigm->add_control_dependency(mem_w);

    auto function = std::make_shared<ngraph::Function>(ngraph::NodeVector{sigm}, ngraph::ParameterVector{input}, "addOutput");

    return InferenceEngine::CNNNetwork{function};
}

std::vector<addOutputsParams> testCases = {
        addOutputsParams(getTargetNetwork(), {"Memory"}, CommonTestUtils::DEVICE_CPU)
};

INSTANTIATE_TEST_SUITE_P(smoke_AddOutputBasic, AddOutputsTest,
        ::testing::ValuesIn(testCases),
        AddOutputsTest::getTestCaseName);
