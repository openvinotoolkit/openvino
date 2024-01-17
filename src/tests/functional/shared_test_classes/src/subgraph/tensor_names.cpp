// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/subgraph/tensor_names.hpp"

namespace SubgraphTestsDefinitions {

std::string TensorNamesTest::getTestCaseName(const testing::TestParamInfo<constResultParams>& obj) {
    std::string targetDevice;
    std::tie(targetDevice) = obj.param;
    std::ostringstream result;
    result << "TargetDevice=" << targetDevice;
    return result.str();
}

void TensorNamesTest::SetUp() {
    std::tie(targetDevice) = this->GetParam();

    auto parameter = std::make_shared<ov::op::v0::Parameter>(ngraph::element::Type_t::f32, ngraph::Shape{1, 3, 10, 10});
    parameter->set_friendly_name("parameter");
    parameter->get_output_tensor(0).set_names({"input"});
    auto relu_prev = std::make_shared<ov::op::v0::Relu>(parameter);
    relu_prev->set_friendly_name("relu_prev");
    relu_prev->get_output_tensor(0).set_names({"relu,prev_t", "identity_prev_t"});
    auto relu = std::make_shared<ov::op::v0::Relu>(relu_prev);
    relu->set_friendly_name("relu");
    relu->get_output_tensor(0).set_names({"relu,t", "identity"});
    const ngraph::ResultVector results{std::make_shared<ov::op::v0::Result>(relu)};
    results[0]->set_friendly_name("out");
    ngraph::ParameterVector params{parameter};
    function = std::make_shared<ngraph::Function>(results, params, "TensorNames");
}

}  // namespace SubgraphTestsDefinitions
