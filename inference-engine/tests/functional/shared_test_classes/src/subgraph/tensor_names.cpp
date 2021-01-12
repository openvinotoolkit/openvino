// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/subgraph/tensor_names.hpp"

namespace SubgraphTestsDefinitions {

std::string TensorNamesTest::getTestCaseName(testing::TestParamInfo<constResultParams> obj) {
    std::string targetDevice;
    std::tie(targetDevice) = obj.param;
    std::ostringstream result;
    result << "TargetDevice=" << targetDevice;
    return result.str();
}

void TensorNamesTest::SetUp() {
    InferenceEngine::SizeVector inputShapes;
    InferenceEngine::Precision inputPrecision;
    std::tie(targetDevice) = this->GetParam();
    std::vector<float> data(300);
    for (size_t i = 0; i < 300; i++)
        data[i] = i;

    auto parameter = std::make_shared<ngraph::opset5::Parameter>(ngraph::element::Type_t::f32, ngraph::Shape{1, 3, 10, 10});
    parameter->set_friendly_name("parameter");
    parameter->output(0).set_names({"input"});
    auto relu = std::make_shared<ngraph::opset5::Relu>(parameter);
    relu->set_friendly_name("relu");
    relu->output(0).set_names({"relu_t", "identity"});
    const ngraph::ResultVector results{std::make_shared<ngraph::opset3::Result>(relu)};
    results[0]->set_friendly_name("out");
    results[0]->get_output_tensor(0).set_name("out_t");
    ngraph::ParameterVector params{parameter};
    function = std::make_shared<ngraph::Function>(results, params, "TensorNames");
}

}  // namespace SubgraphTestsDefinitions


