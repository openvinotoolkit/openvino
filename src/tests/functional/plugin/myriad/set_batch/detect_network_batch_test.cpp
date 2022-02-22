// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "detect_network_batch_test.hpp"
#include "ngraph_functions/subgraph_builders.hpp"
#include "ngraph_functions/builders.hpp"

namespace LayerTestsDefinitions {
std::shared_ptr<ngraph::Function> makeNNWithMultipleInputsDiffDims(ngraph::Shape inputShape = {1, 3, 240, 240},
                                                            ngraph::element::Type_t type = ngraph::element::Type_t::i32) {
    const auto elems = ngraph::shape_size(inputShape);
    auto shape2 = ngraph::Shape({inputShape[0], elems / inputShape[0]});
    auto param0 = std::make_shared<ngraph::opset1::Parameter>(type, inputShape);
    auto param1 = std::make_shared<ngraph::opset1::Parameter>(type, inputShape);
    auto param2 = std::make_shared<ngraph::opset1::Parameter>(type, shape2);
    auto add = std::make_shared<ngraph::opset1::Add>(param0, param1);
    auto reshape = std::make_shared<ngraph::opset6::Reshape>(add, ngraph::op::Constant::create(ngraph::element::i32, {2}, shape2), false);
    auto add2 = std::make_shared<ngraph::opset1::Add>(reshape, param2);
    auto result = std::make_shared<ngraph::opset1::Result>(add2);
    auto fn_ptr = std::make_shared<ngraph::Function>(ngraph::ResultVector{result}, ngraph::ParameterVector{param0, param1, param2});
    fn_ptr->set_friendly_name("MultipleInputsDiffDims");
    return fn_ptr;
}

std::shared_ptr<ngraph::Function> makeNNWithMultipleInputsSameDims(ngraph::Shape inputShape = {1, 3, 240, 240},
                                                            ngraph::element::Type_t type = ngraph::element::Type_t::f32) {
    auto param0 = std::make_shared<ngraph::opset1::Parameter>(type, inputShape);
    auto param1 = std::make_shared<ngraph::opset1::Parameter>(type, inputShape);
    auto param2 = std::make_shared<ngraph::opset1::Parameter>(type, inputShape);
    auto add = std::make_shared<ngraph::opset1::Add>(param0, param1);
    auto add2 = std::make_shared<ngraph::opset1::Add>(add, param2);
    auto result = std::make_shared<ngraph::opset1::Result>(add2);
    auto fn_ptr = std::make_shared<ngraph::Function>(ngraph::ResultVector{result}, ngraph::ParameterVector{param0, param1, param2});
    fn_ptr->set_friendly_name("MultipleInputsSameDims");
    return fn_ptr;
}

std::string DetectNetworkBatch::getTestCaseName(const testing::TestParamInfo<DetectNetworkBatchParams>& obj) {
    LayerTestsUtils::TargetDevice targetDevice;
    unsigned int batchSize;

    std::tie(targetDevice, batchSize) = obj.param;
    std::ostringstream result;
    result << "targetDevice=" << targetDevice << "_";
    result << "batchSize=" << batchSize;
    return result.str();
}

void DetectNetworkBatch::SetUp() {
    std::tie(targetDevice, m_batchSize) = this->GetParam();
}

void DetectNetworkBatch::LoadNetwork() {
    cnnNetwork = InferenceEngine::CNNNetwork{function};
    cnnNetwork.setBatchSize(m_batchSize);
    functionRefs = ngraph::clone_function(*cnnNetwork.getFunction());
    ConfigureNetwork();
    executableNetwork = core->LoadNetwork(cnnNetwork, targetDevice, configuration);
}

TEST_P(DetectNetworkBatch, InferWithOneInput) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    function = ngraph::builder::subgraph::makeSplitConvConcat();
    Run();
};

TEST_P(DetectNetworkBatch, InferWithMultipleInputs_DiffDims) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    function = makeNNWithMultipleInputsDiffDims();
    Run();
};

TEST_P(DetectNetworkBatch, InferWithMultipleInputs_SameDims) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    function = makeNNWithMultipleInputsSameDims();
    Run();
};

} // namespace LayerTestsDefinitions

using namespace LayerTestsDefinitions;

const std::vector<unsigned int> batchSizes = {
        2,
        4,
        8,
};

namespace {
INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, DetectNetworkBatch,
                         ::testing::Combine(
                                 ::testing::Values(CommonTestUtils::DEVICE_MYRIAD),
                                 ::testing::ValuesIn(batchSizes)),
                         DetectNetworkBatch::getTestCaseName);
}  // namespace
