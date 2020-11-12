// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <tuple>
#include <vector>
#include <string>
#include <memory>
#include <functional>
#include <functional_test_utils/skip_tests_config.hpp>

#include "ie_core.hpp"

#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/blob_utils.hpp"
#include "functional_test_utils/plugin_cache.hpp"
#include "functional_test_utils/layer_test_utils.hpp"

#include "subgraph_tests/two_fake_quantize_to_fullyconnected.hpp"


namespace LayerTestsDefinitions {

std::string FakeQuantizeSubgraphTest::getTestCaseName(testing::TestParamInfo<fqSubgraphTestParamsSet> obj) {
    fqSpecificParams fqParams;
    InferenceEngine::Precision netPrecision;
    InferenceEngine::Precision inPrc, outPrc;
    InferenceEngine::Layout inLayout, outLayout;
    InferenceEngine::SizeVector inputShapes;
    std::string targetDevice;
    std::pair<std::string, std::map<std::string, std::string>> config;
    bool biases = false;
    std::tie(fqParams, netPrecision, inPrc, outPrc, inLayout, outLayout, inputShapes, targetDevice, config, biases) = obj.param;
    std::vector<size_t> levels;
    std::vector<std::vector<size_t>> constShape;
    std::vector<float> fqDirectArgs;
    std::vector<float> inputArg;
    std::tie(levels, constShape, fqDirectArgs, inputArg) = fqParams;

    std::ostringstream result;
    result << "IS=" << CommonTestUtils::vec2str(inputShapes) << "_";
    result << "CS=" << CommonTestUtils::vec2str(constShape) << "_";
    result << "LEVELS=" << CommonTestUtils::vec2str(levels) << "_";
    result << "netPRC=" << netPrecision.name() << "_";
    result << "inPRC=" << inPrc.name() << "_";
    result << "outPRC=" << outPrc.name() << "_";
    result << "inL=" << inLayout << "_";
    result << "outL=" << outLayout << "_";
    result << "biases=" << biases << "_";
    result << "trgDev=" << targetDevice;
    if (!config.first.empty()) {
        result << "_targetConfig=" << config.first;
    }
    if (!fqDirectArgs.empty()) {
        result << "_fqArgs=" << fqDirectArgs[0] << "_" << fqDirectArgs[1] << "_" << fqDirectArgs[2] << "_" << fqDirectArgs[3];
    }
    if (inputArg.size() == 3) {
        result << "_inputArg=" << inputArg[0] << "_" << inputArg[1] << "_" << inputArg[2];
    }
    return result.str();
}

void FakeQuantizeSubgraphTest::SetUp() {
    fqSpecificParams fqParams;
    std::vector<size_t> inputShape;
    std::pair<std::string, std::map<std::string, std::string>> config;
    auto netPrecision = InferenceEngine::Precision::UNSPECIFIED;
    bool biases = false;
    std::tie(fqParams, netPrecision, inPrc, outPrc, inLayout, outLayout, inputShape, targetDevice, config, biases) = this->GetParam();
    InferenceEngine::SizeVector kernel, stride, dilation;
    std::vector<size_t> levels;
    std::vector<std::vector<size_t>> constShape;
    std::vector<float> fqDirectArg;
    std::vector<float> inputArg;
    std::tie(levels, constShape, fqDirectArg, inputArg) = fqParams;
    if (inputArg.size() == 3) {
        inputDataMin = inputArg[0];
        inputDataMax = inputArg[1];
        inputDataResolution = inputArg[2];
    }
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    auto params = ngraph::builder::makeParams(ngPrc, {inputShape});
    auto paramOuts = ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));

    std::shared_ptr<ngraph::Node> fakeQNode;
    std::shared_ptr<ngraph::Node> fakeQNode2;
    fakeQNode = ngraph::builder::makeFakeQuantize(
            paramOuts[0],
            ngraph::element::f32,
            levels[0],
            constShape[0],
            {fqDirectArg[0]},
            {fqDirectArg[1]},
            {fqDirectArg[2]},
            {fqDirectArg[3]});
    auto const_param = ngraph::builder::makeConstant(ngPrc, {constShape[1][0], inputShape[1]}, std::vector<float>{1.0f}, false);
    auto inputLowNode = ngraph::builder::makeConstant(ngraph::element::f32, constShape[0], std::vector<float>{fqDirectArg[2]}, false);
    auto inputHighNode = ngraph::builder::makeConstant(ngraph::element::f32, constShape[0], std::vector<float>{fqDirectArg[3]}, false);
    auto outputLowNode = ngraph::builder::makeConstant(ngraph::element::f32, constShape[1], std::vector<float>{fqDirectArg[2]}, false);
    auto outputHighNode = ngraph::builder::makeConstant(ngraph::element::f32, constShape[1], std::vector<float>{fqDirectArg[3]}, false);

    fakeQNode2 = std::make_shared<ngraph::opset1::FakeQuantize>(const_param, inputLowNode, inputHighNode, outputLowNode, outputHighNode, levels[1]);


    auto fq = std::dynamic_pointer_cast<ngraph::opset1::FakeQuantize>(fakeQNode);
    auto fq2 = std::dynamic_pointer_cast<ngraph::opset1::FakeQuantize>(fakeQNode2);
    auto matmul = std::make_shared<ngraph::opset1::MatMul>(fq, fq2, false, true);
    std::shared_ptr<ngraph::Node> biases_node;
    if (biases) {
        auto const_bias = ngraph::builder::makeConstant(ngPrc, {1, constShape[1][0]}, std::vector<float>{-1.0f});
        biases_node = std::make_shared<ngraph::opset1::Add>(matmul, const_bias);
    } else {
        biases_node = matmul;
    }

    auto sigmoid = std::make_shared<ngraph::opset1::Sigmoid>(biases_node);
    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(sigmoid)};
    if (biases) {
        auto sigmoid_2 = std::make_shared<ngraph::opset1::Sigmoid>(fq);
        results.push_back(std::make_shared<ngraph::opset1::Result>(sigmoid_2));
    }
    function = std::make_shared<ngraph::Function>(results, params, "fakeQuantizeSubgraph");

    configuration = config.second;
}

InferenceEngine::Blob::Ptr FakeQuantizeSubgraphTest::GenerateInput(const InferenceEngine::InputInfo &info) const {
    return FuncTestUtils::createAndFillBlob(info.getTensorDesc(), inputDataMax - inputDataMin, inputDataMin, 1 / inputDataResolution,
                                            seed);
}

TEST_P(FakeQuantizeSubgraphTest, CompareWithRefs) {
    Run();
}
}  // namespace LayerTestsDefinitions
