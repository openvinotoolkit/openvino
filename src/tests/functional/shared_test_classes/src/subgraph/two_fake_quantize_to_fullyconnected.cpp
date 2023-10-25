// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/subgraph/two_fake_quantize_to_fullyconnected.hpp"

namespace SubgraphTestsDefinitions {

std::string FakeQuantizeSubgraphTest::getTestCaseName(const testing::TestParamInfo<fqSubgraphTestParamsSet>& obj) {
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
    result << "IS=" << ov::test::utils::vec2str(inputShapes) << "_";
    result << "CS=" << ov::test::utils::vec2str(constShape) << "_";
    result << "LEVELS=" << ov::test::utils::vec2str(levels) << "_";
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
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(inputShape))};
    auto paramOuts = ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));

    const int seed = 0;
    std::mt19937 gen(seed);


    auto generateFloatNumbers = [gen](std::size_t vec_len, float min, float max) mutable {
        std::vector<float> res;

        std::uniform_real_distribution<float> dist(min, max);
        for (std::size_t i = 0; i < vec_len; i++)
            res.emplace_back(static_cast<float>(dist(gen)));

        return res;
    };


    auto weightsRowNum = constShape[1][0];
    auto weightsColNum = inputShape[1];
    auto weightsData = generateFloatNumbers(weightsRowNum * weightsColNum, inputDataMin, inputDataMax);
    auto const_param = ngraph::builder::makeConstant<float>(ngPrc, { constShape[1][0], inputShape[1] }, { 1.0f });
    auto inputMinRange = std::vector<float>{};
    auto inputMaxRange = std::vector<float>{};
    auto channelDataSize = constShape[1];

    if (channelDataSize[0] == 1) {
        // If per tensor data needs to be provided
        inputMinRange.push_back(inputDataMin);
        inputMaxRange.push_back(inputDataMax);
    } else if (channelDataSize[0] == weightsRowNum) {
        // If per channel data needs to be provided
        for (size_t i = 0; i < weightsRowNum; ++i) {
            auto minChannelVal = std::numeric_limits<float>::max();
            auto maxChannelVal = std::numeric_limits<float>::min();
            for (size_t j = 0; j < weightsColNum; ++j) {
                minChannelVal = std::min(minChannelVal, weightsData[i * weightsColNum + j]);
                maxChannelVal = std::max(maxChannelVal, weightsData[i * weightsColNum + j]);
            }

            inputMinRange.push_back(minChannelVal);
            inputMaxRange.push_back(maxChannelVal);
        }
    } else {
        FAIL() << "Invalid test configuration";
    }

    auto lowNode = ngraph::builder::makeConstant(ngraph::element::f32, channelDataSize, inputMinRange, false);
    auto highNode = ngraph::builder::makeConstant(ngraph::element::f32, channelDataSize, inputMaxRange, false);

    auto inputFQNode = ngraph::builder::makeFakeQuantize(paramOuts[0], ngraph::element::f32, levels[0], constShape[0],
        { inputDataMin }, { inputDataMax }, { inputDataMin }, { inputDataMax });

    auto weightsFQNode = std::make_shared<ngraph::opset1::FakeQuantize>(const_param,
        lowNode, highNode, lowNode, highNode, levels[1]);

    auto inputFQ = std::dynamic_pointer_cast<ngraph::opset1::FakeQuantize>(inputFQNode);
    auto weightsFQ = std::dynamic_pointer_cast<ngraph::opset1::FakeQuantize>(weightsFQNode);
    auto matmul = std::make_shared<ngraph::opset1::MatMul>(inputFQ, weightsFQ, false, true);
    std::shared_ptr<ngraph::Node> biases_node;
    if (biases) {
        auto const_bias = ngraph::builder::makeConstant(ngPrc, {1, constShape[1][0]}, std::vector<float>{ -1.0f });
        biases_node = std::make_shared<ngraph::opset1::Add>(matmul, const_bias);
    } else {
        biases_node = matmul;
    }

    auto sigmoid = std::make_shared<ngraph::opset1::Sigmoid>(biases_node);
    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(sigmoid)};
    if (biases) {
        auto sigmoid_2 = std::make_shared<ngraph::opset1::Sigmoid>(inputFQ);
        results.push_back(std::make_shared<ngraph::opset1::Result>(sigmoid_2));
    }
    function = std::make_shared<ngraph::Function>(results, params, "fakeQuantizeSubgraph");
    configuration = config.second;
}

InferenceEngine::Blob::Ptr FakeQuantizeSubgraphTest::GenerateInput(const InferenceEngine::InputInfo &info) const {
    return FuncTestUtils::createAndFillBlob(info.getTensorDesc(), inputDataMax - inputDataMin, inputDataMin, 1 / inputDataResolution,
                                            seed);
}
}  // namespace SubgraphTestsDefinitions
