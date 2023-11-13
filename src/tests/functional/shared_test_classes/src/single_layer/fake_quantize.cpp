// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_layer/fake_quantize.hpp"

namespace LayerTestsDefinitions {


std::string FakeQuantizeLayerTest::getTestCaseName(const testing::TestParamInfo<fqLayerTestParamsSet>& obj) {
    fqSpecificParams fqParams;
    InferenceEngine::Precision netPrecision;
    InferenceEngine::Precision inPrc, outPrc;
    InferenceEngine::Layout inLayout, outLayout;
    InferenceEngine::SizeVector inputShapes;
    std::string targetDevice;
    std::pair<std::string, std::map<std::string, std::string>> config;
    std::tie(fqParams, netPrecision, inPrc, outPrc, inLayout, outLayout, inputShapes, targetDevice, config) = obj.param;
    size_t levels;
    std::vector<size_t> constShape;
    std::vector<float> fqDirectArgs;
    std::vector<float> inputArg;
    ngraph::op::AutoBroadcastSpec broadcast;
    std::tie(levels, constShape, fqDirectArgs, inputArg, broadcast) = fqParams;

    std::ostringstream result;
    result << "IS=" << ov::test::utils::vec2str(inputShapes) << "_";
    result << "CS=" << ov::test::utils::vec2str(constShape) << "_";
    result << "LEVELS=" << levels << "_";
    result << "netPRC=" << netPrecision.name() << "_";
    result << "inPRC=" << inPrc.name() << "_";
    result << "outPRC=" << outPrc.name() << "_";
    result << "inL=" << inLayout << "_";
    result << "outL=" << outLayout << "_";
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
    result << "_" << broadcast.m_type;
    return result.str();
}

void FakeQuantizeLayerTest::SetUp() {
    fqSpecificParams fqParams;
    std::vector<size_t> inputShape;
    std::pair<std::string, std::map<std::string, std::string>> config;
    auto netPrecision = InferenceEngine::Precision::UNSPECIFIED;
    std::tie(fqParams, netPrecision, inPrc, outPrc, inLayout, outLayout, inputShape, targetDevice, config) = this->GetParam();
    InferenceEngine::SizeVector kernel, stride, dilation;
    size_t levels;
    std::vector<size_t> constShape;
    std::vector<float> fqDirectArg;
    std::vector<float> inputArg;
    ngraph::op::AutoBroadcastSpec broadcast;
    std::tie(levels, constShape, fqDirectArg, inputArg, broadcast) = fqParams;
    if (inputArg.size() == 3) {
        inputDataMin = inputArg[0];
        inputDataMax = inputArg[1];
        inputDataResolution = inputArg[2];
    }
    if (fqDirectArg.size() != 0) {
        threshold = (fqDirectArg[3] - fqDirectArg[2]) / levels;
    }
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(inputShape))};

    UpdateSeed();

    std::shared_ptr<ngraph::Node> fakeQNode;
    if (fqDirectArg.empty()) {
        int32_t ngraphSeed = seed;
        if (NGRAPH_SEED != USE_CLOCK_TIME) {
            ngraphSeed = NGRAPH_SEED;
        }
        std::cout << "\033[0;32m" << "[          ] " << "\033[0;0m"
                  << "ngraphSeed = " << ngraphSeed << std::endl;
        fakeQNode = ngraph::builder::makeFakeQuantize(params[0], ngPrc, levels, constShape, ngraphSeed);
    } else {
        fakeQNode = ngraph::builder::makeFakeQuantize(
            params[0],
            ngPrc,
            levels,
            constShape,
            {fqDirectArg[0]},
            {fqDirectArg[1]},
            {fqDirectArg[2]},
            {fqDirectArg[3]});
    }
    auto fq = std::dynamic_pointer_cast<ngraph::opset1::FakeQuantize>(fakeQNode);

    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(fq)};
    function = std::make_shared<ngraph::Function>(results, params, "fakeQuantize");
    configuration = config.second;
}

InferenceEngine::Blob::Ptr FakeQuantizeLayerTest::GenerateInput(const InferenceEngine::InputInfo &info) const {
    return FuncTestUtils::createAndFillBlob(info.getTensorDesc(), inputDataMax - inputDataMin, inputDataMin, 1 / inputDataResolution,
      seed);
}

void FakeQuantizeLayerTest::UpdateSeed() {
    if (BASE_SEED == USE_CLOCK_TIME) {
        seed = std::chrono::system_clock::now().time_since_epoch().count();
    } else if (BASE_SEED == USE_INCREMENTAL_SEED) {
        seed += 9999;
    } else {
        seed = BASE_SEED;
    }
    std::cout << "\033[0;32m" << "[          ] " << "\033[0;0m"
              << "seed = " << seed << std::endl;
}

}  // namespace LayerTestsDefinitions
