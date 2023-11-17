// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_layer/batch_norm.hpp"

namespace LayerTestsDefinitions {
std::string BatchNormLayerTest::getTestCaseName(const testing::TestParamInfo<BatchNormLayerTestParams>& obj) {
    InferenceEngine::Precision netPrecision;
    InferenceEngine::Precision inPrc, outPrc;
    InferenceEngine::Layout inLayout, outLayout;
    InferenceEngine::SizeVector inputShapes;
    double epsilon;
    std::string targetDevice;
    std::tie(epsilon, netPrecision, inPrc, outPrc, inLayout, outLayout, inputShapes, targetDevice) = obj.param;

    std::ostringstream result;
    result << "IS=" << ov::test::utils::vec2str(inputShapes) << "_";
    result << "epsilon=" << epsilon << "_";
    result << "netPRC=" << netPrecision.name() << "_";
    result << "inPRC=" << inPrc.name() << "_";
    result << "outPRC=" << outPrc.name() << "_";
    result << "inL=" << inLayout << "_";
    result << "outL=" << outLayout << "_";
    result << "trgDev=" << targetDevice;
    return result.str();
}

InferenceEngine::Blob::Ptr BatchNormLayerTest::GenerateInput(const InferenceEngine::InputInfo &info) const {
    return FuncTestUtils::createAndFillBlobConsistently(info.getTensorDesc(), 3, 0, 1);
}

void BatchNormLayerTest::SetUp() {
    InferenceEngine::Precision netPrecision;
    InferenceEngine::SizeVector inputShapes;
    double epsilon;
    std::tie(epsilon, netPrecision, inPrc, outPrc, inLayout, outLayout, inputShapes, targetDevice) = this->GetParam();
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

    ov::ParameterVector params {std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(inputShapes))};

    size_t C = inputShapes.at(1);
    bool random = true;
    std::vector<float> values(C);
    auto gamma = ngraph::builder::makeConstant(ngPrc, ov::Shape{C}, values, random, 1.f, 0.f);
    auto beta = ngraph::builder::makeConstant(ngPrc, ov::Shape{C}, values, random, 1.f, 0.f);
    auto mean = ngraph::builder::makeConstant(ngPrc, ov::Shape{C}, values, random, 1.f, 0.f);

    // Fill the vector for variance with positive values
    std::default_random_engine gen;
    std::uniform_real_distribution<float> dis(0.0, 10.0);
    std::generate(values.begin(), values.end(), [&dis, &gen]() {
        return dis(gen);
    });
    auto variance = ngraph::builder::makeConstant(ngPrc, ov::Shape{C}, values, !random);
    auto batchNorm = std::make_shared<ov::op::v5::BatchNormInference>(params[0], gamma, beta, mean, variance, epsilon);

    ngraph::ResultVector results{std::make_shared<ov::op::v0::Result>(batchNorm)};
    function = std::make_shared<ngraph::Function>(results, params, "BatchNormInference");
}

}  // namespace LayerTestsDefinitions
