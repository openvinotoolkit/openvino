// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ov_models/builders.hpp>
#include "shared_test_classes/subgraph/clamp_fq.hpp"

namespace SubgraphTestsDefinitions {

    std::string ClampFakeQuantizeSubgraphTest::getTestCaseName(const testing::TestParamInfo<fqSubgraphTestParamsSet>& obj) {
        fqSpecificParams fqParams;
        InferenceEngine::Precision netPrecision;
        InferenceEngine::Precision inPrc, outPrc;
        InferenceEngine::Layout inLayout, outLayout;
        InferenceEngine::SizeVector inputShapes;
        std::string targetDevice;
        std::pair<std::string, std::map<std::string, std::string>> config;
        std::tie(fqParams, netPrecision, inPrc, outPrc, inLayout, outLayout, inputShapes, targetDevice, config) = obj.param;
        std::vector<size_t> levels;
        std::vector<std::vector<size_t>> constShape;
        std::vector<float> inputParams;
        std::vector<float> clampMinMax;
        std::tie(levels, constShape, clampMinMax, inputParams) = fqParams;

        std::ostringstream result;
        result << "InputShape=" << ov::test::utils::vec2str(inputShapes) << "_";
        result << "CS=" << ov::test::utils::vec2str(constShape) << "_";
        result << "LEVELS=" << ov::test::utils::vec2str(levels) << "_";
        result << "netPRC=" << netPrecision.name() << "_";
        result << "inPRC=" << inPrc.name() << "_";
        result << "outPRC=" << outPrc.name() << "_";
        result << "inL=" << inLayout << "_";
        result << "outL=" << outLayout << "_";
        result << "trgDev=" << targetDevice;
        if (!config.first.empty()) {
            result << "_targetConfig=" << config.first;
        }
        if (inputParams.size() == 3) {
            result << "_inputArg=" << inputParams[0] << "_" << inputParams[1] << "_" << inputParams[2];
        }
        if (clampMinMax.size() == 2) {
            result << "_clampMaxMin=" << clampMinMax[0] << "_" << clampMinMax[1];
        }
        return result.str();
    }
    void ClampFakeQuantizeSubgraphTest::SetUp() {
        fqSpecificParams fqParams;
        std::vector<size_t> inputShape;
        std::pair<std::string, std::map<std::string, std::string>> config;
        auto netPrecision = InferenceEngine::Precision::UNSPECIFIED;
        std::tie(fqParams, netPrecision, inPrc, outPrc, inLayout, outLayout, inputShape, targetDevice, config) = this->GetParam();
        InferenceEngine::SizeVector kernel, stride, dilation;
        std::vector<size_t> levels;
        std::vector<std::vector<size_t>> constShape;
        std::vector<float> clamp_min_max;
        std::vector<float> inputArg;
        std::tie(levels, constShape, clamp_min_max, inputArg) = fqParams;
        if (inputArg.size() == 3) {
            inputDataMin = inputArg[0];
            inputDataMax = inputArg[1];
            inputDataResolution = inputArg[2];
        }
        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
        ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(inputShape))};

        auto clamp = std::make_shared<ngraph::opset1::Clamp>(params[0], clamp_min_max[0], clamp_min_max[1]);

        auto FQNode = ngraph::builder::makeFakeQuantize(clamp, ngraph::element::f32, levels[0], constShape[0],
                                                             { inputDataMin }, { inputDataMax }, { inputDataMin }, { inputDataMax });


        auto FQ = std::dynamic_pointer_cast<ngraph::opset1::FakeQuantize>(FQNode);
        auto sigmoid = std::make_shared<ngraph::opset1::Sigmoid>(FQ);

        ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(sigmoid)};
        function = std::make_shared<ngraph::Function>(results, params, "fakeQuantizeSubgraph");
            configuration = config.second;
    }

InferenceEngine::Blob::Ptr ClampFakeQuantizeSubgraphTest::GenerateInput(const InferenceEngine::InputInfo &info) const {
    return FuncTestUtils::createAndFillBlob(info.getTensorDesc(), inputDataMax - inputDataMin, inputDataMin, 1 / inputDataResolution,
                                            seed);
}
} // namespace SubgraphTestsDefinitions
