// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ov_models/builders.hpp>
#include "shared_test_classes/subgraph/activation_fq.hpp"

namespace SubgraphTestsDefinitions {

    std::string ActivationFakeQuantizeSubgraphTest::getTestCaseName(const testing::TestParamInfo<fqSubgraphTestParamsSet>& obj) {
        fqSpecificParams fqParams;
        ngraph::helpers::ActivationTypes activationType;
        InferenceEngine::Precision netPrecision;
        InferenceEngine::Precision inPrc, outPrc;
        InferenceEngine::Layout inLayout, outLayout;
        InferenceEngine::SizeVector inputShapes;
        std::string targetDevice;
        std::pair<std::string, std::map<std::string, std::string>> config;
        std::tie(fqParams, activationType, netPrecision, inPrc, outPrc, inLayout, outLayout, inputShapes, targetDevice, config) = obj.param;
        std::vector<size_t> levels;
        std::vector<std::vector<size_t>> constShape;
        std::vector<float> inputParams;
        std::tie(levels, constShape, inputParams) = fqParams;

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
        result << "_activation=" << activationNames[activationType];
        return result.str();
    }

    void ActivationFakeQuantizeSubgraphTest::SetUp() {
        fqSpecificParams fqParams;
        ngraph::helpers::ActivationTypes activationType;
        std::vector<size_t> inputShape;
        std::pair<std::string, std::map<std::string, std::string>> config;
        InferenceEngine::Precision netPrecision;
        std::tie(fqParams, activationType, netPrecision, inPrc, outPrc, inLayout, outLayout, inputShape, targetDevice, config) = this->GetParam();
        configuration.insert(config.second.begin(), config.second.end());

        std::vector<size_t> levels;
        std::vector<std::vector<size_t>> constShape;
        std::vector<float> inputArg;
        std::tie(levels, constShape, inputArg) = fqParams;
        if (inputArg.size() == 3) {
            inputDataMin = inputArg[0];
            inputDataMax = inputArg[1];
            inputDataResolution = inputArg[2];
        }

        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
        ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(inputShape))};

        auto act = ngraph::builder::makeActivation(params[0], ngPrc, activationType);

        auto FQNode = ngraph::builder::makeFakeQuantize(act, ngraph::element::f32, levels[0], constShape[0],
                                                        { inputDataMin }, { inputDataMax }, { inputDataMin }, { inputDataMax });

        auto FQ = std::dynamic_pointer_cast<ngraph::opset1::FakeQuantize>(FQNode);

        ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(FQ)};
        function = std::make_shared<ngraph::Function>(results, params, "ActivationFakeQuantizeSubgraph");
    }

InferenceEngine::Blob::Ptr ActivationFakeQuantizeSubgraphTest::GenerateInput(const InferenceEngine::InputInfo &info) const {
    return FuncTestUtils::createAndFillBlob(info.getTensorDesc(), inputDataMax - inputDataMin, inputDataMin, 1 / inputDataResolution,
                                            seed);
}
} // namespace SubgraphTestsDefinitions
