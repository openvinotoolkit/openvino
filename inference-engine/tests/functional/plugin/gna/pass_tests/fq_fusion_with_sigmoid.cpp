// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ie_core.hpp>
#include "ngraph_functions/builders.hpp"
#include "common_test_utils/test_constants.hpp"
#include "shared_test_classes/base/layer_test_utils.hpp"

namespace LayerTestsDefinitions {

typedef std::tuple<
    std::string,                        // Target device name
    InferenceEngine::Precision,         // Network precision
    size_t,                             // Input size
    std::map<std::string, std::string>  // Configuration
> fqFusionWithSigmoidParams;

class FqFusionWithSigmoidTest : public LayerTestsUtils::LayerTestsCommon,
    public testing::WithParamInterface<fqFusionWithSigmoidParams> {
protected:
    void SetUp() override {
        InferenceEngine::Precision netPrecision;
        std::map<std::string, std::string> config;
        size_t inputSize;
        std::tie(targetDevice, netPrecision, inputSize, config) = this->GetParam();
        configuration.insert(config.begin(), config.end());
        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

        auto input = ngraph::builder::makeParams(ngPrc, {{1, inputSize}});
        float left = 0;
        float right = 0.01;
        auto fake1 = ngraph::builder::makeFakeQuantize(input[0], ngPrc, 255, { 1 }, { left }, { right }, { left }, { right });
        auto sigmoid1 = std::make_shared<ngraph::opset1::Sigmoid>(fake1);
        auto mul1 = ngraph::builder::makeEltwise(input[0], sigmoid1, ngraph::helpers::EltwiseTypes::MULTIPLY);
        auto fake2 = ngraph::builder::makeFakeQuantize(mul1, ngPrc, 255, { 1 }, { left }, { right }, { left }, { right });
        auto fake3 = ngraph::builder::makeFakeQuantize(sigmoid1, ngPrc, 255, { 1 }, { left }, { right }, { left }, { right });
        auto mul2 = ngraph::builder::makeEltwise(fake2, fake3, ngraph::helpers::EltwiseTypes::MULTIPLY);
        auto result = std::make_shared<ngraph::opset7::Result>(mul2);
        function = std::make_shared<ngraph::Function>(ngraph::ResultVector{result}, input, "fq_fusion_with_sigmoid");
    }
public:
    static std::string getTestCaseName(const testing::TestParamInfo<fqFusionWithSigmoidParams> &obj) {
        std::string targetDevice;
        InferenceEngine::Precision netPrecision;
        size_t inputSize;
        std::map<std::string, std::string> config;
        std::tie(targetDevice, netPrecision, inputSize, config) = obj.param;
        std::ostringstream result;
        result << "netPrecision=" << netPrecision.name() << "_";
        result << "IS=" << inputSize << "_";
        result << "targetDevice=" << targetDevice;
        return result.str();
    }
}; // class FqFusionWithSigmoidTest

TEST_P(FqFusionWithSigmoidTest, CompareWithRefs) {
    Run();
};

const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32,
    InferenceEngine::Precision::FP16
};

std::vector<size_t> input = {
    64,
};

std::map<std::string, std::string> additional_config = {
    {"GNA_DEVICE_MODE", "GNA_SW_EXACT"},
};

INSTANTIATE_TEST_SUITE_P(smoke_fq_fusion_with_sigmoid, FqFusionWithSigmoidTest,
    ::testing::Combine(
        ::testing::Values(CommonTestUtils::DEVICE_GNA),
        ::testing::ValuesIn(netPrecisions),
        ::testing::ValuesIn(input),
        ::testing::Values(additional_config)),
    FqFusionWithSigmoidTest::getTestCaseName);

} // namespace LayerTestsDefinitions
