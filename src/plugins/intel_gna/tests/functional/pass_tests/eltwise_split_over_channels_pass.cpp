// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ie_core.hpp>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "functional_test_utils/blob_utils.hpp"
#include "ov_models/builders.hpp"
#include "ov_models/utils/ov_helpers.hpp"
#include "shared_test_classes/base/layer_test_utils.hpp"

typedef std::tuple<InferenceEngine::Precision,          // Network Precision
                   std::string,                         // Target Device
                   std::map<std::string, std::string>,  // Configuration
                   std::vector<size_t>                  // Input Shape
                   >
    EltwiseSplitOverChannelsPassParams;

namespace LayerTestsDefinitions {

class EltwiseSplitOverChannelsPassTest : public testing::WithParamInterface<EltwiseSplitOverChannelsPassParams>,
                                         public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<EltwiseSplitOverChannelsPassParams> obj) {
        InferenceEngine::Precision netPrecision;
        std::string targetDevice;
        std::map<std::string, std::string> configuration;
        std::vector<size_t> inputShape;
        std::tie(netPrecision, targetDevice, configuration, inputShape) = obj.param;

        std::ostringstream result;
        result << "netPRC=" << netPrecision.name() << "_";
        result << "targetDevice=" << targetDevice << "_";
        for (auto const& configItem : configuration) {
            result << "_configItem=" << configItem.first << "_" << configItem.second;
        }
        result << "_inputShape=" << ov::test::utils::vec2str(inputShape);
        return result.str();
    }

protected:
    void SetUp() override {
        InferenceEngine::Precision netPrecision;
        std::vector<size_t> inputShape;
        std::tie(netPrecision, targetDevice, configuration, inputShape) = this->GetParam();
        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

        ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(inputShape))};
        auto const_mult2 = ngraph::builder::makeConstant<float>(ngPrc, inputShape, {-1.0f});

        auto mul = ngraph::builder::makeEltwise(params[0], const_mult2, ngraph::helpers::EltwiseTypes::MULTIPLY);
        function = std::make_shared<ngraph::Function>(mul, params, "EltwiseSplitOverChannelsPassTest");
    }
};

TEST_P(EltwiseSplitOverChannelsPassTest, CompareWithRefImpl) {
    Run();
};

const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32,
    InferenceEngine::Precision::FP16,
};

const std::vector<std::map<std::string, std::string>> configs = {
    {{"GNA_DEVICE_MODE", "GNA_SW_EXACT"}, {"GNA_COMPACT_MODE", "NO"}},
    {{"GNA_DEVICE_MODE", "GNA_SW_EXACT"}, {"GNA_COMPACT_MODE", "YES"}},
    {
        {"GNA_DEVICE_MODE", "GNA_SW_FP32"},
    }};

const std::vector<std::vector<size_t>> inputShape = {{1, 67000}, {1, 500000}, {1, 936, 513}, {1, 64, 64, 64}};

INSTANTIATE_TEST_SUITE_P(smoke_EltwiseSplitOverChennels,
                         EltwiseSplitOverChannelsPassTest,
                         ::testing::Combine(::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(ov::test::utils::DEVICE_GNA),
                                            ::testing::ValuesIn(configs),
                                            ::testing::ValuesIn(inputShape)),
                         EltwiseSplitOverChannelsPassTest::getTestCaseName);

}  // namespace LayerTestsDefinitions
