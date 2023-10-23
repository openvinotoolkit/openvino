// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ie_core.hpp>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/blob_utils.hpp"
#include "functional_test_utils/plugin_cache.hpp"
#include "ov_models/builders.hpp"
#include "ov_models/pass/convert_prc.hpp"
#include "ov_models/utils/ov_helpers.hpp"
#include "shared_test_classes/base/layer_test_utils.hpp"

typedef std::tuple<InferenceEngine::Precision,         // Network Precision
                   std::string,                        // Target Device
                   std::map<std::string, std::string>  // Configuration
                   >
    fqWithMultipleOutConnectionsParams;

namespace LayerTestsDefinitions {

class FQWithMultipleOutConnections : public testing::WithParamInterface<fqWithMultipleOutConnectionsParams>,
                                     public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<fqWithMultipleOutConnectionsParams> obj) {
        InferenceEngine::Precision netPrecision;
        std::string targetDevice;
        std::map<std::string, std::string> configuration;
        std::tie(netPrecision, targetDevice, configuration) = obj.param;

        std::ostringstream result;
        result << "netPRC=" << netPrecision.name() << "_";
        result << "targetDevice=" << targetDevice << "_";
        for (auto const& configItem : configuration) {
            result << "_configItem=" << configItem.first << "_" << configItem.second;
        }

        return result.str();
    }

protected:
    void SetUp() override {
        InferenceEngine::Precision netPrecision;

        std::tie(netPrecision, targetDevice, configuration) = this->GetParam();
        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

        const ngraph::Shape shape = {1, 128};
        ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(shape))};

        auto pattern1 = std::make_shared<ngraph::opset8::Constant>(ngraph::element::Type_t::i64,
                                                                   ngraph::Shape{3},
                                                                   ngraph::Shape{1, 2, 64});
        auto reshape1 = std::make_shared<ngraph::opset8::Reshape>(params[0], pattern1, false);

        auto relu1 = std::make_shared<ngraph::opset8::Relu>(reshape1);

        auto lowNode = ngraph::builder::makeConstant<float>(ngPrc, {1}, {-10.0f});
        auto highNode = ngraph::builder::makeConstant<float>(ngPrc, {1}, {10.0f});
        auto fq = std::make_shared<ngraph::opset8::FakeQuantize>(relu1,
                                                                 lowNode,
                                                                 highNode,
                                                                 lowNode,
                                                                 highNode,
                                                                 std::numeric_limits<uint16_t>::max());

        auto pattern2 = std::make_shared<ngraph::opset8::Constant>(ngraph::element::Type_t::i64,
                                                                   ngraph::Shape{shape.size()},
                                                                   shape);
        auto reshape2 = std::make_shared<ngraph::opset8::Reshape>(fq, pattern2, false);

        auto relu2 = std::make_shared<ngraph::opset8::Relu>(fq);
        auto reshape3 = std::make_shared<ngraph::opset8::Reshape>(relu2, pattern2, false);

        ngraph::ResultVector results{std::make_shared<ngraph::opset8::Result>(reshape2),
                                     std::make_shared<ngraph::opset8::Result>(reshape3)};
        function = std::make_shared<ngraph::Function>(results, params, "FQFusionWithMultipleWeights");
    }
};

TEST_P(FQWithMultipleOutConnections, CompareWithRefImpl) {
    Run();
};

const std::vector<InferenceEngine::Precision> netPrecisions = {InferenceEngine::Precision::FP32,
                                                               InferenceEngine::Precision::FP16};

const std::vector<std::map<std::string, std::string>> configs = {{
                                                                     {"GNA_DEVICE_MODE", "GNA_SW_EXACT"},
                                                                 },
                                                                 {
                                                                     {"GNA_DEVICE_MODE", "GNA_SW_FP32"},
                                                                 }};

INSTANTIATE_TEST_SUITE_P(smoke_fq_fusion,
                         FQWithMultipleOutConnections,
                         ::testing::Combine(::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(ov::test::utils::DEVICE_GNA),
                                            ::testing::ValuesIn(configs)),
                         FQWithMultipleOutConnections::getTestCaseName);
}  // namespace LayerTestsDefinitions
