// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <memory>
#include <tuple>
#include <vector>
#include <string>

#include <ie_core.hpp>

#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/plugin_cache.hpp"
#include "functional_test_utils/layer_test_utils.hpp"
#include "functional_test_utils/blob_utils.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"
#include "ngraph_functions/builders.hpp"

#include "ngraph_functions/pass/convert_prc.hpp"

typedef std::tuple<
    InferenceEngine::Precision,         // Network Precision
    std::string,                        // Target Device
    std::map<std::string, std::string>  //Configuration
> removePermutationsPassParams;

namespace LayerTestsDefinitions {

class RemovePermutationsNHWCToNCHWPassTest : public testing::WithParamInterface<removePermutationsPassParams>,
                                             public LayerTestsUtils::LayerTestsCommon {
    public:
        static std::string getTestCaseName(testing::TestParamInfo<removePermutationsPassParams> obj) {
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
            //      Reshape ([1, 336] -> [1, 1, 168, 2])
            //          |
            //      Permute (order: [0, 3, 1, 2])
            //          |
            //      Convolution (weights: [2, 12, 1, 8])
            //          |
            //      Permute (order: [0, 2, 3, 1])
            //          |
            //      Reshape ([1, 1, 161, 12] -> [1, 1932])
            InferenceEngine::Precision netPrecision;
            std::tie(netPrecision, targetDevice, configuration) = this->GetParam();
            auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

            auto params = ngraph::builder::makeParams(ngPrc, { {1, 336} });

            std::vector<size_t> outFormShapes1 = { 1, 1, 168, 2 };
            auto pattern1 = std::make_shared<ngraph::opset1::Constant>(ngraph::element::Type_t::i64, ngraph::Shape{ 4 }, outFormShapes1);
            auto reshape1 = std::make_shared<ngraph::opset1::Reshape>(params[0], pattern1, false);

            auto permute1 = std::make_shared<ngraph::opset1::Transpose>(reshape1,
                ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{ 4 }, { 0, 3, 1, 2 }));
            permute1->set_friendly_name("permute1");

            auto conv1 = ngraph::builder::makeConvolution(permute1, ngPrc, { 1, 8 }, { 1, 1 }, { 0, 0 }, { 0, 0 }, { 1, 1 },
                ngraph::op::PadType::VALID, 12);

            auto permute2 = std::make_shared<ngraph::opset1::Transpose>(conv1,
                ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{ 4 }, { 0, 2, 3, 1 }));
            permute2->set_friendly_name("permute2");

            std::vector<size_t> outFormShapes2 = { 1, 1932 };
            auto pattern2 = std::make_shared<ngraph::opset1::Constant>(ngraph::element::Type_t::i64, ngraph::Shape{ 2 }, outFormShapes2);
            auto reshape2 = std::make_shared<ngraph::opset1::Reshape>(permute2, pattern2, false);

            ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(reshape2) };
            function = std::make_shared<ngraph::Function>(results, params, "RemovePermutationPass");
        }
};

    TEST_P(RemovePermutationsNHWCToNCHWPassTest, CompareWithRefImpl) {
        Run();
    };

    const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::FP16
    };

    const std::vector<std::map<std::string, std::string>> configs = {
        {
            {"GNA_DEVICE_MODE", "GNA_SW_EXACT"},
            {"GNA_SCALE_FACTOR_0", "1638.4"}
        }
    };

    INSTANTIATE_TEST_CASE_P(PermutationPass, RemovePermutationsNHWCToNCHWPassTest,
        ::testing::Combine(
            ::testing::ValuesIn(netPrecisions),
            ::testing::Values(CommonTestUtils::DEVICE_GNA),
            ::testing::ValuesIn(configs)),
        RemovePermutationsNHWCToNCHWPassTest::getTestCaseName);

} // namespace LayerTestsDefinitions

