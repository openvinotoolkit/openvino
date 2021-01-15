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
#include "shared_test_classes/base/layer_test_utils.hpp"
#include "functional_test_utils/blob_utils.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"
#include "ngraph_functions/builders.hpp"

#include "ngraph_functions/pass/convert_prc.hpp"

typedef std::tuple<
    InferenceEngine::Precision,         // Network Precision
    std::string,                        // Target Device
    std::map<std::string, std::string>, // Configuration
    std::vector<size_t>                 // Input shape
> removePermutationsPassParams;

namespace LayerTestsDefinitions {

class RemovePermutationsNHWCToNCHWPassTest : public testing::WithParamInterface<removePermutationsPassParams>,
                                             public LayerTestsUtils::LayerTestsCommon {
    public:
        static std::string getTestCaseName(testing::TestParamInfo<removePermutationsPassParams> obj) {
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
            result << "_IS=" << CommonTestUtils::vec2str(inputShape);
            return result.str();
        }

    protected:
        void SetUp() override {
            //      Reshape
            //          |
            //      Permute (order: [0, 3, 1, 2])
            //          |
            //      Convolution
            //          |
            //      Permute (order: [0, 2, 3, 1])
            //          |
            //      Reshape
            InferenceEngine::Precision netPrecision;
            std::vector<size_t> inputShape;
            std::tie(netPrecision, targetDevice, configuration, inputShape) = this->GetParam();
            auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

            size_t in_total_dims_size = std::accumulate(std::begin(inputShape), std::end(inputShape), 1, std::multiplies<double>());
            auto params = ngraph::builder::makeParams(ngPrc, { {1, in_total_dims_size} });

            auto pattern1 = std::make_shared<ngraph::opset1::Constant>(ngraph::element::Type_t::i64, ngraph::Shape{ 4 }, inputShape);
            auto reshape1 = std::make_shared<ngraph::opset1::Reshape>(params[0], pattern1, false);

            auto permute1 = std::make_shared<ngraph::opset1::Transpose>(reshape1,
                ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{ 4 }, { 0, 3, 1, 2 }));

            size_t num_out_channels = 12;
            size_t kernal_size = 8;
            auto conv1 = ngraph::builder::makeConvolution(permute1, ngPrc, { 1, kernal_size }, { 1, 1 }, { 0, 0 }, { 0, 0 }, { 1, 1 },
                ngraph::op::PadType::VALID, num_out_channels);

            auto permute2 = std::make_shared<ngraph::opset1::Transpose>(conv1,
                ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{ 4 }, { 0, 2, 3, 1 }));

            size_t out_width = (inputShape[2] - kernal_size) + 1;
            std::vector<size_t> outFormShapes = { 1, out_width * num_out_channels };
            auto pattern2 = std::make_shared<ngraph::opset1::Constant>(ngraph::element::Type_t::i64, ngraph::Shape{ 2 }, outFormShapes);
            auto reshape2 = std::make_shared<ngraph::opset1::Reshape>(permute2, pattern2, false);

            ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(reshape2) };
            function = std::make_shared<ngraph::Function>(results, params, "RemovePermutationPass");
        }
};

class RemovePermutationsNHWCToNCHWPass4DOutputTest : public testing::WithParamInterface<removePermutationsPassParams>,
    public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<removePermutationsPassParams> obj) {
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
        result << "_IS=" << CommonTestUtils::vec2str(inputShape);
        return result.str();
    }

protected:
    void SetUp() override {
        InferenceEngine::Precision netPrecision;
        std::vector<size_t> inputShape;
        std::tie(netPrecision, targetDevice, configuration, inputShape) = this->GetParam();
        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

        auto params = ngraph::builder::makeParams(ngPrc, { inputShape });
        auto permute1 = std::make_shared<ngraph::opset1::Transpose>(params[0],
                             ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{ 4 }, { 0, 3, 1, 2 }));

        auto conv1 = ngraph::builder::makeConvolution(permute1, ngPrc, { 1, 8 }, { 1, 1 }, { 0, 0 }, { 0, 0 }, { 1, 1 }, ngraph::op::PadType::VALID, 12);

        auto permute2 = std::make_shared<ngraph::opset1::Transpose>(conv1,
                             ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{ 4 }, { 0, 2, 3, 1 }));

        ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(permute2) };

        function = std::make_shared<ngraph::Function>(results, params, "RemovePermutationPass4DOutput");
    }
};

    TEST_P(RemovePermutationsNHWCToNCHWPassTest, CompareWithRefImpl) {
        Run();
    };

    TEST_P(RemovePermutationsNHWCToNCHWPass4DOutputTest, CompareWithRefImpl) {
        Run();
    };

    const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::FP16
    };

    const std::vector<std::map<std::string, std::string>> configs = {
        {
            {"GNA_DEVICE_MODE", "GNA_SW_EXACT"},
            {"GNA_SCALE_FACTOR_0", "327.67"}
        }
    };

    const std::vector<std::vector<size_t>> inputShapes {
        {1, 1, 168, 1},
        {1, 1, 168, 2},
        {1, 1, 168, 8},
        {1, 1, 32, 1},
        {1, 1, 32, 2},
        {1, 1, 32, 8}
    };

    INSTANTIATE_TEST_CASE_P(smoke_PermutationPass, RemovePermutationsNHWCToNCHWPassTest,
        ::testing::Combine(
            ::testing::ValuesIn(netPrecisions),
            ::testing::Values(CommonTestUtils::DEVICE_GNA),
            ::testing::ValuesIn(configs),
            ::testing::ValuesIn(inputShapes)),
        RemovePermutationsNHWCToNCHWPassTest::getTestCaseName);

    INSTANTIATE_TEST_CASE_P(smoke_PermutationPass, RemovePermutationsNHWCToNCHWPass4DOutputTest,
        ::testing::Combine(
            ::testing::ValuesIn(netPrecisions),
            ::testing::Values(CommonTestUtils::DEVICE_GNA),
            ::testing::ValuesIn(configs),
            ::testing::ValuesIn(inputShapes)),
        RemovePermutationsNHWCToNCHWPass4DOutputTest::getTestCaseName);

} // namespace LayerTestsDefinitions

