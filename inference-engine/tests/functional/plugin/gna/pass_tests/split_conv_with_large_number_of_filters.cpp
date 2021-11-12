// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/test_common.hpp"
#include <string>
#include <sstream>
#include <fstream>
#include <memory>
#include <queue>
#include <map>

#include "transformations/init_node_info.hpp"
#include "ngraph_functions/builders.hpp"
#include "shared_test_classes/base/layer_test_utils.hpp"
#include "../shared_tests_instances/skip_tests_check.hpp"

using namespace ngraph;
using namespace ngraph::opset8;

namespace LayerTestsDefinitions {

typedef std::tuple<
    InferenceEngine::SizeVector,        // Input shape
    InferenceEngine::SizeVector,        // Filter shape
    size_t,                             // Num of filters
    InferenceEngine::Precision,         // Network precision
    std::string,                        // Target device
    std::map<std::string, std::string>, // Configuration
    bool                                // With bias
> splitConvFilterParams;

class SplitConvWithLargeNumOfFiltersTest : public testing::WithParamInterface<splitConvFilterParams>,
    virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<splitConvFilterParams> obj) {
        InferenceEngine::Precision netPrecision;
        std::string targetDevice;
        std::map<std::string, std::string> configuration;
        InferenceEngine::SizeVector inputShape, filterShape;
        size_t numOfFilters;
        bool bias;
        std::tie(inputShape, filterShape, numOfFilters, netPrecision, targetDevice, configuration, bias) = obj.param;

        std::ostringstream result;
        result << "IS=" << CommonTestUtils::vec2str(inputShape) << "_";
        result << "FS" << CommonTestUtils::vec2str(filterShape) << "_";
        result << "O=" << numOfFilters << "_";
        result << "B=" << static_cast<bool>(bias) << "_";
        result << "netPRC=" << netPrecision.name() << "_";
        result << "targetDevice=" << targetDevice << "_";
        for (auto const& configItem : configuration) {
            result << "_configItem=" << configItem.first << "_" << configItem.second;
        }
        return result.str();
    }

protected:
    void SetUp() override {
        threshold = 0.01f;
        InferenceEngine::Precision netPrecision;
        std::map<std::string, std::string> configuration;
        InferenceEngine::SizeVector inputShape, filterShape;
        size_t numOfFilters;
        bool bias;
        std::tie(inputShape, filterShape, numOfFilters, netPrecision, targetDevice, configuration, bias) = this->GetParam();

        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
        auto input = builder::makeParams(ngPrc, {inputShape});
        auto filterSize = std::accumulate(std::begin(filterShape), std::end(filterShape), 1ull, std::multiplies<size_t>());
        auto filterWeights = CommonTestUtils::generate_float_numbers(numOfFilters * inputShape[1] * filterSize, -0.05f, 0.05f);
        auto conv = builder::makeConvolution(input[0], ngPrc, filterShape, {1, 1}, {0, 0}, {0, 0}, {1, 1},
            ngraph::op::PadType::VALID, numOfFilters, false, filterWeights);
        Output<Node> lastOp = conv;

        if (bias) {
            Shape biasShape{1, numOfFilters, 1, 1};
            auto biasWeights = CommonTestUtils::generate_float_numbers(shape_size(biasShape), -1.5f, 1.5f);
            Output<Node> biasConst = std::make_shared<Constant>(ngPrc, biasShape, biasWeights);
            lastOp = std::make_shared<Add>(conv, biasConst);
        }

        auto result = std::make_shared<Result>(lastOp);
        function = std::make_shared<Function>(ResultVector{result}, ParameterVector{input});
    }
};

TEST_P(SplitConvWithLargeNumOfFiltersTest, CompareWithRefs) {
    Run();
}

const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32,
    InferenceEngine::Precision::FP16
};

const std::vector<std::map<std::string, std::string>> configs = {
    {
        {"GNA_DEVICE_MODE", "GNA_SW_EXACT"},
        {"GNA_EXEC_TARGET", "GNA_TARGET_3_0"}
    }
};

const std::vector<std::vector<size_t>> inputShape = {{1, 16, 1, 128}};
const std::vector<std::vector<size_t >> filterShape = {{1, 2}};
const std::vector<size_t> numOfFilters = {64, 256, 8000, 10000, 32000};
const std::vector<bool> bias = {false, true};

INSTANTIATE_TEST_SUITE_P(smoke_SplitConvWithLargeNumOfFilters, SplitConvWithLargeNumOfFiltersTest,
    ::testing::Combine(
        ::testing::ValuesIn(inputShape),
        ::testing::ValuesIn(filterShape),
        ::testing::ValuesIn(numOfFilters),
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(CommonTestUtils::DEVICE_GNA),
        ::testing::ValuesIn(configs),
        ::testing::ValuesIn(bias)),
    SplitConvWithLargeNumOfFiltersTest::getTestCaseName);

} // namespace LayerTestsDefinitions
