// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <fstream>
#include <map>
#include <memory>
#include <queue>
#include <sstream>
#include <string>

#include "common_test_utils/test_common.hpp"
#include "ov_models/builders.hpp"
#include "shared_test_classes/base/layer_test_utils.hpp"
#include "transformations/init_node_info.hpp"

using namespace ngraph;
using namespace opset8;

namespace LayerTestsDefinitions {

typedef std::tuple<bool,            // Normalize variance
                   float,           // Epsilon
                   op::MVNEpsMode,  // Epsilon mode
                   bool,            // Across channels
                   bool             // MVN version, true = v6, false = v1
                   >
    mvnSpecificParams;

typedef std::tuple<mvnSpecificParams,                   // MVN parameters
                   InferenceEngine::Precision,          // Network Precision
                   std::string,                         // Target Device
                   std::map<std::string, std::string>,  // Configuration
                   InferenceEngine::SizeVector          // Input shapes
                   >
    decomposeMVNParams;

class DecomposeMVNTest : public testing::WithParamInterface<decomposeMVNParams>,
                         virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<decomposeMVNParams> obj) {
        mvnSpecificParams mvnParams;
        InferenceEngine::Precision netPrecision;
        std::string targetDevice;
        std::map<std::string, std::string> configuration;
        InferenceEngine::SizeVector inputShape;
        std::tie(mvnParams, netPrecision, targetDevice, configuration, inputShape) = obj.param;
        float eps;
        op::MVNEpsMode epsMode;
        bool normalizeVariance, acrossChannels, mvnVersion6;
        std::tie(normalizeVariance, eps, epsMode, acrossChannels, mvnVersion6) = mvnParams;

        std::ostringstream result;
        result << "IS=" << ov::test::utils::vec2str(inputShape) << "_";
        result << "NV=" << normalizeVariance << "_";
        result << "eps=" << eps << "_";
        result << "mode=" << static_cast<uint32_t>(epsMode) << "_";
        result << "AC=" << acrossChannels << "_";
        result << "version=" << mvnVersion6 << "_";
        result << "netPRC=" << netPrecision.name() << "_";
        result << "targetDevice=" << targetDevice << "_";
        for (auto const& configItem : configuration) {
            result << "_configItem=" << configItem.first << "_" << configItem.second;
        }
        return result.str();
    }

protected:
    void SetUp() override {
        threshold = 0.2f;
        mvnSpecificParams mvnParams;
        InferenceEngine::Precision netPrecision;
        InferenceEngine::SizeVector inputShape;
        std::tie(mvnParams, netPrecision, targetDevice, configuration, inputShape) = this->GetParam();
        float eps;
        op::MVNEpsMode epsMode;
        bool normalizeVariance, acrossChannels, mvnVersion6;
        std::tie(normalizeVariance, eps, epsMode, acrossChannels, mvnVersion6) = mvnParams;
        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
        ov::ParameterVector input{std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(inputShape))};
        InferenceEngine::SizeVector axes(inputShape.size() - 2);
        std::iota(axes.begin(), axes.end(), 2);
        std::shared_ptr<ngraph::Node> mvn;

        if (mvnVersion6) {
            const auto axesConst = std::make_shared<op::v0::Constant>(element::i64, Shape{axes.size()}, axes);
            mvn = std::make_shared<opset8::MVN>(input[0], axesConst, normalizeVariance, eps, epsMode);
        } else {
            mvn = std::make_shared<opset2::MVN>(input[0], acrossChannels, normalizeVariance);
        }

        auto result = std::make_shared<Result>(mvn);
        function = std::make_shared<Function>(ResultVector{result}, ParameterVector{input});
    }
};

TEST_P(DecomposeMVNTest, CompareWithRefs) {
    Run();
}

const std::vector<InferenceEngine::Precision> netPrecisions = {InferenceEngine::Precision::FP32,
                                                               InferenceEngine::Precision::FP16};

const std::vector<std::map<std::string, std::string>> configs = {
    {{"GNA_DEVICE_MODE", "GNA_SW_FP32"}, {"GNA_SCALE_FACTOR_0", "1"}}};

const std::vector<std::vector<size_t>> inputs = {{1, 1, 5, 300}, {1, 6, 256}};
const std::vector<bool> normalizeVariance = {true};
const std::vector<float> eps = {1.0e-09f};
const std::vector<op::MVNEpsMode> epsMode = {op::MVNEpsMode::INSIDE_SQRT};
const std::vector<bool> accrossChannels = {false};

const auto mvnParams_v6 = ::testing::Combine(::testing::ValuesIn(normalizeVariance),
                                             ::testing::ValuesIn(eps),
                                             ::testing::ValuesIn(epsMode),
                                             ::testing::Values(false),
                                             ::testing::Values(true));

const auto mvnParams_v1 = ::testing::Combine(::testing::ValuesIn(normalizeVariance),
                                             ::testing::ValuesIn(eps),
                                             ::testing::ValuesIn(epsMode),
                                             ::testing::ValuesIn(accrossChannels),
                                             ::testing::Values(false));

INSTANTIATE_TEST_SUITE_P(smoke_DecomposeMVN_v6,
                         DecomposeMVNTest,
                         ::testing::Combine(mvnParams_v6,
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(ov::test::utils::DEVICE_GNA),
                                            ::testing::ValuesIn(configs),
                                            ::testing::ValuesIn(inputs)),
                         DecomposeMVNTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_DecomposeMVN_v1,
                         DecomposeMVNTest,
                         ::testing::Combine(mvnParams_v1,
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(ov::test::utils::DEVICE_GNA),
                                            ::testing::ValuesIn(configs),
                                            ::testing::ValuesIn(inputs)),
                         DecomposeMVNTest::getTestCaseName);

}  // namespace LayerTestsDefinitions
