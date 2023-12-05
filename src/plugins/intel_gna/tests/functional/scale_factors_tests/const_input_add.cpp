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

typedef std::tuple<InferenceEngine::Precision,          // Network Precision
                   std::string,                         // Target Device
                   std::map<std::string, std::string>,  // Configuration
                   std::pair<float, float>,             // Input min/max values
                   std::pair<float, float>              // Constant min/max values
                   >
    constInputAddParams;

namespace LayerTestsDefinitions {

class ConstInputAddTest : public testing::WithParamInterface<constInputAddParams>,
                          public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<constInputAddParams> obj) {
        InferenceEngine::Precision netPrecision;
        std::string targetDevice;
        std::map<std::string, std::string> configuration;
        std::pair<float, float> inputRange;
        std::pair<float, float> constRange;
        std::tie(netPrecision, targetDevice, configuration, inputRange, constRange) = obj.param;

        std::ostringstream result;
        result << "netPRC=" << netPrecision.name() << "_";
        result << "targetDevice=" << targetDevice << "_";
        for (auto const& configItem : configuration) {
            result << "_configItem=" << configItem.first << "_" << configItem.second;
        }
        result << "_IR=" << inputRange.first << "," << inputRange.second << "_";
        result << "IR=" << constRange.first << "," << constRange.second;
        return result.str();
    }

    InferenceEngine::Blob::Ptr GenerateInput(const InferenceEngine::InputInfo& info) const override {
        return FuncTestUtils::createAndFillBlob(info.getTensorDesc(),
                                                inputMax - inputMin,
                                                inputMin,
                                                (inputMax - inputMin) / 10);
    }

protected:
    void SetUp() override {
        InferenceEngine::Precision netPrecision;
        std::pair<float, float> inputRange;
        std::pair<float, float> constRange;
        std::tie(netPrecision, targetDevice, configuration, inputRange, constRange) = this->GetParam();
        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
        std::tie(inputMin, inputMax) = inputRange;

        ngraph::Shape shape = {1, 72};
        ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(shape))};

        auto constant =
            ngraph::builder::makeConstant<float>(ngPrc, shape, {}, true, constRange.second, constRange.first);
        auto eltwise = ngraph::builder::makeEltwise(constant, params[0], ngraph::helpers::EltwiseTypes::ADD);

        ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(eltwise)};
        function = std::make_shared<ngraph::Function>(results, params, "InputConstAdd");
    }

private:
    float inputMin = 0.0;
    float inputMax = 0.0;
};

TEST_P(ConstInputAddTest, CompareWithRefImpl) {
    Run();
};

const std::vector<InferenceEngine::Precision> netPrecisions = {InferenceEngine::Precision::FP32,
                                                               InferenceEngine::Precision::FP16};

const std::vector<std::map<std::string, std::string>> configs = {{{"GNA_DEVICE_MODE", "GNA_SW_EXACT"}}};

const std::vector<std::pair<float, float>> inputRange = {{-10, 10}, {-100, 100}};

const std::vector<std::pair<float, float>> constRange = {{-10, 10}, {-0.1, 0.1}, {-1.0e-5, 1.0e-5}};

INSTANTIATE_TEST_SUITE_P(smoke_const_input_add,
                         ConstInputAddTest,
                         ::testing::Combine(::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(ov::test::utils::DEVICE_GNA),
                                            ::testing::ValuesIn(configs),
                                            ::testing::ValuesIn(inputRange),
                                            ::testing::ValuesIn(constRange)),
                         ConstInputAddTest::getTestCaseName);

}  // namespace LayerTestsDefinitions
