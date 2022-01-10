// Copyright (C) 2018-2021 Intel Corporation
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
    ngraph::helpers::EltwiseTypes       // Type of eltwise
> eltwiseParams;

namespace LayerTestsDefinitions {

class Eltwise4dBroadcast : public testing::WithParamInterface<eltwiseParams>,
                  public LayerTestsUtils::LayerTestsCommon {
    public:
        static std::string getTestCaseName(testing::TestParamInfo<eltwiseParams> obj) {
            InferenceEngine::Precision netPrecision;
            std::string targetDevice;
            std::map<std::string, std::string> configuration;
            ngraph::helpers::EltwiseTypes eltwiseType;
            std::tie(netPrecision, targetDevice, configuration, eltwiseType) = obj.param;

            std::ostringstream result;
            result << "netPRC=" << netPrecision.name() << "_";
            result << "targetDevice=" << targetDevice << "_";
            for (auto const& configItem : configuration) {
                result << "_configItem=" << configItem.first << "_" << configItem.second;
            }
            result << "_eltwiseType=" << eltwiseType;
            return result.str();
        }

    protected:
        void SetUp() override {
            InferenceEngine::Precision netPrecision;
            ngraph::helpers::EltwiseTypes eltwiseType;
            std::tie(netPrecision, targetDevice, configuration, eltwiseType) = this->GetParam();
            auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

            outPrc = InferenceEngine::Precision::FP32;

            auto params = ngraph::builder::makeParams(ngPrc, { {1, 72} });

            std::vector<size_t> outFormShapes1 = { 1, 1, 6, 12 };
            auto pattern1 = std::make_shared<ngraph::opset1::Constant>(ngraph::element::Type_t::i64, ngraph::Shape{ 4 }, outFormShapes1);
            auto reshape1 = std::make_shared<ngraph::opset1::Reshape>(params[0], pattern1, false);

            auto constant1 = ngraph::builder::makeConstant<float>(ngPrc, { 1, 1, 1, 12 }, {}, true);
            auto eltwise = ngraph::builder::makeEltwise(reshape1, constant1, eltwiseType);

            std::vector<size_t> outFormShapes2 = { 1, 72 };
            auto pattern2 = std::make_shared<ngraph::opset1::Constant>(ngraph::element::Type_t::i64, ngraph::Shape{ 2 }, outFormShapes2);
            auto reshape2 = std::make_shared<ngraph::opset1::Reshape>(eltwise, pattern2, false);

            ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(reshape2) };
            function = std::make_shared<ngraph::Function>(results, params, "Eltwise4dBroadcast");
        }
};

class Eltwise4dMultipleInput : public testing::WithParamInterface<eltwiseParams>,
    public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<eltwiseParams> obj) {
        InferenceEngine::Precision netPrecision;
        std::string targetDevice;
        std::map<std::string, std::string> configuration;
        ngraph::helpers::EltwiseTypes eltwiseType;
        std::tie(netPrecision, targetDevice, configuration, eltwiseType) = obj.param;

        std::ostringstream result;
        result << "netPRC=" << netPrecision.name() << "_";
        result << "targetDevice=" << targetDevice << "_";
        for (auto const& configItem : configuration) {
            result << "_configItem=" << configItem.first << "_" << configItem.second;
        }
        result << "_eltwiseType=" << eltwiseType;
        return result.str();
    }

protected:
    void SetUp() override {
        InferenceEngine::Precision netPrecision;
        ngraph::helpers::EltwiseTypes eltwiseType;
        std::tie(netPrecision, targetDevice, configuration, eltwiseType) = this->GetParam();
        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

        outPrc = InferenceEngine::Precision::FP32;

        auto params = ngraph::builder::makeParams(ngPrc, { {1, 72}, {1, 72} });

        std::vector<size_t> outFormShapes1 = { 1, 1, 6, 12 };
        auto pattern1 = std::make_shared<ngraph::opset1::Constant>(ngraph::element::Type_t::i64, ngraph::Shape{ 4 }, outFormShapes1);
        auto reshape1 = std::make_shared<ngraph::opset1::Reshape>(params[0], pattern1, false);

        auto reshape2 = std::make_shared<ngraph::opset1::Reshape>(params[1], pattern1, false);

        auto eltwise = ngraph::builder::makeEltwise(reshape1, reshape2, eltwiseType);

        std::vector<size_t> outFormShapes2 = { 1, 72 };
        auto pattern2 = std::make_shared<ngraph::opset1::Constant>(ngraph::element::Type_t::i64, ngraph::Shape{ 2 }, outFormShapes2);
        auto reshape3 = std::make_shared<ngraph::opset1::Reshape>(eltwise, pattern2, false);

        ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(reshape3) };
        function = std::make_shared<ngraph::Function>(results, params, "Eltwise4dMultipleInput");
    }
};

    TEST_P(Eltwise4dBroadcast, CompareWithRefImpl) {
        Run();
    };

    TEST_P(Eltwise4dMultipleInput, CompareWithRefImpl) {
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

    const std::vector<std::map<std::string, std::string>> configsMultiple = {
        {
            {"GNA_DEVICE_MODE", "GNA_SW_EXACT"},
            {"GNA_SCALE_FACTOR_0", "1638.4"},
            {"GNA_SCALE_FACTOR_1", "1638.4"}
        }
    };

    const std::vector<ngraph::helpers::EltwiseTypes> eltwiseOpTypes = {
        ngraph::helpers::EltwiseTypes::MULTIPLY,
        ngraph::helpers::EltwiseTypes::SUBTRACT,
        ngraph::helpers::EltwiseTypes::ADD
    };

    INSTANTIATE_TEST_SUITE_P(smoke_Eltwise4d, Eltwise4dBroadcast,
        ::testing::Combine(
            ::testing::ValuesIn(netPrecisions),
            ::testing::Values(CommonTestUtils::DEVICE_GNA),
            ::testing::ValuesIn(configs),
            ::testing::ValuesIn(eltwiseOpTypes)),
        Eltwise4dBroadcast::getTestCaseName);

    INSTANTIATE_TEST_SUITE_P(smoke_Eltwise4d, Eltwise4dMultipleInput,
        ::testing::Combine(
            ::testing::ValuesIn(netPrecisions),
            ::testing::Values(CommonTestUtils::DEVICE_GNA),
            ::testing::ValuesIn(configsMultiple),
            ::testing::ValuesIn(eltwiseOpTypes)),
        Eltwise4dMultipleInput::getTestCaseName);

} // namespace LayerTestsDefinitions
