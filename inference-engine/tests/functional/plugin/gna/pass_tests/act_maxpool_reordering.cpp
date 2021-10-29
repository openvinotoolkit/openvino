// Copyright (C) 2021 Intel Corporation
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

static std::map<ngraph::helpers::ActivationTypes, std::string> activationNames = {
        {ngraph::helpers::ActivationTypes::Sigmoid,               "Sigmoid"},
        {ngraph::helpers::ActivationTypes::Tanh,                  "Tanh"},
        {ngraph::helpers::ActivationTypes::Relu,                  "Relu"},
        {ngraph::helpers::ActivationTypes::LeakyRelu,             "LeakyRelu"},
        {ngraph::helpers::ActivationTypes::Exp,                   "Exp"},
        {ngraph::helpers::ActivationTypes::Log,                   "Log"},
        {ngraph::helpers::ActivationTypes::Sign,                  "Sign"},
        {ngraph::helpers::ActivationTypes::Abs,                   "Abs"},
        {ngraph::helpers::ActivationTypes::Clamp,                 "Clamp"}
};

typedef std::tuple<
    InferenceEngine::Precision,         // Network Precision
    std::string,                        // Target Device
    std::map<std::string, std::string>, // Configuration
    std::vector<size_t>,                // Input Shape
    bool,                               // add biases
    ngraph::helpers::ActivationTypes    // Activation type
> actMaxpoolReorderingParams;

namespace LayerTestsDefinitions {

class ActMaxpoolReordering : public testing::WithParamInterface<actMaxpoolReorderingParams>,
    public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<actMaxpoolReorderingParams> obj) {
        InferenceEngine::Precision netPrecision;
        std::string targetDevice;
        std::map<std::string, std::string> configuration;
        std::vector<size_t> inputShape;
        bool addBiases;
        ngraph::helpers::ActivationTypes actType;
        std::tie(netPrecision, targetDevice, configuration, inputShape, addBiases, actType) = obj.param;

        std::ostringstream result;
        result << "netPRC=" << netPrecision.name() << "_";
        result << "targetDevice=" << targetDevice << "_";
        for (auto const& configItem : configuration) {
            result << "_configItem=" << configItem.first << "_" << configItem.second;
        }
        result << "_inputShape=" << CommonTestUtils::vec2str(inputShape);
        result << "_bias=" << addBiases;
        result << "_actType=" << activationNames[actType];

        return result.str();
    }

protected:
    void SetUp() override {
        InferenceEngine::Precision netPrecision;
        std::vector<size_t> inputShape;
        bool addBiases;
        ngraph::helpers::ActivationTypes actType;
        std::tie(netPrecision, targetDevice, configuration, inputShape, addBiases, actType) = this->GetParam();
        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

        auto inputVector = ngraph::builder::makeParams(ngPrc, {inputShape});

        size_t num_out_channels = 12;
        size_t kernal_size = 8;
        auto conv = ngraph::builder::makeConvolution(inputVector[0], ngPrc, {1, kernal_size}, { 1, 1 }, { 0, 0 }, { 0, 0 }, { 1, 1 },
                                                     ngraph::op::PadType::VALID, num_out_channels, addBiases);

        auto activation = ngraph::builder::makeActivation(conv, ngPrc, actType);

        auto maxpool = ngraph::builder::makePooling(activation, {1, 2}, {0, 0}, {0, 0}, {1, 2}, ngraph::op::RoundingType::FLOOR,
                                                    ngraph::op::PadType::VALID, false, ngraph::helpers::PoolingTypes::MAX);

        ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(maxpool)};
        function = std::make_shared<ngraph::Function>(results, inputVector, "ActMaxpoolReordering");
    }
};

TEST_P(ActMaxpoolReordering, CompareWithRefImpl) {
    LoadNetwork();
};

const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32,
    InferenceEngine::Precision::FP16
};

const std::vector<std::map<std::string, std::string>> configs = {
    {
        {"GNA_DEVICE_MODE", "GNA_SW_EXACT"}
    }
};

const std::vector<std::vector<size_t>> inputShape = {
    {1, 1, 1, 1024},
    {1, 8, 1, 168},
};

const std::vector<bool> addBiases = {false, true};

const std::vector<ngraph::helpers::ActivationTypes> activationTypes = {
    ngraph::helpers::ActivationTypes::Sigmoid,
    ngraph::helpers::ActivationTypes::Tanh,
    ngraph::helpers::ActivationTypes::Relu,
    ngraph::helpers::ActivationTypes::Exp,
    ngraph::helpers::ActivationTypes::Log,
    ngraph::helpers::ActivationTypes::Sign,
    ngraph::helpers::ActivationTypes::Abs
};

INSTANTIATE_TEST_SUITE_P(smoke_act_maxpool_reordering, ActMaxpoolReordering,
    ::testing::Combine(
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(CommonTestUtils::DEVICE_GNA),
        ::testing::ValuesIn(configs),
        ::testing::ValuesIn(inputShape),
        ::testing::ValuesIn(addBiases),
        ::testing::ValuesIn(activationTypes)),
    ActMaxpoolReordering::getTestCaseName);
} // namespace LayerTestsDefinitions
