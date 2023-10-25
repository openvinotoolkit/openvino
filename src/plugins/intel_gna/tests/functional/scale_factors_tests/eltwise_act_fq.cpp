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

static std::map<ngraph::helpers::ActivationTypes, std::string> activationNames = {
    {ngraph::helpers::ActivationTypes::Sigmoid, "Sigmoid"},
    {ngraph::helpers::ActivationTypes::Tanh, "Tanh"},
    {ngraph::helpers::ActivationTypes::Relu, "Relu"},
    {ngraph::helpers::ActivationTypes::Exp, "Exp"},
    {ngraph::helpers::ActivationTypes::Log, "Log"},
    {ngraph::helpers::ActivationTypes::Sign, "Sign"},
    {ngraph::helpers::ActivationTypes::Abs, "Abs"}};

typedef std::tuple<InferenceEngine::Precision,          // Network Precision
                   std::string,                         // Target Device
                   std::map<std::string, std::string>,  // Configuration
                   std::pair<float, float>,             // Input values
                   ngraph::helpers::ActivationTypes     // Activation type
                   >
    eltwiseActFqParams;

namespace LayerTestsDefinitions {

class EltwiseActFqTest : public testing::WithParamInterface<eltwiseActFqParams>,
                         public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<eltwiseActFqParams> obj) {
        InferenceEngine::Precision netPrecision;
        std::string targetDevice;
        std::map<std::string, std::string> configuration;
        std::pair<float, float> inputValues;
        ngraph::helpers::ActivationTypes act;
        std::tie(netPrecision, targetDevice, configuration, inputValues, act) = obj.param;

        std::ostringstream result;
        result << "netPRC=" << netPrecision.name() << "_";
        result << "targetDevice=" << targetDevice << "_";
        for (auto const& configItem : configuration) {
            result << "_configItem=" << configItem.first << "_" << configItem.second;
        }
        result << "_range=(" << inputValues.first << ", " << inputValues.second << ")";
        result << "_act=" << activationNames[act];

        return result.str();
    }

    InferenceEngine::Blob::Ptr GenerateInput(const InferenceEngine::InputInfo& info) const override {
        InferenceEngine::Blob::Ptr blob = make_blob_with_precision(info.getTensorDesc());
        blob->allocate();

        auto* rawBlobDataPtr = blob->buffer().as<float*>();
        std::vector<float> values = ov::test::utils::generate_float_numbers(blob->size(), inputDataMin, inputDataMax);
        for (size_t i = 0; i < blob->size(); i++) {
            rawBlobDataPtr[i] = values[i];
        }
        return blob;
    }

protected:
    void SetUp() override {
        InferenceEngine::Precision netPrecision;
        std::pair<float, float> inputValues;
        ngraph::helpers::ActivationTypes act;

        std::tie(netPrecision, targetDevice, configuration, inputValues, act) = this->GetParam();
        std::tie(inputDataMin, inputDataMax) = inputValues;
        if (act == ngraph::helpers::ActivationTypes::Log) {
            // clamp not positive values
            inputDataMin = 1.0e-3;
            // get error threshold value from PWL error
            threshold = std::stof(configuration["GNA_PWL_MAX_ERROR_PERCENT"]);
        }
        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

        const ngraph::Shape shape = {1, 128};
        ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(shape))};

        auto lowNodeIn = ngraph::builder::makeConstant<float>(ngPrc, {1}, {100 * -inputDataMax});
        auto highNodeIn = ngraph::builder::makeConstant<float>(ngPrc, {1}, {100 * inputDataMax});
        auto fqIn = std::make_shared<ngraph::opset8::FakeQuantize>(params[0],
                                                                   lowNodeIn,
                                                                   highNodeIn,
                                                                   lowNodeIn,
                                                                   highNodeIn,
                                                                   levels16);

        auto constant = ngraph::builder::makeConstant<float>(
            ngPrc,
            shape,
            ov::test::utils::generate_float_numbers(shape[1], inputDataMin, inputDataMax));
        auto add = std::make_shared<ngraph::opset8::Add>(fqIn, constant);

        auto lowNode = ngraph::builder::makeConstant<float>(ngPrc, {1}, {2 * inputDataMin});
        auto highNode = ngraph::builder::makeConstant<float>(ngPrc, {1}, {2 * inputDataMax});
        auto fq = std::make_shared<ngraph::opset8::FakeQuantize>(add, lowNode, highNode, lowNode, highNode, levels32);

        auto tanh = ngraph::builder::makeActivation(fq, ngPrc, act);

        auto lowNodeOut = ngraph::builder::makeConstant<float>(ngPrc, {1}, {std::tanh(2 * inputDataMin)});
        auto highNodeOut = ngraph::builder::makeConstant<float>(ngPrc, {1}, {std::tanh(2 * inputDataMax)});
        auto fqOut = std::make_shared<ngraph::opset8::FakeQuantize>(tanh,
                                                                    lowNodeOut,
                                                                    highNodeOut,
                                                                    lowNodeOut,
                                                                    highNodeOut,
                                                                    levels16);

        ngraph::ResultVector results{std::make_shared<ngraph::opset8::Result>(fqOut)};
        function = std::make_shared<ngraph::Function>(results, params, "TanhFq");
    }

    float inputDataMax = 1.0;
    float inputDataMin = -1.0;
    const size_t levels16 = std::numeric_limits<uint16_t>::max();
    const size_t levels32 = std::numeric_limits<uint32_t>::max();
    // to reproduce the problem with quite big distance between min int and min value from stats
    const size_t sf_reducer = 100;
};

TEST_P(EltwiseActFqTest, CompareWithRefImpl) {
    Run();
};

const std::vector<InferenceEngine::Precision> netPrecisions = {InferenceEngine::Precision::FP32,
                                                               InferenceEngine::Precision::FP16};

const std::vector<std::map<std::string, std::string>> configs = {
    {{"GNA_DEVICE_MODE", "GNA_SW_EXACT"}, {"GNA_PWL_MAX_ERROR_PERCENT", "0.025"}}};

const std::vector<std::pair<float, float>> inputValues = {{-10.0, 10.0}, {-5.0, 5.0}, {-1.0, 1.0}, {-0.04, 0.04}};

const std::vector<ngraph::helpers::ActivationTypes> activationTypes = {ngraph::helpers::ActivationTypes::Sigmoid,
                                                                       ngraph::helpers::ActivationTypes::Tanh,
                                                                       ngraph::helpers::ActivationTypes::Relu,
                                                                       ngraph::helpers::ActivationTypes::Exp,
                                                                       ngraph::helpers::ActivationTypes::Log,
                                                                       ngraph::helpers::ActivationTypes::Sign,
                                                                       ngraph::helpers::ActivationTypes::Abs};

INSTANTIATE_TEST_SUITE_P(smoke_base,
                         EltwiseActFqTest,
                         ::testing::Combine(::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(ov::test::utils::DEVICE_GNA),
                                            ::testing::ValuesIn(configs),
                                            ::testing::ValuesIn(inputValues),
                                            ::testing::ValuesIn(activationTypes)),
                         EltwiseActFqTest::getTestCaseName);
}  // namespace LayerTestsDefinitions
