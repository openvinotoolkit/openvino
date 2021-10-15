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

typedef std::tuple<
    InferenceEngine::Precision,         // Network Precision
    std::string,                        // Target Device
    std::map<std::string, std::string>, // Configuration
    std::pair<float, float>             // Input values
> tanhFqParams;

namespace LayerTestsDefinitions {

class TanhFqTest : public testing::WithParamInterface<tanhFqParams>,
    public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<tanhFqParams> obj) {
        InferenceEngine::Precision netPrecision;
        std::string targetDevice;
        std::map<std::string, std::string> configuration;
        std::pair<float, float> inputValues;
        std::tie(netPrecision, targetDevice, configuration, inputValues) = obj.param;

        std::ostringstream result;
        result << "netPRC=" << netPrecision.name() << "_";
        result << "targetDevice=" << targetDevice << "_";
        for (auto const& configItem : configuration) {
            result << "_configItem=" << configItem.first << "_" << configItem.second;
        }
        result << "_range=(" << inputValues.first << ", " << inputValues.second << ")";

        return result.str();
    }

    InferenceEngine::Blob::Ptr GenerateInput(const InferenceEngine::InputInfo& info) const override {
        InferenceEngine::Blob::Ptr blob = make_blob_with_precision(info.getTensorDesc());
        blob->allocate();

        auto* rawBlobDataPtr = blob->buffer().as<float*>();
        std::vector<float> values = CommonTestUtils::generate_float_numbers(blob->size(), inputDataMin, inputDataMax);
        for (size_t i = 0; i < blob->size(); i++) {
            rawBlobDataPtr[i] = values[i];
        }
        return blob;
    }

protected:
    void SetUp() override {
        InferenceEngine::Precision netPrecision;
        std::pair<float, float> inputValues;

        std::tie(netPrecision, targetDevice, configuration, inputValues) = this->GetParam();
        std::tie(inputDataMin, inputDataMax) = inputValues;
        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

        const ngraph::Shape shape = {1, 128};
        auto params = ngraph::builder::makeParams(ngPrc, {shape});

        auto lowNodeIn = ngraph::builder::makeConstant<float>(ngPrc, {1}, { 100 * inputDataMin });
        auto highNodeIn = ngraph::builder::makeConstant<float>(ngPrc, {1}, { 100 * inputDataMax });
        auto fqIn = std::make_shared<ngraph::opset8::FakeQuantize>(params[0], lowNodeIn, highNodeIn,
            lowNodeIn, highNodeIn, levels16);

        auto constant = ngraph::builder::makeConstant<float>(ngPrc, shape,
            CommonTestUtils::generate_float_numbers(shape[1], inputDataMin, inputDataMax));
        auto add = std::make_shared<ngraph::opset8::Add>(fqIn, constant);

        auto lowNode = ngraph::builder::makeConstant<float>(ngPrc, {1}, { 2 * inputDataMin });
        auto highNode = ngraph::builder::makeConstant<float>(ngPrc, {1}, { 2 * inputDataMax });
        auto fq = std::make_shared<ngraph::opset8::FakeQuantize>(add, lowNode, highNode,
            lowNode, highNode, levels32);

        auto tanh = std::make_shared<ngraph::opset8::Tanh>(fq);

        auto lowNodeOut = ngraph::builder::makeConstant<float>(ngPrc, {1}, { std::tanh(2 * inputDataMin) });
        auto highNodeOut = ngraph::builder::makeConstant<float>(ngPrc, {1}, { std::tanh(2 * inputDataMax) });
        auto fqOut = std::make_shared<ngraph::opset8::FakeQuantize>(tanh, lowNodeOut, highNodeOut,
            lowNodeOut, highNodeOut, levels16);

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

TEST_P(TanhFqTest, CompareWithRefImpl) {
    Run();
};

const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32,
    InferenceEngine::Precision::FP16
};

const std::vector<std::map<std::string, std::string>> configs = {
    {
        {"GNA_DEVICE_MODE", "GNA_SW_EXACT"},
    }
};

const std::vector<std::pair<float, float>> inputValues = {
    {-10.0, 10.0},
    {-5.0, 5.0},
    {-1.0, 1.0},
    {-0.04, 0.04}
};

INSTANTIATE_TEST_SUITE_P(smoke_base, TanhFqTest,
    ::testing::Combine(
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(CommonTestUtils::DEVICE_GNA),
        ::testing::ValuesIn(configs),
        ::testing::ValuesIn(inputValues)),
    TanhFqTest::getTestCaseName);
} // namespace LayerTestsDefinitions