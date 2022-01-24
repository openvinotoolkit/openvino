// Copyright (C) 2018-2022 Intel Corporation
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
> fqScaleFactorParams;

namespace LayerTestsDefinitions {

class TestFQScaleFactorsTest : public testing::WithParamInterface<fqScaleFactorParams>,
    public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<fqScaleFactorParams> obj) {
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

protected:
    void SetUp() override {
        InferenceEngine::Precision netPrecision;
        std::pair<float, float> inputValues;

        std::tie(netPrecision, targetDevice, configuration, inputValues) = this->GetParam();
        std::tie(inputDataMin, inputDataMax) = inputValues;
        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

        const ngraph::Shape shape = {1, 128};
        auto params = ngraph::builder::makeParams(ngPrc, {shape});

        auto lowNodeIn = ngraph::builder::makeConstant<float>(ngPrc, {1}, { inputDataMin });
        auto highNodeIn = ngraph::builder::makeConstant<float>(ngPrc, {1}, { inputDataMax });
        auto fqIn = std::make_shared<ngraph::opset8::FakeQuantize>(params[0], lowNodeIn, highNodeIn,
            lowNodeIn, highNodeIn, levels);

        auto mul = std::make_shared<ngraph::opset8::Multiply>(fqIn, params[0]);

        auto lowNodeOut = ngraph::builder::makeConstant<float>(ngPrc, {1}, { -inputDataMin * inputDataMin });
        auto highNodeOut = ngraph::builder::makeConstant<float>(ngPrc, {1}, { inputDataMax * inputDataMax });
        auto fqOut = std::make_shared<ngraph::opset8::FakeQuantize>(mul, lowNodeOut, highNodeOut,
            lowNodeOut, highNodeOut, levels);

        ngraph::ResultVector results{std::make_shared<ngraph::opset8::Result>(fqOut)};
        function = std::make_shared<ngraph::Function>(results, params, "FQWithSmallScaleFactor");
        functionRefs = ngraph::clone_function(*function);
    }

    float inputDataMax = 1.0;
    float inputDataMin = -1.0;
    size_t levels = std::numeric_limits<uint16_t>::max();
};

TEST_P(TestFQScaleFactorsTest, CompareWithRefImpl) {
    LoadNetwork();
    GenerateInputs();
    Infer();
    auto refs = CalculateRefs();
    auto results = GetOutputs();
    const auto expected = reinterpret_cast<const float*>(refs.front().second.data());
    size_t size = results.front()->size();
    auto memory = InferenceEngine::as<InferenceEngine::MemoryBlob>(results.front());
    IE_ASSERT(memory);
    const auto lockedMemory = memory->wmap();
    const auto actualBuffer = lockedMemory.as<const float*>();

    /* the absolute threshold is calculated as 1.25 * (1 / last_fq_out_scale_factor) = 1.25 * (2 * maxValue) / (levels - 1),
    the most of accuracy degradation in this model is introduced by the output scale factor of FakeQuantize,
    1 / sf is a part of the value which can be represented by one level, so we can't get more accurate resolution than this part,
    maxValue = inputDataMax * inputDataMax since this model multiplies input values with itself,
    1.25 is a reserve factor to cover other errors in this model */
    abs_threshold = 2.5 * inputDataMax * inputDataMax / (levels - 1);

    for (size_t i = 0; i < size; ++i) {
        const auto &ref = expected[i];
        const auto &res = actualBuffer[i];
        if (CommonTestUtils::ie_abs(res - ref) > abs_threshold) {
            IE_THROW() << "Absolute comparison of values expected: " << ref << " and actual: " << res
                        << " at index " << i << " with absolute threshold " << abs_threshold
                        << " failed";
        }
    }
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
    {-188.0, 188.0},
    {-90.0, 90.0},
    {-20.0, 20.0},
    {-10.0, 10.0}
};

INSTANTIATE_TEST_SUITE_P(smoke_base, TestFQScaleFactorsTest,
    ::testing::Combine(
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(CommonTestUtils::DEVICE_GNA),
        ::testing::ValuesIn(configs),
        ::testing::ValuesIn(inputValues)),
    TestFQScaleFactorsTest::getTestCaseName);
} // namespace LayerTestsDefinitions