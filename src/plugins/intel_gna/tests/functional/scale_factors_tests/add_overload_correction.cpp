// Copyright (C) 2022 Intel Corporation
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
                   std::vector<size_t>>
    addOverloadCorrectionParams;

namespace LayerTestsDefinitions {

class AddOverloadCorrectionTest : public testing::WithParamInterface<addOverloadCorrectionParams>,
                                  public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<addOverloadCorrectionParams> obj) {
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
        result << "_IS=" << ov::test::utils::vec2str(inputShape);

        return result.str();
    }

protected:
    InferenceEngine::Blob::Ptr GenerateInput(const InferenceEngine::InputInfo& info) const override {
        InferenceEngine::Blob::Ptr blob = make_blob_with_precision(info.getTensorDesc());
        blob->allocate();

        auto* rawBlobDataPtr = blob->buffer().as<float*>();
        // generate values with different dynamic ranges for different inputs to produce integer overload after Add
        const float valueLimit = (info.name() == "input1") ? 10.0 : 1.0;
        std::vector<float> values = ov::test::utils::generate_float_numbers(blob->size(), -valueLimit, valueLimit);
        for (size_t i = 0; i < blob->size(); i++) {
            rawBlobDataPtr[i] = values[i];
        }
        return blob;
    }

    void SetUp() override {
        InferenceEngine::Precision netPrecision;
        std::vector<size_t> inputShape;

        std::tie(netPrecision, targetDevice, configuration, inputShape) = this->GetParam();
        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

        ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(inputShape)),
                                   std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(inputShape))};
        params[0]->set_friendly_name("input1");
        params[1]->set_friendly_name("input2");

        auto lowNodeIn = ngraph::builder::makeConstant<float>(ngPrc, {1}, {-10.0f});
        auto highNodeIn = ngraph::builder::makeConstant<float>(ngPrc, {1}, {10.0f});
        auto fqIn = std::make_shared<ngraph::opset8::FakeQuantize>(params[0],
                                                                   lowNodeIn,
                                                                   highNodeIn,
                                                                   lowNodeIn,
                                                                   highNodeIn,
                                                                   levels16);

        auto constant =
            ngraph::builder::makeConstant<float>(ngPrc,
                                                 inputShape,
                                                 ov::test::utils::generate_float_numbers(inputShape[1], -1.0f, 1.0f));
        auto mul = std::make_shared<ngraph::opset8::Multiply>(params[1], constant);
        auto lowNodeMul = ngraph::builder::makeConstant<float>(ngPrc, {1}, {-1.0f});
        auto highNodeMul = ngraph::builder::makeConstant<float>(ngPrc, {1}, {1.0f});
        auto fqMul = std::make_shared<ngraph::opset8::FakeQuantize>(mul,
                                                                    lowNodeMul,
                                                                    highNodeMul,
                                                                    lowNodeMul,
                                                                    highNodeMul,
                                                                    levels16);

        auto add = std::make_shared<ngraph::opset8::Add>(fqIn, fqMul);

        auto lowNodeOut = ngraph::builder::makeConstant<float>(ngPrc, {1}, {-11.0f});
        auto highNodeOut = ngraph::builder::makeConstant<float>(ngPrc, {1}, {11.0f});
        auto fqOut = std::make_shared<ngraph::opset8::FakeQuantize>(add,
                                                                    lowNodeOut,
                                                                    highNodeOut,
                                                                    lowNodeOut,
                                                                    highNodeOut,
                                                                    levels16);

        ngraph::ResultVector results{std::make_shared<ngraph::opset8::Result>(fqOut)};
        function = std::make_shared<ngraph::Function>(results, params, "AddOverloadCorrection");
    }

    const size_t levels16 = std::numeric_limits<uint16_t>::max();
};

TEST_P(AddOverloadCorrectionTest, CompareWithRefImpl) {
    Run();
};

const std::vector<InferenceEngine::Precision> netPrecisions = {InferenceEngine::Precision::FP32,
                                                               InferenceEngine::Precision::FP16};

const std::vector<std::map<std::string, std::string>> configs = {{{"GNA_DEVICE_MODE", "GNA_SW_EXACT"}}};

const std::vector<std::vector<size_t>> inputShapes = {{1, 128}};

INSTANTIATE_TEST_SUITE_P(smoke_base,
                         AddOverloadCorrectionTest,
                         ::testing::Combine(::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(ov::test::utils::DEVICE_GNA),
                                            ::testing::ValuesIn(configs),
                                            ::testing::ValuesIn(inputShapes)),
                         AddOverloadCorrectionTest::getTestCaseName);
}  // namespace LayerTestsDefinitions
