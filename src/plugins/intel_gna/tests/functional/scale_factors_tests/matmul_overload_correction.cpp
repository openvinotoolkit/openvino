// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gmock/gmock.h>

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
                   std::vector<size_t>,                 // Input shape
                   bool,                                // Constant second input
                   bool                                 // Swap inputs
                   >
    matmulOverloadCorrectionParams;

namespace LayerTestsDefinitions {

class MatMulOverloadCorrectionNegTest : public testing::WithParamInterface<matmulOverloadCorrectionParams>,
                                        public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<matmulOverloadCorrectionParams> obj) {
        InferenceEngine::Precision netPrecision;
        std::string targetDevice;
        std::map<std::string, std::string> configuration;
        std::vector<size_t> inputShape;
        bool isSecondInputConst, swapInputs;
        std::tie(netPrecision, targetDevice, configuration, inputShape, isSecondInputConst, swapInputs) = obj.param;

        std::ostringstream result;
        result << "netPRC=" << netPrecision.name() << "_";
        result << "targetDevice=" << targetDevice << "_";
        for (auto const& configItem : configuration) {
            result << "_configItem=" << configItem.first << "_" << configItem.second;
        }
        result << "_IS=" << ov::test::utils::vec2str(inputShape);
        result << "_secondInput=" << (isSecondInputConst ? "const" : "param");
        result << "_swapInputs=" << swapInputs;

        return result.str();
    }

protected:
    void SetUp() override {
        InferenceEngine::Precision netPrecision;
        bool isSecondInputConst, swapInputs;
        std::vector<size_t> inputShape;

        std::tie(netPrecision, targetDevice, configuration, inputShape, isSecondInputConst, swapInputs) =
            this->GetParam();
        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

        const ngraph::Shape shape1 = inputShape;
        const ngraph::Shape shape2 = {1, inputShape[1] * inputShape[1]};
        const float maxInputValue = 10.0f;
        ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(shape1))};
        auto relu = std::make_shared<ngraph::opset8::Relu>(params[0]);

        std::shared_ptr<ngraph::Node> input2;
        if (isSecondInputConst) {
            input2 = ngraph::builder::makeConstant<float>(
                ngPrc,
                ngraph::Shape{shape1[1], shape1[1]},
                ov::test::utils::generate_float_numbers(shape2[1], 0.0f, maxInputValue));
        } else {
            input2 = std::make_shared<ov::op::v0::Parameter>(ngPrc, shape2);
            params.push_back(std::dynamic_pointer_cast<ngraph::opset8::Parameter>(input2));
        }

        auto lowNodeIn1 = ngraph::builder::makeConstant<float>(ngPrc, {1}, {-maxInputValue});
        auto highNodeIn1 = ngraph::builder::makeConstant<float>(ngPrc, {1}, {maxInputValue});
        auto fqIn1 = std::make_shared<ngraph::opset8::FakeQuantize>(relu,
                                                                    lowNodeIn1,
                                                                    highNodeIn1,
                                                                    lowNodeIn1,
                                                                    highNodeIn1,
                                                                    levels16);

        auto lowNodeIn2 = ngraph::builder::makeConstant<float>(ngPrc, {1}, {-maxInputValue});
        auto highNodeIn2 = ngraph::builder::makeConstant<float>(ngPrc, {1}, {maxInputValue});
        auto fqIn2 = std::make_shared<ngraph::opset8::FakeQuantize>(input2,
                                                                    lowNodeIn2,
                                                                    highNodeIn2,
                                                                    lowNodeIn2,
                                                                    highNodeIn2,
                                                                    levels16);

        std::shared_ptr<ngraph::Node> matmul_input2 = fqIn2;
        if (!isSecondInputConst) {
            auto pattern = std::make_shared<ngraph::opset8::Constant>(ngraph::element::Type_t::i64,
                                                                      ngraph::Shape{2},
                                                                      ngraph::Shape{shape1[1], shape1[1]});
            matmul_input2 = std::make_shared<ngraph::opset8::Reshape>(fqIn2, pattern, false);
        }

        auto matmul = swapInputs ? std::make_shared<ov::op::v0::MatMul>(matmul_input2, fqIn1, false, true)
                                 : std::make_shared<ov::op::v0::MatMul>(fqIn1, matmul_input2, false, true);

        auto lowNodeOut =
            ngraph::builder::makeConstant<float>(ngPrc, {1}, {-maxInputValue * maxInputValue * inputShape[1] / 10});
        auto highNodeOut =
            ngraph::builder::makeConstant<float>(ngPrc, {1}, {maxInputValue * maxInputValue * inputShape[1] / 10});
        auto fqOut = std::make_shared<ngraph::opset8::FakeQuantize>(matmul,
                                                                    lowNodeOut,
                                                                    highNodeOut,
                                                                    lowNodeOut,
                                                                    highNodeOut,
                                                                    levels32);

        ngraph::ResultVector results{std::make_shared<ngraph::opset8::Result>(fqOut)};
        function = std::make_shared<ngraph::Function>(results, params, "MatMulOverloadCorrection");
    }

    const size_t levels16 = std::numeric_limits<uint16_t>::max();
    const size_t levels32 = std::numeric_limits<uint32_t>::max();
};

TEST_P(MatMulOverloadCorrectionNegTest, CompareWithRefImpl) {
    std::stringstream what;
    std::streambuf* sbuf = std::cout.rdbuf();
    std::streambuf* ebuf = std::cerr.rdbuf();
    std::cout.rdbuf(what.rdbuf());
    std::cerr.rdbuf(what.rdbuf());
    LoadNetwork();
    const auto expected = "Potential overload correction issue at layer ";
    EXPECT_THAT(what.str(), ::testing::HasSubstr(expected));
    std::cout.rdbuf(sbuf);
    std::cerr.rdbuf(ebuf);
    Run();
};

const std::vector<InferenceEngine::Precision> netPrecisions = {InferenceEngine::Precision::FP32,
                                                               InferenceEngine::Precision::FP16};

const std::vector<std::map<std::string, std::string>> configs = {
    {{"GNA_DEVICE_MODE", "GNA_SW_EXACT"}, {"LOG_LEVEL", "LOG_WARNING"}}};

const std::vector<std::vector<size_t>> inputShapes = {{1, 128}, {1, 256}};

INSTANTIATE_TEST_SUITE_P(smoke_base,
                         MatMulOverloadCorrectionNegTest,
                         ::testing::Combine(::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(ov::test::utils::DEVICE_GNA),
                                            ::testing::ValuesIn(configs),
                                            ::testing::ValuesIn(inputShapes),
                                            ::testing::ValuesIn({true}),
                                            ::testing::ValuesIn({true, false})),
                         MatMulOverloadCorrectionNegTest::getTestCaseName);
}  // namespace LayerTestsDefinitions
