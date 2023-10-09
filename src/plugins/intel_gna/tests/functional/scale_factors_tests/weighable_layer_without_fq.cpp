// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "ov_models/builders.hpp"
#include "ov_models/utils/ov_helpers.hpp"
#include "shared_test_classes/base/layer_test_utils.hpp"

namespace SubgraphTestsDefinitions {

typedef std::tuple<InferenceEngine::Precision,         // Net precision
                   std::vector<size_t>,                // Input shape;
                   std::vector<size_t>,                // Constant shape;
                   LayerTestsUtils::TargetDevice,      // Device name
                   std::map<std::string, std::string>  // Additional backend configuration and alis name to it
                   >
    WeighableLayerWithoutFqParamsSet;

/*
 * This test emulates cases in which the ConcatAlignFilter layer is created and the model has FakeQuantize layers.
 */
class WeighableLayerWithoutFqTest : public testing::WithParamInterface<WeighableLayerWithoutFqParamsSet>,
                                    virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<WeighableLayerWithoutFqParamsSet>& obj) {
        InferenceEngine::Precision netPrecision;
        std::string targetDevice;
        std::map<std::string, std::string> config;
        std::vector<size_t> inputShape;
        std::vector<size_t> constantShape;
        std::tie(netPrecision, constantShape, inputShape, targetDevice, config) = obj.param;

        std::ostringstream result;
        result << "netPRC=" << netPrecision.name() << "_";
        result << "trgDev=" << targetDevice;
        for (auto const& configItem : config) {
            result << "_configItem=" << configItem.first << "_" << configItem.second;
        }
        return result.str();
    }

protected:
    void SetUp() override {
        std::map<std::string, std::string> config;
        InferenceEngine::Precision netPrecision;
        std::vector<size_t> inputShape;
        std::vector<size_t> constantShape;
        std::tie(netPrecision, constantShape, inputShape, targetDevice, config) = this->GetParam();
        configuration.insert(config.begin(), config.end());

        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
        ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(inputShape))};
        auto relu = std::make_shared<ngraph::opset8::Relu>(params[0]);
        auto fq1 = std::make_shared<ngraph::opset8::FakeQuantize>(
            relu,
            ngraph::opset8::Constant::create(ngraph::element::f32, {1}, {-10.}),
            ngraph::opset8::Constant::create(ngraph::element::f32, {1}, {10.}),
            ngraph::opset8::Constant::create(ngraph::element::f32, {1}, {-10.}),
            ngraph::opset8::Constant::create(ngraph::element::f32, {1}, {10.}),
            static_cast<uint32_t>(std::numeric_limits<uint16_t>::max()) + 1);
        auto constant = ngraph::builder::makeConstant(ngPrc, constantShape, std::vector<float>{}, true);
        auto fq2 = std::make_shared<ngraph::opset8::FakeQuantize>(
            constant,
            ngraph::opset8::Constant::create(ngraph::element::f32, {1}, {-10}),
            ngraph::opset8::Constant::create(ngraph::element::f32, {1}, {10.}),
            ngraph::opset8::Constant::create(ngraph::element::f32, {1}, {-10.}),
            ngraph::opset8::Constant::create(ngraph::element::f32, {1}, {10.}),
            static_cast<uint32_t>(std::numeric_limits<uint16_t>::max()) + 1);
        auto concat = ngraph::builder::makeConcat({fq1, fq2}, 1);
        function = std::make_shared<ngraph::Function>(concat, params, "WeighableLayerWithoutFq");
    }
};  // class WeighableLayerWithoutFqTest

TEST_P(WeighableLayerWithoutFqTest, CompareWithRefs) {
    Run();
}

namespace {
const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32,
    InferenceEngine::Precision::FP16,
};

const std::vector<std::vector<size_t>> inputShapes = {{{1, 5}}};

const std::vector<std::vector<size_t>> constantShapes = {{{1, 16}}};

const std::vector<std::map<std::string, std::string>> configs = {{{"GNA_DEVICE_MODE", "GNA_SW_FP32"}},
                                                                 {{"GNA_DEVICE_MODE", "GNA_SW_EXACT"}}};

INSTANTIATE_TEST_SUITE_P(smoke_WeighableLayerWithoutFqTest,
                         WeighableLayerWithoutFqTest,
                         ::testing::Combine(::testing::ValuesIn(netPrecisions),
                                            ::testing::ValuesIn(inputShapes),
                                            ::testing::ValuesIn(constantShapes),
                                            ::testing::Values(ov::test::utils::DEVICE_GNA),
                                            ::testing::ValuesIn(configs)),
                         WeighableLayerWithoutFqTest::getTestCaseName);
}  // namespace
}  // namespace SubgraphTestsDefinitions
