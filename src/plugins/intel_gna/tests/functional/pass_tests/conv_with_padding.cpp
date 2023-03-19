// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <util/type_prop.hpp>

#include "../shared_tests_instances/skip_tests_check.hpp"
#include "ngraph_functions/builders.hpp"
#include "shared_test_classes/base/layer_test_utils.hpp"

typedef std::tuple<InferenceEngine::Precision,          // Network Precision
                   std::string,                         // Target Device
                   std::map<std::string, std::string>,  // Configuration
                   std::vector<size_t>,                 // Input Shape
                   std::vector<size_t>,                 // Filter Shape
                   std::vector<std::ptrdiff_t>          // Padding Size
                   >
    ConvWithPaddingParams;

namespace LayerTestsDefinitions {

class ConvWithPadding : public testing::WithParamInterface<ConvWithPaddingParams>,
                        public LayerTestsUtils::LayerTestsCommon,
                        public GnaLayerTestCheck {
public:
    static std::string getTestCaseName(testing::TestParamInfo<ConvWithPaddingParams> obj) {
        InferenceEngine::Precision precision;
        std::string targetDevice;
        std::map<std::string, std::string> configuration;
        std::vector<size_t> input_shape;
        std::vector<size_t> filter_shape;
        std::vector<std::ptrdiff_t> padding_size;

        std::tie(precision, targetDevice, configuration, input_shape, filter_shape, padding_size) = obj.param;

        std::ostringstream result;
        result << "netPRC=" << precision.name() << "_";
        result << "targetDevice=" << targetDevice << "_";

        for (auto const& configItem : configuration) {
            result << "_configItem=" << configItem.first << "_" << configItem.second;
        }
        result << "_inputShape=" << CommonTestUtils::vec2str(input_shape) << "_";
        result << "_filterShape=" << CommonTestUtils::vec2str(filter_shape) << "_";
        result << "_paddingSize=" << CommonTestUtils::vec2str(padding_size);

        return result.str();
    }

protected:
    InferenceEngine::Blob::Ptr GenerateInput(const InferenceEngine::InputInfo& info) const override {
        InferenceEngine::Blob::Ptr blob = make_blob_with_precision(info.getTensorDesc());
        blob->allocate();

        auto* rawBlobDataPtr = blob->buffer().as<float*>();
        std::vector<float> values = CommonTestUtils::generate_float_numbers(blob->size(), -0.01f, 0.01f);
        for (size_t i = 0; i < blob->size(); i++) {
            rawBlobDataPtr[i] = values[i];
        }
        return blob;
    }

    void SetUp() override {
        InferenceEngine::Precision precision;
        std::vector<size_t> input_shape;
        std::vector<size_t> filter_shape;
        std::vector<std::ptrdiff_t> padding_size;

        std::tie(precision, targetDevice, configuration, input_shape, filter_shape, padding_size) = this->GetParam();

        GnaLayerTestCheck::SetUp(targetDevice);
        if (GnaLayerTestCheck::gnaLibVersionLessThan("3.5")) {
            GTEST_SKIP() << GnaLayerTestCheck::getLastCmpResultMsg() << std::endl;
        }

        auto ng_precision = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(precision);
        auto input = std::make_shared<ngraph::opset8::Parameter>(ng_precision, ngraph::Shape{input_shape});
        auto filter = ngraph::builder::makeConstant<float>(ng_precision, filter_shape, {1.f});
        auto conv = std::make_shared<ngraph::opset8::Convolution>(input,
                                                                  filter,
                                                                  ov::Strides{1, 1},
                                                                  padding_size,
                                                                  padding_size,
                                                                  ov::Strides{});

        auto res = std::make_shared<ngraph::opset8::Result>(conv);
        function = std::make_shared<ngraph::Function>(ngraph::ResultVector{res}, ngraph::ParameterVector{input});
    }
};

using ConvWithPaddingTestPos = ConvWithPadding;
using ConvWithPaddingTestNeg = ConvWithPadding;

TEST_P(ConvWithPaddingTestPos, CompareWithRefImpl) {
    Run();
};

TEST_P(ConvWithPaddingTestNeg, CompareWithRefImpl) {
    std::string what;
    try {
        LoadNetwork();
    } catch (const std::runtime_error& re) {
        what.assign(re.what());
    } catch (const std::exception& e) {
        what.assign(e.what());
    } catch (...) {
        what.assign("Unknown failure occurred.");
    }
    EXPECT_HAS_SUBSTRING(what, std::string("Unsupported convolution input padding"));
};

const InferenceEngine::Precision net_precisions{InferenceEngine::Precision::FP32};

const std::vector<std::map<std::string, std::string>> configs_gna_3_0_to_3_5 = {
    {{"GNA_DEVICE_MODE", "GNA_SW_EXACT"}, {"GNA_EXEC_TARGET", "GNA_TARGET_3_0"}},
    {{"GNA_DEVICE_MODE", "GNA_SW_EXACT"}, {"GNA_EXEC_TARGET", "GNA_TARGET_3_5"}}};

const std::vector<std::map<std::string, std::string>> configs_gna_3_0 = {
    {{"GNA_DEVICE_MODE", "GNA_SW_EXACT"}, {"GNA_EXEC_TARGET", "GNA_TARGET_3_0"}}};

const std::vector<std::map<std::string, std::string>> configs_gna_3_5 = {
    {{"GNA_DEVICE_MODE", "GNA_SW_EXACT"}, {"GNA_EXEC_TARGET", "GNA_TARGET_3_5"}}};

const std::vector<size_t> input2D = {1, 8, 16, 16};
const std::vector<size_t> filter2D = {8, 8, 2, 2};
const std::vector<std::vector<size_t>> inputs2D_gna_3_5 = {{1, 1, 4, 16}, {1, 1, 16, 16}};
const std::vector<std::vector<size_t>> inputs1D_gna_3_5 = {{1, 1, 1, 16}};
const std::vector<std::vector<size_t>> filters1D_gna_3_5 = {{1, 1, 1, 2}, {1, 1, 1, 16}};
const std::vector<std::vector<size_t>> filters2D_mappable_to_1D_gna_3_5 = {{1, 1, 2, 16}};
const std::vector<std::ptrdiff_t> no_padding = {0, 0};
const std::vector<std::ptrdiff_t> padding1D = {0, 1};
const std::vector<std::ptrdiff_t> padding2D = {1, 1};

INSTANTIATE_TEST_SUITE_P(smoke_conv_without_padding,
                         ConvWithPaddingTestPos,
                         ::testing::Combine(::testing::Values(net_precisions),
                                            ::testing::Values(CommonTestUtils::DEVICE_GNA),
                                            ::testing::ValuesIn(configs_gna_3_0_to_3_5),
                                            ::testing::Values(input2D),
                                            ::testing::Values(filter2D),
                                            ::testing::Values(no_padding)),
                         ConvWithPaddingTestPos::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_conv_with_padding_input1D_filter1D_gna_3_5,
                         ConvWithPaddingTestPos,
                         ::testing::Combine(::testing::Values(net_precisions),
                                            ::testing::Values(CommonTestUtils::DEVICE_GNA),
                                            ::testing::ValuesIn(configs_gna_3_5),
                                            ::testing::ValuesIn(inputs1D_gna_3_5),
                                            ::testing::ValuesIn(filters1D_gna_3_5),
                                            ::testing::Values(padding1D)),
                         ConvWithPaddingTestPos::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_conv_with_padding_input2D_filter1D_gna_3_5,
                         ConvWithPaddingTestPos,
                         ::testing::Combine(::testing::Values(net_precisions),
                                            ::testing::Values(CommonTestUtils::DEVICE_GNA),
                                            ::testing::ValuesIn(configs_gna_3_5),
                                            ::testing::ValuesIn(inputs2D_gna_3_5),
                                            ::testing::ValuesIn(filters1D_gna_3_5),
                                            ::testing::Values(padding1D)),
                         ConvWithPaddingTestPos::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_conv_with_padding_2D_mappable_to_1D_gna_3_5,
                         ConvWithPaddingTestPos,
                         ::testing::Combine(::testing::Values(net_precisions),
                                            ::testing::Values(CommonTestUtils::DEVICE_GNA),
                                            ::testing::ValuesIn(configs_gna_3_5),
                                            ::testing::ValuesIn(inputs2D_gna_3_5),
                                            ::testing::ValuesIn(filters2D_mappable_to_1D_gna_3_5),
                                            ::testing::Values(padding1D)),
                         ConvWithPaddingTestPos::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_conv_with_padding_2D_gna_3_5,
                         ConvWithPaddingTestPos,
                         ::testing::Combine(::testing::Values(net_precisions),
                                            ::testing::Values(CommonTestUtils::DEVICE_GNA),
                                            ::testing::ValuesIn(configs_gna_3_5),
                                            ::testing::Values(input2D),
                                            ::testing::Values(filter2D),
                                            ::testing::Values(padding2D)),
                         ConvWithPaddingTestPos::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_expect_exception_for_conv_with_padding_when_gna_3_0,
                         ConvWithPaddingTestNeg,
                         ::testing::Combine(::testing::Values(net_precisions),
                                            ::testing::Values(CommonTestUtils::DEVICE_GNA),
                                            ::testing::ValuesIn(configs_gna_3_0),
                                            ::testing::Values(input2D),
                                            ::testing::Values(filter2D),
                                            ::testing::Values(padding2D)),
                         ConvWithPaddingTestNeg::getTestCaseName);
}  // namespace LayerTestsDefinitions
