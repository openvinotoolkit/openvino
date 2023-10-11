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
                   std::vector<size_t>                  // input shape
                   >
    cccmParams;

namespace LayerTestsDefinitions {

class CropAfterConvolutionTest : public testing::WithParamInterface<cccmParams>,
                                 public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<cccmParams> obj) {
        InferenceEngine::Precision netPrecision;
        std::string targetDevice;
        std::map<std::string, std::string> configuration;
        std::vector<size_t> cropInputShape;
        std::tie(netPrecision, targetDevice, configuration, cropInputShape) = obj.param;

        std::ostringstream result;
        result << "netPRC=" << netPrecision.name() << "_";
        result << "targetDevice=" << targetDevice << "_";
        for (auto const& configItem : configuration) {
            result << "_configItem=" << configItem.first << "_" << configItem.second;
        }
        result << "_CIS=" << ov::test::utils::vec2str(cropInputShape);
        return result.str();
    }

protected:
    void SetUp() override {
        InferenceEngine::Precision netPrecision;
        std::vector<size_t> inputShape;
        std::tie(netPrecision, targetDevice, configuration, inputShape) = this->GetParam();

        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
        auto reshape_pattern_size = ngraph::Shape{inputShape.size()};
        auto reshape_pattern = ngraph::builder::makeConstant(ov::element::i64, reshape_pattern_size, inputShape);
        ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(inputShape))};
        auto input_reshape = std::make_shared<ngraph::opset9::Reshape>(params[0], reshape_pattern, false);

        const std::vector<size_t> filterSize{1, 1};
        const std::vector<size_t> strides{1, 1};
        const std::vector<ptrdiff_t> padsBegin{0, 0};
        const std::vector<ptrdiff_t> padsEnd{0, 0};
        const std::vector<size_t> dilations{1, 1};
        const auto pad_type = ngraph::op::PadType::EXPLICIT;
        const size_t numOutChannels = 8;
        constexpr auto c_index_in_nchw = 1;
        constexpr auto h_index_in_nchw = 2;
        const auto weights_size = ngraph::shape_size(filterSize) * numOutChannels * inputShape[c_index_in_nchw];
        auto weights_values = ov::test::utils::generate_float_numbers(weights_size, -0.2f, 0.2f);
        const auto weights2_size = ngraph::shape_size(filterSize) * numOutChannels * numOutChannels;
        auto weights2_values = ov::test::utils::generate_float_numbers(weights2_size, -0.2f, 0.2f);

        auto convolution_node = ngraph::builder::makeConvolution(input_reshape,
                                                                 ngPrc,
                                                                 filterSize,
                                                                 strides,
                                                                 padsBegin,
                                                                 padsEnd,
                                                                 dilations,
                                                                 pad_type,
                                                                 numOutChannels,
                                                                 false,
                                                                 weights_values);

        const std::vector<int64_t> crop_begin{4};
        const std::vector<int64_t> crop_end{20};
        const std::vector<int64_t> crop_stride{1};
        const std::vector<int64_t> axes{h_index_in_nchw};
        auto split_node = ngraph::builder::makeSlice(convolution_node, crop_begin, crop_end, crop_stride, axes, ngPrc);

        auto convolution_node2 = ngraph::builder::makeConvolution(split_node,
                                                                  ngPrc,
                                                                  filterSize,
                                                                  strides,
                                                                  padsBegin,
                                                                  padsEnd,
                                                                  dilations,
                                                                  pad_type,
                                                                  numOutChannels,
                                                                  false,
                                                                  weights2_values);
        ngraph::ResultVector results{std::make_shared<ngraph::opset9::Result>(convolution_node2)};
        function = std::make_shared<ngraph::Function>(results, params, "CropAfterConvolutionTest");
    }
};

TEST_P(CropAfterConvolutionTest, CompareWithRefImpl) {
    Run();
};

const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP16,
    InferenceEngine::Precision::FP32,
};

const std::vector<std::map<std::string, std::string>> configs = {
    {{"GNA_DEVICE_MODE", "GNA_SW_EXACT"}, {"GNA_EXEC_TARGET", "GNA_TARGET_3_0"}},
    {{"GNA_DEVICE_MODE", "GNA_SW_EXACT"}, {"GNA_EXEC_TARGET", "GNA_TARGET_3_5"}},
    {{"GNA_DEVICE_MODE", "GNA_SW_FP32"}}};

const std::vector<std::vector<size_t>> input_shapes{
    {1, 8, 32, 16},
    {1, 16, 32, 16},
    {1, 8, 128, 32},
    {1, 16, 32, 32},
};

INSTANTIATE_TEST_SUITE_P(smoke_crop_after_conv,
                         CropAfterConvolutionTest,
                         ::testing::Combine(::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(ov::test::utils::DEVICE_GNA),
                                            ::testing::ValuesIn(configs),
                                            ::testing::ValuesIn(input_shapes)),
                         CropAfterConvolutionTest::getTestCaseName);

}  // namespace LayerTestsDefinitions
