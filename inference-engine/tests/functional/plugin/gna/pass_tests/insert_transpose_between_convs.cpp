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
    std::vector<size_t>                 // Input Shape
> insertTransposeBetweenConvsParams;

namespace LayerTestsDefinitions {

class InsertTransposeBetweenConvs : public testing::WithParamInterface<insertTransposeBetweenConvsParams>,
    public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<insertTransposeBetweenConvsParams> obj) {
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
        result << "_inputShape=" << CommonTestUtils::vec2str(inputShape);
        return result.str();
    }

    InferenceEngine::Blob::Ptr GenerateInput(const InferenceEngine::InputInfo& info) const {
        InferenceEngine::Blob::Ptr blob = make_blob_with_precision(info.getTensorDesc());
        blob->allocate();

        auto* rawBlobDataPtr = blob->buffer().as<float*>();
        std::vector<float> values = CommonTestUtils::generate_float_numbers(blob->size(), -0.2f, 0.2f);
        for (size_t i = 0; i < blob->size(); i++) {
            rawBlobDataPtr[i] = values[i];
        }
        return blob;
    }

protected:
    void SetUp() override {
        InferenceEngine::Precision netPrecision;
        std::vector<size_t> inputShape;
        std::tie(netPrecision, targetDevice, configuration, inputShape) = this->GetParam();
        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

        ngraph::Shape inputShape_2d = {inputShape[0], inputShape[1] * inputShape[2] * inputShape[3]};
        auto params = ngraph::builder::makeParams(ngPrc, {inputShape_2d});

        auto pattern1 = std::make_shared<ngraph::opset1::Constant>(ngraph::element::Type_t::i64, ngraph::Shape{ 4 }, inputShape);
        auto reshape1 = std::make_shared<ngraph::opset1::Reshape>(params[0], pattern1, false);

        size_t num_out_channels = 8;
        size_t kernal_size = 8;
        size_t stride_size = 8;
        std::vector<float> filter_weights_1 = CommonTestUtils::generate_float_numbers(num_out_channels * inputShape[1] * kernal_size,
                                                                                        -0.2f, 0.2f);
        auto conv1 = ngraph::builder::makeConvolution(reshape1, ngPrc, {1, kernal_size}, { 1, stride_size }, { 0, 0 }, { 0, 0 }, { 1, 1 },
            ngraph::op::PadType::VALID, num_out_channels, false, filter_weights_1);
        size_t out_width = ((inputShape[3] - kernal_size) / stride_size + 1);
        size_t out_height = ((inputShape[2] - 1) + 1);

        ngraph::Shape pattern2_shape = {1, 1, 1, num_out_channels * out_height * out_width};
        auto pattern2 = std::make_shared<ngraph::opset1::Constant>(ngraph::element::Type_t::i64, ngraph::Shape{ 4 }, pattern2_shape);
        auto reshape2 = std::make_shared<ngraph::opset1::Reshape>(conv1, pattern2, false);

        std::vector<float> filter_weights_2 = CommonTestUtils::generate_float_numbers(num_out_channels * kernal_size,
                                                                                      -0.2f, 0.2f);
        auto conv2 = ngraph::builder::makeConvolution(reshape2, ngPrc, {1, kernal_size}, { 1, 1 }, { 0, 0 }, { 0, 0 }, { 1, 1 },
            ngraph::op::PadType::VALID, num_out_channels, false, filter_weights_2);
        out_width = ((pattern2_shape[3] - kernal_size) + 1);
        out_height = ((pattern2_shape[2] - 1) + 1);

        ngraph::Shape pattern3_shape = {1, num_out_channels * out_height * out_width};
        auto pattern3 = std::make_shared<ngraph::opset1::Constant>(ngraph::element::Type_t::i64, ngraph::Shape{ 2 }, pattern3_shape);
        auto reshape3 = std::make_shared<ngraph::opset1::Reshape>(conv2, pattern3, false);

        ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(reshape3)};
        function = std::make_shared<ngraph::Function>(results, params, "InsertTransposeBetweenConvs");
    }
};

class InsertTransposeBetweenConvsWithPool : public testing::WithParamInterface<insertTransposeBetweenConvsParams>,
    public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<insertTransposeBetweenConvsParams> obj) {
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
        result << "_inputShape=" << CommonTestUtils::vec2str(inputShape);
        return result.str();
    }

    InferenceEngine::Blob::Ptr GenerateInput(const InferenceEngine::InputInfo& info) const {
        InferenceEngine::Blob::Ptr blob = make_blob_with_precision(info.getTensorDesc());
        blob->allocate();

        auto* rawBlobDataPtr = blob->buffer().as<float*>();
        std::vector<float> values = CommonTestUtils::generate_float_numbers(blob->size(), -0.2f, 0.2f);
        for (size_t i = 0; i < blob->size(); i++) {
            rawBlobDataPtr[i] = values[i];
        }
        return blob;
    }

protected:
    void SetUp() override {
        InferenceEngine::Precision netPrecision;
        std::vector<size_t> inputShape;
        std::tie(netPrecision, targetDevice, configuration, inputShape) = this->GetParam();
        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

        ngraph::Shape inputShape_2d = {inputShape[0], inputShape[1] * inputShape[2] * inputShape[3]};
        auto params = ngraph::builder::makeParams(ngPrc, {inputShape_2d});

        auto pattern1 = std::make_shared<ngraph::opset1::Constant>(ngraph::element::Type_t::i64, ngraph::Shape{ 4 }, inputShape);
        auto reshape1 = std::make_shared<ngraph::opset1::Reshape>(params[0], pattern1, false);

        size_t num_out_channels = 8;
        size_t kernal_size = 8;
        size_t stride_size = 8;
        std::vector<float> filter_weights_1 = CommonTestUtils::generate_float_numbers(num_out_channels * inputShape[1] * kernal_size,
                                                                                        -0.2f, 0.2f);
        auto conv1 = ngraph::builder::makeConvolution(reshape1, ngPrc, {1, kernal_size}, { 1, stride_size }, { 0, 0 }, { 0, 0 }, { 1, 1 },
            ngraph::op::PadType::VALID, num_out_channels, false, filter_weights_1);
        size_t out_width = ((inputShape[3] - kernal_size) / stride_size + 1);
        size_t out_height = ((inputShape[2] - 1) + 1);
        auto pool = ngraph::builder::makePooling(conv1, {1, 2}, {0, 0}, {0, 0}, {1, 2}, ngraph::op::RoundingType::FLOOR,
                                                     ngraph::op::PadType::VALID, false, ngraph::helpers::PoolingTypes::MAX);
        out_width /= 2;

        ngraph::Shape pattern2_shape = {1, 1, 1, num_out_channels * out_height * out_width};
        auto pattern2 = std::make_shared<ngraph::opset1::Constant>(ngraph::element::Type_t::i64, ngraph::Shape{ 4 }, pattern2_shape);
        auto reshape2 = std::make_shared<ngraph::opset1::Reshape>(pool, pattern2, false);

        std::vector<float> filter_weights_2 = CommonTestUtils::generate_float_numbers(num_out_channels * kernal_size,
                                                                                      -0.2f, 0.2f);
        auto conv2 = ngraph::builder::makeConvolution(reshape2, ngPrc, {1, kernal_size}, { 1, 1 }, { 0, 0 }, { 0, 0 }, { 1, 1 },
            ngraph::op::PadType::VALID, num_out_channels, false, filter_weights_2);
        out_width = ((pattern2_shape[3] - kernal_size) + 1);
        out_height = ((pattern2_shape[2] - 1) + 1);

        ngraph::Shape pattern3_shape = {1, num_out_channels * out_height * out_width};
        auto pattern3 = std::make_shared<ngraph::opset1::Constant>(ngraph::element::Type_t::i64, ngraph::Shape{ 2 }, pattern3_shape);
        auto reshape3 = std::make_shared<ngraph::opset1::Reshape>(conv2, pattern3, false);

        ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(reshape3)};
        function = std::make_shared<ngraph::Function>(results, params, "InsertTransposeBetweenConvs");
    }
};

TEST_P(InsertTransposeBetweenConvs, CompareWithRefImpl) {
    Run();
};

TEST_P(InsertTransposeBetweenConvsWithPool, CompareWithRefImpl) {
    Run();
};

const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32,
    InferenceEngine::Precision::FP16
};

const std::vector<std::map<std::string, std::string>> configs = {
    {
        {"GNA_DEVICE_MODE", "GNA_SW_EXACT"}
    },
    {
        {"GNA_DEVICE_MODE", "GNA_SW_FP32"}
    }
};

const std::vector<std::vector<size_t>> inputShape = {
    {1, 1, 1, 32},
    {1, 8, 1, 64},
    {1, 4, 1, 128}
};

INSTANTIATE_TEST_SUITE_P(smoke_InsertTransposeBetweenConvsTest, InsertTransposeBetweenConvs,
    ::testing::Combine(
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(CommonTestUtils::DEVICE_GNA),
        ::testing::ValuesIn(configs),
        ::testing::ValuesIn(inputShape)),
    InsertTransposeBetweenConvs::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_InsertTransposeBetweenConvsWithPoolTest, InsertTransposeBetweenConvsWithPool,
    ::testing::Combine(
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(CommonTestUtils::DEVICE_GNA),
        ::testing::ValuesIn(configs),
        ::testing::ValuesIn(inputShape)),
    InsertTransposeBetweenConvsWithPool::getTestCaseName);

} // namespace LayerTestsDefinitions
