// Copyright (C) 2018-2021 Intel Corporation
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
    std::vector<size_t>                 // Input shape
> removePermutationsPassParams;

typedef std::tuple<
    InferenceEngine::Precision,         // Network Precision
    std::string,                        // Target Device
    std::map<std::string, std::string>, // Configuration
    std::vector<size_t>,                // Input shape
    bool                                // with activation
> removePermutationsWithPoolPassParams;

namespace LayerTestsDefinitions {

class RemovePermutationsNHWCToNCHWPassTest : public testing::WithParamInterface<removePermutationsPassParams>,
                                             public LayerTestsUtils::LayerTestsCommon {
    public:
        static std::string getTestCaseName(testing::TestParamInfo<removePermutationsPassParams> obj) {
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
            result << "_IS=" << CommonTestUtils::vec2str(inputShape);
            return result.str();
        }

    protected:
        void SetUp() override {
            //      Reshape
            //          |
            //      Permute (order: [0, 3, 1, 2])
            //          |
            //      Convolution
            //          |
            //      Permute (order: [0, 2, 3, 1])
            //          |
            //      Reshape
            InferenceEngine::Precision netPrecision;
            std::vector<size_t> inputShape;
            std::tie(netPrecision, targetDevice, configuration, inputShape) = this->GetParam();
            auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

            size_t in_total_dims_size = std::accumulate(std::begin(inputShape), std::end(inputShape), 1, std::multiplies<double>());
            auto params = ngraph::builder::makeParams(ngPrc, { {1, in_total_dims_size} });

            auto pattern1 = std::make_shared<ngraph::opset1::Constant>(ngraph::element::Type_t::i64, ngraph::Shape{ 4 }, inputShape);
            auto reshape1 = std::make_shared<ngraph::opset1::Reshape>(params[0], pattern1, false);

            auto permute1 = std::make_shared<ngraph::opset1::Transpose>(reshape1,
                ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{ 4 }, { 0, 3, 1, 2 }));

            size_t num_out_channels = 12;
            size_t kernel_size = 8;
            std::vector<size_t> kernal_shape = (inputShape[1] == 1 ? std::vector<size_t>{1, kernel_size} : std::vector<size_t>{kernel_size, 1});
            auto conv1 = ngraph::builder::makeConvolution(permute1, ngPrc, kernal_shape, { 1, 1 }, { 0, 0 }, { 0, 0 }, { 1, 1 },
                                                          ngraph::op::PadType::VALID, num_out_channels);

            auto permute2 = std::make_shared<ngraph::opset1::Transpose>(conv1,
                ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{ 4 }, { 0, 2, 3, 1 }));

            size_t out_width = (inputShape[2] - kernal_shape[1]) + 1;
            size_t out_height = (inputShape[1] - kernal_shape[0]) + 1;
            std::vector<size_t> outFormShapes = { 1, out_width * out_height * num_out_channels };
            auto pattern2 = std::make_shared<ngraph::opset1::Constant>(ngraph::element::Type_t::i64, ngraph::Shape{ 2 }, outFormShapes);
            auto reshape2 = std::make_shared<ngraph::opset1::Reshape>(permute2, pattern2, false);

            ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(reshape2) };
            function = std::make_shared<ngraph::Function>(results, params, "RemovePermutationPass");
        }
};

class RemovePermutationsNHWCToNCHWPass4DOutputTest : public testing::WithParamInterface<removePermutationsPassParams>,
    public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<removePermutationsPassParams> obj) {
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
        result << "_IS=" << CommonTestUtils::vec2str(inputShape);
        return result.str();
    }

protected:
    void SetUp() override {
        InferenceEngine::Precision netPrecision;
        std::vector<size_t> inputShape;
        std::tie(netPrecision, targetDevice, configuration, inputShape) = this->GetParam();
        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

        auto params = ngraph::builder::makeParams(ngPrc, { inputShape });
        auto permute1 = std::make_shared<ngraph::opset1::Transpose>(params[0],
                             ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{ 4 }, { 0, 3, 1, 2 }));

        size_t kernal_size = 8;
        std::vector<size_t> kernal_shape = (inputShape[1] == 1 ? std::vector<size_t>{1, kernal_size} : std::vector<size_t>{kernal_size, 1});
        auto conv1 = ngraph::builder::makeConvolution(permute1, ngPrc, kernal_shape, { 1, 1 }, { 0, 0 }, { 0, 0 }, { 1, 1 }, ngraph::op::PadType::VALID, 12);

        auto permute2 = std::make_shared<ngraph::opset1::Transpose>(conv1,
                             ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{ 4 }, { 0, 2, 3, 1 }));

        ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(permute2) };

        function = std::make_shared<ngraph::Function>(results, params, "RemovePermutationPass4DOutput");
    }
};

class RemovePermutationsWithPoolAndActTest : public testing::WithParamInterface<removePermutationsWithPoolPassParams>,
                                             public LayerTestsUtils::LayerTestsCommon {
    public:
        static std::string getTestCaseName(testing::TestParamInfo<removePermutationsWithPoolPassParams> obj) {
            InferenceEngine::Precision netPrecision;
            std::string targetDevice;
            std::map<std::string, std::string> configuration;
            std::vector<size_t> inputShape;
            bool withActivation;
            std::tie(netPrecision, targetDevice, configuration, inputShape, withActivation) = obj.param;

            std::ostringstream result;
            result << "netPRC=" << netPrecision.name() << "_";
            result << "targetDevice=" << targetDevice << "_";
            for (auto const& configItem : configuration) {
                result << "_configItem=" << configItem.first << "_" << configItem.second;
            }
            result << "_IS=" << CommonTestUtils::vec2str(inputShape);
            result << "_withActivation=" << withActivation;
            return result.str();
        }

    protected:
        InferenceEngine::Blob::Ptr GenerateInput(const InferenceEngine::InputInfo& info) const override {
            InferenceEngine::Blob::Ptr blob = make_blob_with_precision(info.getTensorDesc());
            blob->allocate();

            auto* rawBlobDataPtr = blob->buffer().as<float*>();
            std::vector<float> values = CommonTestUtils::generate_float_numbers(blob->size(), -0.2f, 0.2f);
            for (size_t i = 0; i < blob->size(); i++) {
                rawBlobDataPtr[i] = values[i];
            }
            return blob;
        }

        void SetUp() override {
            //      Reshape
            //          |
            //      Permute (order: [0, 3, 1, 2])
            //          |
            //      Convolution
            //          |
            //       Pooling
            //          |
            //        Relu
            //          |
            //      Permute (order: [0, 2, 3, 1])
            //          |
            //      Reshape
            InferenceEngine::Precision netPrecision;
            std::vector<size_t> inputShape;
            bool withActivation;
            std::tie(netPrecision, targetDevice, configuration, inputShape, withActivation) = this->GetParam();
            auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

            size_t in_total_dims_size = std::accumulate(std::begin(inputShape), std::end(inputShape), 1, std::multiplies<double>());
            auto params = ngraph::builder::makeParams(ngPrc, { {1, in_total_dims_size} });

            auto pattern1 = std::make_shared<ngraph::opset1::Constant>(ngraph::element::Type_t::i64, ngraph::Shape{ 4 }, inputShape);
            auto reshape1 = std::make_shared<ngraph::opset1::Reshape>(params[0], pattern1, false);
            auto permute1 = std::make_shared<ngraph::opset1::Transpose>(reshape1,
                ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{ 4 }, { 0, 3, 1, 2 }));

            size_t num_out_channels = 12;
            size_t kernal_size = 8;
            auto kernal_shape = (inputShape[1] == 1 ? std::vector<size_t>{1, kernal_size} : std::vector<size_t>{kernal_size, 1});
            std::vector<float> filter_weights = CommonTestUtils::generate_float_numbers(num_out_channels * inputShape[3] * kernal_size,
                                                                                        -0.2f, 0.2f);
            auto conv1 = ngraph::builder::makeConvolution(permute1, ngPrc, kernal_shape, { 1, 1 }, { 0, 0 }, { 0, 0 }, { 1, 1 },
                ngraph::op::PadType::VALID, num_out_channels, false, filter_weights);
            auto pool_kernal_shape = (inputShape[1] == 1 ? std::vector<size_t>{1, 2} : std::vector<size_t>{2, 1});
            auto pool = ngraph::builder::makePooling(conv1, pool_kernal_shape, {0, 0}, {0, 0}, pool_kernal_shape, ngraph::op::RoundingType::FLOOR,
                                                     ngraph::op::PadType::VALID, false, ngraph::helpers::PoolingTypes::MAX);

            size_t out_width = ((inputShape[2] - kernal_shape[1]) + 1) / pool_kernal_shape[1];
            size_t out_height = ((inputShape[1] - kernal_shape[0]) + 1) / pool_kernal_shape[0];

            auto pool_output = pool;
            if (withActivation) {
                auto relu2 = std::make_shared<ngraph::opset3::Relu>(pool);
                pool_output = relu2;
            }

            auto permute2 = std::make_shared<ngraph::opset1::Transpose>(pool_output,
                ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{ 4 }, { 0, 2, 3, 1 }));

            std::vector<size_t> outFormShapes = { 1, out_width * out_height * num_out_channels };
            auto pattern2 = std::make_shared<ngraph::opset1::Constant>(ngraph::element::Type_t::i64, ngraph::Shape{ 2 }, outFormShapes);
            auto reshape2 = std::make_shared<ngraph::opset1::Reshape>(permute2, pattern2, false);

            ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(reshape2) };
            function = std::make_shared<ngraph::Function>(results, params, "RemovePermutationPass");
        }
};

class RemovePermutationsWithTwoConvTest : public testing::WithParamInterface<removePermutationsPassParams>,
                                          public LayerTestsUtils::LayerTestsCommon {
    public:
        static std::string getTestCaseName(testing::TestParamInfo<removePermutationsPassParams> obj) {
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
            result << "_IS=" << CommonTestUtils::vec2str(inputShape);
            return result.str();
        }

    protected:
        InferenceEngine::Blob::Ptr GenerateInput(const InferenceEngine::InputInfo& info) const override {
            InferenceEngine::Blob::Ptr blob = make_blob_with_precision(info.getTensorDesc());
            blob->allocate();

            auto* rawBlobDataPtr = blob->buffer().as<float*>();
            std::vector<float> values = CommonTestUtils::generate_float_numbers(blob->size(), 0.0f, 0.5f);
            for (size_t i = 0; i < blob->size(); i++) {
                rawBlobDataPtr[i] = values[i];
            }
            return blob;
        }

        void SetUp() override {
            //      Reshape
            //          |
            //      Permute (order: [0, 3, 1, 2])
            //          |
            //      Convolution
            //          |
            //      Convolution
            //          |
            //      Permute (order: [0, 2, 3, 1])
            //          |
            //      Reshape
            InferenceEngine::Precision netPrecision;
            std::vector<size_t> inputShape;
            std::tie(netPrecision, targetDevice, configuration, inputShape) = this->GetParam();
            auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

            size_t in_total_dims_size = std::accumulate(std::begin(inputShape), std::end(inputShape), 1, std::multiplies<double>());
            auto params = ngraph::builder::makeParams(ngPrc, { {1, in_total_dims_size} });

            auto pattern1 = std::make_shared<ngraph::opset1::Constant>(ngraph::element::Type_t::i64, ngraph::Shape{ 4 }, inputShape);
            auto reshape1 = std::make_shared<ngraph::opset1::Reshape>(params[0], pattern1, false);
            auto permute1 = std::make_shared<ngraph::opset1::Transpose>(reshape1,
                ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{ 4 }, { 0, 3, 1, 2 }));

            size_t num_out_channels = 12;
            size_t kernal_size = 8;
            std::vector<size_t> kernal_shape = (inputShape[1] == 1 ? std::vector<size_t>{1, kernal_size} : std::vector<size_t>{kernal_size, 1});
            std::vector<float> filter_weights_1 = CommonTestUtils::generate_float_numbers(num_out_channels * inputShape[3] * kernal_size,
                                                                                          0.0f, 0.5f);
            auto conv1 = ngraph::builder::makeConvolution(permute1, ngPrc, kernal_shape, { 1, 1 }, { 0, 0 }, { 0, 0 }, { 1, 1 },
                ngraph::op::PadType::VALID, num_out_channels, false, filter_weights_1);
            size_t out_width = ((inputShape[2] - kernal_shape[1]) + 1);
            size_t out_height = ((inputShape[1] - kernal_shape[0]) + 1);

            std::vector<float> filter_weights_2 = CommonTestUtils::generate_float_numbers(num_out_channels * num_out_channels * kernal_size,
                                                                                          -0.2f, 0.2f);
            auto conv2 = ngraph::builder::makeConvolution(conv1, ngPrc, kernal_shape, { 1, 1 }, { 0, 0 }, { 0, 0 }, { 1, 1 },
                ngraph::op::PadType::VALID, num_out_channels, false, filter_weights_2);
            out_width = ((out_width - kernal_shape[1]) + 1);
            out_height = ((out_height - kernal_shape[0]) + 1);

            auto permute2 = std::make_shared<ngraph::opset1::Transpose>(conv2,
                ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{ 4 }, { 0, 2, 3, 1 }));

            std::vector<size_t> outFormShapes = { 1, out_width * out_height * num_out_channels };
            auto pattern2 = std::make_shared<ngraph::opset1::Constant>(ngraph::element::Type_t::i64, ngraph::Shape{ 2 }, outFormShapes);
            auto reshape2 = std::make_shared<ngraph::opset1::Reshape>(permute2, pattern2, false);

            ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(reshape2) };
            function = std::make_shared<ngraph::Function>(results, params, "RemovePermutationPass");
        }
};

class RemovePermutationsWithEltwiseTest : public testing::WithParamInterface<removePermutationsPassParams>,
                                          public LayerTestsUtils::LayerTestsCommon {
    public:
        static std::string getTestCaseName(testing::TestParamInfo<removePermutationsPassParams> obj) {
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
            result << "_IS=" << CommonTestUtils::vec2str(inputShape);
            return result.str();
        }

    protected:
        InferenceEngine::Blob::Ptr GenerateInput(const InferenceEngine::InputInfo& info) const override {
            InferenceEngine::Blob::Ptr blob = make_blob_with_precision(info.getTensorDesc());
            blob->allocate();

            auto* rawBlobDataPtr = blob->buffer().as<float*>();
            std::vector<float> values = CommonTestUtils::generate_float_numbers(blob->size(), -0.2f, 0.2f);
            for (size_t i = 0; i < blob->size(); i++) {
                rawBlobDataPtr[i] = values[i];
            }
            return blob;
        }

        void SetUp() override {
            //      Reshape                                 Reshape
            //          |                                      |
            //      Permute (order: [0, 3, 1, 2])          Permute (order: [0, 3, 1, 2])
            //          |                                      |
            //      Convolution                            Convolution
            //          |______________________________________|
            //                              |
            //                             Add
            //                              |
            //                  Permute (order: [0, 2, 3, 1])
            //                              |
            //                            Reshape
            InferenceEngine::Precision netPrecision;
            std::vector<size_t> inputShape;
            std::tie(netPrecision, targetDevice, configuration, inputShape) = this->GetParam();
            auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

            size_t in_total_dims_size = std::accumulate(std::begin(inputShape), std::end(inputShape), 1, std::multiplies<double>());
            auto params = ngraph::builder::makeParams(ngPrc, { {1, 2 * in_total_dims_size} });
            auto split = ngraph::builder::makeSplit(params[0], ngPrc, 2, 1);
            auto in_width = inputShape[2];
            auto in_height = inputShape[1];
            auto in_channels = inputShape[3];

            auto pattern1 = std::make_shared<ngraph::opset1::Constant>(ngraph::element::Type_t::i64, ngraph::Shape{ 4 }, inputShape);
            auto reshape1 = std::make_shared<ngraph::opset1::Reshape>(split->output(0), pattern1, false);
            auto permute1 = std::make_shared<ngraph::opset1::Transpose>(reshape1,
                ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{ 4 }, { 0, 3, 1, 2 }));

            size_t num_out_channels = 12;
            size_t kernal_size = 8;
            std::vector<size_t> kernal_shape = (inputShape[1] == 1 ? std::vector<size_t>{1, kernal_size} : std::vector<size_t>{kernal_size, 1});
            std::vector<float> filter_weights_1 = CommonTestUtils::generate_float_numbers(num_out_channels * in_channels * kernal_size,
                                                                                          -0.2f, 0.2f);
            auto conv1 = ngraph::builder::makeConvolution(permute1, ngPrc, kernal_shape, { 1, 1 }, { 0, 0 }, { 0, 0 }, { 1, 1 },
                ngraph::op::PadType::VALID, num_out_channels, false, filter_weights_1);

            auto pattern2 = std::make_shared<ngraph::opset1::Constant>(ngraph::element::Type_t::i64, ngraph::Shape{ 4 }, inputShape);
            auto reshape2 = std::make_shared<ngraph::opset1::Reshape>(split->output(1), pattern2, false);
            auto permute2 = std::make_shared<ngraph::opset1::Transpose>(reshape2,
                ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{ 4 }, { 0, 3, 1, 2 }));

            std::vector<float> filter_weights_2 = CommonTestUtils::generate_float_numbers(num_out_channels * in_channels * kernal_size,
                                                                                          -0.2f, 0.2f);
            auto conv2 = ngraph::builder::makeConvolution(permute2, ngPrc, kernal_shape, { 1, 1 }, { 0, 0 }, { 0, 0 }, { 1, 1 },
                ngraph::op::PadType::VALID, num_out_channels, false, filter_weights_2);

            auto add = std::make_shared<ngraph::opset1::Add>(conv1, conv2);

            auto permute3 = std::make_shared<ngraph::opset1::Transpose>(add,
                ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{ 4 }, { 0, 2, 3, 1 }));

            size_t out_width = ((in_width - kernal_shape[1]) + 1);
            size_t out_height = ((in_height - kernal_shape[0]) + 1);
            std::vector<size_t> outFormShapes = { 1, out_width * out_height * num_out_channels };
            auto pattern3 = std::make_shared<ngraph::opset1::Constant>(ngraph::element::Type_t::i64, ngraph::Shape{ 2 }, outFormShapes);
            auto reshape3 = std::make_shared<ngraph::opset1::Reshape>(permute3, pattern3, false);

            ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(reshape3) };
            function = std::make_shared<ngraph::Function>(results, params, "RemovePermutationPass");
        }
};

    TEST_P(RemovePermutationsNHWCToNCHWPassTest, CompareWithRefImpl) {
        Run();
    };

    TEST_P(RemovePermutationsNHWCToNCHWPass4DOutputTest, CompareWithRefImpl) {
        Run();
    };

    TEST_P(RemovePermutationsWithPoolAndActTest, CompareWithRefImpl) {
        Run();
    };

    TEST_P(RemovePermutationsWithTwoConvTest, CompareWithRefImpl) {
        Run();
    };

    TEST_P(RemovePermutationsWithEltwiseTest, CompareWithRefImpl) {
        Run();
    };

    const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::FP16
    };

    const std::vector<std::map<std::string, std::string>> configs = {
        {
            {"GNA_DEVICE_MODE", "GNA_SW_EXACT"},
            {"GNA_SCALE_FACTOR_0", "327.67"}
        }
    };

    const std::vector<std::vector<size_t>> inputShapes {
        {1, 1, 168, 1},
        {1, 1, 168, 2},
        {1, 1, 168, 8},
        {1, 1, 32, 1},
        {1, 1, 32, 2},
        {1, 1, 32, 8},
        {1, 168, 1, 1},
        {1, 168, 1, 2},
        {1, 168, 1, 8},
        {1, 32, 1, 1},
        {1, 32, 1, 2},
        {1, 32, 1, 8}
    };

    INSTANTIATE_TEST_SUITE_P(smoke_PermutationPass, RemovePermutationsNHWCToNCHWPassTest,
        ::testing::Combine(
            ::testing::ValuesIn(netPrecisions),
            ::testing::Values(CommonTestUtils::DEVICE_GNA),
            ::testing::ValuesIn(configs),
            ::testing::ValuesIn(inputShapes)),
        RemovePermutationsNHWCToNCHWPassTest::getTestCaseName);

    INSTANTIATE_TEST_SUITE_P(smoke_PermutationPass, RemovePermutationsNHWCToNCHWPass4DOutputTest,
        ::testing::Combine(
            ::testing::ValuesIn(netPrecisions),
            ::testing::Values(CommonTestUtils::DEVICE_GNA),
            ::testing::ValuesIn(configs),
            ::testing::ValuesIn(inputShapes)),
        RemovePermutationsNHWCToNCHWPass4DOutputTest::getTestCaseName);

    INSTANTIATE_TEST_SUITE_P(smoke_PermutationPass, RemovePermutationsWithPoolAndActTest,
        ::testing::Combine(
            ::testing::ValuesIn(netPrecisions),
            ::testing::Values(CommonTestUtils::DEVICE_GNA),
            ::testing::ValuesIn(configs),
            ::testing::ValuesIn(inputShapes),
            ::testing::ValuesIn(std::vector<bool>{false, true})), // with activation
        RemovePermutationsWithPoolAndActTest::getTestCaseName);

    INSTANTIATE_TEST_SUITE_P(smoke_PermutationPass, RemovePermutationsWithTwoConvTest,
        ::testing::Combine(
            ::testing::ValuesIn(netPrecisions),
            ::testing::Values(CommonTestUtils::DEVICE_GNA),
            ::testing::ValuesIn(configs),
            ::testing::ValuesIn(inputShapes)),
        RemovePermutationsWithTwoConvTest::getTestCaseName);

    INSTANTIATE_TEST_SUITE_P(smoke_PermutationPass, RemovePermutationsWithEltwiseTest,
        ::testing::Combine(
            ::testing::ValuesIn(netPrecisions),
            ::testing::Values(CommonTestUtils::DEVICE_GNA),
            ::testing::ValuesIn(configs),
            ::testing::ValuesIn(inputShapes)),
        RemovePermutationsWithEltwiseTest::getTestCaseName);

} // namespace LayerTestsDefinitions

