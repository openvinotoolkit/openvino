// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ie_core.hpp>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "functional_test_utils/blob_utils.hpp"
#include "ov_models/builders.hpp"
#include "ov_models/utils/ov_helpers.hpp"
#include "shared_test_classes/base/layer_test_utils.hpp"

typedef std::tuple<std::vector<std::vector<size_t>>,   // Input shape
                   InferenceEngine::Precision,         // Network Precision
                   std::string,                        // Target Device
                   std::map<std::string, std::string>  // Configuration
                   >
    ConvertMatmulToFcPassParams;

namespace LayerTestsDefinitions {
class ConvertMatmulToFcPass : public testing::WithParamInterface<ConvertMatmulToFcPassParams>,
                              public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<ConvertMatmulToFcPassParams> obj) {
        InferenceEngine::Precision netPrecision;
        std::string targetDevice;
        std::map<std::string, std::string> configuration;
        std::vector<std::vector<size_t>> inputShape;
        std::tie(inputShape, netPrecision, targetDevice, configuration) = obj.param;

        std::ostringstream result;
        result << "netPRC=" << netPrecision.name() << "_";
        result << "targetDevice=" << targetDevice << "_";
        for (auto const& configItem : configuration) {
            result << "_configItem=" << configItem.first << "_" << configItem.second;
        }
        result << "_IS=" << ov::test::utils::vec2str(inputShape[1]) << "_";
        result << "_CS=" << ov::test::utils::vec2str(inputShape[0]) << "_";
        return result.str();
    }

    InferenceEngine::Blob::Ptr GenerateInput(const InferenceEngine::InputInfo& info) const override {
        InferenceEngine::Blob::Ptr blob = make_blob_with_precision(info.getTensorDesc());
        blob->allocate();

        auto* rawBlobDataPtr = blob->buffer().as<float*>();
        std::vector<float> values = ov::test::utils::generate_float_numbers(blob->size(), -0.2f, 0.2f);
        for (size_t i = 0; i < blob->size(); i++) {
            rawBlobDataPtr[i] = values[i];
        }
        return blob;
    }

protected:
    void SetUp() override {
        InferenceEngine::Precision netPrecision;
        std::vector<std::vector<size_t>> inputShape;
        std::tie(inputShape, netPrecision, targetDevice, configuration) = this->GetParam();
        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

        ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(inputShape[1]))};
        std::vector<float> weights =
            ov::test::utils::generate_float_numbers(inputShape[0][0] * inputShape[0][1], -0.2f, 0.2f);
        auto const_mult2 = ngraph::builder::makeConstant<float>(ngPrc, inputShape[0], weights);

        auto const_eltwise = ngraph::builder::makeConstant<float>(ngPrc, {inputShape[0][0], inputShape[1][1]}, {1.0f});
        auto matmul = std::make_shared<ngraph::opset1::MatMul>(const_mult2, params[0], false, false);

        auto eltwise = std::make_shared<ngraph::opset1::Multiply>(matmul, const_eltwise);
        function = std::make_shared<ngraph::Function>(eltwise, params, "ConvertMatmulToFC");
    }
};

class ConvertMatmulToFcWithTransposesPass : public testing::WithParamInterface<ConvertMatmulToFcPassParams>,
                                            public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<ConvertMatmulToFcPassParams> obj) {
        InferenceEngine::Precision netPrecision;
        std::string targetDevice;
        std::map<std::string, std::string> configuration;
        std::vector<std::vector<size_t>> inputShape;
        std::tie(inputShape, netPrecision, targetDevice, configuration) = obj.param;

        std::ostringstream result;
        result << "netPRC=" << netPrecision.name() << "_";
        result << "targetDevice=" << targetDevice << "_";
        for (auto const& configItem : configuration) {
            result << "_configItem=" << configItem.first << "_" << configItem.second;
        }
        result << "_IS=" << ov::test::utils::vec2str(inputShape[1]) << "_";
        result << "_CS=" << ov::test::utils::vec2str(inputShape[0]) << "_";
        return result.str();
    }

    InferenceEngine::Blob::Ptr GenerateInput(const InferenceEngine::InputInfo& info) const override {
        InferenceEngine::Blob::Ptr blob = make_blob_with_precision(info.getTensorDesc());
        blob->allocate();

        auto* rawBlobDataPtr = blob->buffer().as<float*>();
        std::vector<float> values = ov::test::utils::generate_float_numbers(blob->size(), -0.1f, 0.1f);
        for (size_t i = 0; i < blob->size(); i++) {
            rawBlobDataPtr[i] = values[i];
        }
        return blob;
    }

protected:
    void SetUp() override {
        InferenceEngine::Precision netPrecision;
        std::vector<std::vector<size_t>> inputShape;
        std::tie(inputShape, netPrecision, targetDevice, configuration) = this->GetParam();
        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

        ov::ParameterVector params{
            std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape({1, inputShape[1][0] * inputShape[1][1]}))};
        auto reshape1 = std::make_shared<ngraph::opset1::Reshape>(
            params[0],
            ngraph::builder::makeConstant(ngraph::element::i64, {inputShape[1].size()}, inputShape[1]),
            false);
        auto transpose1 = std::make_shared<ngraph::opset1::Transpose>(
            reshape1,
            ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{2}, std::vector<int64_t>{1, 0}));

        std::vector<float> weights =
            ov::test::utils::generate_float_numbers(inputShape[0][0] * inputShape[0][1], -0.1f, 0.1f);
        auto const_mult2 = ngraph::builder::makeConstant<float>(ngPrc, inputShape[0], weights);
        auto matmul = std::make_shared<ngraph::opset1::MatMul>(const_mult2, transpose1, false, false);
        auto relu = std::make_shared<ngraph::opset1::Relu>(matmul);

        auto transpose2 = std::make_shared<ngraph::opset1::Transpose>(
            relu,
            ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{2}, std::vector<int64_t>{1, 0}));
        auto transpose_output_shape = transpose2->get_output_shape(0);
        ngraph::Shape output_shape = {1, transpose_output_shape[0] * transpose_output_shape[1]};
        auto reshape2 = std::make_shared<ngraph::opset1::Reshape>(
            transpose2,
            ngraph::builder::makeConstant(ngraph::element::i64, {output_shape.size()}, output_shape),
            false);

        function = std::make_shared<ngraph::Function>(reshape2, params, "ConvertMatmulToFCWithTransposes");
    }
};

TEST_P(ConvertMatmulToFcPass, CompareWithRefImpl) {
    Run();
}

TEST_P(ConvertMatmulToFcWithTransposesPass, CompareWithRefImpl) {
    Run();
}

const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32,
    InferenceEngine::Precision::FP16,
};

const std::vector<std::map<std::string, std::string>> configs = {
    {{"GNA_DEVICE_MODE", "GNA_SW_FP32"}, {"GNA_COMPACT_MODE", "NO"}},
    {{"GNA_DEVICE_MODE", "GNA_SW_EXACT"}, {"GNA_COMPACT_MODE", "NO"}}};

const std::vector<std::vector<std::vector<size_t>>> input_shapes = {{{1, 8}, {8, 1}},
                                                                    {{128, 8}, {8, 1}},
                                                                    {{8, 8}, {8, 8}},
                                                                    {{1, 16}, {16, 8}},
                                                                    {{6, 16}, {16, 8}}};

const std::vector<std::vector<std::vector<size_t>>> input_shapes_transposes = {{{1, 8}, {1, 8}},
                                                                               {{128, 8}, {1, 8}},
                                                                               {{8, 8}, {8, 8}},
                                                                               {{1, 16}, {8, 16}},
                                                                               {{6, 16}, {1, 16}}};

INSTANTIATE_TEST_SUITE_P(smoke_convert_matmul_to_fc,
                         ConvertMatmulToFcPass,
                         ::testing::Combine(::testing::ValuesIn(input_shapes),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(ov::test::utils::DEVICE_GNA),
                                            ::testing::ValuesIn(configs)),
                         ConvertMatmulToFcPass::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_convert_matmul_to_fc,
                         ConvertMatmulToFcWithTransposesPass,
                         ::testing::Combine(::testing::ValuesIn(input_shapes_transposes),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(ov::test::utils::DEVICE_GNA),
                                            ::testing::ValuesIn(configs)),
                         ConvertMatmulToFcWithTransposesPass::getTestCaseName);
}  // namespace LayerTestsDefinitions
