// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/gru_cell.hpp"

#include <transformations/op_conversions/gru_cell_decomposition.hpp>
#include <vector>

#include "common_test_utils/test_constants.hpp"

namespace LayerTestsDefinitions {

using ngraph::helpers::InputLayerType;

class GRUCellGNATest : public GRUCellTest {
protected:
    void SetUp() override {
        bool should_decompose;
        size_t batch;
        size_t hidden_size;
        size_t input_size;
        std::vector<std::string> activations;
        std::vector<float> activations_alpha;
        std::vector<float> activations_beta;
        float clip;
        bool linear_before_reset;
        InputLayerType WType;
        InputLayerType RType;
        InputLayerType BType;
        InferenceEngine::Precision netPrecision;
        std::tie(should_decompose,
                 batch,
                 hidden_size,
                 input_size,
                 activations,
                 clip,
                 linear_before_reset,
                 WType,
                 RType,
                 BType,
                 netPrecision,
                 targetDevice) = this->GetParam();

        std::vector<std::vector<size_t>> inputShapes = {
            {{batch, input_size},
             {batch, hidden_size},
             {3 * hidden_size, input_size},
             {3 * hidden_size, hidden_size},
             {(linear_before_reset ? 4 : 3) * hidden_size}},
        };

        ASSERT_EQ(InputLayerType::CONSTANT, WType);
        ASSERT_EQ(InputLayerType::CONSTANT, RType);
        ASSERT_EQ(InputLayerType::CONSTANT, BType);

        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
        ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(inputShapes[0])),
                                   std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(inputShapes[1]))};
        std::vector<ngraph::Shape> WRB = {inputShapes[2], inputShapes[3], inputShapes[4]};
        std::vector<float> weights_vals =
            ov::test::utils::generate_float_numbers(ngraph::shape_size(WRB[0]), -0.0001f, 0.0001f);
        std::vector<float> reccurrenceWeights_vals =
            ov::test::utils::generate_float_numbers(ngraph::shape_size(WRB[1]), -0.0001f, 0.0001f);
        std::vector<float> bias_vals =
            ov::test::utils::generate_float_numbers(ngraph::shape_size(WRB[2]), -0.0001f, 0.0001f);

        auto weightsNode = ngraph::builder::makeConstant<float>(ngPrc, WRB[0], weights_vals);
        auto reccurrenceWeightsNode = ngraph::builder::makeConstant<float>(ngPrc, WRB[1], reccurrenceWeights_vals);
        auto biasNode = ngraph::builder::makeConstant<float>(ngPrc, WRB[2], bias_vals);

        auto gru_cell = std::make_shared<ngraph::opset8::GRUCell>(params[0],
                                                                  params[1],
                                                                  weightsNode,
                                                                  reccurrenceWeightsNode,
                                                                  biasNode,
                                                                  hidden_size,
                                                                  activations,
                                                                  activations_alpha,
                                                                  activations_beta,
                                                                  clip,
                                                                  linear_before_reset);

        ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(gru_cell->output(0))};
        function = std::make_shared<ngraph::Function>(results, params, "gru_cell");
        if (should_decompose) {
            ngraph::pass::Manager m;
            m.register_pass<ov::pass::GRUCellDecomposition>();
            m.run_passes(function);
        }
    }

    InferenceEngine::Blob::Ptr GenerateInput(const InferenceEngine::InputInfo& info) const override {
        InferenceEngine::Blob::Ptr blob = make_blob_with_precision(info.getTensorDesc());
        blob->allocate();
        auto* rawBlobDataPtr = blob->buffer().as<float*>();
        std::vector<float> values = ov::test::utils::generate_float_numbers(blob->size(), -0.002f, 0.002f);
        for (size_t i = 0; i < blob->size(); i++) {
            rawBlobDataPtr[i] = values[i];
        }
        return blob;
    }
};

TEST_P(GRUCellGNATest, CompareWithRefs) {
    Run();
}

}  //  namespace LayerTestsDefinitions

using namespace LayerTestsDefinitions;

namespace {
std::vector<bool> should_decompose{false, true};
std::vector<size_t> batch{1};
std::vector<size_t> hidden_size{1, 5};
std::vector<size_t> input_size{1, 10};
std::vector<std::vector<std::string>> activations = {{"relu", "tanh"},
                                                     {"tanh", "sigmoid"},
                                                     {"sigmoid", "tanh"},
                                                     {"tanh", "relu"}};
std::vector<float> clip = {0.0f, 0.7f};
std::vector<bool> linear_before_reset = {true, false};
std::vector<InferenceEngine::Precision> netPrecisions = {InferenceEngine::Precision::FP32,
                                                         InferenceEngine::Precision::FP16};

INSTANTIATE_TEST_SUITE_P(smoke_GRUCellCommon,
                         GRUCellGNATest,
                         ::testing::Combine(::testing::ValuesIn(should_decompose),
                                            ::testing::ValuesIn(batch),
                                            ::testing::ValuesIn(hidden_size),
                                            ::testing::ValuesIn(input_size),
                                            ::testing::ValuesIn(activations),
                                            ::testing::ValuesIn(clip),
                                            ::testing::ValuesIn(linear_before_reset),
                                            ::testing::Values(InputLayerType::CONSTANT),
                                            ::testing::Values(InputLayerType::CONSTANT),
                                            ::testing::Values(InputLayerType::CONSTANT),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(ov::test::utils::DEVICE_GNA)),
                         GRUCellTest::getTestCaseName);

}  // namespace
