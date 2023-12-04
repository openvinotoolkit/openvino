// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/gru_sequence.hpp"

#include <ngraph/op/util/attr_types.hpp>
#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "transformations/op_conversions/bidirectional_sequences_decomposition.hpp"
#include "transformations/op_conversions/convert_sequences_to_tensor_iterator.hpp"

namespace LayerTestsDefinitions {

using ngraph::helpers::InputLayerType;

class GRUSequenceGNATest : public GRUSequenceTest {
protected:
    void SetUp() override {
        using namespace ngraph::helpers;
        threshold = 0.015;
        size_t seq_lengths;
        size_t batch;
        size_t hidden_size;
        size_t input_size = 10;
        std::vector<std::string> activations;
        std::vector<float> activations_alpha;
        std::vector<float> activations_beta;
        float clip;
        bool linear_before_reset;
        InputLayerType WRBType;
        ngraph::op::RecurrentSequenceDirection direction;
        InferenceEngine::Precision netPrecision;
        std::tie(m_mode,
                 seq_lengths,
                 batch,
                 hidden_size,
                 activations,
                 clip,
                 linear_before_reset,
                 direction,
                 WRBType,
                 netPrecision,
                 targetDevice) = this->GetParam();

        ASSERT_EQ(InputLayerType::CONSTANT, WRBType);

        size_t num_directions = direction == ngraph::op::RecurrentSequenceDirection::BIDIRECTIONAL ? 2 : 1;
        std::vector<std::vector<size_t>> inputShapes = {
            {{batch, seq_lengths, input_size},
             {batch, num_directions, hidden_size},
             {batch},
             {num_directions, 3 * hidden_size, input_size},
             {num_directions, 3 * hidden_size, hidden_size},
             {num_directions, (linear_before_reset ? 4 : 3) * hidden_size}},
        };
        m_max_seq_len = seq_lengths;
        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
        ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(inputShapes[0])),
                                   std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(inputShapes[1]))};

        std::vector<ngraph::Shape> WRB = {inputShapes[3], inputShapes[4], inputShapes[5], inputShapes[2]};

        std::vector<float> weights_vals =
            ov::test::utils::generate_float_numbers(ngraph::shape_size(WRB[0]), -0.0001f, 0.0001f);
        std::vector<float> reccurrenceWeights_vals =
            ov::test::utils::generate_float_numbers(ngraph::shape_size(WRB[1]), -0.0001f, 0.0001f);
        std::vector<float> bias_vals =
            ov::test::utils::generate_float_numbers(ngraph::shape_size(WRB[2]), -0.0001f, 0.0001f);

        auto weightsNode = ngraph::builder::makeConstant<float>(ngPrc, WRB[0], weights_vals);
        auto reccurrenceWeightsNode = ngraph::builder::makeConstant<float>(ngPrc, WRB[1], reccurrenceWeights_vals);
        auto biasNode = ngraph::builder::makeConstant<float>(ngPrc, WRB[2], bias_vals);

        std::vector<float> lengths(params[0]->get_partial_shape()[0].get_min_length(),
                                   params[0]->get_partial_shape()[1].get_min_length());
        std::shared_ptr<ngraph::Node> seq_length =
            ngraph::builder::makeConstant(ngraph::element::i64, WRB[3], lengths, false);

        auto gru_sequence = std::make_shared<ngraph::opset8::GRUSequence>(params[0],
                                                                          params[1],
                                                                          seq_length,
                                                                          weightsNode,
                                                                          reccurrenceWeightsNode,
                                                                          biasNode,
                                                                          hidden_size,
                                                                          direction,
                                                                          activations,
                                                                          activations_alpha,
                                                                          activations_beta,
                                                                          clip,
                                                                          linear_before_reset);

        ngraph::ResultVector results{std::make_shared<ngraph::opset8::Result>(gru_sequence->output(0)),
                                     std::make_shared<ngraph::opset8::Result>(gru_sequence->output(1))};
        function = std::make_shared<ngraph::Function>(results, params, "gru_sequence");

        bool is_pure_sequence = m_mode == SequenceTestsMode::PURE_SEQ;
        if (!is_pure_sequence) {
            ngraph::pass::Manager manager;
            if (direction == ngraph::op::RecurrentSequenceDirection::BIDIRECTIONAL)
                manager.register_pass<ov::pass::BidirectionalGRUSequenceDecomposition>();
            manager.register_pass<ov::pass::ConvertGRUSequenceToTensorIterator>();
            manager.run_passes(function);
            bool ti_found = is_tensor_iterator_exist(function);
            EXPECT_EQ(ti_found, true);
        } else {
            bool ti_found = is_tensor_iterator_exist(function);
            EXPECT_EQ(ti_found, false);
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

TEST_P(GRUSequenceGNATest, CompareWithRefs) {
    Run();
}

}  //  namespace LayerTestsDefinitions

using namespace LayerTestsDefinitions;

namespace {
std::vector<ngraph::helpers::SequenceTestsMode> mode{
    ngraph::helpers::SequenceTestsMode::CONVERT_TO_TI_MAX_SEQ_LEN_CONST,
    ngraph::helpers::SequenceTestsMode::PURE_SEQ};
// output values increase rapidly without clip, so use only seq_lengths = 1
std::vector<size_t> seq_lengths_zero_clip{1};
std::vector<size_t> seq_lengths_clip_non_zero{1};
std::vector<size_t> batch{1};
std::vector<size_t> hidden_size{1, 10};
// std::vector<size_t> input_size{10};
std::vector<std::vector<std::string>> activations = {{"relu", "tanh"},
                                                     {"tanh", "sigmoid"},
                                                     {"sigmoid", "tanh"},
                                                     {"tanh", "relu"}};
std::vector<bool> linear_before_reset = {true, false};
std::vector<float> clip{0.f};
std::vector<float> clip_non_zeros{0.7f};
std::vector<ngraph::op::RecurrentSequenceDirection> direction = {ngraph::op::RecurrentSequenceDirection::FORWARD,
                                                                 ngraph::op::RecurrentSequenceDirection::REVERSE,
                                                                 ngraph::op::RecurrentSequenceDirection::BIDIRECTIONAL};
std::vector<InferenceEngine::Precision> netPrecisions = {InferenceEngine::Precision::FP32,
                                                         InferenceEngine::Precision::FP16};

INSTANTIATE_TEST_SUITE_P(
    smoke_GRUSequenceCommonZeroClip,
    GRUSequenceGNATest,
    ::testing::Combine(::testing::ValuesIn(mode),
                       ::testing::ValuesIn(seq_lengths_zero_clip),
                       ::testing::ValuesIn(batch),
                       ::testing::ValuesIn(hidden_size),
                       // ::testing::ValuesIn(input_size), // hardcoded to 10 due to Combine supports up to 10 args
                       ::testing::ValuesIn(activations),
                       ::testing::ValuesIn(clip),
                       ::testing::ValuesIn(linear_before_reset),
                       ::testing::ValuesIn(direction),
                       ::testing::Values(InputLayerType::CONSTANT),
                       ::testing::ValuesIn(netPrecisions),
                       ::testing::Values(ov::test::utils::DEVICE_GNA)),
    GRUSequenceTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
    smoke_GRUSequenceCommonClip,
    GRUSequenceGNATest,
    ::testing::Combine(::testing::ValuesIn(mode),
                       ::testing::ValuesIn(seq_lengths_clip_non_zero),
                       ::testing::ValuesIn(batch),
                       ::testing::ValuesIn(hidden_size),
                       // ::testing::ValuesIn(input_size),  // hardcoded to 10 due to Combine supports up to 10 args
                       ::testing::ValuesIn(activations),
                       ::testing::ValuesIn(clip_non_zeros),
                       ::testing::ValuesIn(linear_before_reset),
                       ::testing::ValuesIn(direction),
                       ::testing::Values(InputLayerType::CONSTANT),
                       ::testing::ValuesIn(netPrecisions),
                       ::testing::Values(ov::test::utils::DEVICE_GNA)),
    GRUSequenceTest::getTestCaseName);

}  // namespace
