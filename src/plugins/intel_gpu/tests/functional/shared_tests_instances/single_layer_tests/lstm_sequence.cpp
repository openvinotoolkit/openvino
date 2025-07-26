// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/lstm_sequence.hpp"
#include "openvino/op/lstm_sequence.hpp"
#include "openvino/op/concat.hpp"
#include "common_test_utils/test_constants.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "common_test_utils/ov_test_utils.hpp"
#include "transformations/op_conversions/bidirectional_sequences_decomposition.hpp"
#include "transformations/op_conversions/convert_sequences_to_tensor_iterator.hpp"

namespace {
using ov::test::LSTMSequenceTest;
using ov::test::utils::SequenceTestsMode;
using ov::test::utils::InputLayerType;
class LSTMSequenceGPUTest : public LSTMSequenceTest {
     void SetUp() override {
        LSTMSequenceTest::SetUp();
        ov::test::LSTMSequenceParams params;
        params = this->GetParam();
        const auto network_precision = std::get<9>(params);
        const auto activations = std::get<5>(params);
        if (network_precision == ov::element::f16) {
            rel_threshold = 0.03f;
            abs_threshold = 0.0025f;
            if (activations == std::vector<std::string>{"tanh", "tanh", "tanh"}) {
                rel_threshold = 0.05f;
                abs_threshold = 0.005f;
            }
        }
     }
};

std::vector<ov::test::utils::SequenceTestsMode> mode{ov::test::utils::SequenceTestsMode::CONVERT_TO_TI_MAX_SEQ_LEN_CONST,
                                                     ov::test::utils::SequenceTestsMode::CONVERT_TO_TI_RAND_SEQ_LEN_CONST,
                                                     ov::test::utils::SequenceTestsMode::CONVERT_TO_TI_RAND_SEQ_LEN_PARAM,
                                                     ov::test::utils::SequenceTestsMode::PURE_SEQ_RAND_SEQ_LEN_CONST,
                                                     ov::test::utils::SequenceTestsMode::PURE_SEQ_RAND_SEQ_LEN_PARAM,
                                                     ov::test::utils::SequenceTestsMode::PURE_SEQ};
// output values increase rapidly without clip, so use only seq_lengths = 2
std::vector<size_t> seq_lengths_zero_clip{2};
std::vector<size_t> seq_lengths_clip_non_zero{20};
std::vector<size_t> batch{10};
std::vector<size_t> hidden_size{1, 4, 10};
std::vector<size_t> hidden_size_smoke{1};
std::vector<size_t> input_size{10};
std::vector<std::vector<std::string>> activations = {{"relu", "sigmoid", "tanh"}, {"sigmoid", "tanh", "tanh"},
                                                     {"tanh", "relu", "sigmoid"}, {"sigmoid", "sigmoid", "sigmoid"},
                                                     {"tanh", "tanh", "tanh"}, {"relu", "relu", "relu"}};
std::vector<std::vector<std::string>> activations_smoke = {{"relu", "sigmoid", "tanh"}};
std::vector<float> clip{0.f};
std::vector<float> clip_non_zeros{0.7f};
std::vector<ov::op::RecurrentSequenceDirection> direction = {ov::op::RecurrentSequenceDirection::FORWARD,
                                                             ov::op::RecurrentSequenceDirection::REVERSE,
                                                             ov::op::RecurrentSequenceDirection::BIDIRECTIONAL
};
std::vector<ov::element::Type> netPrecisions = {ov::element::f32,
                                                ov::element::f16};
TEST_P(LSTMSequenceGPUTest, Inference) {
    run();
};

INSTANTIATE_TEST_SUITE_P(LSTMSequenceCommonZeroClip, LSTMSequenceGPUTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(mode),
                                ::testing::ValuesIn(seq_lengths_zero_clip),
                                ::testing::ValuesIn(batch),
                                ::testing::ValuesIn(hidden_size),
                                ::testing::ValuesIn(input_size),
                                ::testing::ValuesIn(activations),
                                ::testing::ValuesIn(clip),
                                ::testing::ValuesIn(direction),
                                ::testing::Values(ov::test::utils::InputLayerType::CONSTANT),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(ov::test::utils::DEVICE_GPU)),
                        LSTMSequenceGPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(LSTMSequenceCommonZeroClipNonConstantWRB, LSTMSequenceGPUTest,
                        ::testing::Combine(
                                ::testing::Values(ov::test::utils::SequenceTestsMode::PURE_SEQ),
                                ::testing::ValuesIn(seq_lengths_zero_clip),
                                ::testing::ValuesIn(batch),
                                ::testing::ValuesIn(hidden_size),
                                ::testing::ValuesIn(input_size),
                                ::testing::ValuesIn(activations),
                                ::testing::ValuesIn(clip),
                                ::testing::ValuesIn(direction),
                                ::testing::Values(ov::test::utils::InputLayerType::PARAMETER),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(ov::test::utils::DEVICE_GPU)),
                        LSTMSequenceGPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(LSTMSequenceCommonClip, LSTMSequenceGPUTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(mode),
                                ::testing::ValuesIn(seq_lengths_clip_non_zero),
                                ::testing::ValuesIn(batch),
                                ::testing::ValuesIn(hidden_size),
                                ::testing::ValuesIn(input_size),
                                ::testing::ValuesIn(activations),
                                ::testing::ValuesIn(clip_non_zeros),
                                ::testing::ValuesIn(direction),
                                ::testing::Values(ov::test::utils::InputLayerType::CONSTANT),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(ov::test::utils::DEVICE_GPU)),
                        LSTMSequenceGPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_LSTMSequenceCommonClip, LSTMSequenceGPUTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(mode),
                                ::testing::ValuesIn(seq_lengths_clip_non_zero),
                                ::testing::ValuesIn(batch),
                                ::testing::ValuesIn(hidden_size_smoke),
                                ::testing::ValuesIn(input_size),
                                ::testing::ValuesIn(activations_smoke),
                                ::testing::ValuesIn(clip_non_zeros),
                                ::testing::ValuesIn(direction),
                                ::testing::Values(ov::test::utils::InputLayerType::CONSTANT),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(ov::test::utils::DEVICE_GPU)),
                        LSTMSequenceGPUTest::getTestCaseName);


std::vector<size_t> seq_lengths_cm{2};
std::vector<size_t> batch_cm{1};
std::vector<size_t> hidden_size_cm{128};
std::vector<size_t> input_size_cm{64, 256};
std::vector<std::vector<std::string>> activations_cm = {{"sigmoid", "tanh", "tanh"}};
std::vector<float> clip_cm{0};
std::vector<ov::element::Type> netPrecisions_cm = {ov::element::f16};

INSTANTIATE_TEST_SUITE_P(LSTMSequenceCM, LSTMSequenceGPUTest,
                        ::testing::Combine(
                                ::testing::Values(ov::test::utils::SequenceTestsMode::PURE_SEQ),
                                ::testing::ValuesIn(seq_lengths_cm),
                                ::testing::ValuesIn(batch_cm),
                                ::testing::ValuesIn(hidden_size_cm),
                                ::testing::ValuesIn(input_size_cm),
                                ::testing::ValuesIn(activations_cm),
                                ::testing::ValuesIn(clip_cm),
                                ::testing::Values(ov::op::RecurrentSequenceDirection::BIDIRECTIONAL),
                                ::testing::Values(ov::test::utils::InputLayerType::CONSTANT),
                                ::testing::ValuesIn(netPrecisions_cm),
                                ::testing::Values(ov::test::utils::DEVICE_GPU)),
                        LSTMSequenceGPUTest::getTestCaseName);

class MultipleLSTMSequenceGPUTest : public LSTMSequenceTest {
private:
    size_t max_seq_lengths;

public:
    void SetUp() override {
    SequenceTestsMode mode;
    size_t seq_lengths;
    size_t batch;
    size_t hidden_size;
    size_t input_size;
    std::vector<std::string> activations;
    std::vector<float> activations_alpha;
    std::vector<float> activations_beta;
    float clip;
    ov::op::RecurrentSequenceDirection direction;
    InputLayerType WRBType;
    ov::element::Type model_type;
    std::tie(mode, seq_lengths, batch, hidden_size, input_size, activations, clip, direction,
                WRBType, model_type, targetDevice) = this->GetParam();

    max_seq_lengths = seq_lengths;
    size_t num_directions = direction == ov::op::RecurrentSequenceDirection::BIDIRECTIONAL ? 2 : 1;
    std::vector<ov::Shape> inputShapes = {
            {batch, seq_lengths, input_size},
            {batch, num_directions, hidden_size},
            {batch, num_directions, hidden_size},
            {batch},
            {num_directions, 4 * hidden_size, input_size},
            {num_directions, 4 * hidden_size, hidden_size},
            {num_directions, 4 * hidden_size},
    };

    const auto& W_shape = inputShapes[4];
    const auto& R_shape = inputShapes[5];
    const auto& B_shape = inputShapes[6];

    std::vector<ov::Shape> param_shapes{inputShapes[0], inputShapes[1], inputShapes[2]};
    std::vector<ov::Shape> const_input_shapes;
    if (mode == SequenceTestsMode::CONVERT_TO_TI_MAX_SEQ_LEN_PARAM ||
        mode == SequenceTestsMode::CONVERT_TO_TI_RAND_SEQ_LEN_PARAM ||
        mode == SequenceTestsMode::PURE_SEQ_RAND_SEQ_LEN_PARAM) {
        param_shapes.push_back(inputShapes[3]);
    }

    if (WRBType == InputLayerType::PARAMETER) {
        param_shapes.push_back(inputShapes[4]);
        param_shapes.push_back(inputShapes[5]);
        param_shapes.push_back(inputShapes[6]);
    }
    init_input_shapes(ov::test::static_shapes_to_test_representation(param_shapes));

    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(model_type, inputDynamicShapes[0]),
                               std::make_shared<ov::op::v0::Parameter>(model_type, inputDynamicShapes[1]),
                               std::make_shared<ov::op::v0::Parameter>(model_type, inputDynamicShapes[2])};

    std::shared_ptr<ov::Node> seq_lengths_node;
    if (mode == SequenceTestsMode::CONVERT_TO_TI_MAX_SEQ_LEN_PARAM ||
        mode == SequenceTestsMode::CONVERT_TO_TI_RAND_SEQ_LEN_PARAM ||
        mode == SequenceTestsMode::PURE_SEQ_RAND_SEQ_LEN_PARAM) {
        auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, inputDynamicShapes[3]);
        seq_lengths_node = param;
        seq_lengths_node->set_friendly_name("seq_lengths");
        params.push_back(param);
    } else if (mode == SequenceTestsMode::CONVERT_TO_TI_RAND_SEQ_LEN_CONST ||
               mode == SequenceTestsMode::PURE_SEQ_RAND_SEQ_LEN_CONST) {
        ov::test::utils::InputGenerateData in_data;
        in_data.start_from = 0;
        in_data.range = seq_lengths;
        auto tensor = ov::test::utils::create_and_fill_tensor(ov::element::i64, inputShapes[3], in_data);
        seq_lengths_node = std::make_shared<ov::op::v0::Constant>(tensor);
    } else {
        std::vector<int64_t> lengths(inputShapes[3][0], seq_lengths);
        seq_lengths_node = ov::op::v0::Constant::create(ov::element::i64, inputShapes[3], lengths);
    }

    std::shared_ptr<ov::Node> W, R, B;
    if (WRBType == InputLayerType::PARAMETER) {
        auto param_num = inputDynamicShapes.size();
        const auto W_param = std::make_shared<ov::op::v0::Parameter>(model_type, inputDynamicShapes[param_num - 3]);
        const auto R_param = std::make_shared<ov::op::v0::Parameter>(model_type, inputDynamicShapes[param_num - 2]);
        const auto B_param = std::make_shared<ov::op::v0::Parameter>(model_type, inputDynamicShapes[param_num - 1]);
        W = W_param;
        R = R_param;
        B = B_param;
        params.push_back(W_param);
        params.push_back(R_param);
        params.push_back(B_param);
    } else {
        auto tensor_w = ov::test::utils::create_and_fill_tensor_real_distribution(model_type, W_shape, -1, 1, 0);
        W = std::make_shared<ov::op::v0::Constant>(tensor_w);

        auto tensor_r = ov::test::utils::create_and_fill_tensor_real_distribution(model_type, R_shape, -1, 1, 0);
        R = std::make_shared<ov::op::v0::Constant>(tensor_r);

        auto tensor_b = ov::test::utils::create_and_fill_tensor_real_distribution(model_type, B_shape, -1, 1, 0);
        B = std::make_shared<ov::op::v0::Constant>(tensor_b);
    }

    auto lstm_sequence0 = std::make_shared<ov::op::v5::LSTMSequence>(params[0], params[1], params[2], seq_lengths_node, W, R, B, hidden_size, direction,
                                                                    std::vector<float>{}, std::vector<float>{}, activations, clip);
    auto lstm_sequence1 = std::make_shared<ov::op::v5::LSTMSequence>(params[0], lstm_sequence0->output(1), lstm_sequence0->output(2),
                                                                    seq_lengths_node, W, R, B, hidden_size, direction,
                                                                    std::vector<float>{}, std::vector<float>{}, activations, clip);
    auto lstm_sequence2 = std::make_shared<ov::op::v5::LSTMSequence>(params[0], lstm_sequence0->output(1), lstm_sequence0->output(2),
                                                                    seq_lengths_node, W, R, B, hidden_size, direction,
                                                                    std::vector<float>{}, std::vector<float>{}, activations, clip);
    auto concat0 = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{lstm_sequence1->output(0), lstm_sequence2->output(0)}, 0);
    auto concat1 = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{lstm_sequence1->output(1), lstm_sequence2->output(1)}, 0);
    auto concat2 = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{lstm_sequence1->output(2), lstm_sequence2->output(2)}, 0);

    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(concat0),
                             std::make_shared<ov::op::v0::Result>(concat1),
                             std::make_shared<ov::op::v0::Result>(concat2)};

    function = std::make_shared<ov::Model>(results, params, "multiple_lstm_sequence");
    bool is_pure_sequence = mode == SequenceTestsMode::PURE_SEQ ||
                            mode == SequenceTestsMode::PURE_SEQ_RAND_SEQ_LEN_PARAM ||
                            mode == SequenceTestsMode::PURE_SEQ_RAND_SEQ_LEN_CONST;

    if (!is_pure_sequence) {
        ov::pass::Manager manager;
        if (direction == ov::op::RecurrentSequenceDirection::BIDIRECTIONAL)
            manager.register_pass<ov::pass::BidirectionalLSTMSequenceDecomposition>();
        manager.register_pass<ov::pass::ConvertLSTMSequenceToTensorIterator>();
        manager.run_passes(function);
        bool ti_found = ov::test::utils::is_tensor_iterator_exist(function);
        EXPECT_EQ(ti_found, true);
    } else {
        bool ti_found = ov::test::utils::is_tensor_iterator_exist(function);
        EXPECT_EQ(ti_found, false);
    }
    if (model_type == ov::element::f16) {
    rel_threshold = 0.03f;
    abs_threshold = 0.0025f;
        if (activations == std::vector<std::string>{"tanh", "tanh", "tanh"}) {
            rel_threshold = 0.05f;
            abs_threshold = 0.005f;
        }
    }
}

void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
    inputs.clear();
    const auto& func_inputs = function->inputs();
    ov::test::utils::InputGenerateData in_data;
    in_data.start_from = 0;
    in_data.range = 10;

    for (size_t i = 0; i < func_inputs.size(); ++i) {
        ov::Tensor tensor;

        if (i == 3) {
            in_data.range = max_seq_lengths;
        } else {
            in_data.range = 10;
        }
        if (i < 3) {
            tensor = ov::test::utils::create_and_fill_tensor_real_distribution(func_inputs[i].get_element_type(), targetInputStaticShapes[i], 0, 1, 0);
        } else {
            tensor = ov::test::utils::create_and_fill_tensor(func_inputs[i].get_element_type(), targetInputStaticShapes[i], in_data);
        }
        inputs.insert({func_inputs[i].get_node_shared_ptr(), tensor});
    }
}
};

std::vector<std::vector<std::string>> multilstmseq_activations = {{"sigmoid", "tanh", "tanh"}};

TEST_P(MultipleLSTMSequenceGPUTest, Inference) {
    run();
};

// Check whether inputs of LSTMSequence have right idx and weight optimization works properly when constant is shared by multiple LSTMSequence
// for onednn primitive
INSTANTIATE_TEST_SUITE_P(MultipleLSTMSequenceGPUTest, MultipleLSTMSequenceGPUTest,
                        ::testing::Combine(
                                ::testing::Values(ov::test::utils::SequenceTestsMode::PURE_SEQ),
                                ::testing::ValuesIn(seq_lengths_zero_clip),
                                ::testing::ValuesIn(batch),
                                ::testing::ValuesIn(hidden_size),
                                ::testing::ValuesIn(input_size),
                                ::testing::ValuesIn(multilstmseq_activations),
                                ::testing::ValuesIn(clip),
                                ::testing::Values(ov::op::RecurrentSequenceDirection::FORWARD),
                                ::testing::Values(ov::test::utils::InputLayerType::CONSTANT),
                                ::testing::Values(ov::element::f16),
                                ::testing::Values(ov::test::utils::DEVICE_GPU)),
                        LSTMSequenceGPUTest::getTestCaseName);
}  // namespace
