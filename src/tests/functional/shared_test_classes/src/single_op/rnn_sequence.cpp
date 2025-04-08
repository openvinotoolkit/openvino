// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/pass/manager.hpp"
#include "transformations/op_conversions/bidirectional_sequences_decomposition.hpp"
#include "transformations/op_conversions/convert_sequences_to_tensor_iterator.hpp"
#include "shared_test_classes/single_op/rnn_sequence.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "common_test_utils/ov_test_utils.hpp"

using ov::test::utils::InputLayerType;
using ov::test::utils::SequenceTestsMode;

namespace ov {
namespace test {

std::string RNNSequenceTest::getTestCaseName(const testing::TestParamInfo<RNNSequenceParams> &obj) {
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
    ov::element::Type model_type;
    InputLayerType WRBType;
    std::string target_device;
    std::tie(mode, seq_lengths, batch, hidden_size, input_size, activations, clip, direction, WRBType,
            model_type, target_device) = obj.param;
    std::vector<std::vector<size_t>> input_shapes = {
            {{batch, input_size}, {batch, hidden_size}, {batch, hidden_size}, {hidden_size, input_size},
                    {hidden_size, hidden_size}, {hidden_size}},
    };
    std::ostringstream result;
    result << "mode=" << mode << "_";
    result << "seq_lengths=" << seq_lengths << "_";
    result << "batch=" << batch << "_";
    result << "hidden_size=" << hidden_size << "_";
    result << "input_size=" << input_size << "_";
    result << "IS=" << ov::test::utils::vec2str(input_shapes) << "_";
    result << "activations=" << ov::test::utils::vec2str(activations) << "_";
    result << "direction=" << direction << "_";
    result << "clip=" << clip << "_";
    result << "modelType=" << model_type.to_string() << "_";
    result << "targetDevice=" << target_device;
    return result.str();
}

void RNNSequenceTest::SetUp() {
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
    std::tie(mode, seq_lengths, batch, hidden_size, input_size, activations, clip, direction, WRBType,
            model_type, targetDevice) = this->GetParam();

    size_t num_directions = direction == ov::op::RecurrentSequenceDirection::BIDIRECTIONAL ? 2 : 1;
    std::vector<ov::Shape> input_shapes = {
            {{batch, seq_lengths, input_size}, {batch, num_directions, hidden_size}, {batch},
                {num_directions, hidden_size, input_size}, {num_directions, hidden_size, hidden_size},
                {num_directions, hidden_size}},
    };

    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(model_type, ov::Shape(input_shapes[0])),
                               std::make_shared<ov::op::v0::Parameter>(model_type, ov::Shape(input_shapes[1]))};
    std::shared_ptr<ov::Node> seq_lengths_node;
    if (mode == SequenceTestsMode::CONVERT_TO_TI_MAX_SEQ_LEN_PARAM ||
        mode == SequenceTestsMode::CONVERT_TO_TI_RAND_SEQ_LEN_PARAM ||
        mode == SequenceTestsMode::PURE_SEQ_RAND_SEQ_LEN_PARAM) {
        auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, input_shapes[2]);
        param->set_friendly_name("seq_lengths");
        params.push_back(param);
        seq_lengths_node = param;
    } else if (mode == ov::test::utils::SequenceTestsMode::CONVERT_TO_TI_RAND_SEQ_LEN_CONST ||
                mode == ov::test::utils::SequenceTestsMode::PURE_SEQ_RAND_SEQ_LEN_CONST) {
        ov::test::utils::InputGenerateData in_data;
        in_data.start_from = 0;
        in_data.range = seq_lengths;
        auto tensor = ov::test::utils::create_and_fill_tensor(ov::element::i64, input_shapes[2], in_data);
        seq_lengths_node = std::make_shared<ov::op::v0::Constant>(tensor);
    } else {
        std::vector<float> lengths(batch, seq_lengths);
        seq_lengths_node = std::make_shared<ov::op::v0::Constant>(ov::element::i64, input_shapes[2], lengths);
    }

    const auto& W_shape = input_shapes[3];
    const auto& R_shape = input_shapes[4];
    const auto& B_shape = input_shapes[5];

    std::shared_ptr<ov::Node> W, R, B;
    if (WRBType == InputLayerType::PARAMETER) {
        const auto W_param = std::make_shared<ov::op::v0::Parameter>(model_type, W_shape);
        const auto R_param = std::make_shared<ov::op::v0::Parameter>(model_type, R_shape);
        const auto B_param = std::make_shared<ov::op::v0::Parameter>(model_type, B_shape);
        W = W_param;
        R = R_param;
        B = B_param;
        params.push_back(W_param);
        params.push_back(R_param);
        params.push_back(B_param);
    } else {
        const auto W_tensor = ov::test::utils::create_and_fill_tensor(model_type, W_shape);
        const auto R_tensor = ov::test::utils::create_and_fill_tensor(model_type, R_shape);
        const auto B_tensor = ov::test::utils::create_and_fill_tensor(model_type, B_shape);
        W = std::make_shared<ov::op::v0::Constant>(W_tensor);
        R = std::make_shared<ov::op::v0::Constant>(R_tensor);
        B = std::make_shared<ov::op::v0::Constant>(B_tensor);
    }

    auto rnn_sequence = std::make_shared<ov::op::v5::RNNSequence>(params[0], params[1], seq_lengths_node, W, R, B, hidden_size, direction,
                                                                activations, activations_alpha, activations_beta, clip);
    function = std::make_shared<ov::Model>(rnn_sequence->outputs(), params, "rnn_sequence");
    bool is_pure_sequence = (mode == SequenceTestsMode::PURE_SEQ ||
                             mode == SequenceTestsMode::PURE_SEQ_RAND_SEQ_LEN_PARAM ||
                             mode == SequenceTestsMode::PURE_SEQ_RAND_SEQ_LEN_CONST);
    if (!is_pure_sequence) {
        ov::pass::Manager manager;
        if (direction == ov::op::RecurrentSequenceDirection::BIDIRECTIONAL)
            manager.register_pass<ov::pass::BidirectionalRNNSequenceDecomposition>();
        manager.register_pass<ov::pass::ConvertRNNSequenceToTensorIterator>();
        manager.run_passes(function);
        bool ti_found = ov::test::utils::is_tensor_iterator_exist(function);
        EXPECT_EQ(ti_found, true);
    } else {
        bool ti_found = ov::test::utils::is_tensor_iterator_exist(function);
        EXPECT_EQ(ti_found, false);
    }
}
}  // namespace test
}  // namespace ov
