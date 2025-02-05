// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_op/lstm_sequence.hpp"

#include "transformations/op_conversions/bidirectional_sequences_decomposition.hpp"
#include "transformations/op_conversions/convert_sequences_to_tensor_iterator.hpp"
#include "openvino/pass/visualize_tree.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/lstm_sequence.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "common_test_utils/ov_test_utils.hpp"


namespace ov {
namespace test {
using ov::test::utils::SequenceTestsMode;
using ov::test::utils::InputLayerType;

std::string LSTMSequenceTest::getTestCaseName(const testing::TestParamInfo<LSTMSequenceParams> &obj) {
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
    std::string targetDevice;
    std::tie(mode, seq_lengths, batch, hidden_size, input_size, activations, clip, direction,
                WRBType, model_type, targetDevice) = obj.param;
    std::vector<std::vector<size_t>> input_shapes = {
            {{batch, input_size}, {batch, hidden_size}, {batch, hidden_size}, {4 * hidden_size, input_size},
                    {4 * hidden_size, hidden_size}, {4 * hidden_size}},
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
    result << "WRBType=" << WRBType << "_";
    result << "modelType=" << model_type.get_type_name() << "_";
    result << "targetDevice=" << targetDevice << "_";
    return result.str();
}

void LSTMSequenceTest::SetUp() {
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
        auto tensor_w = ov::test::utils::create_and_fill_tensor(model_type, W_shape);
        W = std::make_shared<ov::op::v0::Constant>(tensor_w);

        auto tensor_r = ov::test::utils::create_and_fill_tensor(model_type, R_shape);
        R = std::make_shared<ov::op::v0::Constant>(tensor_r);

        auto tensor_b = ov::test::utils::create_and_fill_tensor(model_type, B_shape);
        B = std::make_shared<ov::op::v0::Constant>(tensor_b);
    }

    auto lstm_sequence = std::make_shared<ov::op::v5::LSTMSequence>(params[0], params[1], params[2], seq_lengths_node, W, R, B, hidden_size, direction,
                                                                    std::vector<float>{}, std::vector<float>{}, activations, clip);

    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(lstm_sequence->output(0)),
                             std::make_shared<ov::op::v0::Result>(lstm_sequence->output(1)),
                             std::make_shared<ov::op::v0::Result>(lstm_sequence->output(2))};

    function = std::make_shared<ov::Model>(results, params, "lstm_sequence");
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
}

void LSTMSequenceTest::generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) {
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

        tensor = ov::test::utils::create_and_fill_tensor(func_inputs[i].get_element_type(), targetInputStaticShapes[i], in_data);

        inputs.insert({func_inputs[i].get_node_shared_ptr(), tensor});
    }}
}  // namespace test
}  // namespace ov
