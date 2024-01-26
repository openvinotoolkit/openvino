// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/lstmsequence_to_multilstmsequence_fusion.hpp"

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/op/abs.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/lstm_cell.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multi_lstm_sequence.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/sigmoid.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/tanh.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/pass/visualize_tree.hpp"

using namespace ov;

using LSTMSequenceFusionParam = std::tuple<int,   // rank of bias (B)
                                           int>;  // split axis

class LSTMSequenceFusionTestSuite : public testing::WithParamInterface<LSTMSequenceFusionParam>,
                                    public TransformationTestsF {};

TEST_P(LSTMSequenceFusionTestSuite, SubgraphFusedToMultiLSTMSequence) {
    size_t input_size = 3;
    size_t hidden_size = 2;
    size_t batch = 1;
    size_t cells_cnt = 2;

    {
        auto axis_0 = std::make_shared<op::v0::Constant>(element::i64, Shape{}, 0);
        auto axis_1 = std::make_shared<op::v0::Constant>(element::i64, Shape{}, 1);
        auto seq_len = std::make_shared<op::v0::Constant>(element::i64, Shape{batch}, 1);

        auto X = std::make_shared<op::v0::Parameter>(element::f32, Shape{batch, input_size});
        auto H = std::make_shared<op::v0::Parameter>(element::f32, Shape{batch, hidden_size});
        auto C = std::make_shared<op::v0::Parameter>(element::f32, Shape{batch, hidden_size});
        auto W = std::make_shared<op::v0::Parameter>(element::f32, Shape{4 * hidden_size, input_size});
        auto R = std::make_shared<op::v0::Parameter>(element::f32, Shape{4 * hidden_size, hidden_size});
        auto B = std::make_shared<op::v0::Parameter>(element::f32, Shape{4 * hidden_size});

        auto unH = std::make_shared<op::v0::Unsqueeze>(H, axis_1);
        auto unC = std::make_shared<op::v0::Unsqueeze>(C, axis_1);
        auto unW = std::make_shared<op::v0::Unsqueeze>(W, axis_0);
        auto unR = std::make_shared<op::v0::Unsqueeze>(R, axis_0);
        auto unB = std::make_shared<op::v0::Unsqueeze>(B, axis_0);

        OutputVector in_X;
        OutputVector in_A;
        for (size_t i = 0; i < cells_cnt; ++i) {
            in_X.push_back(std::make_shared<op::v0::Unsqueeze>(X, axis_1));
        }
        auto concat_X = std::make_shared<op::v0::Concat>(in_X, 1);

        auto lstm_sequence_1 = std::make_shared<op::v5::LSTMSequence>(concat_X,
                                                                      unH,
                                                                      unC,
                                                                      seq_len,
                                                                      unW,
                                                                      unR,
                                                                      unB,
                                                                      hidden_size,
                                                                      op::RecurrentSequenceDirection::FORWARD);

        auto squeeze_1 = std::make_shared<op::v0::Squeeze>(lstm_sequence_1->output(0), axis_1);

        auto axis_0_1 = std::make_shared<op::v0::Constant>(element::i64, Shape{}, 0);
        auto axis_1_1 = std::make_shared<op::v0::Constant>(element::i64, Shape{}, 1);
        auto seq_len_2 = std::make_shared<op::v0::Constant>(element::i64, Shape{batch}, 1);

        auto X_2 = std::make_shared<op::v0::Parameter>(element::f32, Shape{batch, input_size});
        auto H_2 = std::make_shared<op::v0::Parameter>(element::f32, Shape{batch, hidden_size});
        auto C_2 = std::make_shared<op::v0::Parameter>(element::f32, Shape{batch, hidden_size});
        auto W_2 = std::make_shared<op::v0::Parameter>(element::f32, Shape{4 * hidden_size, input_size});
        auto R_2 = std::make_shared<op::v0::Parameter>(element::f32, Shape{4 * hidden_size, hidden_size});
        auto B_2 = std::make_shared<op::v0::Parameter>(element::f32, Shape{4 * hidden_size});

        auto unH_2 = std::make_shared<op::v0::Unsqueeze>(H_2, axis_1_1);
        auto unC_2 = std::make_shared<op::v0::Unsqueeze>(C_2, axis_1_1);
        auto unW_2 = std::make_shared<op::v0::Unsqueeze>(W_2, axis_0_1);
        auto unR_2 = std::make_shared<op::v0::Unsqueeze>(R_2, axis_0_1);
        auto unB_2 = std::make_shared<op::v0::Unsqueeze>(B_2, axis_0_1);

        auto lstm_sequence_2 = std::make_shared<op::v5::LSTMSequence>(squeeze_1->output(0),
                                                                      unH_2,
                                                                      unC_2,
                                                                      seq_len_2,
                                                                      unW_2,
                                                                      unR_2,
                                                                      unB_2,
                                                                      hidden_size,
                                                                      op::RecurrentSequenceDirection::FORWARD);

        auto squeeze_2 = std::make_shared<op::v0::Squeeze>(lstm_sequence_2->output(0), axis_1_1);

        auto abs = std::make_shared<op::v0::Abs>(squeeze_2->output(0));
        //std::cout << "ONE\n";
        ParameterVector params = {X, H, C, W, R, B, X_2, H_2, C_2, W_2, R_2, B_2};
        model = std::make_shared<Model>(NodeVector{abs}, params);
        pass::VisualizeTree(std::string("/home/pwysocki/") + "multi.svg").run_on_model(model);
        manager.register_pass<ov::pass::LSTMSequenceToMultiLSTMSequenceFusion>();
        //std::cout << "TWO\n";
    }

    {
        size_t sequences_count = 2;
        auto axis_0 = std::make_shared<op::v0::Constant>(element::i64, Shape{}, 0);
        auto axis_1 = std::make_shared<op::v0::Constant>(element::i64, Shape{}, 1);
        auto seq_len = std::make_shared<op::v0::Constant>(element::i64, Shape{batch}, 1);

        auto X = std::make_shared<op::v0::Parameter>(element::f32, Shape{batch, input_size});
        auto H = std::make_shared<op::v0::Parameter>(element::f32, Shape{batch, hidden_size});
        auto C = std::make_shared<op::v0::Parameter>(element::f32, Shape{batch, hidden_size});
        auto W = std::make_shared<op::v0::Parameter>(element::f32, Shape{4 * hidden_size, input_size});
        auto R = std::make_shared<op::v0::Parameter>(element::f32, Shape{4 * hidden_size, hidden_size});
        auto B = std::make_shared<op::v0::Parameter>(element::f32, Shape{4 * hidden_size});
        auto A = std::make_shared<op::v0::Parameter>(element::f32, Shape{batch, 1});

        auto unH = std::make_shared<op::v0::Unsqueeze>(H, axis_1);
        auto unC = std::make_shared<op::v0::Unsqueeze>(C, axis_1);
        auto unW = std::make_shared<op::v0::Unsqueeze>(W, axis_0);
        auto unR = std::make_shared<op::v0::Unsqueeze>(R, axis_0);
        auto unB = std::make_shared<op::v0::Unsqueeze>(B, axis_0);

        OutputVector in_X;
        OutputVector in_A;
        for (size_t i = 0; i < cells_cnt; ++i) {
            in_X.push_back(std::make_shared<op::v0::Unsqueeze>(X, axis_1));
            in_A.push_back(std::make_shared<op::v0::Unsqueeze>(A, axis_1));
        }
        auto concat_X = std::make_shared<op::v0::Concat>(in_X, 1);
        auto concat_A = std::make_shared<op::v0::Concat>(in_A, 1);

        auto multi_lstm_sequence =
            std::make_shared<op::v13::MultiLSTMSequence>(concat_X,
                                                         unH,
                                                         unC,
                                                         seq_len,
                                                         unW,
                                                         unR,
                                                         unB,
                                                         sequences_count,
                                                         hidden_size,
                                                         op::RecurrentSequenceDirection::FORWARD);
        auto abs = std::make_shared<op::v0::Abs>(multi_lstm_sequence->output(0));
        //std::cout << "THREE\n";
        model_ref = std::make_shared<Model>(NodeVector{abs}, ParameterVector{X, H, C, W, R, B, A});
        pass::VisualizeTree(std::string("/home/pwysocki/") + "model_ref.svg").run_on_model(model_ref);
        manager.register_pass<ov::pass::LSTMSequenceToMultiLSTMSequenceFusion>();
        //std::cout << "FOUR\n";
    }

    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

INSTANTIATE_TEST_SUITE_P(LSTMSequenceToMultiLSTMSequenceFusion,
                         LSTMSequenceFusionTestSuite,
                         testing::Combine(testing::Values(1, 2), testing::Values(1, -1)));
