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

using namespace ov;

using LSTMSequenceFusionParam = std::tuple<int,   // rank of bias (B)
                                           int>;  // split axis

class LSTMSequenceFusionTestSuite : public testing::WithParamInterface<LSTMSequenceFusionParam>,
                                    public TransformationTestsF {};

TEST_P(LSTMSequenceFusionTestSuite, SubgraphFusedToMultiLSTMSequence) {
    size_t input_size = 3;
    size_t hidden_size = 2;

    {
        auto X_1 = std::make_shared<op::v0::Parameter>(element::f32, Shape{1, input_size});
        auto H_1 = std::make_shared<op::v0::Parameter>(element::f32, Shape{1, hidden_size});
        auto C_1 = std::make_shared<op::v0::Parameter>(element::f32, Shape{1, hidden_size});
        auto sequence_length_1 = std::make_shared<op::v0::Parameter>(element::i64, Shape{1});
        Shape W_shape{4 * hidden_size, input_size};
        Shape R_shape{4 * hidden_size, hidden_size};
        std::vector<float> W_values =
            std::vector<float>{4, 12, 20, 5, 13, 21, 0, 8, 16, 1, 9, 17, 2, 10, 18, 3, 11, 19, 6, 14, 22, 7, 15, 23};
        auto W_1 = op::v0::Constant::create(element::f32, W_shape, W_values);
        std::vector<float> R_values =
            std::vector<float>{28, 36, 29, 37, 24, 32, 25, 33, 26, 34, 27, 35, 30, 38, 31, 39};
        auto R_1 = op::v0::Constant::create(element::f32, R_shape, R_values);
        Shape B_shape{4 * hidden_size};
        std::vector<float> B_values{5, 6, 0, 1, 2, 3, 6, 7};
        auto B_1 = op::v0::Constant::create(element::f32, B_shape, B_values);
        auto lstm_sequence_1 = std::make_shared<op::v5::LSTMSequence>(X_1,
                                                                      H_1,
                                                                      C_1,
                                                                      sequence_length_1,
                                                                      W_1,
                                                                      R_1,
                                                                      B_1,
                                                                      hidden_size,
                                                                      op::RecurrentSequenceDirection::FORWARD);
        auto squeeze_1 = std::make_shared<op::v0::Squeeze>(lstm_sequence_1->output(0));

        auto H_2 = std::make_shared<op::v0::Parameter>(element::f32, Shape{1, hidden_size});
        auto C_2 = std::make_shared<op::v0::Parameter>(element::f32, Shape{1, hidden_size});
        auto sequence_length_2 = std::make_shared<op::v0::Parameter>(element::i64, Shape{1});
        auto W_2 = op::v0::Constant::create(element::f32, W_shape, W_values);
        auto R_2 = op::v0::Constant::create(element::f32, R_shape, R_values);
        auto B_2 = op::v0::Constant::create(element::f32, B_shape, B_values);
        auto lstm_sequence_2 = std::make_shared<op::v5::LSTMSequence>(squeeze_1->output(0),
                                                                      H_2,
                                                                      C_2,
                                                                      sequence_length_2,
                                                                      W_2,
                                                                      R_2,
                                                                      B_2,
                                                                      hidden_size,
                                                                      op::RecurrentSequenceDirection::FORWARD);
        auto squeeze_2 = std::make_shared<op::v0::Squeeze>(lstm_sequence_2->output(0));

        auto abs = std::make_shared<op::v0::Abs>(squeeze_2->output(0));
        model = std::make_shared<Model>(NodeVector{abs}, ParameterVector{X_1, H_1, H_2, C_1, C_2});
        manager.register_pass<ov::pass::LSTMSequenceToMultiLSTMSequenceFusion>();
    }

    {
        auto X = std::make_shared<op::v0::Parameter>(element::f32, Shape{1, input_size});
        auto H = std::make_shared<op::v0::Parameter>(element::f32, Shape{1, hidden_size});
        auto C = std::make_shared<op::v0::Parameter>(element::f32, Shape{1, hidden_size});
        // auto lstm_count = std::make_shared<op::v0::Parameter>(element::i64, Shape{1});
        Shape W_shape{4 * hidden_size, input_size};
        Shape R_shape{4 * hidden_size, hidden_size};
        std::vector<float> W_values =
            std::vector<float>{4, 12, 20, 5, 13, 21, 0, 8, 16, 1, 9, 17, 2, 10, 18, 3, 11, 19, 6, 14, 22, 7, 15, 23};
        auto W = op::v0::Constant::create(element::f32, W_shape, W_values);
        std::vector<float> R_values =
            std::vector<float>{28, 36, 29, 37, 24, 32, 25, 33, 26, 34, 27, 35, 30, 38, 31, 39};
        auto R = op::v0::Constant::create(element::f32, R_shape, R_values);
        Shape B_shape{4 * hidden_size};
        std::vector<float> B_values{5, 6, 0, 1, 2, 3, 6, 7};
        auto B = op::v0::Constant::create(element::f32, B_shape, B_values);
        auto multi_lstm_sequence = std::make_shared<
            op::v13::MultiLSTMSequence>(X, H, C, 4, W, R, B, hidden_size, op::RecurrentSequenceDirection::FORWARD);
        auto abs = std::make_shared<op::v0::Abs>(multi_lstm_sequence->output(0));
        model_ref = std::make_shared<Model>(NodeVector{abs}, ParameterVector{X, H, C});
        manager.register_pass<ov::pass::LSTMSequenceToMultiLSTMSequenceFusion>();
    }

    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

INSTANTIATE_TEST_SUITE_P(LSTMSequenceToMultiLSTMSequenceFusion,
                         LSTMSequenceFusionTestSuite,
                         testing::Combine(testing::Values(1, 2), testing::Values(1, -1)));
