// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/opsets/opset9.hpp"

using namespace std;
using namespace ov::opset9;

struct LSTMStatesAttributes {
    ov::element::Type_t data_et;
    ov::Dimension data_batch_size, new_batch_size;
    ov::Dimension input_size, hidden_size;
};

class LSTMStatesBroadcastTest : public testing::WithParamInterface<LSTMStatesAttributes>,
                                public ov::test::TestsCommon {};

TEST_P(LSTMStatesBroadcastTest, BareLSTM) {
    auto p = GetParam();

    shared_ptr<ov::Model> model(nullptr);
    {
        auto parameter = make_shared<Parameter>(p.data_et, ov::PartialShape{p.data_batch_size, p.input_size});
        auto initial_hidden_state =
            ov::op::v0::Constant::create(p.data_et, ov::PartialShape{1, p.hidden_size}.to_shape(), {0});
        auto initial_cell_state =
            ov::op::v0::Constant::create(p.data_et, ov::PartialShape{1, p.hidden_size}.to_shape(), {0});
        auto W =
            ov::op::v0::Constant::create(p.data_et, ov::PartialShape{p.hidden_size * 4, p.input_size}.to_shape(), {0});
        auto R =
            ov::op::v0::Constant::create(p.data_et, ov::PartialShape{p.hidden_size * 4, p.hidden_size}.to_shape(), {0});

        auto cell = make_shared<LSTMCell>(parameter,
                                          initial_hidden_state,
                                          initial_cell_state,
                                          W,
                                          R,
                                          static_cast<size_t>(p.hidden_size.get_length()));

        model = make_shared<ov::Model>(ov::NodeVector{cell}, ov::ParameterVector{parameter});
    }
    OV_ASSERT_NO_THROW(model->reshape(ov::PartialShape{p.new_batch_size, p.input_size}));
}

class LSTMStatesBroadcastTestWithTI : public testing::WithParamInterface<LSTMStatesAttributes>,
                                      public ov::test::TestsCommon {};

TEST_P(LSTMStatesBroadcastTestWithTI, TI_With_LSTM) {
    auto p = GetParam();

    shared_ptr<ov::Model> model(nullptr);
    {
        auto X = make_shared<Parameter>(p.data_et, ov::PartialShape{p.data_batch_size, 1, p.input_size});
        auto H_init =
            ov::op::v0::Constant::create(ov::element::i64, ov::PartialShape{1, p.hidden_size}.to_shape(), {0});
        auto C_init =
            ov::op::v0::Constant::create(ov::element::i64, ov::PartialShape{1, p.hidden_size}.to_shape(), {0});

        auto Xi = make_shared<Parameter>(p.data_et, ov::PartialShape{1, 1, p.input_size});
        auto H_t = make_shared<Parameter>(p.data_et, ov::PartialShape{1, p.hidden_size});
        auto C_t = make_shared<Parameter>(p.data_et, ov::PartialShape{1, p.hidden_size});

        // Body
        auto squeeze = make_shared<Squeeze>(Xi, create_constant<int64_t>({1}));
        auto W =
            ov::op::v0::Constant::create(p.data_et, ov::PartialShape{p.hidden_size * 4, p.input_size}.to_shape(), {0});
        auto R =
            ov::op::v0::Constant::create(p.data_et, ov::PartialShape{p.hidden_size * 4, p.hidden_size}.to_shape(), {0});

        auto lstm_cell =
            make_shared<LSTMCell>(squeeze, H_t, C_t, W, R, static_cast<size_t>(p.hidden_size.get_length()));
        auto res_1 = make_shared<Result>(lstm_cell->output(0));
        auto unsqueeze = make_shared<Unsqueeze>(lstm_cell->output(0), create_constant<int64_t>({1}));
        auto res_2 = make_shared<Result>(unsqueeze);
        auto res_3 = make_shared<Result>(lstm_cell->output(1));
        auto body = make_shared<ov::Model>(ov::OutputVector{res_1, res_2, res_3}, ov::ParameterVector{Xi, H_t, C_t});

        auto tensor_iterator = make_shared<TensorIterator>();
        tensor_iterator->set_body(body);

        tensor_iterator->set_merged_input(C_t, C_init, res_3);
        tensor_iterator->set_sliced_input(Xi, X, 0, 1, 1, -1, 1);
        tensor_iterator->set_merged_input(H_t, H_init, res_1);

        auto out0 = tensor_iterator->get_iter_value(res_1, -1);
        auto out1 = tensor_iterator->get_concatenated_slices(res_2, 0, 1, 1, -1, 0);

        auto res_ti_1 = make_shared<Result>(tensor_iterator->output(1));
        auto res_ti_2 = make_shared<Result>(tensor_iterator->output(0));
        model = make_shared<ov::Model>(ov::NodeVector{res_ti_1, res_ti_2}, ov::ParameterVector{X});
    }
    OV_ASSERT_NO_THROW(model->reshape(ov::PartialShape{p.new_batch_size, 1, p.input_size}));
}

static vector<LSTMStatesAttributes> params = {
    LSTMStatesAttributes{ov::element::f32, {1}, {2}, {512}, {256}},
    LSTMStatesAttributes{ov::element::f32, {-1}, {2}, {512}, {256}},
};

INSTANTIATE_TEST_SUITE_P(SmartReshapeTests, LSTMStatesBroadcastTest, ::testing::ValuesIn(params));
INSTANTIATE_TEST_SUITE_P(SmartReshapeTests, LSTMStatesBroadcastTestWithTI, ::testing::ValuesIn(params));
