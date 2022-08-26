// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <openvino/core/model.hpp>
#include <openvino/opsets/opset9.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

template<class T>
std::shared_ptr<ov::opset9::Constant> create_constant(const std::vector<T>& data, const ov::element::Type_t et = ov::element::i64, bool scalar = false) {
    ov::Shape shape = scalar ? ov::Shape{} : ov::Shape{data.size()};
    return ov::opset9::Constant::create(et, shape, data);
}

std::shared_ptr<ov::opset9::Constant> create_zero_constant(const ov::element::Type_t et, ov::PartialShape shape) {
    return ov::opset9::Constant::create(et, shape.to_shape(), {0});
}

struct LSTMStatesAttributes {
    ov::element::Type_t data_et;
    ov::Dimension data_batch_size, new_batch_size;
    ov::Dimension input_size, hidden_size;
};

class LSTMStatesBroadcastTest
        : public testing::WithParamInterface<LSTMStatesAttributes>, public CommonTestUtils::TestsCommon {
};

TEST_P(LSTMStatesBroadcastTest, BareLSTM) {
    auto p = GetParam();

    std::shared_ptr<ov::Model> model(nullptr);
    {
        auto parameter = std::make_shared<ov::opset9::Parameter>(p.data_et, ov::PartialShape{p.data_batch_size, p.input_size});
        auto initial_hidden_state = create_zero_constant(p.data_et, {1, p.hidden_size});
        auto initial_cell_state = create_zero_constant(p.data_et, {1, p.hidden_size});
        auto W = create_zero_constant(p.data_et, {p.hidden_size * 4, p.input_size});
        auto R = create_zero_constant(p.data_et, {p.hidden_size * 4, p.hidden_size});

        auto cell = std::make_shared<ov::opset9::LSTMCell>(
                parameter, initial_hidden_state, initial_cell_state, W, R, static_cast<size_t>(p.hidden_size.get_length()));

        model = std::make_shared<ov::Model>(ov::NodeVector{cell}, ov::ParameterVector{parameter});
    }
    ASSERT_NO_THROW(model->reshape(ov::PartialShape{p.new_batch_size, p.input_size}));
}


class LSTMStatesBroadcastTestWithTI
        : public testing::WithParamInterface<LSTMStatesAttributes>, public CommonTestUtils::TestsCommon {
};

TEST_P(LSTMStatesBroadcastTestWithTI, TI_With_LSTM) {
    auto p = GetParam();

    std::shared_ptr<ov::Model> model(nullptr);
    {
        auto X = std::make_shared<ov::opset9::Parameter>(p.data_et, ov::PartialShape{p.data_batch_size, 1,  p.input_size});
        auto H_init = create_zero_constant(ov::element::i64, {1, p.hidden_size});
        auto C_init = create_zero_constant(ov::element::i64, {1, p.hidden_size});

        auto Xi = std::make_shared<ov::opset9::Parameter>(p.data_et, ov::PartialShape{1, 1, p.input_size});
        auto H_t = std::make_shared<ov::opset9::Parameter>(p.data_et, ov::PartialShape{1, p.hidden_size});
        auto C_t = std::make_shared<ov::opset9::Parameter>(p.data_et, ov::PartialShape{1, p.hidden_size});

        // Body
        auto squeeze = std::make_shared<ov::opset9::Squeeze>(Xi, create_constant<int64_t>({1}));
        auto W = create_zero_constant(p.data_et, {p.hidden_size * 4, p.input_size});
        auto R = create_zero_constant(p.data_et, {p.hidden_size * 4, p.hidden_size});

        auto lstm_cell = std::make_shared<ov::opset9::LSTMCell>(squeeze, H_t, C_t, W, R, static_cast<size_t>(p.hidden_size.get_length()));
        auto res_1 = std::make_shared<ov::opset9::Result>(lstm_cell->output(0));
        auto unsqueeze = std::make_shared<ov::opset9::Unsqueeze>(lstm_cell->output(0), create_constant<int64_t>({1}));
        auto res_2 = std::make_shared<ov::opset9::Result>(unsqueeze);
        auto res_3 = std::make_shared<ov::opset9::Result>(lstm_cell->output(1));
        auto body = std::make_shared<ov::Model>(ov::OutputVector{res_1, res_2, res_3}, ov::ParameterVector{Xi, H_t, C_t});

        auto tensor_iterator = std::make_shared<ov::opset9::TensorIterator>();
        tensor_iterator->set_body(body);

        tensor_iterator->set_merged_input(C_t, C_init, res_3);
        tensor_iterator->set_sliced_input(Xi, X, 0, 1, 1, -1, 1);
        tensor_iterator->set_merged_input(H_t, H_init, res_1);

        auto out0 = tensor_iterator->get_iter_value(res_1, -1);
        auto out1 = tensor_iterator->get_concatenated_slices(res_2, 0, 1, 1, -1, 0);

        auto res_ti_1 = std::make_shared<ov::opset9::Result>(tensor_iterator->output(1));
        auto res_ti_2 = std::make_shared<ov::opset9::Result>(tensor_iterator->output(0));
        model = std::make_shared<ov::Model>(ov::NodeVector{res_ti_1, res_ti_2},
                                               ov::ParameterVector{X});
    }
    model->reshape(ov::PartialShape{p.new_batch_size, 1, p.input_size});
    ASSERT_NO_THROW(model->reshape(ov::PartialShape{p.new_batch_size, 1, p.input_size}));
}

static std::vector<LSTMStatesAttributes> params = {
        LSTMStatesAttributes{ov::element::f32, {1}, {2}, {512}, {256}},
        LSTMStatesAttributes{ov::element::f32, {-1}, {2}, {512}, {256}},
};

INSTANTIATE_TEST_SUITE_P(SmartReshapeTests, LSTMStatesBroadcastTest, ::testing::ValuesIn(params));
INSTANTIATE_TEST_SUITE_P(SmartReshapeTests, LSTMStatesBroadcastTestWithTI, ::testing::ValuesIn(params));
