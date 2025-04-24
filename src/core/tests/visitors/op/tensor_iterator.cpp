// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/tensor_iterator.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/node_builders/reshape.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/lstm_cell.hpp"
#include "openvino/op/multiply.hpp"
#include "visitors/visitors.hpp"

using namespace std;
using namespace ov;
using ov::test::NodeBuilder;

TEST(attributes, tensor_iterator_lstm) {
    // That which we iterate over
    const size_t N = 32;  // Batch size
    const size_t L = 10;  // Sequence length
    const size_t I = 8;   // Input size
    const size_t H = 32;  // Hidden size

    NodeBuilder::opset().insert<ov::op::v0::TensorIterator>();

    auto SENT = make_shared<ov::op::v0::Parameter>(element::f32, Shape{N, L, I});

    auto H_init = make_shared<ov::op::v0::Parameter>(element::f32, Shape{N, 1, H});
    auto C_init = make_shared<ov::op::v0::Parameter>(element::f32, Shape{N, 1, H});

    auto W = make_shared<ov::op::v0::Parameter>(element::f32, Shape{4 * H, I});
    auto R = make_shared<ov::op::v0::Parameter>(element::f32, Shape{4 * H, H});
    auto H_t = make_shared<ov::op::v0::Parameter>(element::f32, Shape{N, 1, H});
    auto C_t = make_shared<ov::op::v0::Parameter>(element::f32, Shape{N, 1, H});

    // Body
    auto X = make_shared<ov::op::v0::Parameter>(element::f32, Shape{N, 1, I});
    auto W_body = make_shared<ov::op::v0::Parameter>(element::f32, Shape{4 * H, I});
    auto R_body = make_shared<ov::op::v0::Parameter>(element::f32, Shape{4 * H, H});
    auto LSTM_cell = make_shared<ov::op::v4::LSTMCell>(ov::test::utils::reshape(X, Shape{N, I}),
                                                       ov::test::utils::reshape(H_t, Shape{N, H}),
                                                       ov::test::utils::reshape(C_t, Shape{N, H}),
                                                       W_body,
                                                       R_body,
                                                       H);
    auto H_o = ov::test::utils::reshape(LSTM_cell->output(0), Shape{N, 1, H});
    auto C_o = ov::test::utils::reshape(LSTM_cell->output(1), Shape{N, 1, H});
    auto body = make_shared<ov::Model>(OutputVector{H_o, C_o}, ParameterVector{X, H_t, C_t, W_body, R_body});

    auto tensor_iterator = make_shared<op::v0::TensorIterator>();
    tensor_iterator->set_body(body);
    // start=0, stride=1, part_size=1, end=39, axis=1
    tensor_iterator->set_sliced_input(X, SENT, 0, 1, 1, -1, 1);
    // H_t is Hinit on the first iteration, Ho after that
    tensor_iterator->set_merged_input(H_t, H_init, H_o);
    tensor_iterator->set_merged_input(C_t, C_init, C_o);
    tensor_iterator->set_invariant_input(W_body, W);
    tensor_iterator->set_invariant_input(R_body, R);

    NodeBuilder builder(tensor_iterator);
    const auto g_tensor_iterator = ov::as_type_ptr<op::v0::TensorIterator>(builder.create());

    EXPECT_EQ(g_tensor_iterator->get_body(), tensor_iterator->get_body());
    EXPECT_EQ(g_tensor_iterator->get_input_descriptions(), tensor_iterator->get_input_descriptions());
    EXPECT_EQ(g_tensor_iterator->get_output_descriptions(), tensor_iterator->get_output_descriptions());
}

TEST(attributes, tensor_iterator_2_slice_inputs_part_size_2) {
    NodeBuilder::opset().insert<op::v0::TensorIterator>();
    // That which we iterate over
    auto X = make_shared<ov::op::v0::Parameter>(element::f32, Shape{32, 40, 10});
    auto Y = make_shared<ov::op::v0::Parameter>(element::f32, Shape{32, 40, 10});
    auto M = make_shared<ov::op::v0::Parameter>(element::f32, Shape{32, 2, 10});

    // Set up the cell body, a function from (Xi, Yi) -> (Zo)
    // Body parameters
    auto Xi = make_shared<ov::op::v0::Parameter>(element::f32, Shape{32, 2, 10});
    auto Yi = make_shared<ov::op::v0::Parameter>(element::f32, Shape{32, 2, 10});
    auto M_body = make_shared<ov::op::v0::Parameter>(element::f32, Shape{32, 2, 10});

    // Body
    auto Zo = std::make_shared<op::v1::Multiply>(std::make_shared<op::v1::Add>(Xi, Yi), M_body);
    auto body = make_shared<ov::Model>(OutputVector{Zo}, ParameterVector{Xi, Yi, M_body});

    auto tensor_iterator = make_shared<op::v0::TensorIterator>();

    tensor_iterator->set_body(body);
    // The Xi are the elements of Xseq
    // start=0, stride=2, part_size=2, end=39, axis=1
    tensor_iterator->set_sliced_input(Xi, X, 0, 2, 2, 39, 1);
    // The Yi are the elements of Yseq
    // start=0, stride=2, part_size=2, end=-1, axis=1
    tensor_iterator->set_sliced_input(Yi, Y, 0, 2, 2, -1, 1);
    tensor_iterator->set_invariant_input(M_body, M);

    NodeBuilder builder(tensor_iterator);
    const auto g_tensor_iterator = ov::as_type_ptr<op::v0::TensorIterator>(builder.create());

    EXPECT_EQ(g_tensor_iterator->get_body(), tensor_iterator->get_body());
    EXPECT_EQ(g_tensor_iterator->get_input_descriptions(), tensor_iterator->get_input_descriptions());
    EXPECT_EQ(g_tensor_iterator->get_output_descriptions(), tensor_iterator->get_output_descriptions());
}

TEST(attributes, tensor_iterator_2_slice_inputs_part_size_2_dynamic) {
    NodeBuilder::opset().insert<op::v0::TensorIterator>();
    // That which we iterate over
    auto X = make_shared<ov::op::v0::Parameter>(element::f32, Shape{32, 40, 10});
    auto Y = make_shared<ov::op::v0::Parameter>(element::f32, Shape{32, 40, 10});
    auto M = make_shared<ov::op::v0::Parameter>(element::f32, Shape{32, 2, 10});

    // Set up the cell body, a function from (Xi, Yi) -> (Zo)
    // Body parameters
    auto Xi = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic());
    auto Yi = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic());
    auto M_body = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic());

    // Body
    auto Zo = std::make_shared<op::v1::Multiply>(std::make_shared<op::v1::Add>(Xi, Yi), M_body);
    auto body = make_shared<ov::Model>(OutputVector{Zo}, ParameterVector{Xi, Yi, M_body});

    auto tensor_iterator = make_shared<op::v0::TensorIterator>();
    tensor_iterator->set_body(body);
    // The Xi are the elements of Xseq
    // start=0, stride=2, part_size=2, end=38, axis=1
    tensor_iterator->set_sliced_input(Xi, X, 0, 2, 2, 38, 1);
    // The Yi are the elements of Yseq
    // start=0, stride=2, part_size=2, end=-2, axis=1
    tensor_iterator->set_sliced_input(Yi, Y, 0, 2, 2, -2, 1);
    tensor_iterator->set_invariant_input(M_body, M);

    NodeBuilder builder(tensor_iterator);
    const auto g_tensor_iterator = ov::as_type_ptr<op::v0::TensorIterator>(builder.create());

    EXPECT_EQ(g_tensor_iterator->get_body(), tensor_iterator->get_body());
    EXPECT_EQ(g_tensor_iterator->get_input_descriptions(), tensor_iterator->get_input_descriptions());
    EXPECT_EQ(g_tensor_iterator->get_output_descriptions(), tensor_iterator->get_output_descriptions());
}
