// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/tensor_iterator.hpp"

#include <map>

#include "common_test_utils/node_builders/reshape.hpp"
#include "common_test_utils/type_prop.hpp"
#include "openvino/core/model.hpp"
#include "openvino/opsets/opset5.hpp"

using namespace std;
using namespace ov;

TEST(type_prop, tensor_iterator_lstm) {
    // That which we iterate over
    const size_t N = 32;  // Batch size
    const size_t L = 10;  // Sequence length
    const size_t I = 8;   // Input size
    const size_t H = 32;  // Hidden size
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
    auto LSTM_cell = make_shared<opset5::LSTMCell>(ov::test::utils::reshape(X, Shape{N, I}),
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

    // Output 0 is last Ho, result 0 of body
    auto out0 = tensor_iterator->get_iter_value(H_o, -1);
    // Output 1 is last Co, result 1 of body
    auto out1 = tensor_iterator->get_iter_value(C_o, -1);

    auto results = ResultVector{make_shared<ov::op::v0::Result>(out0), make_shared<ov::op::v0::Result>(out1)};
    auto f = make_shared<Model>(results, ParameterVector{SENT, H_init, C_init, W, R});
}

TEST(type_prop, tensor_iterator_2_slice_inputs_part_size_2) {
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

    // Output 0 is last Zo
    auto out0 = tensor_iterator->get_iter_value(Zo, -1);
    // Output 1 is concat of Zos
    // start=0, stride=2, part_size=2, end=39, axis=1
    auto out1 = tensor_iterator->get_concatenated_slices(Zo, 0, 2, 2, 39, 1);
    auto result0 = make_shared<ov::op::v0::Result>(out0);
    auto result1 = make_shared<ov::op::v0::Result>(out1);
    Shape out0_shape{32, 2, 10};
    Shape out1_shape{32, 40, 10};

    auto results = ResultVector{result0, result1};
    auto f = make_shared<Model>(results, ParameterVector{X, Y, M});
    EXPECT_EQ(result0->get_output_shape(0), out0_shape);
    EXPECT_EQ(result1->get_output_shape(0), out1_shape);
}

TEST(type_prop, tensor_iterator_2_slice_inputs_part_size_2_dynamic) {
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

    // check input descriptors
    for (auto& desc : tensor_iterator->get_input_descriptions()) {
        auto type_info = desc->get_type_info();
        if (std::strcmp(type_info.name, "InvariantInputDescription") == 0) {
            auto input_desc = ov::as_type_ptr<ov::op::v0::TensorIterator::InvariantInputDescription>(desc);
            EXPECT_NE(input_desc, nullptr);
        } else if (std::strcmp(type_info.name, "SliceInputDescription") == 0) {
            auto input_desc = ov::as_type_ptr<ov::op::v0::TensorIterator::SliceInputDescription>(desc);
            EXPECT_NE(input_desc, nullptr);
        } else if (std::strcmp(type_info.name, "MergedInputDescription") == 0) {
            auto input_desc = ov::as_type_ptr<ov::op::v0::TensorIterator::MergedInputDescription>(desc);
            EXPECT_NE(input_desc, nullptr);
        }
    }

    // Output 0 is last Zo
    auto out0 = tensor_iterator->get_iter_value(Zo, -1);
    // Output 1 is concat of Zos
    // start=0, stride=2, part_size=2, end=38, axis=1
    auto out1 = tensor_iterator->get_concatenated_slices(Zo, 0, 2, 2, 38, 1);

    // check output descriptors
    for (auto& desc : tensor_iterator->get_output_descriptions()) {
        auto type_info = desc->get_type_info();
        if (std::strcmp(type_info.name, "ConcatOutputDescription") == 0) {
            auto output_desc = ov::as_type_ptr<ov::op::v0::TensorIterator::ConcatOutputDescription>(desc);
            EXPECT_NE(output_desc, nullptr);
        } else if (std::strcmp(type_info.name, "BodyOutputDescription") == 0) {
            auto output_desc = ov::as_type_ptr<ov::op::v0::TensorIterator::BodyOutputDescription>(desc);
            EXPECT_NE(output_desc, nullptr);
        }
    }
    auto result0 = make_shared<ov::op::v0::Result>(out0);
    auto result1 = make_shared<ov::op::v0::Result>(out1);
    Shape out0_shape{32, 2, 10};
    Shape out1_shape{32, 38, 10};

    auto results = ResultVector{result0, result1};
    auto f = make_shared<Model>(results, ParameterVector{X, Y, M});
    EXPECT_EQ(result0->get_output_shape(0), out0_shape);
    EXPECT_EQ(result1->get_output_shape(0), out1_shape);

    EXPECT_EQ(body->get_results()[0]->get_output_shape(0), out0_shape);
}

TEST(type_prop, tensor_iterator_with_dynamic_reshape) {
    // That which we iterate over
    const size_t N = 32;  // Batch size
    const size_t L = 10;  // Sequence length
    const size_t I = 8;   // Input size
    const size_t H = 32;  // Hidden size

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
    auto LSTM_cell = make_shared<opset5::LSTMCell>(ov::test::utils::reshape(X, Shape{N, I}),
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

    // Output 0 is last Ho, result 0 of body
    auto out0 = tensor_iterator->get_iter_value(H_o, -1);
    // Output 1 is last Co, result 1 of body
    auto out1 = tensor_iterator->get_iter_value(C_o, -1);

    auto results = ResultVector{make_shared<ov::op::v0::Result>(out0), make_shared<ov::op::v0::Result>(out1)};
    auto f = make_shared<Model>(results, ParameterVector{SENT, H_init, C_init, W, R});
    ASSERT_EQ(tensor_iterator->get_num_iterations(), 10);

    std::map<ov::Output<ov::Node>, ov::PartialShape> dyn;
    dyn[SENT->output(0)] = {-1, -1, -1};
    f->reshape(dyn);
    f->validate_nodes_and_infer_types();

    ASSERT_EQ(tensor_iterator->get_num_iterations(), -1);
}

TEST(type_prop, tensor_iterator_dyn_slice) {
    const size_t N = 32;  // Batch size
    const size_t I = 8;   // Input size

    ov::PartialShape ps = {N, ov::Dimension::dynamic(), I};
    auto SENT = make_shared<ov::op::v0::Parameter>(element::f32, ps);

    // Body
    auto X = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic());
    auto Res = make_shared<ov::op::v0::Result>(X);
    auto body = make_shared<ov::Model>(Res, ParameterVector{X});
    auto tensor_iterator = make_shared<op::v0::TensorIterator>();
    tensor_iterator->set_body(body);

    // start=0, stride=1, part_size=1, end=39, axis=1
    const size_t part_size = 1;
    tensor_iterator->set_sliced_input(X, SENT, 0, 1, part_size, -1, 1);

    // Output 0 is last Ho, result 0 of body
    auto out0 = tensor_iterator->get_iter_value(Res, -1);

    auto results = ResultVector{make_shared<ov::op::v0::Result>(out0)};
    auto model = make_shared<ov::Model>(results, ParameterVector{SENT});

    EXPECT_EQ(tensor_iterator->get_num_iterations(), -1);
    PartialShape ref_ps = {N, part_size, I};
    EXPECT_EQ(X->get_partial_shape(), ref_ps);
}
