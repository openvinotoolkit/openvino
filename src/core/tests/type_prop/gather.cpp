// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/gather.hpp"

#include "common_test_utils/test_assertions.hpp"
#include "common_test_utils/type_prop.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/shape_of.hpp"

using namespace std;
using namespace ov;
using namespace testing;

// ------------------------------ V1 ------------------------------

TEST(type_prop, gather_v1_axis_0) {
    Shape params_shape{3, 2};
    Shape indices_shape{2, 2};
    Shape out_shape{2, 2, 2};
    auto P = make_shared<ov::op::v0::Parameter>(element::f32, params_shape);
    auto I = make_shared<ov::op::v0::Parameter>(element::i32, indices_shape);
    auto A = ov::op::v0::Constant::create(element::i64, Shape{}, {0});
    auto G = make_shared<op::v1::Gather>(P, I, A);
    EXPECT_EQ(G->get_element_type(), element::f32);
    EXPECT_EQ(G->get_shape(), out_shape);
    EXPECT_EQ(G->get_axis(), 0);
}

TEST(type_prop, gather_v1_uint8) {
    // Gather_1 must allow even if indices is not int32/int64
    PartialShape data_shape{3, 2};
    PartialShape indices_shape{2, 2};
    PartialShape out_shape{2, 2, 2};

    auto D = make_shared<ov::op::v0::Parameter>(element::f32, data_shape);
    auto I = make_shared<ov::op::v0::Parameter>(element::u8, indices_shape);
    auto A = ov::op::v0::Constant::create(element::i64, Shape{}, {0});
    auto G = make_shared<op::v1::Gather>(D, I, A);

    EXPECT_EQ(G->get_element_type(), element::f32);
    EXPECT_EQ(G->get_output_partial_shape(0), out_shape);
    EXPECT_EQ(G->get_axis(), 0);
}

TEST(type_prop, gather_v1_float32) {
    // Gather_1 should allow non int32/int64 indices
    PartialShape data_shape{3, 2};
    PartialShape indices_shape{2, 2};
    PartialShape out_shape{2, 2, 2};

    auto D = make_shared<ov::op::v0::Parameter>(element::f32, data_shape);
    auto I = make_shared<ov::op::v0::Parameter>(element::f32, indices_shape);
    auto A = ov::op::v0::Constant::create(element::i64, Shape{}, {0});
    auto G = make_shared<op::v1::Gather>(D, I, A);

    EXPECT_EQ(G->get_element_type(), element::f32);
    EXPECT_EQ(G->get_output_partial_shape(0), out_shape);
    EXPECT_EQ(G->get_axis(), 0);
}

TEST(type_prop, gather_axis_1) {
    Shape params_shape{3, 3};
    Shape indices_shape{1, 2};
    Shape out_shape{3, 1, 2};
    auto P = make_shared<ov::op::v0::Parameter>(element::f32, params_shape);
    auto I = make_shared<ov::op::v0::Parameter>(element::i32, indices_shape);
    auto A = ov::op::v0::Constant::create(element::i64, Shape{}, {1});
    auto G = make_shared<op::v1::Gather>(P, I, A);
    EXPECT_EQ(G->get_element_type(), element::f32);
    EXPECT_EQ(G->get_shape(), out_shape);
    EXPECT_EQ(G->get_axis(), 1);
}

TEST(type_prop, gather_v1_incorrect_axis_shape) {
    auto params = make_shared<ov::op::v0::Parameter>(element::f32, Shape{5, 6});
    auto indices = make_shared<ov::op::v0::Parameter>(element::i64, Shape{4});
    auto axis = make_shared<ov::op::v0::Parameter>(element::i64, Shape{2});

    OV_EXPECT_THROW(auto g = make_shared<op::v1::Gather>(params, indices, axis),
                    NodeValidationFailure,
                    HasSubstr("Axis input must be scalar or have 1 element"));
}

TEST(type_prop, gather_v1_axis_out_of_input_rank) {
    auto params = make_shared<ov::op::v0::Parameter>(element::f32, Shape{5, 6});
    auto indices = make_shared<ov::op::v0::Parameter>(element::i64, Shape{4});
    auto axis = make_shared<ov::op::v0::Constant>(element::i64, Shape{1}, vector<int64_t>{2});

    OV_EXPECT_THROW(auto g = make_shared<op::v1::Gather>(params, indices, axis),
                    ov::AssertFailure,
                    HasSubstr("out of the tensor rank range"));
}

TEST(type_prop, gather_v1_negative_axis) {
    auto params = make_shared<ov::op::v0::Parameter>(element::f32, Shape{5, 6, 7});
    auto indices = make_shared<ov::op::v0::Parameter>(element::i64, Shape{4});
    int64_t axis = -2;
    auto axis_node = make_shared<ov::op::v0::Constant>(element::i64, Shape{1}, vector<int64_t>{axis});
    auto gather_v1 = make_shared<op::v1::Gather>(params, indices, axis_node);
    EXPECT_EQ(gather_v1->get_axis(), 1);
}

TEST(type_prop, gather_1_dynamic_value_and_symbol_propagation) {
    Dimension marked_0 = Dimension(3);
    auto symbol = std::make_shared<ov::Symbol>();
    marked_0.set_symbol(symbol);
    PartialShape target_0 = PartialShape{marked_0, 4};

    auto param = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{1});
    auto param_0 = std::make_shared<ov::op::v0::Parameter>(element::f32, target_0);
    auto shape_0 = std::make_shared<op::v0::ShapeOf>(param_0);

    const auto& et = element::i64;
    std::vector<int64_t> zero{0};
    const auto indices = std::make_shared<op::v0::Constant>(et, Shape{zero.size()}, zero);
    const auto axis = std::make_shared<op::v0::Constant>(et, Shape{}, zero);
    const auto gather = std::make_shared<op::v1::Gather>(shape_0, indices, axis);

    auto bc = std::make_shared<op::v1::Broadcast>(param, gather);
    EXPECT_EQ(bc->get_shape(), (Shape{3}));

    const auto& output_shape = bc->get_output_partial_shape(0);
    EXPECT_EQ(output_shape[0].get_symbol(), symbol);
}

TEST(type_prop, dynamic_value_propagation) {
    auto param = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, 3, -1, -1});
    auto shape_of = std::make_shared<op::v3::ShapeOf>(param, element::i32);

    auto indices = ov::op::v0::Constant::create(element::i32, {}, {1});
    auto axis = ov::op::v0::Constant::create(element::i32, {}, {0});
    auto gather = std::make_shared<op::v1::Gather>(shape_of, indices, axis);

    auto add = std::make_shared<op::v1::Add>(gather, ov::op::v0::Constant::create(element::i32, {}, {0}));

    auto range = std::make_shared<op::v4::Range>(ov::op::v0::Constant::create(element::i32, {}, {0}),
                                                 add,
                                                 ov::op::v0::Constant::create(element::i32, {}, {1}),
                                                 element::i64);

    auto RIC = std::make_shared<op::v1::Gather>(param, range, ov::op::v0::Constant::create(element::i32, {}, {1}));

    EXPECT_EQ(RIC->get_element_type(), element::f32);
    EXPECT_EQ(RIC->get_output_partial_shape(0), (PartialShape{-1, 3, -1, -1}));
}

// ------------------------------ V7 ------------------------------

TEST(type_prop, gather_7_axis_0) {
    PartialShape data_shape{3, 2};
    PartialShape indices_shape{2, 2};
    PartialShape out_shape{2, 2, 2};
    int64_t batch_dims = 0;

    auto D = make_shared<ov::op::v0::Parameter>(element::f32, data_shape);
    auto I = make_shared<ov::op::v0::Parameter>(element::i32, indices_shape);
    auto A = ov::op::v0::Constant::create(element::i64, Shape{}, {0});
    auto G = make_shared<op::v7::Gather>(D, I, A, batch_dims);

    EXPECT_EQ(G->get_element_type(), element::f32);
    EXPECT_EQ(G->get_output_partial_shape(0), out_shape);
    EXPECT_EQ(G->get_axis(), 0);
}

TEST(type_prop, gather_7_axis_1) {
    PartialShape data_shape{3, 3};
    PartialShape indices_shape{1, 2};
    PartialShape out_shape{3, 1, 2};
    int64_t axis = 1;

    auto D = make_shared<ov::op::v0::Parameter>(element::f32, data_shape);
    auto I = make_shared<ov::op::v0::Parameter>(element::i32, indices_shape);
    auto A = ov::op::v0::Constant::create(element::i64, Shape{}, {axis});
    auto G = make_shared<op::v7::Gather>(D, I, A);

    EXPECT_EQ(G->get_element_type(), element::f32);
    EXPECT_EQ(G->get_output_partial_shape(0), out_shape);
    EXPECT_EQ(G->get_axis(), 1);
}

TEST(type_prop, gather_7_negative_axis) {
    PartialShape data_shape{5, 6, 7};
    PartialShape indices_shape{4};
    PartialShape out_shape{5, 4, 7};
    int64_t axis = -2;

    auto D = make_shared<ov::op::v0::Parameter>(element::f32, data_shape);
    auto I = make_shared<ov::op::v0::Parameter>(element::i64, indices_shape);
    auto A = make_shared<ov::op::v0::Constant>(element::i64, Shape{1}, vector<int64_t>{axis});
    auto G = make_shared<op::v7::Gather>(D, I, A);

    EXPECT_EQ(G->get_axis(), 1);
    EXPECT_EQ(G->get_output_partial_shape(0), out_shape);
}

TEST(type_prop, gather_7_dynamic_pshape_batch_dims_1_axis_1) {
    PartialShape data_shape{Dimension(1, 7), 20, 20};
    PartialShape indices_shape{Dimension(7, 10), 3, 8};
    PartialShape out_shape{7, 3, 8, 20};
    int64_t axis = 1;
    int64_t batch_dims = 1;

    auto D = make_shared<ov::op::v0::Parameter>(element::f32, data_shape);
    auto I = make_shared<ov::op::v0::Parameter>(element::i64, indices_shape);
    auto A = make_shared<ov::op::v0::Constant>(element::i64, Shape{1}, vector<int64_t>{axis});
    auto G = make_shared<op::v7::Gather>(D, I, A, batch_dims);

    EXPECT_EQ(G->get_element_type(), element::f32);
    EXPECT_EQ(G->get_output_partial_shape(0), out_shape);
}

TEST(type_prop, gather_7_dynamic_pshape_batch_dims_1_axis_3) {
    PartialShape data_shape{Dimension(1, 7), Dimension(1, 3), 200, 400};
    PartialShape indices_shape{Dimension(7, 10), Dimension(2, 10), 3, 8};
    PartialShape out_shape{7, Dimension(1, 3), 200, Dimension(2, 10), 3, 8};
    int64_t axis = 3;
    int64_t batch_dims = 1;

    auto D = make_shared<ov::op::v0::Parameter>(element::f32, data_shape);
    auto I = make_shared<ov::op::v0::Parameter>(element::i64, indices_shape);
    auto A = make_shared<ov::op::v0::Constant>(element::i64, Shape{1}, vector<int64_t>{axis});
    auto G = make_shared<op::v7::Gather>(D, I, A, batch_dims);

    EXPECT_EQ(G->get_element_type(), element::f32);
    EXPECT_EQ(G->get_output_partial_shape(0), out_shape);
}

TEST(type_prop, gather_7_dynamic_2d_pshape_batch_dim) {
    PartialShape data_shape{Dimension(1, 7), Dimension(1, 3), 200, 400};
    PartialShape indices_shape{Dimension(7, 10), Dimension(2, 10), 3, 8};
    PartialShape out_shape{7, Dimension(2, 3), 3, 8, 400};
    int64_t axis = 2;
    int64_t batch_dims = 2;

    auto D = make_shared<ov::op::v0::Parameter>(element::f32, data_shape);
    auto I = make_shared<ov::op::v0::Parameter>(element::i64, indices_shape);
    auto A = make_shared<ov::op::v0::Constant>(element::i64, Shape{1}, vector<int64_t>{axis});
    auto G = make_shared<op::v7::Gather>(D, I, A, batch_dims);

    EXPECT_EQ(G->get_element_type(), element::f32);
    EXPECT_EQ(G->get_output_partial_shape(0), out_shape);
}

TEST(type_prop, gather_7_dynamic_2d_pshape_batch_dim_axis_3) {
    PartialShape data_shape{Dimension(1, 7), Dimension(1, 3), 200, 400};
    PartialShape indices_shape{Dimension(7, 10), Dimension(2, 10), 3, 8};
    PartialShape out_shape{7, Dimension(2, 3), 200, 3, 8};
    int64_t axis = 3;
    int64_t batch_dims = 2;

    auto D = make_shared<ov::op::v0::Parameter>(element::f32, data_shape);
    auto I = make_shared<ov::op::v0::Parameter>(element::i64, indices_shape);
    auto A = make_shared<ov::op::v0::Constant>(element::i64, Shape{1}, vector<int64_t>{axis});
    auto G = make_shared<op::v7::Gather>(D, I, A, batch_dims);

    EXPECT_EQ(G->get_element_type(), element::f32);
    EXPECT_EQ(G->get_output_partial_shape(0), out_shape);
}

TEST(type_prop, gather_7_dynamic_rank) {
    PartialShape data_shape{Dimension(1, 7), Dimension(1, 3), 200, 400};
    PartialShape indices_shape = PartialShape::dynamic(Rank(3, 5));
    PartialShape out_shape = PartialShape::dynamic(Rank(4, 6));
    int64_t axis = 3;
    int64_t batch_dims = 2;

    auto D = make_shared<ov::op::v0::Parameter>(element::f32, data_shape);
    auto I = make_shared<ov::op::v0::Parameter>(element::i64, indices_shape);
    auto A = make_shared<ov::op::v0::Constant>(element::i64, Shape{1}, vector<int64_t>{axis});
    auto G = make_shared<op::v7::Gather>(D, I, A, batch_dims);

    EXPECT_EQ(G->get_element_type(), element::f32);
    EXPECT_EQ(G->get_output_partial_shape(0), out_shape);
}

TEST(type_prop, gather_7_axis_boundcheck_for_dynamic_data_rank) {
    PartialShape data_shape = PartialShape::dynamic();
    PartialShape indices_shape{7, 3, 8};
    PartialShape out_shape = PartialShape::dynamic();
    int64_t axis = 3;
    int64_t batch_dims = 2;

    auto D = make_shared<ov::op::v0::Parameter>(element::f32, data_shape);
    auto I = make_shared<ov::op::v0::Parameter>(element::i64, indices_shape);
    auto A = make_shared<ov::op::v0::Constant>(element::i64, Shape{1}, vector<int64_t>{axis});
    auto G = make_shared<op::v7::Gather>(D, I, A, batch_dims);

    EXPECT_EQ(G->get_element_type(), element::f32);
    EXPECT_EQ(G->get_output_partial_shape(0), out_shape);
}

TEST(type_prop, gather_7_dynamic_rank_negative_batch_dims) {
    PartialShape data_shape{Dimension(1, 7), Dimension(1, 3), 200, 400};
    PartialShape indices_shape = PartialShape::dynamic(Rank(3, 5));
    PartialShape out_shape = PartialShape::dynamic(Rank(3, 5));
    int64_t axis = 3;
    int64_t batch_dims = -2;

    auto D = make_shared<ov::op::v0::Parameter>(element::f32, data_shape);
    auto I = make_shared<ov::op::v0::Parameter>(element::i64, indices_shape);
    auto A = make_shared<ov::op::v0::Constant>(element::i64, Shape{1}, vector<int64_t>{axis});
    auto G = make_shared<op::v7::Gather>(D, I, A, batch_dims);

    EXPECT_EQ(G->get_element_type(), element::f32);
    EXPECT_EQ(G->get_output_partial_shape(0), out_shape);
}

TEST(type_prop, gather_7_axis_not_set) {
    PartialShape data_shape{1, 1, 200, 400};
    PartialShape indices_shape{2, 2};
    // default batch_dims = 0
    PartialShape out_shape = PartialShape::dynamic(5);  // out_rank = data_rank + indices_rank - 1 - batch_dims

    auto D = make_shared<ov::op::v0::Parameter>(element::f32, data_shape);
    auto I = make_shared<ov::op::v0::Parameter>(element::i64, indices_shape);
    auto A = make_shared<ov::op::v0::Parameter>(element::i32, Shape{1});
    auto G = make_shared<op::v7::Gather>(D, I, A);

    EXPECT_EQ(G->get_element_type(), element::f32);
    EXPECT_EQ(G->get_output_partial_shape(0), out_shape);
}

TEST(type_prop, gather_7_axis_not_set_positive_batch_dims) {
    PartialShape data_shape{2, 1, 200, 400};
    PartialShape indices_shape{2, 2};
    int64_t batch_dims = 1;
    PartialShape out_shape = PartialShape({2, Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic()});

    auto D = make_shared<ov::op::v0::Parameter>(element::f32, data_shape);
    auto I = make_shared<ov::op::v0::Parameter>(element::i64, indices_shape);
    auto A = make_shared<ov::op::v0::Parameter>(element::i32, Shape{1});
    auto G = make_shared<op::v7::Gather>(D, I, A, batch_dims);

    EXPECT_EQ(G->get_element_type(), element::f32);
    EXPECT_EQ(G->get_output_partial_shape(0), out_shape);
}

TEST(type_prop, gather_7_dynamic_value_and_symbol_propagation) {
    Dimension marked_0 = Dimension(3);
    auto symbol = std::make_shared<ov::Symbol>();
    marked_0.set_symbol(symbol);
    PartialShape target_0 = PartialShape{marked_0, 4};

    auto param = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{1});
    auto param_0 = std::make_shared<ov::op::v0::Parameter>(element::f32, target_0);
    auto shape_0 = std::make_shared<op::v0::ShapeOf>(param_0);

    const auto& et = element::i64;
    std::vector<int64_t> zero{0};
    const auto indices = std::make_shared<op::v0::Constant>(et, Shape{zero.size()}, zero);
    const auto axis = std::make_shared<op::v0::Constant>(et, Shape{}, zero);
    const auto gather = std::make_shared<op::v7::Gather>(shape_0, indices, axis);

    auto bc = std::make_shared<op::v1::Broadcast>(param, gather);
    EXPECT_EQ(bc->get_shape(), (Shape{3}));

    const auto& output_shape = bc->get_output_partial_shape(0);
    EXPECT_EQ(output_shape[0].get_symbol(), symbol);
}

// --------------------- V7 Negative tests ------------------------------

TEST(type_prop, gather_7_incorrect_axis_shape) {
    auto D = make_shared<ov::op::v0::Parameter>(element::f32, Shape{5, 6});
    auto I = make_shared<ov::op::v0::Parameter>(element::i64, Shape{4});
    auto A = make_shared<ov::op::v0::Parameter>(element::i64, Shape{2});

    OV_EXPECT_THROW(auto g = make_shared<op::v7::Gather>(D, I, A),
                    NodeValidationFailure,
                    HasSubstr("Axis input must be scalar or have 1 element"));
}

TEST(type_prop, gather_7_axis_out_of_input_rank) {
    auto D = make_shared<ov::op::v0::Parameter>(element::f32, Shape{5, 6});
    auto I = make_shared<ov::op::v0::Parameter>(element::i64, Shape{4});
    auto A = make_shared<ov::op::v0::Constant>(element::i64, Shape{1}, vector<int64_t>{2});
    int64_t batch_dims = 0;

    OV_EXPECT_THROW(auto g = make_shared<op::v7::Gather>(D, I, A, batch_dims),
                    ov::AssertFailure,
                    HasSubstr("out of the tensor rank range"));
}

TEST(type_prop, gather_7_dynamic_batch_dims_inconsistent) {
    PartialShape data_shape{Dimension(1, 7), 20, 20};
    PartialShape indices_shape{Dimension(8, 10), 3, 8};

    auto D = make_shared<ov::op::v0::Parameter>(element::f32, data_shape);
    auto I = make_shared<ov::op::v0::Parameter>(element::i64, indices_shape);
    int64_t axis = 1;
    auto A = make_shared<ov::op::v0::Constant>(element::i64, Shape{1}, vector<int64_t>{axis});
    int64_t batch_dims = 1;

    OV_EXPECT_THROW(auto g = make_shared<op::v7::Gather>(D, I, A, batch_dims),
                    NodeValidationFailure,
                    HasSubstr("data and indices must have equal or intersecting sizes until batch_dims"));
}

TEST(type_prop, gather_7_batch_dims_less_check) {
    PartialShape data_shape{1, 3, 20};
    PartialShape indices_shape{1, 3, 8};

    auto D = make_shared<ov::op::v0::Parameter>(element::f32, data_shape);
    auto I = make_shared<ov::op::v0::Parameter>(element::i64, indices_shape);
    int64_t axis = 1;
    auto A = make_shared<ov::op::v0::Constant>(element::i64, Shape{1}, vector<int64_t>{axis});
    int64_t batch_dims = 2;

    OV_EXPECT_THROW(auto g = make_shared<op::v7::Gather>(D, I, A, batch_dims),
                    NodeValidationFailure,
                    HasSubstr("After normalization batch_dims must be <= axis. But instead got: batch_dims ="));
}

TEST(type_prop, gather_7_batch_dims_less_indices_rank_check) {
    PartialShape data_shape{1, 20, 20, 22, 22};
    PartialShape indices_shape{1, 3};

    auto D = make_shared<ov::op::v0::Parameter>(element::f32, data_shape);
    auto I = make_shared<ov::op::v0::Parameter>(element::i64, indices_shape);
    int64_t axis = 4;
    auto A = make_shared<ov::op::v0::Constant>(element::i64, Shape{1}, vector<int64_t>{axis});
    int64_t batch_dims = 3;

    OV_EXPECT_THROW(auto g = make_shared<op::v7::Gather>(D, I, A, batch_dims),
                    NodeValidationFailure,
                    HasSubstr("batch_dims must be <= indices_rank"));
}

TEST(type_prop, gather_7_indices_type_check) {
    PartialShape data_shape{1, 20, 20, 22, 22};
    PartialShape indices_shape{1, 3};

    auto D = make_shared<ov::op::v0::Parameter>(element::f32, data_shape);
    auto I = make_shared<ov::op::v0::Parameter>(element::f32, indices_shape);
    int64_t axis = 4;
    auto A = make_shared<ov::op::v0::Constant>(element::i64, Shape{1}, vector<int64_t>{axis});
    int64_t batch_dims = 0;

    OV_EXPECT_THROW(auto g = make_shared<op::v7::Gather>(D, I, A, batch_dims),
                    NodeValidationFailure,
                    HasSubstr("Indices element type must be of an integral number type"));
}

TEST(type_prop, gather_7_axis_type_check) {
    PartialShape data_shape{1, 20, 20, 22, 22};
    PartialShape indices_shape{1, 3};

    auto D = make_shared<ov::op::v0::Parameter>(element::f32, data_shape);
    auto I = make_shared<ov::op::v0::Parameter>(element::i32, indices_shape);
    int64_t axis = 4;
    auto A = make_shared<ov::op::v0::Constant>(element::f32, Shape{1}, vector<int64_t>{axis});
    int64_t batch_dims = 0;

    OV_EXPECT_THROW(auto g = make_shared<op::v7::Gather>(D, I, A, batch_dims),
                    NodeValidationFailure,
                    HasSubstr("Axis element type must be of an integral number type"));
}

// ------------------------------ V8 ------------------------------

TEST(type_prop, gather_v8_axis_0) {
    PartialShape data_shape{3, 2};
    PartialShape indices_shape{2, 2};
    PartialShape out_shape{2, 2, 2};
    int64_t batch_dims = 0;

    auto D = make_shared<ov::op::v0::Parameter>(element::f32, data_shape);
    auto I = make_shared<ov::op::v0::Parameter>(element::i32, indices_shape);
    auto A = ov::op::v0::Constant::create(element::i64, Shape{}, {0});
    auto G = make_shared<op::v8::Gather>(D, I, A, batch_dims);

    EXPECT_EQ(G->get_element_type(), element::f32);
    EXPECT_EQ(G->get_output_partial_shape(0), out_shape);
    EXPECT_EQ(G->get_axis(), 0);
}

TEST(type_prop, gather_v8_axis_1) {
    PartialShape data_shape{3, 3};
    PartialShape indices_shape{1, 2};
    PartialShape out_shape{3, 1, 2};
    int64_t axis = 1;

    auto D = make_shared<ov::op::v0::Parameter>(element::f32, data_shape);
    auto I = make_shared<ov::op::v0::Parameter>(element::i32, indices_shape);
    auto A = ov::op::v0::Constant::create(element::i64, Shape{}, {axis});
    auto G = make_shared<op::v8::Gather>(D, I, A);

    EXPECT_EQ(G->get_element_type(), element::f32);
    EXPECT_EQ(G->get_output_partial_shape(0), out_shape);
    EXPECT_EQ(G->get_axis(), 1);
}

TEST(type_prop, gather_v8_negative_axis) {
    PartialShape data_shape{5, 6, 7};
    PartialShape indices_shape{4};
    PartialShape out_shape{5, 4, 7};
    int64_t axis = -2;

    auto D = make_shared<ov::op::v0::Parameter>(element::f32, data_shape);
    auto I = make_shared<ov::op::v0::Parameter>(element::i64, indices_shape);
    auto A = make_shared<ov::op::v0::Constant>(element::i64, Shape{1}, vector<int64_t>{axis});
    auto G = make_shared<op::v8::Gather>(D, I, A);

    EXPECT_EQ(G->get_axis(), 1);
    EXPECT_EQ(G->get_output_partial_shape(0), out_shape);
}

TEST(type_prop, gather_v8_dynamic_pshape_batch_dims_1_axis_1) {
    PartialShape data_shape{Dimension(1, 7), 20, 20};
    PartialShape indices_shape{Dimension(7, 10), 3, 8};
    PartialShape out_shape{7, 3, 8, 20};
    int64_t axis = 1;
    int64_t batch_dims = 1;

    auto D = make_shared<ov::op::v0::Parameter>(element::f32, data_shape);
    auto I = make_shared<ov::op::v0::Parameter>(element::i64, indices_shape);
    auto A = make_shared<ov::op::v0::Constant>(element::i64, Shape{1}, vector<int64_t>{axis});
    auto G = make_shared<op::v8::Gather>(D, I, A, batch_dims);

    EXPECT_EQ(G->get_element_type(), element::f32);
    EXPECT_EQ(G->get_output_partial_shape(0), out_shape);
}

TEST(type_prop, gather_v8_dynamic_pshape_batch_dims_1_axis_3) {
    PartialShape data_shape{Dimension(1, 7), Dimension(1, 3), 200, 400};
    PartialShape indices_shape{Dimension(7, 10), Dimension(2, 10), 3, 8};
    PartialShape out_shape{7, Dimension(1, 3), 200, Dimension(2, 10), 3, 8};
    int64_t axis = 3;
    int64_t batch_dims = 1;

    auto D = make_shared<ov::op::v0::Parameter>(element::f32, data_shape);
    auto I = make_shared<ov::op::v0::Parameter>(element::i64, indices_shape);
    auto A = make_shared<ov::op::v0::Constant>(element::i64, Shape{1}, vector<int64_t>{axis});
    auto G = make_shared<op::v8::Gather>(D, I, A, batch_dims);

    EXPECT_EQ(G->get_element_type(), element::f32);
    EXPECT_EQ(G->get_output_partial_shape(0), out_shape);
}

TEST(type_prop, gather_v8_dim_no_bound_pshape_batch_dims_1_axis_3) {
    PartialShape data_shape{Dimension(7, -1), Dimension(-1, 3), 200, 400};
    PartialShape indices_shape{Dimension(7, 10), Dimension(2, 10), 3, 8};
    PartialShape out_shape{Dimension(7, 10), Dimension(-1, 3), 200, Dimension(2, 10), 3, 8};
    int64_t axis = 3;
    int64_t batch_dims = 1;

    auto D = make_shared<ov::op::v0::Parameter>(element::f32, data_shape);
    auto I = make_shared<ov::op::v0::Parameter>(element::i64, indices_shape);
    auto A = make_shared<ov::op::v0::Constant>(element::i64, Shape{1}, vector<int64_t>{axis});
    auto G = make_shared<op::v8::Gather>(D, I, A, batch_dims);

    EXPECT_EQ(G->get_element_type(), element::f32);
    EXPECT_EQ(G->get_output_partial_shape(0), out_shape);
}

TEST(type_prop, gather_v8_dynamic_2d_pshape_batch_dim) {
    PartialShape data_shape{Dimension(1, 7), Dimension(1, 3), 200, 400};
    PartialShape indices_shape{Dimension(7, 10), Dimension(2, 10), 3, 8};
    PartialShape out_shape{7, Dimension(2, 3), 3, 8, 400};
    int64_t axis = 2;
    int64_t batch_dims = 2;

    auto D = make_shared<ov::op::v0::Parameter>(element::f32, data_shape);
    auto I = make_shared<ov::op::v0::Parameter>(element::i64, indices_shape);
    auto A = make_shared<ov::op::v0::Constant>(element::i64, Shape{1}, vector<int64_t>{axis});
    auto G = make_shared<op::v8::Gather>(D, I, A, batch_dims);

    EXPECT_EQ(G->get_element_type(), element::f32);
    EXPECT_EQ(G->get_output_partial_shape(0), out_shape);
}

TEST(type_prop, gather_v8_dynamic_2d_pshape_batch_dim_axis_3) {
    PartialShape data_shape{Dimension(1, 7), Dimension(1, 3), 200, 400};
    PartialShape indices_shape{Dimension(7, 10), Dimension(2, 10), 3, 8};
    PartialShape out_shape{7, Dimension(2, 3), 200, 3, 8};
    int64_t axis = 3;
    int64_t batch_dims = 2;

    auto D = make_shared<ov::op::v0::Parameter>(element::f32, data_shape);
    auto I = make_shared<ov::op::v0::Parameter>(element::i64, indices_shape);
    auto A = make_shared<ov::op::v0::Constant>(element::i64, Shape{1}, vector<int64_t>{axis});
    auto G = make_shared<op::v8::Gather>(D, I, A, batch_dims);

    EXPECT_EQ(G->get_element_type(), element::f32);
    EXPECT_EQ(G->get_output_partial_shape(0), out_shape);
}

TEST(type_prop, gather_v8_dynamic_rank) {
    PartialShape data_shape{Dimension(1, 7), Dimension(1, 3), 200, 400};
    PartialShape indices_shape = PartialShape::dynamic(Rank(3, 5));
    PartialShape out_shape = PartialShape::dynamic(Rank(4, 6));
    int64_t axis = 3;
    int64_t batch_dims = 2;

    auto D = make_shared<ov::op::v0::Parameter>(element::f32, data_shape);
    auto I = make_shared<ov::op::v0::Parameter>(element::i64, indices_shape);
    auto A = make_shared<ov::op::v0::Constant>(element::i64, Shape{1}, vector<int64_t>{axis});
    auto G = make_shared<op::v8::Gather>(D, I, A, batch_dims);

    EXPECT_EQ(G->get_element_type(), element::f32);
    EXPECT_EQ(G->get_output_partial_shape(0), out_shape);
}

TEST(type_prop, gather_v8_axis_boundcheck_for_dynamic_data_rank) {
    PartialShape data_shape = PartialShape::dynamic();
    PartialShape indices_shape{7, 3, 8};
    PartialShape out_shape = PartialShape::dynamic();
    int64_t axis = 3;
    int64_t batch_dims = 2;

    auto D = make_shared<ov::op::v0::Parameter>(element::f32, data_shape);
    auto I = make_shared<ov::op::v0::Parameter>(element::i64, indices_shape);
    auto A = make_shared<ov::op::v0::Constant>(element::i64, Shape{1}, vector<int64_t>{axis});
    auto G = make_shared<op::v8::Gather>(D, I, A, batch_dims);

    EXPECT_EQ(G->get_element_type(), element::f32);
    EXPECT_EQ(G->get_output_partial_shape(0), out_shape);
}

TEST(type_prop, gather_v8_dynamic_rank_negative_batch_dims) {
    PartialShape data_shape{Dimension(1, 7), Dimension(1, 3), 200, 400};
    PartialShape indices_shape = PartialShape::dynamic(Rank(3, 5));
    PartialShape out_shape = PartialShape::dynamic(Rank(3, 5));
    int64_t axis = 3;
    int64_t batch_dims = -2;

    auto D = make_shared<ov::op::v0::Parameter>(element::f32, data_shape);
    auto I = make_shared<ov::op::v0::Parameter>(element::i64, indices_shape);
    auto A = make_shared<ov::op::v0::Constant>(element::i64, Shape{1}, vector<int64_t>{axis});
    auto G = make_shared<op::v8::Gather>(D, I, A, batch_dims);

    EXPECT_EQ(G->get_element_type(), element::f32);
    EXPECT_EQ(G->get_output_partial_shape(0), out_shape);
}

TEST(type_prop, gather_v8_axis_not_set) {
    PartialShape data_shape{1, 1, 200, 400};
    PartialShape indices_shape{2, 2};
    // default batch_dims = 0
    PartialShape out_shape = PartialShape::dynamic(5);  // out_rank = data_rank + indices_rank - 1 - batch_dims

    auto D = make_shared<ov::op::v0::Parameter>(element::f32, data_shape);
    auto I = make_shared<ov::op::v0::Parameter>(element::i64, indices_shape);
    auto A = make_shared<ov::op::v0::Parameter>(element::i32, Shape{1});
    auto G = make_shared<op::v8::Gather>(D, I, A);

    EXPECT_EQ(G->get_element_type(), element::f32);
    EXPECT_EQ(G->get_output_partial_shape(0), out_shape);
}

TEST(type_prop, gather_v8_axis_not_set_positive_batch_dims) {
    PartialShape data_shape{2, 1, 200, 400};
    PartialShape indices_shape{2, 2};
    int64_t batch_dims = 1;
    PartialShape out_shape = PartialShape({2, Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic()});

    auto D = make_shared<ov::op::v0::Parameter>(element::f32, data_shape);
    auto I = make_shared<ov::op::v0::Parameter>(element::i64, indices_shape);
    auto A = make_shared<ov::op::v0::Parameter>(element::i32, Shape{1});
    auto G = make_shared<op::v8::Gather>(D, I, A, batch_dims);

    EXPECT_EQ(G->get_element_type(), element::f32);
    EXPECT_EQ(G->get_output_partial_shape(0), out_shape);
}

/** \brief Check usage of evaluate lower and symbol on shape inference. */
TEST(type_prop, gather_v8_dynamic_value_and_symbol_propagation) {
    Dimension marked_0 = Dimension(3);
    auto symbol = std::make_shared<ov::Symbol>();
    marked_0.set_symbol(symbol);
    PartialShape target_0 = PartialShape{marked_0, 4};

    auto param = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{1});
    auto param_0 = std::make_shared<ov::op::v0::Parameter>(element::f32, target_0);
    auto shape_0 = std::make_shared<op::v0::ShapeOf>(param_0);

    const auto& et = element::i64;
    std::vector<int64_t> zero{0};
    const auto indices = std::make_shared<op::v0::Constant>(et, Shape{zero.size()}, zero);
    const auto axis = std::make_shared<op::v0::Constant>(et, Shape{}, zero);
    const auto gather = std::make_shared<op::v8::Gather>(shape_0, indices, axis);

    auto bc = std::make_shared<op::v1::Broadcast>(param, gather);
    EXPECT_EQ(bc->get_shape(), (Shape{3}));

    const auto& output_shape = bc->get_output_partial_shape(0);
    EXPECT_EQ(output_shape[0].get_symbol(), symbol);
}

/** \brief Check usage of evaluate lower/upper and symbol on shape inference. */
TEST(type_prop, gather_v8_dynamic_value_and_symbol_propagation_interval_dim) {
    Dimension marked_0 = Dimension(2, 4);
    auto symbol = std::make_shared<ov::Symbol>();
    marked_0.set_symbol(symbol);
    PartialShape target_0 = PartialShape{marked_0, 4};

    auto param = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{1});
    auto param_0 = std::make_shared<ov::op::v0::Parameter>(element::f32, target_0);
    auto shape_0 = std::make_shared<op::v0::ShapeOf>(param_0);

    const auto& et = element::i64;
    std::vector<int64_t> zero{0};
    const auto indices = std::make_shared<op::v0::Constant>(et, Shape{zero.size()}, zero);
    const auto axis = std::make_shared<op::v0::Constant>(et, Shape{}, zero);
    const auto gather = std::make_shared<op::v8::Gather>(shape_0, indices, axis);

    auto bc = std::make_shared<op::v1::Broadcast>(param, gather);
    EXPECT_EQ(bc->get_output_partial_shape(0), PartialShape({marked_0}));

    const auto& output_shape = bc->get_output_partial_shape(0);
    EXPECT_EQ(output_shape[0].get_symbol(), symbol);
}

TEST(type_prop, gather_v8_use_default_ctor) {
    auto D = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{2, 1, 200, 400});
    auto I = make_shared<ov::op::v0::Parameter>(element::i64, PartialShape{2, 2});
    auto A = make_shared<ov::op::v0::Constant>(element::i64, Shape{1}, vector<int64_t>{-1});
    constexpr int64_t batch_dims = 1;

    auto G = make_shared<op::v8::Gather>();
    G->set_arguments(NodeVector{D, I, A});
    G->set_batch_dims(batch_dims);
    G->validate_and_infer_types();

    EXPECT_EQ(G->get_element_type(), element::f32);
    EXPECT_EQ(G->get_batch_dims(), batch_dims);
    EXPECT_EQ(G->get_axis(), 3);
    EXPECT_EQ(G->get_output_partial_shape(0), PartialShape({2, 1, 200, 2}));
}

// --------------------- V8 Negative tests ------------------------------

TEST(type_prop, gather_v8_incorrect_axis_shape) {
    auto D = make_shared<ov::op::v0::Parameter>(element::f32, Shape{5, 6});
    auto I = make_shared<ov::op::v0::Parameter>(element::i64, Shape{4});
    auto A = make_shared<ov::op::v0::Parameter>(element::i64, Shape{2});

    OV_EXPECT_THROW(auto g = make_shared<op::v7::Gather>(D, I, A),
                    NodeValidationFailure,
                    HasSubstr("Axis input must be scalar or have 1 element"));
}

TEST(type_prop, gather_v8_axis_out_of_input_rank) {
    auto D = make_shared<ov::op::v0::Parameter>(element::f32, Shape{5, 6});
    auto I = make_shared<ov::op::v0::Parameter>(element::i64, Shape{4});
    auto A = make_shared<ov::op::v0::Constant>(element::i64, Shape{1}, vector<int64_t>{2});
    int64_t batch_dims = 0;

    OV_EXPECT_THROW(auto g = make_shared<op::v8::Gather>(D, I, A, batch_dims),
                    ov::AssertFailure,
                    HasSubstr("out of the tensor rank range"));
}

TEST(type_prop, gather_v8_dynamic_batch_dims_inconsistent) {
    PartialShape data_shape{Dimension(1, 7), 20, 20};
    PartialShape indices_shape{Dimension(8, 10), 3, 8};

    auto D = make_shared<ov::op::v0::Parameter>(element::f32, data_shape);
    auto I = make_shared<ov::op::v0::Parameter>(element::i64, indices_shape);
    int64_t axis = 1;
    auto A = make_shared<ov::op::v0::Constant>(element::i64, Shape{1}, vector<int64_t>{axis});
    int64_t batch_dims = 1;

    OV_EXPECT_THROW(auto g = make_shared<op::v7::Gather>(D, I, A, batch_dims),
                    NodeValidationFailure,
                    HasSubstr("data and indices must have equal or intersecting sizes until batch_dims"));
}

TEST(type_prop, gather_v8_batch_dims_less_check) {
    PartialShape data_shape{1, 3, 20};
    PartialShape indices_shape{1, 3, 8};

    auto D = make_shared<ov::op::v0::Parameter>(element::f32, data_shape);
    auto I = make_shared<ov::op::v0::Parameter>(element::i64, indices_shape);
    int64_t axis = 1;
    auto A = make_shared<ov::op::v0::Constant>(element::i64, Shape{1}, vector<int64_t>{axis});
    int64_t batch_dims = 2;

    OV_EXPECT_THROW(auto g = make_shared<op::v7::Gather>(D, I, A, batch_dims),
                    NodeValidationFailure,
                    HasSubstr("After normalization batch_dims must be <= axis. But instead got: batch_dims ="));
}

TEST(type_prop, gather_v8_batch_dims_less_indices_rank_check) {
    PartialShape data_shape{1, 20, 20, 22, 22};
    PartialShape indices_shape{1, 3};

    auto D = make_shared<ov::op::v0::Parameter>(element::f32, data_shape);
    auto I = make_shared<ov::op::v0::Parameter>(element::i64, indices_shape);
    int64_t axis = 4;
    auto A = make_shared<ov::op::v0::Constant>(element::i64, Shape{1}, vector<int64_t>{axis});
    int64_t batch_dims = 3;

    OV_EXPECT_THROW(auto g = make_shared<op::v7::Gather>(D, I, A, batch_dims),
                    NodeValidationFailure,
                    HasSubstr("batch_dims must be <= indices_rank"));
}

TEST(type_prop, gather_v8_indices_type_check) {
    PartialShape data_shape{1, 20, 20, 22, 22};
    PartialShape indices_shape{1, 3};

    auto D = make_shared<ov::op::v0::Parameter>(element::f32, data_shape);
    auto I = make_shared<ov::op::v0::Parameter>(element::f32, indices_shape);
    int64_t axis = 4;
    auto A = make_shared<ov::op::v0::Constant>(element::i64, Shape{1}, vector<int64_t>{axis});
    int64_t batch_dims = 0;

    OV_EXPECT_THROW(auto g = make_shared<op::v7::Gather>(D, I, A, batch_dims),
                    NodeValidationFailure,
                    HasSubstr("Indices element type must be of an integral number type"));
}

TEST(type_prop, gather_v8_axis_type_check) {
    PartialShape data_shape{1, 20, 20, 22, 22};
    PartialShape indices_shape{1, 3};

    auto D = make_shared<ov::op::v0::Parameter>(element::f32, data_shape);
    auto I = make_shared<ov::op::v0::Parameter>(element::i32, indices_shape);
    int64_t axis = 4;
    auto A = make_shared<ov::op::v0::Constant>(element::f32, Shape{1}, vector<int64_t>{axis});
    int64_t batch_dims = 0;

    OV_EXPECT_THROW(auto g = make_shared<op::v7::Gather>(D, I, A, batch_dims),
                    NodeValidationFailure,
                    HasSubstr("Axis element type must be of an integral number type"));
}
