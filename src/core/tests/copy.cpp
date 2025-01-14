// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>
#include <string>

#include "openvino/core/shape.hpp"
#include "openvino/op/abs.hpp"
#include "openvino/op/acos.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/asin.hpp"
#include "openvino/op/atan.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/ceiling.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/cos.hpp"
#include "openvino/op/cosh.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/equal.hpp"
#include "openvino/op/exp.hpp"
#include "openvino/op/floor.hpp"
#include "openvino/op/greater.hpp"
#include "openvino/op/greater_eq.hpp"
#include "openvino/op/less.hpp"
#include "openvino/op/less_eq.hpp"
#include "openvino/op/log.hpp"
#include "openvino/op/loop.hpp"
#include "openvino/op/maximum.hpp"
#include "openvino/op/minimum.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/negative.hpp"
#include "openvino/op/not_equal.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/power.hpp"
#include "openvino/op/random_uniform.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/sign.hpp"
#include "openvino/op/sin.hpp"
#include "openvino/op/sinh.hpp"
#include "openvino/op/strided_slice.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/tan.hpp"
#include "openvino/op/tanh.hpp"

using namespace std;
using namespace ov;

template <typename OP>
bool check_unary() {
    ov::Shape shape{1};
    auto arg0 = make_shared<ov::op::v0::Parameter>(element::f32, shape);
    OutputVector new_args{make_shared<ov::op::v0::Parameter>(element::f32, shape)};

    auto node = make_shared<OP>(arg0);
    auto new_node = node->copy_with_new_inputs(new_args);

    return (nullptr != new_node) && (new_args == new_node->input_values());
}

template <typename OP>
bool check_binary() {
    Shape shape{1};
    auto arg0 = make_shared<ov::op::v0::Parameter>(element::f32, shape);
    auto arg1 = make_shared<ov::op::v0::Parameter>(element::f32, shape);
    OutputVector new_args{make_shared<ov::op::v0::Parameter>(element::f32, shape),
                          make_shared<ov::op::v0::Parameter>(element::f32, shape)};

    auto node = make_shared<OP>(arg0, arg1);
    auto new_node = node->copy_with_new_inputs(new_args);

    return (nullptr != new_node) && (new_args == new_node->input_values());
}

TEST(copy, abs) {
    ASSERT_TRUE(check_unary<op::v0::Abs>());
}

TEST(copy, acos) {
    ASSERT_TRUE(check_unary<op::v0::Acos>());
}

TEST(copy, add) {
    ASSERT_TRUE(check_binary<ov::op::v1::Add>());
}

TEST(copy, asin) {
    ASSERT_TRUE(check_unary<op::v0::Asin>());
}

TEST(copy, atan) {
    ASSERT_TRUE(check_unary<op::v0::Atan>());
}

TEST(copy, broadcast) {
    Shape shape{1, 3};
    Shape new_shape{4, 1, 3};
    AxisSet axes{1, 2};
    auto arg0 = make_shared<ov::op::v0::Parameter>(element::f32, shape);
    OutputVector new_args{make_shared<ov::op::v0::Parameter>(element::f32, shape),
                          ov::op::v0::Constant::create(element::u64, Shape{new_shape.size()}, new_shape),
                          ov::op::v0::Constant::create(element::i64, Shape{axes.size()}, axes.to_vector())};

    auto node = make_shared<op::v1::Broadcast>(
        arg0,
        ov::op::v0::Constant::create(element::u64, Shape{new_shape.size()}, new_shape),
        ov::op::v0::Constant::create(element::i64, Shape{axes.size()}, axes.to_vector()));
    auto new_node = node->copy_with_new_inputs(new_args);
    auto node_cast = ov::as_type_ptr<op::v1::Broadcast>(new_node);
    ASSERT_NE(node_cast, nullptr);

    ASSERT_NE(nullptr, new_node);
    ASSERT_EQ(new_args, new_node->input_values());
    bool axes_determined;
    AxisSet broadcast_axes;
    std::tie(axes_determined, broadcast_axes) = node_cast->get_broadcast_axes();
    ASSERT_EQ(true, axes_determined);
    ASSERT_EQ(AxisSet{0}, broadcast_axes);
}

TEST(copy, ceiling) {
    ASSERT_TRUE(check_unary<op::v0::Ceiling>());
}

TEST(copy, concat) {
    Shape shape{1};
    auto arg0 = make_shared<ov::op::v0::Parameter>(element::f32, shape);
    auto arg1 = make_shared<ov::op::v0::Parameter>(element::f32, shape);
    OutputVector new_args{make_shared<ov::op::v0::Parameter>(element::f32, shape),
                          make_shared<ov::op::v0::Parameter>(element::f32, shape)};
    int64_t axis = 0;
    auto node = make_shared<ov::op::v0::Concat>(NodeVector{arg0, arg1}, axis);
    auto new_node = node->clone_with_new_inputs(new_args);
    auto node_cast = ov::as_type_ptr<ov::op::v0::Concat>(new_node);
    ASSERT_NE(node_cast, nullptr);

    ASSERT_TRUE(nullptr != new_node);
    ASSERT_TRUE(new_args == new_node->input_values());
    ASSERT_TRUE(node_cast->get_axis() == axis);
}

TEST(copy, constant) {
    Shape shape{};
    vector<float> c{2.4f};
    auto& et = element::f32;
    auto node = ov::op::v0::Constant::create(et, shape, c);
    auto new_node = node->clone_with_new_inputs(OutputVector{});
    auto node_cast = ov::as_type_ptr<ov::op::v0::Constant>(new_node);
    ASSERT_NE(node_cast, nullptr);
    ASSERT_TRUE(nullptr != new_node);
    ASSERT_TRUE(OutputVector{} == new_node->input_values());
    ASSERT_TRUE(node_cast->get_vector<float>() == c);
    ASSERT_TRUE(node_cast->get_shape() == shape);
    ASSERT_TRUE(node_cast->get_element_type() == et);
}

TEST(copy, convert) {
    Shape shape;
    auto& et = element::f64;
    auto arg0 = make_shared<ov::op::v0::Parameter>(element::f32, shape);
    OutputVector new_args{make_shared<ov::op::v0::Parameter>(element::f32, shape)};

    auto node = make_shared<op::v0::Convert>(arg0, et);
    auto new_node = node->clone_with_new_inputs(new_args);
    auto node_cast = ov::as_type_ptr<op::v0::Convert>(new_node);
    ASSERT_NE(node_cast, nullptr);

    ASSERT_TRUE(nullptr != new_node);
    ASSERT_TRUE(new_args == new_node->input_values());
    ASSERT_TRUE(et == node_cast->get_convert_element_type());
}

TEST(copy, cos) {
    ASSERT_TRUE(check_unary<op::v0::Cos>());
}

TEST(copy, cosh) {
    ASSERT_TRUE(check_unary<op::v0::Cosh>());
}

TEST(copy, divide) {
    ASSERT_TRUE(check_binary<op::v1::Divide>());
}

TEST(copy, equal) {
    ASSERT_TRUE(check_binary<op::v1::Equal>());
}

TEST(copy, exp) {
    ASSERT_TRUE(check_unary<op::v0::Exp>());
}

TEST(copy, floor) {
    ASSERT_TRUE(check_unary<op::v0::Floor>());
}

TEST(copy, greater_eq) {
    ASSERT_TRUE(check_binary<op::v1::GreaterEqual>());
}

TEST(copy, greater) {
    ASSERT_TRUE(check_binary<op::v1::Greater>());
}

TEST(copy, less_eq) {
    ASSERT_TRUE(check_binary<op::v1::LessEqual>());
}

TEST(copy, less) {
    ASSERT_TRUE(check_binary<op::v1::Less>());
}

TEST(copy, log) {
    ASSERT_TRUE(check_unary<op::v0::Log>());
}

TEST(copy, maximum) {
    ASSERT_TRUE(check_binary<op::v1::Maximum>());
}

TEST(copy, minimum) {
    ASSERT_TRUE(check_binary<op::v1::Minimum>());
}

TEST(copy, multiply) {
    ASSERT_TRUE(check_binary<op::v1::Multiply>());
}

TEST(copy, negative) {
    ASSERT_TRUE(check_unary<op::v0::Negative>());
}

TEST(copy, not_equal) {
    ASSERT_TRUE(check_binary<op::v1::NotEqual>());
}

TEST(copy, parameter) {
    Shape shape{1};
    auto node = make_shared<ov::op::v0::Parameter>(element::f32, shape);
    auto new_node = node->clone_with_new_inputs({});
    auto node_cast = ov::as_type_ptr<ov::op::v0::Parameter>(new_node);
    ASSERT_NE(node_cast, nullptr);

    ASSERT_TRUE(nullptr != new_node);
    ASSERT_TRUE(new_node->input_values().size() == 0);
    ASSERT_TRUE(node->has_same_type(new_node));
}

TEST(copy, power) {
    ASSERT_TRUE(check_binary<op::v1::Power>());
}

TEST(copy, reduce_sum) {
    Shape shape{4, 3};
    AxisSet axes{1};
    auto arg0 = make_shared<ov::op::v0::Parameter>(element::f32, shape);

    auto axes_node = ov::op::v0::Constant::create(element::i64, {axes.size()}, axes.to_vector());
    auto node = make_shared<op::v1::ReduceSum>(arg0, axes_node, true);
    OutputVector new_args{make_shared<ov::op::v0::Parameter>(element::f32, shape),
                          ov::op::v0::Constant::create(element::i64, {axes.size()}, axes.to_vector())};
    auto new_node = node->clone_with_new_inputs(new_args);
    auto node_cast = ov::as_type_ptr<op::v1::ReduceSum>(new_node);
    ASSERT_NE(node_cast, nullptr);

    ASSERT_TRUE(nullptr != new_node);
    ASSERT_TRUE(new_args == new_node->input_values());
    ASSERT_TRUE(axes == node_cast->get_reduction_axes());
    ASSERT_TRUE(true == node_cast->get_keep_dims());
}

TEST(copy, reshape) {
    Shape shape_in{2, 3, 4};
    Shape shape_out{6, 4};

    auto arg0 = make_shared<ov::op::v0::Parameter>(element::f32, shape_in);
    OutputVector new_args{make_shared<ov::op::v0::Parameter>(element::f32, shape_in),
                          ov::op::v0::Constant::create(element::u64, {shape_out.size()}, shape_out)};

    auto shape_pattern = ov::op::v0::Constant::create(element::u64, {shape_out.size()}, shape_out);
    auto node = make_shared<op::v1::Reshape>(arg0, shape_pattern, false);
    auto new_node = node->clone_with_new_inputs(new_args);
    auto node_cast = ov::as_type_ptr<op::v1::Reshape>(new_node);
    ASSERT_NE(node_cast, nullptr);

    ASSERT_TRUE(nullptr != new_node);
    ASSERT_TRUE(new_args == new_node->input_values());
    ASSERT_TRUE(shape_out == node_cast->get_output_shape(0));
}

TEST(copy, select) {
    Shape shape{1};
    auto arg0 = make_shared<ov::op::v0::Parameter>(element::boolean, shape);
    auto arg1 = make_shared<ov::op::v0::Parameter>(element::f32, shape);
    auto arg2 = make_shared<ov::op::v0::Parameter>(element::f32, shape);
    OutputVector new_args{make_shared<ov::op::v0::Parameter>(element::boolean, shape),
                          make_shared<ov::op::v0::Parameter>(element::f32, shape),
                          make_shared<ov::op::v0::Parameter>(element::f32, shape)};

    auto node = make_shared<op::v1::Select>(arg0, arg1, arg2);
    auto new_node = node->clone_with_new_inputs(new_args);
    auto node_cast = ov::as_type_ptr<op::v1::Select>(new_node);
    ASSERT_NE(node_cast, nullptr);

    ASSERT_TRUE(nullptr != new_node);
    ASSERT_TRUE(new_args == new_node->input_values());
}

TEST(copy, sign) {
    ASSERT_TRUE(check_unary<op::v0::Sign>());
}

TEST(copy, sin) {
    ASSERT_TRUE(check_unary<op::v0::Sin>());
}

TEST(copy, sinh) {
    ASSERT_TRUE(check_unary<op::v0::Sinh>());
}

TEST(copy, strided_slice) {
    Shape shape_in{2, 3, 4};
    Coordinate lower{0, 0, 0};
    Coordinate upper{2, 3, 4};
    Strides strides{1, 1, 1};

    auto arg0 = make_shared<ov::op::v0::Parameter>(element::f32, shape_in);
    OutputVector new_args{make_shared<ov::op::v0::Parameter>(element::f32, shape_in),
                          ov::op::v0::Constant::create(element::u64, {lower.size()}, lower),
                          ov::op::v0::Constant::create(element::u64, {upper.size()}, upper),
                          ov::op::v0::Constant::create(element::i64, {strides.size()}, strides)};

    auto begin_node = ov::op::v0::Constant::create(element::i64, {lower.size()}, lower);
    auto end_node = ov::op::v0::Constant::create(element::i64, {upper.size()}, upper);
    auto strides_node = ov::op::v0::Constant::create(element::i64, {strides.size()}, strides);
    auto node = make_shared<op::v1::StridedSlice>(arg0,
                                                  begin_node,
                                                  end_node,
                                                  strides_node,
                                                  std::vector<int64_t>{0, 0, 1},
                                                  std::vector<int64_t>{1, 0, 0},
                                                  std::vector<int64_t>{0, 1, 0},
                                                  std::vector<int64_t>{0, 0, 1},
                                                  std::vector<int64_t>{1, 0, 0});
    auto new_node = node->clone_with_new_inputs(new_args);
    auto node_cast = ov::as_type_ptr<op::v1::StridedSlice>(new_node);
    ASSERT_NE(node_cast, nullptr);

    ASSERT_TRUE(nullptr != new_node);
    ASSERT_TRUE(new_args == new_node->input_values());
    std::vector<int64_t> expected_begin_mask{0, 0, 1};
    std::vector<int64_t> expected_end_mask{1, 0, 0};
    std::vector<int64_t> expected_new_axis_mask{0, 1, 0};
    std::vector<int64_t> expected_shrink_axis_mask{0, 0, 1};
    std::vector<int64_t> expected_ellipsis_mask{1, 0, 0};
    ASSERT_TRUE(expected_begin_mask == node_cast->get_begin_mask());
    ASSERT_TRUE(expected_end_mask == node_cast->get_end_mask());
    ASSERT_TRUE(expected_new_axis_mask == node_cast->get_new_axis_mask());
    ASSERT_TRUE(expected_shrink_axis_mask == node_cast->get_shrink_axis_mask());
    ASSERT_TRUE(expected_ellipsis_mask == node_cast->get_ellipsis_mask());
}

TEST(copy, subtract) {
    ASSERT_TRUE(check_binary<op::v1::Subtract>());
}

TEST(copy, tan) {
    ASSERT_TRUE(check_unary<op::v0::Tan>());
}

TEST(copy, tanh) {
    ASSERT_TRUE(check_unary<op::v0::Tanh>());
}

TEST(copy, loop) {
    // That which we iterate over
    auto X = make_shared<op::v0::Parameter>(element::f32, Shape{32, 1, 10});
    auto Y = make_shared<op::v0::Parameter>(element::f32, Shape{32, 1, 10});
    auto M = make_shared<op::v0::Parameter>(element::f32, Shape{32, 1, 10});

    // Set up the cell body, a function from (Xi, Yi) -> (Zo)
    // Body parameters
    auto current_iteration = make_shared<op::v0::Parameter>(element::i64, Shape{});
    auto Xi = make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());
    auto Yi = make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());
    auto M_body = make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());
    auto body_condition = std::make_shared<ov::op::v0::Constant>(ov::element::boolean, ov::Shape{}, true);

    auto trip_count = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{}, 10);
    auto exec_condition = std::make_shared<ov::op::v0::Constant>(ov::element::boolean, ov::Shape{}, true);
    // Body
    auto sum = make_shared<ov::op::v1::Add>(Xi, Yi);
    auto Zo = make_shared<ov::op::v1::Multiply>(sum, M_body);
    auto body =
        make_shared<ov::Model>(OutputVector{Zo, body_condition}, ParameterVector{Xi, current_iteration, Yi, M_body});

    auto loop = make_shared<op::v5::Loop>(trip_count, exec_condition);
    loop->set_function(body);
    loop->set_special_body_ports(ov::op::v5::Loop::SpecialBodyPorts{1, 1});

    loop->set_invariant_input(Xi, X);
    loop->set_invariant_input(Yi, Y);
    loop->set_merged_input(M_body, M, Zo);

    // Output 0 is last Zo
    auto out0 = loop->get_iter_value(body_condition, -1);
    auto out1 = loop->get_iter_value(Zo, -1);
    // Output 1 is concat of Zos
    // start=0, stride=1, part_size=1, end=-1, axis=1
    auto out2 = loop->get_concatenated_slices(Zo, 0, 1, 1, -1, 1);
    loop->validate_and_infer_types();
    // That which we iterate over
    auto X_new = make_shared<op::v0::Parameter>(element::f32, Shape{3, 2, 5});
    auto Y_new = make_shared<op::v0::Parameter>(element::f32, Shape{3, 2, 5});
    auto M_new = make_shared<op::v0::Parameter>(element::f32, Shape{3, 2, 5});
    OutputVector new_args = {trip_count, exec_condition, X_new, Y_new, M_new};
    auto loop_copy = loop->clone_with_new_inputs(new_args);

    auto node_cast = ov::as_type_ptr<op::v5::Loop>(loop_copy);
    ASSERT_NE(node_cast, nullptr);
    ASSERT_TRUE(nullptr != loop_copy);
    EXPECT_EQ(loop->get_num_iterations(), node_cast->get_num_iterations());
    EXPECT_EQ(loop->get_special_body_ports().body_condition_output_idx,
              node_cast->get_special_body_ports().body_condition_output_idx);
    EXPECT_EQ(loop->get_special_body_ports().current_iteration_input_idx,
              node_cast->get_special_body_ports().current_iteration_input_idx);
    ASSERT_TRUE(new_args == loop_copy->input_values());

    loop_copy->validate_and_infer_types();
    Shape out0_shape{};
    Shape out1_shape{3, 2, 5};
    Shape out2_shape{3, 20, 5};
    EXPECT_EQ(loop_copy->get_output_shape(0), out0_shape);
    EXPECT_EQ(loop_copy->get_output_shape(1), out1_shape);
    EXPECT_EQ(loop_copy->get_output_shape(2), out2_shape);
}

TEST(copy, random_uniform) {
    auto shape = std::vector<int64_t>{1, 2, 3};
    float min = 0., max = 1.;

    const auto min_val_param = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1});
    const auto max_val_param = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1});
    auto out_shape = make_shared<ov::op::v0::Constant>(element::i64, Shape{3}, shape);
    auto ru =
        std::make_shared<ov::op::v8::RandomUniform>(out_shape, min_val_param, max_val_param, element::f32, 150, 10);

    // Call `evaluate` to update m_state
    auto outputs = ov::TensorVector{{element::i64, {1lu, 2lu, 3lu}}};
    ru->evaluate(outputs,
                 ov::TensorVector{{element::i64, out_shape->get_shape(), shape.data()},
                                  {element::f32, min_val_param->get_shape(), &min},
                                  {element::f32, max_val_param->get_shape(), &max}});

    auto out_shape_c = make_shared<ov::op::v0::Constant>(element::i64, Shape{4}, std::vector<int64_t>{4, 3, 2, 1});
    const auto min_val_param_c = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1});
    const auto max_val_param_c = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1});
    OutputVector new_args{out_shape_c, min_val_param_c, max_val_param_c};
    auto new_ru = ru->clone_with_new_inputs(new_args);
    auto node_cast = ov::as_type_ptr<ov::op::v8::RandomUniform>(new_ru);
    ASSERT_NE(node_cast, nullptr);

    ASSERT_TRUE(nullptr != new_ru);
    ASSERT_TRUE(new_args == new_ru->input_values());
    ASSERT_TRUE(ru->get_out_type() == node_cast->get_out_type());
    ASSERT_TRUE(out_shape_c->get_shape_val() == node_cast->get_shape());
    ASSERT_TRUE(ru->get_global_seed() == node_cast->get_global_seed());
    ASSERT_TRUE(ru->get_op_seed() == node_cast->get_op_seed());
    ASSERT_TRUE(ru->get_state() == node_cast->get_state());
}
