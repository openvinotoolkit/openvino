//*****************************************************************************
// Copyright 2017-2021 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include "gtest/gtest.h"

#include "ngraph/ngraph.hpp"
#include "ngraph/opsets/opset5.hpp"
#include "ngraph/pass/constant_folding.hpp"
#include "ngraph/pass/manager.hpp"
#include "util/all_close_f.hpp"
#include "util/test_tools.hpp"

using namespace ngraph;
using namespace std;

template <typename T>
static std::vector<T> get_result_constant(std::shared_ptr<Function> f, size_t pos)
{
    auto new_const =
        as_type_ptr<op::Constant>(f->get_results().at(pos)->input_value(0).get_node_shared_ptr());
    return new_const->cast_vector<T>();
}

void range_test_check(const vector<double>& values_out, const vector<double>& values_expected)
{
    ASSERT_TRUE(test::all_close_f(values_out, values_expected, MIN_FLOAT_TOLERANCE_BITS));
}

void range_test_check(const vector<float>& values_out, const vector<float>& values_expected)
{
    ASSERT_TRUE(test::all_close_f(values_out, values_expected, MIN_FLOAT_TOLERANCE_BITS));
}

template <typename T>
typename std::enable_if<std::is_integral<T>::value>::type
    range_test_check(const vector<T>& values_out, const vector<T>& values_expected)
{
    ASSERT_EQ(values_out, values_expected);
}

TEST(constant_folding, acosh)
{
    Shape shape_in{2, 4, 1};

    vector<float> values_in{0, 1, 2, 3, 4, 5, 6, 7};
    vector<float> expected;
    for (float f : values_in)
    {
        expected.push_back(std::acosh(f));
    }
    auto constant = make_shared<op::Constant>(element::f32, shape_in, values_in);
    auto acosh = make_shared<op::Acosh>(constant);
    acosh->set_friendly_name("test");
    auto f = make_shared<Function>(acosh, ParameterVector{});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(f);

    EXPECT_EQ(count_ops_of_type<op::Acosh>(f), 0);
    EXPECT_EQ(count_ops_of_type<op::Constant>(f), 1);
    ASSERT_EQ(f->get_results().size(), 1);

    auto new_const =
        as_type_ptr<op::Constant>(f->get_results()[0]->input_value(0).get_node_shared_ptr());
    EXPECT_TRUE(new_const);
    ASSERT_EQ(new_const->get_friendly_name(), "test");

    auto values_out = new_const->get_vector<float>();
    EXPECT_TRUE(test::all_close_f(expected, values_out, MIN_FLOAT_TOLERANCE_BITS));
}

TEST(constant_folding, asinh)
{
    Shape shape_in{2, 4, 1};

    vector<float> values_in{0, 1, 2, 3, 4, 5, 6, 7};
    vector<float> expected;
    for (float f : values_in)
    {
        expected.push_back(std::asinh(f));
    }
    auto constant = make_shared<op::Constant>(element::f32, shape_in, values_in);
    auto asinh = make_shared<op::Asinh>(constant);
    asinh->set_friendly_name("test");
    auto f = make_shared<Function>(asinh, ParameterVector{});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(f);

    EXPECT_EQ(count_ops_of_type<op::Asinh>(f), 0);
    EXPECT_EQ(count_ops_of_type<op::Constant>(f), 1);
    ASSERT_EQ(f->get_results().size(), 1);

    auto new_const =
        as_type_ptr<op::Constant>(f->get_results()[0]->input_value(0).get_node_shared_ptr());
    EXPECT_TRUE(new_const);
    ASSERT_EQ(new_const->get_friendly_name(), "test");

    auto values_out = new_const->get_vector<float>();
    EXPECT_TRUE(test::all_close_f(expected, values_out, MIN_FLOAT_TOLERANCE_BITS));
}

TEST(constant_folding, atanh)
{
    Shape shape_in{2, 4, 1};

    vector<float> values_in{0, 1, 2, 3, 4, 5, 6, 7};
    vector<float> expected;
    for (float f : values_in)
    {
        expected.push_back(std::atanh(f));
    }
    auto constant = make_shared<op::Constant>(element::f32, shape_in, values_in);
    auto atanh = make_shared<op::Atanh>(constant);
    atanh->set_friendly_name("test");
    auto f = make_shared<Function>(atanh, ParameterVector{});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(f);

    EXPECT_EQ(count_ops_of_type<op::Atanh>(f), 0);
    EXPECT_EQ(count_ops_of_type<op::Constant>(f), 1);
    ASSERT_EQ(f->get_results().size(), 1);

    auto new_const =
        as_type_ptr<op::Constant>(f->get_results()[0]->input_value(0).get_node_shared_ptr());
    EXPECT_TRUE(new_const);
    ASSERT_EQ(new_const->get_friendly_name(), "test");

    auto values_out = new_const->get_vector<float>();
    EXPECT_TRUE(test::all_close_f(expected, values_out, MIN_FLOAT_TOLERANCE_BITS));
}

TEST(constant_folding, constant_squeeze)
{
    Shape shape_in{2, 4, 1};
    Shape shape_out{2, 4};
    Shape axes_shape{1};

    vector<float> values_in{0, 1, 2, 3, 4, 5, 6, 7};
    auto constant = make_shared<op::Constant>(element::f32, shape_in, values_in);
    vector<int64_t> values_axes{2};
    auto constant_axes = op::Constant::create(element::i64, axes_shape, values_axes);
    auto squeeze = make_shared<op::Squeeze>(constant, constant_axes);
    squeeze->set_friendly_name("test");
    auto f = make_shared<Function>(squeeze, ParameterVector{});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::Squeeze>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::Constant>(f), 1);

    auto new_const =
        as_type_ptr<op::Constant>(f->get_results().at(0)->input_value(0).get_node_shared_ptr());
    ASSERT_TRUE(new_const);
    ASSERT_EQ(new_const->get_friendly_name(), "test");
    ASSERT_EQ(new_const->get_shape(), shape_out);

    auto values_out = new_const->get_vector<float>();
    ASSERT_TRUE(test::all_close_f(values_in, values_out, MIN_FLOAT_TOLERANCE_BITS));
}

TEST(constant_folding, constant_unsqueeze)
{
    Shape shape_in{2, 4};
    Shape shape_out{2, 4, 1, 1};
    Shape axes_shape{2};

    vector<float> values_in{0, 1, 2, 3, 4, 5, 6, 7};
    auto constant = make_shared<op::Constant>(element::f32, shape_in, values_in);
    vector<int64_t> values_axes{2, 3};
    auto constant_axes = op::Constant::create(element::i64, axes_shape, values_axes);
    auto unsqueeze = make_shared<op::v0::Unsqueeze>(constant, constant_axes);
    unsqueeze->set_friendly_name("test");
    auto f = make_shared<Function>(unsqueeze, ParameterVector{});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::v0::Unsqueeze>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::Constant>(f), 1);

    auto new_const =
        as_type_ptr<op::Constant>(f->get_results().at(0)->input_value(0).get_node_shared_ptr());
    ASSERT_TRUE(new_const);
    ASSERT_EQ(new_const->get_friendly_name(), "test");
    ASSERT_EQ(new_const->get_shape(), shape_out);

    auto values_out = new_const->get_vector<float>();
    ASSERT_TRUE(test::all_close_f(values_in, values_out, MIN_FLOAT_TOLERANCE_BITS));
}

TEST(constant_folding, constant_broadcast_v1)
{
    vector<int32_t> values_in{0, 1};
    auto constant_in = make_shared<op::Constant>(element::i32, Shape{2}, values_in);
    vector<int64_t> shape_in{2, 4};
    auto constant_shape = make_shared<op::Constant>(element::i64, Shape{2}, shape_in);
    vector<int64_t> axes_in{0};
    auto constant_axes = make_shared<op::Constant>(element::i64, Shape{1}, axes_in);
    auto broadcast_v1 = make_shared<op::v1::Broadcast>(constant_in, constant_shape, constant_axes);
    broadcast_v1->set_friendly_name("test");
    auto f = make_shared<Function>(broadcast_v1, ParameterVector{});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::v1::Broadcast>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::Constant>(f), 1);

    auto new_const =
        as_type_ptr<op::Constant>(f->get_results().at(0)->input_value(0).get_node_shared_ptr());
    ASSERT_TRUE(new_const);
    ASSERT_EQ(new_const->get_friendly_name(), "test");
    auto values_out = new_const->get_vector<int32_t>();

    vector<int32_t> values_expected{0, 0, 0, 0, 1, 1, 1, 1};
    ASSERT_EQ(values_expected, values_out);
}

TEST(constant_folding, constant_broadcast_v1_with_target_shape)
{
    vector<int32_t> values_in{1};
    auto constant_in = make_shared<op::Constant>(element::i32, Shape{1, 1, 1, 1}, values_in);
    vector<int64_t> shape_in{1, 3, 1, 1};
    auto target_shape = make_shared<op::Constant>(element::i64, Shape{4}, shape_in);
    auto broadcast_v1 = make_shared<op::v1::Broadcast>(constant_in, target_shape);
    broadcast_v1->set_friendly_name("test");
    auto f = make_shared<Function>(broadcast_v1, ParameterVector{});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::v1::Broadcast>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::Constant>(f), 1);

    auto new_const =
        as_type_ptr<op::Constant>(f->get_results().at(0)->input_value(0).get_node_shared_ptr());
    ASSERT_TRUE(new_const);
    ASSERT_EQ(new_const->get_friendly_name(), "test");
    auto values_out = new_const->get_vector<int32_t>();

    vector<int32_t> values_expected{1, 1, 1};
    ASSERT_EQ(values_expected, values_out);
}

TEST(constant_folding, constant_broadcast_v1_numpy)
{
    vector<int32_t> values_in{0, 1};
    auto constant_in = make_shared<op::Constant>(element::i32, Shape{2}, values_in);
    vector<int64_t> shape_in{4, 2};
    auto constant_shape = make_shared<op::Constant>(element::i64, Shape{2}, shape_in);
    auto broadcast_v1 = make_shared<op::v1::Broadcast>(constant_in, constant_shape);
    broadcast_v1->set_friendly_name("test");
    auto f = make_shared<Function>(broadcast_v1, ParameterVector{});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::v1::Broadcast>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::Constant>(f), 1);

    auto new_const =
        as_type_ptr<op::Constant>(f->get_results().at(0)->input_value(0).get_node_shared_ptr());
    ASSERT_TRUE(new_const);
    ASSERT_EQ(new_const->get_friendly_name(), "test");
    auto values_out = new_const->get_vector<int32_t>();

    vector<int32_t> values_expected{0, 1, 0, 1, 0, 1, 0, 1};
    ASSERT_EQ(values_expected, values_out);
}

TEST(constant_folding, constant_unary_binary)
{
    vector<int> values_a{1, 2, 3, 4};
    vector<int> values_b{1, 2, 3, 4};
    vector<int> values_c{-1, -1, -1, -1};
    vector<int> values_d{1, 4, 9, 16};
    vector<int> values_e{5, 6};
    vector<int> values_f{0, 10};
    vector<int> values_g{1, 4};
    vector<char> values_h{0, 0, 1, 1};
    vector<char> values_i{0, 1};
    auto a = make_shared<op::Constant>(element::i32, Shape{2, 2}, values_a);
    auto b = make_shared<op::Constant>(element::i32, Shape{2, 2}, values_b);
    auto c = make_shared<op::Constant>(element::i32, Shape{2, 2}, values_c);
    auto d = make_shared<op::Constant>(element::i32, Shape{2, 2}, values_d);
    auto e = make_shared<op::Constant>(element::i32, Shape{2}, values_e);
    auto f = make_shared<op::Constant>(element::i32, Shape{2}, values_f);
    auto g = make_shared<op::Constant>(element::i32, Shape{2}, values_g);
    auto h = make_shared<op::Constant>(element::boolean, Shape{2, 2}, values_h);
    auto i = make_shared<op::Constant>(element::boolean, Shape{2}, values_i);

    auto add = make_shared<op::v1::Add>(a, b);
    auto sub = make_shared<op::v1::Subtract>(a, b);
    auto mul = make_shared<op::v1::Multiply>(a, b);
    auto divn = make_shared<op::v1::Divide>(a, b);
    auto pow = make_shared<op::v1::Power>(a, b);
    auto min = make_shared<op::v1::Minimum>(c, a);
    auto max = make_shared<op::v1::Maximum>(a, c);
    auto absn = make_shared<op::Abs>(c);
    auto neg = make_shared<op::Negative>(c);
    auto sqrt = make_shared<op::Sqrt>(d);
    auto add_autob_numpy = make_shared<op::v1::Add>(a, e, op::AutoBroadcastType::NUMPY);
    auto sub_autob_numpy = make_shared<op::v1::Subtract>(a, e, op::AutoBroadcastType::NUMPY);
    auto mul_autob_numpy = make_shared<op::v1::Multiply>(a, e, op::AutoBroadcastType::NUMPY);
    auto div_autob_numpy = make_shared<op::v1::Divide>(a, g, op::AutoBroadcastType::NUMPY);
    auto pow_autob_numpy = make_shared<op::v1::Power>(a, g, op::AutoBroadcastType::NUMPY);
    auto min_autob_numpy = make_shared<op::v1::Minimum>(a, f, op::AutoBroadcastType::NUMPY);
    auto max_autob_numpy = make_shared<op::v1::Maximum>(a, f, op::AutoBroadcastType::NUMPY);
    auto equal_autob_numpy = make_shared<op::v1::Equal>(a, g, op::AutoBroadcastType::NUMPY);
    auto not_equal_autob_numpy = make_shared<op::v1::NotEqual>(a, g, op::AutoBroadcastType::NUMPY);
    auto greater_autob_numpy = make_shared<op::v1::Greater>(a, g, op::AutoBroadcastType::NUMPY);
    auto greater_eq_autob_numpy =
        make_shared<op::v1::GreaterEqual>(a, g, op::AutoBroadcastType::NUMPY);
    auto less_autob_numpy = make_shared<op::v1::Less>(a, g, op::AutoBroadcastType::NUMPY);
    auto less_eq_autob_numpy = make_shared<op::v1::LessEqual>(a, g, op::AutoBroadcastType::NUMPY);
    auto logical_or_autob_numpy =
        make_shared<op::v1::LogicalOr>(h, i, op::AutoBroadcastType::NUMPY);
    auto logical_xor_autob_numpy = make_shared<op::Xor>(h, i, op::AutoBroadcastType::NUMPY);

    auto neg_sqrt = make_shared<op::Sqrt>(c);

    auto func = make_shared<Function>(NodeVector{add,
                                                 sub,
                                                 mul,
                                                 divn,
                                                 pow,
                                                 min,
                                                 max,
                                                 absn,
                                                 neg,
                                                 sqrt,
                                                 add_autob_numpy,
                                                 sub_autob_numpy,
                                                 mul_autob_numpy,
                                                 div_autob_numpy,
                                                 pow_autob_numpy,
                                                 min_autob_numpy,
                                                 max_autob_numpy,
                                                 equal_autob_numpy,
                                                 not_equal_autob_numpy,
                                                 greater_autob_numpy,
                                                 greater_eq_autob_numpy,
                                                 less_autob_numpy,
                                                 less_eq_autob_numpy,
                                                 logical_or_autob_numpy,
                                                 logical_xor_autob_numpy},
                                      ParameterVector{});
    auto func_error = make_shared<Function>(NodeVector{neg_sqrt}, ParameterVector{});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(func);

    // expected values
    vector<int> add_expected{2, 4, 6, 8};
    vector<int> sub_expected{0, 0, 0, 0};
    vector<int> mul_expected{1, 4, 9, 16};
    vector<int> div_expected{1, 1, 1, 1};
    vector<int> pow_expected{1, 4, 27, 256};
    vector<int> min_expected{-1, -1, -1, -1};
    vector<int> max_expected{1, 2, 3, 4};
    vector<int> abs_neg_expected{1, 1, 1, 1};
    vector<int> sqrt_expected{1, 2, 3, 4};
    vector<int> add_autob_numpy_expected{6, 8, 8, 10};
    vector<int> sub_autob_numpy_expected{-4, -4, -2, -2};
    vector<int> mul_autob_numpy_expected{5, 12, 15, 24};
    vector<int> div_autob_numpy_expected{1, 0, 3, 1};
    vector<int> pow_autob_numpy_expected{1, 16, 3, 256};
    vector<int> min_autob_numpy_expected{0, 2, 0, 4};
    vector<int> max_autob_numpy_expected{1, 10, 3, 10};
    vector<char> equal_autob_numpy_expected{1, 0, 0, 1};
    vector<char> not_equal_autob_numpy_expected{0, 1, 1, 0};
    vector<char> greater_autob_numpy_expected{0, 0, 1, 0};
    vector<char> greater_eq_autob_numpy_expected{1, 0, 1, 1};
    vector<char> less_autob_numpy_expected{0, 1, 0, 0};
    vector<char> less_eq_autob_numpy_expected{1, 1, 0, 1};
    vector<char> logical_or_autob_numpy_expected{0, 1, 1, 1};
    vector<char> logical_xor_autob_numpy_expected{0, 1, 1, 0};

    ASSERT_EQ(get_result_constant<int>(func, 0), add_expected);
    ASSERT_EQ(get_result_constant<int>(func, 1), sub_expected);
    ASSERT_EQ(get_result_constant<int>(func, 2), mul_expected);
    ASSERT_EQ(get_result_constant<int>(func, 3), div_expected);
    ASSERT_EQ(get_result_constant<int>(func, 4), pow_expected);
    ASSERT_EQ(get_result_constant<int>(func, 5), min_expected);
    ASSERT_EQ(get_result_constant<int>(func, 6), max_expected);
    ASSERT_EQ(get_result_constant<int>(func, 7), abs_neg_expected);
    ASSERT_EQ(get_result_constant<int>(func, 8), abs_neg_expected);
    ASSERT_EQ(get_result_constant<int>(func, 9), sqrt_expected);
    ASSERT_EQ(get_result_constant<int>(func, 10), add_autob_numpy_expected);
    ASSERT_EQ(get_result_constant<int>(func, 11), sub_autob_numpy_expected);
    ASSERT_EQ(get_result_constant<int>(func, 12), mul_autob_numpy_expected);
    ASSERT_EQ(get_result_constant<int>(func, 13), div_autob_numpy_expected);
    ASSERT_EQ(get_result_constant<int>(func, 14), pow_autob_numpy_expected);
    ASSERT_EQ(get_result_constant<int>(func, 15), min_autob_numpy_expected);
    ASSERT_EQ(get_result_constant<int>(func, 16), max_autob_numpy_expected);
    ASSERT_EQ(get_result_constant<char>(func, 17), equal_autob_numpy_expected);
    ASSERT_EQ(get_result_constant<char>(func, 18), not_equal_autob_numpy_expected);
    ASSERT_EQ(get_result_constant<char>(func, 19), greater_autob_numpy_expected);
    ASSERT_EQ(get_result_constant<char>(func, 20), greater_eq_autob_numpy_expected);
    ASSERT_EQ(get_result_constant<char>(func, 21), less_autob_numpy_expected);
    ASSERT_EQ(get_result_constant<char>(func, 22), less_eq_autob_numpy_expected);
    ASSERT_EQ(get_result_constant<char>(func, 23), logical_or_autob_numpy_expected);
    ASSERT_EQ(get_result_constant<char>(func, 24), logical_xor_autob_numpy_expected);
    ASSERT_NO_THROW(pass_manager.run_passes(func_error));
}

TEST(constant_folding, const_convert)
{
    Shape input_shape{3, 4};

    vector<int32_t> values_in{1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7};
    auto constant = op::Constant::create(element::f32, input_shape, values_in);
    auto convert = make_shared<op::Convert>(constant, element::u64);
    convert->set_friendly_name("test");
    auto f = make_shared<Function>(convert, ParameterVector{});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::Convert>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::Constant>(f), 1);

    auto new_const =
        as_type_ptr<op::Constant>(f->get_results().at(0)->input_value(0).get_node_shared_ptr());
    ASSERT_TRUE(new_const);
    ASSERT_EQ(new_const->get_friendly_name(), "test");
    ASSERT_EQ(new_const->get_output_element_type(0), element::u64);
    auto values_out = new_const->get_vector<uint64_t>();

    vector<uint64_t> values_expected{1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7};
    ASSERT_EQ(values_expected, values_out);
}

TEST(constant_folding, shape_of_v0)
{
    Shape input_shape{3, 4, 0, 22, 608, 909, 3};

    auto param = make_shared<op::Parameter>(element::boolean, input_shape);
    auto shape_of = make_shared<op::v0::ShapeOf>(param);
    shape_of->set_friendly_name("test");
    auto f = make_shared<Function>(shape_of, ParameterVector{param});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::v0::ShapeOf>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::Constant>(f), 1);

    auto new_const =
        as_type_ptr<op::Constant>(f->get_results().at(0)->input_value(0).get_node_shared_ptr());
    ASSERT_TRUE(new_const);
    ASSERT_EQ(new_const->get_friendly_name(), "test");
    ASSERT_EQ(new_const->get_output_element_type(0), element::i64);
    auto values_out = new_const->get_vector<int64_t>();

    ASSERT_EQ((vector<int64_t>{3, 4, 0, 22, 608, 909, 3}), values_out);
}

TEST(constant_folding, shape_of_v3)
{
    Shape input_shape{3, 4, 0, 22, 608, 909, 3};

    auto param = make_shared<op::Parameter>(element::boolean, input_shape);
    auto shape_of = make_shared<op::v3::ShapeOf>(param);
    shape_of->set_friendly_name("test");
    auto f = make_shared<Function>(shape_of, ParameterVector{param});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::v3::ShapeOf>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::Constant>(f), 1);

    auto new_const =
        as_type_ptr<op::Constant>(f->get_results().at(0)->input_value(0).get_node_shared_ptr());
    ASSERT_TRUE(new_const);
    ASSERT_EQ(new_const->get_friendly_name(), "test");
    ASSERT_EQ(new_const->get_output_element_type(0), element::i64);
    auto values_out = new_const->get_vector<int64_t>();

    ASSERT_EQ((vector<int64_t>{3, 4, 0, 22, 608, 909, 3}), values_out);
}

TEST(constant_folding, shape_of_i32_v3)
{
    Shape input_shape{3, 4, 0, 22, 608, 909, 3};

    auto param = make_shared<op::Parameter>(element::boolean, input_shape);
    auto shape_of = make_shared<op::v3::ShapeOf>(param, element::i32);
    shape_of->set_friendly_name("test");
    auto f = make_shared<Function>(shape_of, ParameterVector{param});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::v3::ShapeOf>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::Constant>(f), 1);

    auto new_const =
        as_type_ptr<op::Constant>(f->get_results().at(0)->input_value(0).get_node_shared_ptr());
    ASSERT_TRUE(new_const);
    ASSERT_EQ(new_const->get_friendly_name(), "test");
    ASSERT_EQ(new_const->get_output_element_type(0), element::i32);
    auto values_out = new_const->get_vector<int32_t>();

    ASSERT_EQ((vector<int32_t>{3, 4, 0, 22, 608, 909, 3}), values_out);
}

TEST(constant_folding, shape_of_dynamic_v0)
{
    PartialShape input_shape{3, 4, Dimension::dynamic(), 22, 608, 909, 3};

    auto param = make_shared<op::Parameter>(element::boolean, input_shape);
    auto shape_of = make_shared<op::v0::ShapeOf>(param);
    shape_of->set_friendly_name("test");
    auto f = make_shared<Function>(shape_of, ParameterVector{param});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::v0::ShapeOf>(f), 1);
    ASSERT_EQ(count_ops_of_type<op::v1::Gather>(f), 1);
    ASSERT_EQ(count_ops_of_type<op::Concat>(f), 1);
    ASSERT_EQ(count_ops_of_type<op::Constant>(f), 8);

    auto result_as_concat =
        as_type_ptr<op::Concat>(f->get_results().at(0)->input_value(0).get_node_shared_ptr());
    ASSERT_TRUE(result_as_concat);
    ASSERT_EQ(result_as_concat->get_friendly_name(), "test");
    ASSERT_EQ(result_as_concat->get_output_shape(0), Shape{7});
}

TEST(constant_folding, shape_of_dynamic_v3)
{
    PartialShape input_shape{3, 4, Dimension::dynamic(), 22, 608, 909, 3};

    auto param = make_shared<op::Parameter>(element::boolean, input_shape);
    auto shape_of = make_shared<op::v3::ShapeOf>(param);
    shape_of->set_friendly_name("test");
    auto f = make_shared<Function>(shape_of, ParameterVector{param});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::v3::ShapeOf>(f), 1);
    ASSERT_EQ(count_ops_of_type<op::v1::Gather>(f), 1);
    ASSERT_EQ(count_ops_of_type<op::Concat>(f), 1);
    ASSERT_EQ(count_ops_of_type<op::Constant>(f), 8);

    auto result_as_concat =
        as_type_ptr<op::Concat>(f->get_results().at(0)->input_value(0).get_node_shared_ptr());
    ASSERT_TRUE(result_as_concat);
    ASSERT_EQ(result_as_concat->get_friendly_name(), "test");
    ASSERT_EQ(result_as_concat->get_output_shape(0), Shape{7});
    ASSERT_EQ(result_as_concat->get_output_element_type(0), element::i64);
}

TEST(constant_folding, shape_of_dynamic_i32_v3)
{
    PartialShape input_shape{3, 4, Dimension::dynamic(), 22, 608, 909, 3};

    auto param = make_shared<op::Parameter>(element::boolean, input_shape);
    auto shape_of = make_shared<op::v3::ShapeOf>(param, element::i32);
    shape_of->set_friendly_name("test");
    auto f = make_shared<Function>(shape_of, ParameterVector{param});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::v3::ShapeOf>(f), 1);
    ASSERT_EQ(count_ops_of_type<op::v1::Gather>(f), 1);
    ASSERT_EQ(count_ops_of_type<op::Concat>(f), 1);
    ASSERT_EQ(count_ops_of_type<op::Constant>(f), 8);

    auto result_as_concat =
        as_type_ptr<op::Concat>(f->get_results().at(0)->input_value(0).get_node_shared_ptr());
    ASSERT_TRUE(result_as_concat);
    ASSERT_EQ(result_as_concat->get_friendly_name(), "test");
    ASSERT_EQ(result_as_concat->get_output_shape(0), Shape{7});
    ASSERT_EQ(result_as_concat->get_output_element_type(0), element::i32);
}

// We need to be sure that constant folding won't be calculated endlessly.
TEST(constant_folding, shape_of_dynamic_double_folding_v0)
{
    PartialShape input_shape{3, 4, Dimension::dynamic(), 22, 608, 909, 3};

    auto param = make_shared<op::Parameter>(element::boolean, input_shape);
    auto shape_of = make_shared<op::v0::ShapeOf>(param);
    shape_of->set_friendly_name("test");
    auto f = make_shared<Function>(shape_of, ParameterVector{param});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(f);
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::v0::ShapeOf>(f), 1);
    ASSERT_EQ(count_ops_of_type<op::v1::Gather>(f), 1);
    ASSERT_EQ(count_ops_of_type<op::Concat>(f), 1);
    ASSERT_EQ(count_ops_of_type<op::Constant>(f), 8);

    auto result_as_concat =
        as_type_ptr<op::Concat>(f->get_results().at(0)->input_value(0).get_node_shared_ptr());
    ASSERT_TRUE(result_as_concat);
    ASSERT_EQ(result_as_concat->get_friendly_name(), "test");
    ASSERT_EQ(result_as_concat->get_output_shape(0), Shape{7});
}

TEST(constant_folding, shape_of_dynamic_double_folding_v3)
{
    PartialShape input_shape{3, 4, Dimension::dynamic(), 22, 608, 909, 3};

    auto param = make_shared<op::Parameter>(element::boolean, input_shape);
    auto shape_of = make_shared<op::v3::ShapeOf>(param);
    shape_of->set_friendly_name("test");
    auto f = make_shared<Function>(shape_of, ParameterVector{param});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(f);
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::v3::ShapeOf>(f), 1);
    ASSERT_EQ(count_ops_of_type<op::v1::Gather>(f), 1);
    ASSERT_EQ(count_ops_of_type<op::Concat>(f), 1);
    ASSERT_EQ(count_ops_of_type<op::Constant>(f), 8);

    auto result_as_concat =
        as_type_ptr<op::Concat>(f->get_results().at(0)->input_value(0).get_node_shared_ptr());
    ASSERT_TRUE(result_as_concat);
    ASSERT_EQ(result_as_concat->get_friendly_name(), "test");
    ASSERT_EQ(result_as_concat->get_output_shape(0), Shape{7});
}

// Constant folding will not succeed on ShapeOf if the argument rank is dynamic.
// We want to make sure it fails gracefully, leaving the ShapeOf op in place.
TEST(constant_folding, shape_of_rank_dynamic_v0)
{
    PartialShape input_shape{PartialShape::dynamic()};

    auto param = make_shared<op::Parameter>(element::boolean, input_shape);
    auto shape_of = make_shared<op::v0::ShapeOf>(param);
    shape_of->set_friendly_name("test");
    auto f = make_shared<Function>(shape_of, ParameterVector{param});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::v0::ShapeOf>(f), 1);
    ASSERT_EQ(count_ops_of_type<op::Constant>(f), 0);

    auto result_shape_of = f->get_results().at(0)->get_input_node_shared_ptr(0);
    ASSERT_EQ(result_shape_of, shape_of);
    ASSERT_EQ(result_shape_of->get_friendly_name(), "test");
}

TEST(constant_folding, shape_of_rank_dynamic_v3)
{
    PartialShape input_shape{PartialShape::dynamic()};

    auto param = make_shared<op::Parameter>(element::boolean, input_shape);
    auto shape_of = make_shared<op::v3::ShapeOf>(param);
    shape_of->set_friendly_name("test");
    auto f = make_shared<Function>(shape_of, ParameterVector{param});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::v3::ShapeOf>(f), 1);
    ASSERT_EQ(count_ops_of_type<op::Constant>(f), 0);

    auto result_shape_of = f->get_results().at(0)->get_input_node_shared_ptr(0);
    ASSERT_EQ(result_shape_of, shape_of);
    ASSERT_EQ(result_shape_of->get_friendly_name(), "test");
}

void const_reverse(const element::Type& axes_elem_type)
{
    Shape input_shape{3, 3};

    vector<int32_t> values_in{1, 2, 3, 4, 5, 6, 7, 8, 9};
    auto constant = op::Constant::create(element::i32, input_shape, values_in);
    auto axes = op::Constant::create(axes_elem_type, {1}, {1});
    auto convert = make_shared<op::v1::Reverse>(constant, axes, op::v1::Reverse::Mode::INDEX);
    convert->set_friendly_name("test");
    auto f = make_shared<Function>(convert, ParameterVector{});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::v1::Reverse>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::Constant>(f), 1);

    auto new_const =
        as_type_ptr<op::Constant>(f->get_results().at(0)->input_value(0).get_node_shared_ptr());
    ASSERT_TRUE(new_const);
    ASSERT_EQ(new_const->get_friendly_name(), "test");
    auto values_out = new_const->get_vector<int32_t>();

    vector<int32_t> values_expected{3, 2, 1, 6, 5, 4, 9, 8, 7};
    ASSERT_EQ(values_expected, values_out);
}

TEST(constant_folding, const_reverse)
{
    for (auto&& axes_elem_type : {element::i8,
                                  element::u8,
                                  element::i16,
                                  element::u16,
                                  element::i32,
                                  element::u32,
                                  element::i64,
                                  element::u64})
    {
        const_reverse(axes_elem_type);
    }
}

TEST(constant_folding, const_reduceprod)
{
    Shape input_shape{3, 3};
    Shape output_shape{3};

    vector<int32_t> values_in{1, 2, 3, 4, 5, 6, 7, 8, 9};
    auto constant = op::Constant::create(element::i32, input_shape, values_in);
    Shape axes_shape{1};
    vector<int32_t> values_axes{1};
    auto constant_axes = op::Constant::create(element::i64, axes_shape, values_axes);
    auto convert = make_shared<op::v1::ReduceProd>(constant, constant_axes);
    convert->set_friendly_name("test");
    auto f = make_shared<Function>(convert, ParameterVector{});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::v1::ReduceProd>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::Constant>(f), 1);

    auto new_const =
        as_type_ptr<op::Constant>(f->get_results().at(0)->input_value(0).get_node_shared_ptr());
    ASSERT_TRUE(new_const);
    ASSERT_EQ(new_const->get_friendly_name(), "test");
    ASSERT_EQ(new_const->get_shape(), output_shape);

    auto values_out = new_const->get_vector<int32_t>();

    vector<int32_t> values_expected{6, 120, 504};

    ASSERT_EQ(values_expected, values_out);
}

TEST(constant_folding, const_reduceprod_keepdims)
{
    Shape input_shape{3, 3};
    Shape output_shape{3, 1};

    vector<int32_t> values_in{1, 2, 3, 4, 5, 6, 7, 8, 9};
    auto constant = op::Constant::create(element::i32, input_shape, values_in);
    Shape axes_shape{1};
    vector<int32_t> values_axes{1};
    auto constant_axes = op::Constant::create(element::i64, axes_shape, values_axes);
    auto convert = make_shared<op::v1::ReduceProd>(constant, constant_axes, true);
    convert->set_friendly_name("test");
    auto f = make_shared<Function>(convert, ParameterVector{});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::v1::ReduceProd>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::Constant>(f), 1);

    auto new_const =
        as_type_ptr<op::Constant>(f->get_results().at(0)->input_value(0).get_node_shared_ptr());
    ASSERT_TRUE(new_const);
    ASSERT_EQ(new_const->get_friendly_name(), "test");
    ASSERT_EQ(new_const->get_shape(), output_shape);

    auto values_out = new_const->get_vector<int32_t>();

    vector<int32_t> values_expected{6, 120, 504};

    ASSERT_EQ(values_expected, values_out);
}

TEST(constant_folding, const_reducesum)
{
    Shape input_shape{3, 3};
    Shape output_shape{3};

    vector<int32_t> values_in{1, 2, 3, 4, 5, 6, 7, 8, 9};
    auto constant = op::Constant::create(element::i32, input_shape, values_in);
    Shape axes_shape{1};
    vector<int32_t> values_axes{1};
    auto constant_axes = op::Constant::create(element::i64, axes_shape, values_axes);
    auto convert = make_shared<op::v1::ReduceSum>(constant, constant_axes);
    convert->set_friendly_name("test");
    auto f = make_shared<Function>(convert, ParameterVector{});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::v1::ReduceSum>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::Constant>(f), 1);

    auto new_const =
        as_type_ptr<op::Constant>(f->get_results().at(0)->input_value(0).get_node_shared_ptr());
    ASSERT_TRUE(new_const);
    ASSERT_EQ(new_const->get_friendly_name(), "test");
    ASSERT_EQ(new_const->get_shape(), output_shape);

    auto values_out = new_const->get_vector<int32_t>();

    vector<int32_t> values_expected{6, 15, 24};

    ASSERT_EQ(values_expected, values_out);
}

TEST(constant_folding, const_reducesum_keepdims)
{
    Shape input_shape{3, 3};
    Shape output_shape{3, 1};

    vector<int32_t> values_in{1, 2, 3, 4, 5, 6, 7, 8, 9};
    auto constant = op::Constant::create(element::i32, input_shape, values_in);
    Shape axes_shape{1};
    vector<int32_t> values_axes{1};
    auto constant_axes = op::Constant::create(element::i64, axes_shape, values_axes);
    auto convert = make_shared<op::v1::ReduceSum>(constant, constant_axes, true);
    convert->set_friendly_name("test");
    auto f = make_shared<Function>(convert, ParameterVector{});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::v1::ReduceSum>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::Constant>(f), 1);

    auto new_const =
        as_type_ptr<op::Constant>(f->get_results().at(0)->input_value(0).get_node_shared_ptr());
    ASSERT_TRUE(new_const);
    ASSERT_EQ(new_const->get_friendly_name(), "test");
    ASSERT_EQ(new_const->get_shape(), output_shape);

    auto values_out = new_const->get_vector<int32_t>();

    vector<int32_t> values_expected{6, 15, 24};

    ASSERT_EQ(values_expected, values_out);
}

TEST(constant_folding, const_reducemax)
{
    Shape input_shape{3, 2};
    Shape output_shape{3};

    vector<int32_t> values_in{1, 2, 3, 4, 5, 6};
    auto constant = op::Constant::create(element::i32, input_shape, values_in);
    Shape axes_shape{1};
    vector<int32_t> values_axes{1};
    auto constant_axes = op::Constant::create(element::i64, axes_shape, values_axes);
    auto convert = make_shared<op::v1::ReduceMax>(constant, constant_axes);
    convert->set_friendly_name("test");
    auto f = make_shared<Function>(convert, ParameterVector{});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::v1::ReduceMax>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::Constant>(f), 1);

    auto new_const =
        as_type_ptr<op::Constant>(f->get_results().at(0)->input_value(0).get_node_shared_ptr());
    ASSERT_TRUE(new_const);
    ASSERT_EQ(new_const->get_friendly_name(), "test");
    ASSERT_EQ(new_const->get_shape(), output_shape);

    auto values_out = new_const->get_vector<int32_t>();

    vector<int32_t> values_expected{2, 4, 6};

    ASSERT_EQ(values_expected, values_out);
}

TEST(constant_folding, const_reducemax_keepdims)
{
    Shape input_shape{3, 2};
    Shape output_shape{3, 1};

    vector<int32_t> values_in{1, 2, 3, 4, 5, 6};
    auto constant = op::Constant::create(element::i32, input_shape, values_in);
    Shape axes_shape{1};
    vector<int32_t> values_axes{1};
    auto constant_axes = op::Constant::create(element::i64, axes_shape, values_axes);
    auto convert = make_shared<op::v1::ReduceMax>(constant, constant_axes, true);
    convert->set_friendly_name("test");
    auto f = make_shared<Function>(convert, ParameterVector{});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::v1::ReduceMax>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::Constant>(f), 1);

    auto new_const =
        as_type_ptr<op::Constant>(f->get_results().at(0)->input_value(0).get_node_shared_ptr());
    ASSERT_TRUE(new_const);
    ASSERT_EQ(new_const->get_friendly_name(), "test");
    ASSERT_EQ(new_const->get_shape(), output_shape);

    auto values_out = new_const->get_vector<int32_t>();

    vector<int32_t> values_expected{2, 4, 6};

    ASSERT_EQ(values_expected, values_out);
}

TEST(constant_folding, const_reducemin)
{
    Shape input_shape{3, 2};
    Shape output_shape{3};

    vector<int32_t> values_in{1, 2, 3, 4, 5, 6};
    auto constant = op::Constant::create(element::i32, input_shape, values_in);
    Shape axes_shape{1};
    vector<int32_t> values_axes{1};
    auto constant_axes = op::Constant::create(element::i64, axes_shape, values_axes);
    auto convert = make_shared<op::v1::ReduceMin>(constant, constant_axes);
    convert->set_friendly_name("test");
    auto f = make_shared<Function>(convert, ParameterVector{});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::v1::ReduceMin>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::Constant>(f), 1);

    auto new_const =
        as_type_ptr<op::Constant>(f->get_results().at(0)->input_value(0).get_node_shared_ptr());
    ASSERT_TRUE(new_const);
    ASSERT_EQ(new_const->get_friendly_name(), "test");
    ASSERT_EQ(new_const->get_shape(), output_shape);

    auto values_out = new_const->get_vector<int32_t>();

    vector<int32_t> values_expected{1, 3, 5};

    ASSERT_EQ(values_expected, values_out);
}

TEST(constant_folding, const_reducemin_keepdims)
{
    Shape input_shape{3, 2};
    Shape output_shape{3, 1};

    vector<int32_t> values_in{1, 2, 3, 4, 5, 6};
    auto constant = op::Constant::create(element::i32, input_shape, values_in);
    Shape axes_shape{1};
    vector<int32_t> values_axes{1};
    auto constant_axes = op::Constant::create(element::i64, axes_shape, values_axes);
    auto convert = make_shared<op::v1::ReduceMin>(constant, constant_axes, true);
    convert->set_friendly_name("test");
    auto f = make_shared<Function>(convert, ParameterVector{});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::v1::ReduceMin>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::Constant>(f), 1);

    auto new_const =
        as_type_ptr<op::Constant>(f->get_results().at(0)->input_value(0).get_node_shared_ptr());
    ASSERT_TRUE(new_const);
    ASSERT_EQ(new_const->get_friendly_name(), "test");
    ASSERT_EQ(new_const->get_shape(), output_shape);

    auto values_out = new_const->get_vector<int32_t>();

    vector<int32_t> values_expected{1, 3, 5};

    ASSERT_EQ(values_expected, values_out);
}

TEST(constant_folding, const_reducemean)
{
    Shape input_shape{3, 3};
    Shape output_shape{3};

    vector<int32_t> values_in{1, 2, 3, 4, 5, 6, 7, 8, 9};
    auto constant = op::Constant::create(element::i32, input_shape, values_in);
    Shape axes_shape{1};
    vector<int32_t> values_axes{1};
    auto constant_axes = op::Constant::create(element::i64, axes_shape, values_axes);
    auto convert = make_shared<op::v1::ReduceMean>(constant, constant_axes);
    convert->set_friendly_name("test");
    auto f = make_shared<Function>(convert, ParameterVector{});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::v1::ReduceMean>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::Constant>(f), 1);

    auto new_const =
        as_type_ptr<op::Constant>(f->get_results().at(0)->input_value(0).get_node_shared_ptr());
    ASSERT_TRUE(new_const);
    ASSERT_EQ(new_const->get_friendly_name(), "test");
    ASSERT_EQ(new_const->get_shape(), output_shape);

    auto values_out = new_const->get_vector<int32_t>();

    vector<int32_t> values_expected{2, 5, 8};

    ASSERT_EQ(values_expected, values_out);
}

TEST(constant_folding, const_reducemean_keepdims)
{
    Shape input_shape{3, 3};
    Shape output_shape{3, 1};

    vector<int32_t> values_in{1, 2, 3, 4, 5, 6, 7, 8, 9};
    auto constant = op::Constant::create(element::i32, input_shape, values_in);
    Shape axes_shape{1};
    vector<int32_t> values_axes{1};
    auto constant_axes = op::Constant::create(element::i64, axes_shape, values_axes);
    auto convert = make_shared<op::v1::ReduceMean>(constant, constant_axes, true);
    convert->set_friendly_name("test");
    auto f = make_shared<Function>(convert, ParameterVector{});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::v1::ReduceMean>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::Constant>(f), 1);

    auto new_const =
        as_type_ptr<op::Constant>(f->get_results().at(0)->input_value(0).get_node_shared_ptr());
    ASSERT_TRUE(new_const);
    ASSERT_EQ(new_const->get_friendly_name(), "test");
    ASSERT_EQ(new_const->get_shape(), output_shape);

    auto values_out = new_const->get_vector<int32_t>();

    vector<int32_t> values_expected{2, 5, 8};

    ASSERT_EQ(values_expected, values_out);
}

TEST(constant_folding, const_reduce_logical_and__no_keepdims)
{
    const Shape input_shape{3, 3};

    const vector<char> values_in{0, 1, 1, 0, 1, 0, 1, 1, 1};
    const auto data = op::Constant::create(element::boolean, input_shape, values_in);
    const auto axes = op::Constant::create(element::i64, {1}, {1});
    const auto convert = make_shared<op::v1::ReduceLogicalAnd>(data, axes, false);
    convert->set_friendly_name("test");
    auto f = make_shared<Function>(convert, ParameterVector{});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::v1::ReduceLogicalAnd>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::Constant>(f), 1);

    const auto new_const =
        as_type_ptr<op::Constant>(f->get_results().at(0)->input_value(0).get_node_shared_ptr());
    ASSERT_TRUE(new_const);
    ASSERT_EQ(new_const->get_friendly_name(), "test");

    const Shape expected_out_shape{3};
    ASSERT_EQ(new_const->get_shape(), expected_out_shape);

    const auto values_out = new_const->get_vector<char>();

    const vector<char> values_expected{0, 0, 1};

    ASSERT_EQ(values_expected, values_out);
}

TEST(constant_folding, const_reduce_logical_and__keepdims)
{
    const Shape input_shape{3, 3};

    const vector<char> values_in{0, 1, 1, 0, 1, 0, 1, 1, 1};
    const auto data = op::Constant::create(element::boolean, input_shape, values_in);
    const auto axes = op::Constant::create(element::i64, {1}, {1});
    const auto convert = make_shared<op::v1::ReduceLogicalAnd>(data, axes, true);
    convert->set_friendly_name("test");
    auto f = make_shared<Function>(convert, ParameterVector{});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::v1::ReduceLogicalAnd>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::Constant>(f), 1);

    const auto new_const =
        as_type_ptr<op::Constant>(f->get_results().at(0)->input_value(0).get_node_shared_ptr());
    ASSERT_TRUE(new_const);
    ASSERT_EQ(new_const->get_friendly_name(), "test");

    // the output shape is expected to have 'ones' at the positions specified in the reduction axes
    // in case the keep_dims attribute of ReduceLogicalAnd is set to true
    const Shape expected_out_shape{3, 1};
    ASSERT_EQ(new_const->get_shape(), expected_out_shape);

    const auto values_out = new_const->get_vector<char>();

    const vector<char> values_expected{0, 0, 1};

    ASSERT_EQ(values_expected, values_out);
}

TEST(constant_folding, const_reduce_logical_and__keepdims_3d)
{
    const Shape input_shape{2, 2, 2};

    const vector<char> values_in{1, 1, 0, 0, 1, 0, 0, 1};
    const auto data = op::Constant::create(element::boolean, input_shape, values_in);
    const auto axes = op::Constant::create(element::i64, {2}, {0, 2});
    const auto convert = make_shared<op::v1::ReduceLogicalAnd>(data, axes, true);
    convert->set_friendly_name("test");
    auto f = make_shared<Function>(convert, ParameterVector{});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::v1::ReduceLogicalAnd>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::Constant>(f), 1);

    const auto new_const =
        as_type_ptr<op::Constant>(f->get_results().at(0)->input_value(0).get_node_shared_ptr());
    ASSERT_TRUE(new_const);
    ASSERT_EQ(new_const->get_friendly_name(), "test");

    const Shape expected_out_shape{1, 2, 1};
    ASSERT_EQ(new_const->get_shape(), expected_out_shape);

    const auto values_out = new_const->get_vector<char>();

    const vector<char> values_expected{0, 0};

    ASSERT_EQ(values_expected, values_out);
}

TEST(constant_folding, const_reduce_logical_or__no_keepdims)
{
    const Shape input_shape{3, 3};

    const vector<char> values_in{1, 0, 0, 1, 0, 1, 0, 0, 0};
    const auto data = op::Constant::create(element::boolean, input_shape, values_in);
    const auto axes = op::Constant::create(element::i64, {1}, {1});
    const auto convert = make_shared<op::v1::ReduceLogicalOr>(data, axes, false);
    convert->set_friendly_name("test");
    auto f = make_shared<Function>(convert, ParameterVector{});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::v1::ReduceLogicalAnd>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::Constant>(f), 1);

    const auto new_const =
        as_type_ptr<op::Constant>(f->get_results().at(0)->input_value(0).get_node_shared_ptr());
    ASSERT_TRUE(new_const);
    ASSERT_EQ(new_const->get_friendly_name(), "test");

    const Shape expected_out_shape{3};
    ASSERT_EQ(new_const->get_shape(), expected_out_shape);

    const auto values_out = new_const->get_vector<char>();

    const vector<char> values_expected{1, 1, 0};

    ASSERT_EQ(values_expected, values_out);
}

TEST(constant_folding, const_concat)
{
    auto constant0 =
        op::Constant::create(element::i32, Shape{2, 3}, vector<int32_t>{1, 2, 3, 4, 5, 6});
    auto constant1 = op::Constant::create(element::i32, Shape{2, 1}, vector<int32_t>{7, 8});
    auto concat = make_shared<op::Concat>(NodeVector{constant0, constant1}, 1);
    concat->set_friendly_name("test");
    auto f = make_shared<Function>(concat, ParameterVector{});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::Concat>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::Constant>(f), 1);

    auto new_const =
        as_type_ptr<op::Constant>(f->get_results().at(0)->input_value(0).get_node_shared_ptr());
    ASSERT_TRUE(new_const);
    ASSERT_EQ(new_const->get_friendly_name(), "test");
    auto values_out = new_const->get_vector<int32_t>();

    vector<int32_t> values_expected{1, 2, 3, 7, 4, 5, 6, 8};

    ASSERT_EQ(values_expected, values_out);
}

TEST(constant_folding, const_concat_3d_single_elem)
{
    auto constant_1 = op::Constant::create(element::i32, Shape{1, 1, 1}, vector<int32_t>{1});
    auto constant_2 = op::Constant::create(element::i32, Shape{1, 1, 1}, vector<int32_t>{2});
    auto concat = make_shared<op::Concat>(NodeVector{constant_1, constant_2}, 0);
    concat->set_friendly_name("test");
    auto f = make_shared<Function>(concat, ParameterVector{});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::Concat>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::Constant>(f), 1);

    auto new_const =
        as_type_ptr<op::Constant>(f->get_results().at(0)->input_value(0).get_node_shared_ptr());

    ASSERT_TRUE(new_const);
    ASSERT_EQ(new_const->get_friendly_name(), "test");
    ASSERT_EQ(new_const->get_output_shape(0), (Shape{2, 1, 1}));

    auto values_out = new_const->get_vector<int32_t>();
    vector<int32_t> values_expected{1, 2};
    ASSERT_EQ(values_expected, values_out);
}

TEST(constant_folding, const_concat_axis_2)
{
    auto constant_1 =
        op::Constant::create(element::i32, Shape{3, 1, 2}, vector<int32_t>{1, 2, 3, 4, 5, 6});
    auto constant_2 = op::Constant::create(
        element::i32, Shape{3, 1, 4}, vector<int32_t>{7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18});
    auto concat = make_shared<op::Concat>(NodeVector{constant_1, constant_2}, 2);
    concat->set_friendly_name("test");
    auto f = make_shared<Function>(concat, ParameterVector{});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::Concat>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::Constant>(f), 1);

    auto new_const =
        as_type_ptr<op::Constant>(f->get_results().at(0)->input_value(0).get_node_shared_ptr());

    ASSERT_TRUE(new_const);
    ASSERT_EQ(new_const->get_friendly_name(), "test");
    ASSERT_EQ(new_const->get_output_shape(0), (Shape{3, 1, 6}));

    auto values_out = new_const->get_vector<int32_t>();
    vector<int32_t> values_expected{1, 2, 7, 8, 9, 10, 3, 4, 11, 12, 13, 14, 5, 6, 15, 16, 17, 18};
    ASSERT_EQ(values_expected, values_out);
}

TEST(constant_folding, const_concat_axis_1_bool_type)
{
    auto constant_1 =
        op::Constant::create(element::boolean, Shape{1, 1, 2}, vector<int32_t>{true, true});
    auto constant_2 = op::Constant::create(
        element::boolean, Shape{1, 2, 2}, vector<char>{true, false, true, false});
    auto constant_3 = op::Constant::create(
        element::boolean, Shape{1, 3, 2}, vector<char>{true, false, true, false, true, false});
    auto concat = make_shared<op::Concat>(NodeVector{constant_1, constant_2, constant_3}, 1);
    concat->set_friendly_name("test");
    auto f = make_shared<Function>(concat, ParameterVector{});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::Concat>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::Constant>(f), 1);

    auto new_const =
        as_type_ptr<op::Constant>(f->get_results().at(0)->input_value(0).get_node_shared_ptr());

    ASSERT_TRUE(new_const);
    ASSERT_EQ(new_const->get_friendly_name(), "test");
    ASSERT_EQ(new_const->get_output_shape(0), (Shape{1, 6, 2}));

    auto values_out = new_const->get_vector<char>();
    vector<char> values_expected{
        true, true, true, false, true, false, true, false, true, false, true, false};
    ASSERT_EQ(values_expected, values_out);
}

TEST(constant_folding, const_logical_not)
{
    auto constant =
        op::Constant::create(element::boolean, Shape{2, 3}, vector<char>{0, 1, 0, 0, 1, 1});
    auto logical_not = make_shared<op::v1::LogicalNot>(constant);
    logical_not->set_friendly_name("test");
    auto f = make_shared<Function>(logical_not, ParameterVector{});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::v1::LogicalNot>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::Constant>(f), 1);

    auto new_const =
        as_type_ptr<op::Constant>(f->get_results().at(0)->input_value(0).get_node_shared_ptr());
    ASSERT_TRUE(new_const);
    ASSERT_EQ(new_const->get_friendly_name(), "test");
    auto values_out = new_const->get_vector<char>();

    vector<char> values_expected{1, 0, 1, 1, 0, 0};

    ASSERT_EQ(values_expected, values_out);
}

TEST(constant_folding, const_equal)
{
    auto constant0 =
        op::Constant::create(element::i32, Shape{2, 3}, vector<int32_t>{1, 2, 3, 4, 5, 6});
    auto constant1 =
        op::Constant::create(element::i32, Shape{2, 3}, vector<int32_t>{1, 2, 2, 3, 5, 6});
    auto eq = make_shared<op::v1::Equal>(constant0, constant1);
    eq->set_friendly_name("test");
    auto f = make_shared<Function>(eq, ParameterVector{});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::v1::Equal>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::Constant>(f), 1);

    auto new_const =
        as_type_ptr<op::Constant>(f->get_results().at(0)->input_value(0).get_node_shared_ptr());
    ASSERT_TRUE(new_const);
    ASSERT_EQ(new_const->get_friendly_name(), "test");
    auto values_out = new_const->get_vector<char>();

    vector<char> values_expected{1, 1, 0, 0, 1, 1};

    ASSERT_EQ(values_expected, values_out);
}

TEST(constant_folding, const_not_equal)
{
    auto constant0 =
        op::Constant::create(element::i32, Shape{2, 3}, vector<int32_t>{1, 2, 3, 4, 5, 6});
    auto constant1 =
        op::Constant::create(element::i32, Shape{2, 3}, vector<int32_t>{1, 2, 2, 3, 5, 6});
    auto eq = make_shared<op::v1::NotEqual>(constant0, constant1);
    eq->set_friendly_name("test");
    auto f = make_shared<Function>(eq, ParameterVector{});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::v1::NotEqual>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::Constant>(f), 1);

    auto new_const =
        as_type_ptr<op::Constant>(f->get_results().at(0)->input_value(0).get_node_shared_ptr());
    ASSERT_TRUE(new_const);
    ASSERT_EQ(new_const->get_friendly_name(), "test");
    auto values_out = new_const->get_vector<char>();

    vector<char> values_expected{0, 0, 1, 1, 0, 0};

    ASSERT_EQ(values_expected, values_out);
}

TEST(constant_folding, const_greater)
{
    auto constant0 =
        op::Constant::create(element::i32, Shape{2, 3}, vector<int32_t>{1, 2, 3, 4, 5, 6});
    auto constant1 =
        op::Constant::create(element::i32, Shape{2, 3}, vector<int32_t>{2, 2, 2, 5, 5, 5});
    auto eq = make_shared<op::v1::Greater>(constant0, constant1);
    eq->set_friendly_name("test");
    auto f = make_shared<Function>(eq, ParameterVector{});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::v1::Greater>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::Constant>(f), 1);

    auto new_const =
        as_type_ptr<op::Constant>(f->get_results().at(0)->input_value(0).get_node_shared_ptr());
    ASSERT_TRUE(new_const);
    ASSERT_EQ(new_const->get_friendly_name(), "test");
    auto values_out = new_const->get_vector<char>();

    vector<char> values_expected{0, 0, 1, 0, 0, 1};

    ASSERT_EQ(values_expected, values_out);
}

TEST(constant_folding, const_greater_eq)
{
    auto constant0 =
        op::Constant::create(element::i32, Shape{2, 3}, vector<int32_t>{1, 2, 3, 4, 5, 6});
    auto constant1 =
        op::Constant::create(element::i32, Shape{2, 3}, vector<int32_t>{2, 2, 2, 5, 5, 5});
    auto eq = make_shared<op::v1::GreaterEqual>(constant0, constant1);
    eq->set_friendly_name("test");
    auto f = make_shared<Function>(eq, ParameterVector{});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::v1::GreaterEqual>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::Constant>(f), 1);

    auto new_const =
        as_type_ptr<op::Constant>(f->get_results().at(0)->input_value(0).get_node_shared_ptr());
    ASSERT_TRUE(new_const);
    ASSERT_EQ(new_const->get_friendly_name(), "test");
    auto values_out = new_const->get_vector<char>();

    vector<char> values_expected{0, 1, 1, 0, 1, 1};

    ASSERT_EQ(values_expected, values_out);
}

TEST(constant_folding, const_less)
{
    auto constant0 =
        op::Constant::create(element::i32, Shape{2, 3}, vector<int32_t>{1, 2, 3, 4, 5, 6});
    auto constant1 =
        op::Constant::create(element::i32, Shape{2, 3}, vector<int32_t>{2, 2, 2, 5, 5, 5});
    auto eq = make_shared<op::v1::Less>(constant0, constant1);
    eq->set_friendly_name("test");
    auto f = make_shared<Function>(eq, ParameterVector{});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::v1::Less>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::Constant>(f), 1);

    auto new_const =
        as_type_ptr<op::Constant>(f->get_results().at(0)->input_value(0).get_node_shared_ptr());
    ASSERT_TRUE(new_const);
    ASSERT_EQ(new_const->get_friendly_name(), "test");
    auto values_out = new_const->get_vector<char>();

    vector<char> values_expected{1, 0, 0, 1, 0, 0};

    ASSERT_EQ(values_expected, values_out);
}

TEST(constant_folding, const_less_eq)
{
    auto constant0 =
        op::Constant::create(element::i32, Shape{2, 3}, vector<int32_t>{1, 2, 3, 4, 5, 6});
    auto constant1 =
        op::Constant::create(element::i32, Shape{2, 3}, vector<int32_t>{2, 2, 2, 5, 5, 5});
    auto eq = make_shared<op::v1::LessEqual>(constant0, constant1);
    eq->set_friendly_name("test");
    auto f = make_shared<Function>(eq, ParameterVector{});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::v1::LessEqual>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::Constant>(f), 1);

    auto new_const =
        as_type_ptr<op::Constant>(f->get_results().at(0)->input_value(0).get_node_shared_ptr());
    ASSERT_TRUE(new_const);
    ASSERT_EQ(new_const->get_friendly_name(), "test");
    auto values_out = new_const->get_vector<char>();

    vector<char> values_expected{1, 1, 0, 1, 1, 0};

    ASSERT_EQ(values_expected, values_out);
}

TEST(constant_folding, const_or)
{
    auto constant0 =
        op::Constant::create(element::boolean, Shape{2, 3}, vector<int32_t>{0, 0, 1, 0, 1, 1});
    auto constant1 =
        op::Constant::create(element::boolean, Shape{2, 3}, vector<int32_t>{0, 1, 1, 1, 0, 1});
    auto eq = make_shared<op::v1::LogicalOr>(constant0, constant1);
    eq->set_friendly_name("test");
    auto f = make_shared<Function>(eq, ParameterVector{});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::v1::LogicalOr>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::Constant>(f), 1);

    auto new_const =
        as_type_ptr<op::Constant>(f->get_results().at(0)->input_value(0).get_node_shared_ptr());
    ASSERT_TRUE(new_const);
    ASSERT_EQ(new_const->get_friendly_name(), "test");
    auto values_out = new_const->get_vector<char>();

    vector<char> values_expected{0, 1, 1, 1, 1, 1};

    ASSERT_EQ(values_expected, values_out);
}

TEST(constant_folding, const_xor)
{
    auto constant0 =
        op::Constant::create(element::boolean, Shape{2, 3}, vector<int32_t>{0, 0, 1, 0, 1, 1});
    auto constant1 =
        op::Constant::create(element::boolean, Shape{2, 3}, vector<int32_t>{0, 1, 1, 1, 0, 1});
    auto eq = make_shared<op::Xor>(constant0, constant1);
    eq->set_friendly_name("test");
    auto f = make_shared<Function>(eq, ParameterVector{});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::Xor>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::Constant>(f), 1);

    auto new_const =
        as_type_ptr<op::Constant>(f->get_results().at(0)->input_value(0).get_node_shared_ptr());
    ASSERT_TRUE(new_const);
    ASSERT_EQ(new_const->get_friendly_name(), "test");
    auto values_out = new_const->get_vector<char>();

    vector<char> values_expected{0, 1, 0, 1, 1, 0};

    ASSERT_EQ(values_expected, values_out);
}

TEST(constant_folding, const_ceiling)
{
    auto constant = op::Constant::create(
        element::f32, Shape{2, 3}, vector<float>{0.0f, 0.1f, -0.1f, -2.5f, 2.5f, 3.0f});
    auto ceil = make_shared<op::Ceiling>(constant);
    ceil->set_friendly_name("test");
    auto f = make_shared<Function>(ceil, ParameterVector{});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::Ceiling>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::Constant>(f), 1);

    auto new_const =
        as_type_ptr<op::Constant>(f->get_results().at(0)->input_value(0).get_node_shared_ptr());
    ASSERT_TRUE(new_const);
    ASSERT_EQ(new_const->get_friendly_name(), "test");
    auto values_out = new_const->get_vector<float>();

    vector<float> values_expected{0.0f, 1.0f, 0.0f, -2.0f, 3.0f, 3.0f};

    ASSERT_TRUE(test::all_close_f(values_out, values_expected, MIN_FLOAT_TOLERANCE_BITS));
}

TEST(constant_folding, const_floor)
{
    auto constant = op::Constant::create(
        element::f32, Shape{2, 3}, vector<float>{0.0f, 0.1f, -0.1f, -2.5f, 2.5f, 3.0f});
    auto floor = make_shared<op::Floor>(constant);
    floor->set_friendly_name("test");
    auto f = make_shared<Function>(floor, ParameterVector{});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::Floor>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::Constant>(f), 1);

    auto new_const =
        as_type_ptr<op::Constant>(f->get_results().at(0)->input_value(0).get_node_shared_ptr());
    ASSERT_TRUE(new_const);
    ASSERT_EQ(new_const->get_friendly_name(), "test");
    auto values_out = new_const->get_vector<float>();

    vector<float> values_expected{0.0f, 0.0f, -1.0f, -3.0f, 2.0f, 3.0f};

    ASSERT_TRUE(test::all_close_f(values_out, values_expected, MIN_FLOAT_TOLERANCE_BITS));
}

TEST(constant_folding, const_gather_v1)
{
    auto constant_data = op::Constant::create(
        element::f32,
        Shape{2, 5},
        vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f});
    auto constant_indices =
        op::Constant::create(element::i64, Shape{4}, vector<int64_t>{0, 3, 2, 2});
    auto constant_axis = op::Constant::create(element::i64, Shape{1}, vector<int64_t>{1});
    auto gather = make_shared<op::v1::Gather>(constant_data, constant_indices, constant_axis);
    gather->set_friendly_name("test");
    auto f = make_shared<Function>(gather, ParameterVector{});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::v1::Gather>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::Constant>(f), 1);

    auto new_const =
        as_type_ptr<op::Constant>(f->get_results().at(0)->input_value(0).get_node_shared_ptr());
    ASSERT_TRUE(new_const);
    ASSERT_EQ(new_const->get_friendly_name(), "test");
    auto values_out = new_const->get_vector<float>();

    vector<float> values_expected{1.0f, 4.0f, 3.0f, 3.0f, 6.0f, 9.0f, 8.0f, 8.0f};

    ASSERT_TRUE(test::all_close_f(values_out, values_expected, MIN_FLOAT_TOLERANCE_BITS));
}

TEST(constant_folding, const_gather_v1_scalar)
{
    auto constant_data = op::Constant::create(
        element::f32,
        Shape{2, 5},
        vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f});
    auto constant_indices =
        op::Constant::create(element::i64, Shape{4}, vector<int64_t>{0, 3, 2, 2});
    auto constant_axis = op::Constant::create(element::i64, Shape{}, vector<int64_t>{1});
    auto gather = make_shared<op::v1::Gather>(constant_data, constant_indices, constant_axis);
    gather->set_friendly_name("test");
    auto f = make_shared<Function>(gather, ParameterVector{});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::v1::Gather>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::Constant>(f), 1);

    auto new_const =
        as_type_ptr<op::Constant>(f->get_results().at(0)->input_value(0).get_node_shared_ptr());
    ASSERT_TRUE(new_const);
    ASSERT_EQ(new_const->get_friendly_name(), "test");
    auto values_out = new_const->get_vector<float>();

    vector<float> values_expected{1.0f, 4.0f, 3.0f, 3.0f, 6.0f, 9.0f, 8.0f, 8.0f};

    ASSERT_TRUE(test::all_close_f(values_out, values_expected, MIN_FLOAT_TOLERANCE_BITS));
}

TEST(constant_folding, const_gather_v1_subgraph)
{
    const auto A = make_shared<op::Parameter>(element::f32, Shape{1});
    const float b_value = 3.21f;
    const auto B_const = op::Constant::create(element::f32, {1}, {b_value});
    const auto C = make_shared<op::Parameter>(element::f32, Shape{1});
    const int64_t axis = 0;
    const auto axis_const = op::Constant::create(element::i64, {}, {axis});

    const auto concat = make_shared<op::Concat>(NodeVector{A, B_const, C}, axis);

    const vector<int64_t> indices{1};
    const auto indices_const = op::Constant::create(element::i64, {indices.size()}, indices);
    const auto gather = make_shared<op::v1::Gather>(concat, indices_const, axis_const);
    gather->set_friendly_name("test");
    auto f = make_shared<Function>(gather, ParameterVector{A, C});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::Concat>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::v1::Gather>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::Constant>(f), 1);

    const auto new_const =
        as_type_ptr<op::Constant>(f->get_results().at(0)->input_value(0).get_node_shared_ptr());
    ASSERT_TRUE(new_const);
    ASSERT_EQ(new_const->get_friendly_name(), "test");

    const auto values_out = new_const->get_vector<float>();
    ASSERT_TRUE(test::all_close_f(values_out, {b_value}, MIN_FLOAT_TOLERANCE_BITS));
}

TEST(constant_folding, const_gather_v1_subgraph_neg_axis)
{
    const auto A = make_shared<op::Parameter>(element::f32, Shape{1});
    const float b_value = 1.23f;
    const auto B = make_shared<op::Parameter>(element::f32, Shape{1});
    const auto C_const = op::Constant::create(element::f32, {1}, {b_value});
    const int64_t axis = 0;
    const auto axis_const = op::Constant::create(element::i64, {}, {axis});

    const auto concat = make_shared<op::Concat>(NodeVector{A, B, C_const}, axis);

    const vector<int64_t> indices{-1};
    const auto indices_const = op::Constant::create(element::i64, {indices.size()}, indices);
    const auto gather = make_shared<op::v1::Gather>(concat, indices_const, axis_const);
    gather->set_friendly_name("test");
    auto f = make_shared<Function>(gather, ParameterVector{A, B});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::Concat>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::v1::Gather>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::Constant>(f), 1);

    const auto new_const =
        as_type_ptr<op::Constant>(f->get_results().at(0)->input_value(0).get_node_shared_ptr());
    ASSERT_TRUE(new_const);
    ASSERT_EQ(new_const->get_friendly_name(), "test");

    const auto values_out = new_const->get_vector<float>();
    ASSERT_TRUE(test::all_close_f(values_out, {b_value}, MIN_FLOAT_TOLERANCE_BITS));
}

TEST(constant_folding, const_gather_v1_subgraph_no_constant_input)
{
    const auto A = make_shared<op::Parameter>(element::f32, Shape{1});
    const auto B = make_shared<op::Parameter>(element::f32, Shape{1});
    const auto C = make_shared<op::Parameter>(element::f32, Shape{1});
    const int64_t axis = 0;
    const auto axis_const = op::Constant::create(element::i64, {}, {axis});

    const auto concat = make_shared<op::Concat>(NodeVector{A, B, C}, axis);

    const vector<int64_t> indices{1};
    const auto indices_const = op::Constant::create(element::i64, {indices.size()}, indices);
    const auto gather = make_shared<op::v1::Gather>(concat, indices_const, axis_const);
    gather->set_friendly_name("test");
    auto f = make_shared<Function>(gather, ParameterVector{A, B, C});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::Concat>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::v1::Gather>(f), 0);
}

TEST(constant_folding, const_gather_v1_subgraph_no_constant_input_scalar)
{
    const auto A = make_shared<op::Parameter>(element::f32, Shape{1});
    const auto B = make_shared<op::Parameter>(element::f32, Shape{1});
    const auto C = make_shared<op::Parameter>(element::f32, Shape{1});
    const int64_t axis = 0;
    const auto axis_const = op::Constant::create(element::i64, {}, {axis});

    const auto concat = make_shared<op::Concat>(NodeVector{A, B, C}, axis);

    const vector<int64_t> indices{1};
    const auto indices_const = op::Constant::create(element::i64, {}, indices);
    const auto gather = make_shared<op::v1::Gather>(concat, indices_const, axis_const);
    auto f = make_shared<Function>(gather, ParameterVector{A, B, C});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::Concat>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::v1::Gather>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::v0::Squeeze>(f), 1);
}

TEST(constant_folding, const_gather_v1_subgraph_skip_if_non_zero_axis)
{
    const auto A = make_shared<op::Parameter>(element::f32, Shape{2, 2});
    const auto B = make_shared<op::Parameter>(element::f32, Shape{2, 2});
    const auto C = make_shared<op::Parameter>(element::f32, Shape{2, 2});
    const int64_t axis = 1;
    const auto axis_const = op::Constant::create(element::i64, {}, {axis});

    const auto concat = make_shared<op::Concat>(NodeVector{A, B, C}, axis);

    const vector<int64_t> indices{1};
    const auto indices_const = op::Constant::create(element::i64, {indices.size()}, indices);
    const auto gather = make_shared<op::v1::Gather>(concat, indices_const, axis_const);
    auto f = make_shared<Function>(gather, ParameterVector{A, B, C});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::Concat>(f), 1);
    ASSERT_EQ(count_ops_of_type<op::v1::Gather>(f), 1);
}

TEST(constant_folding, const_gather_v1_subgraph_skip_if_non_single_indices)
{
    const auto A = make_shared<op::Parameter>(element::f32, Shape{1});
    const auto B = make_shared<op::Parameter>(element::f32, Shape{1});
    const auto C = make_shared<op::Parameter>(element::f32, Shape{1});
    const int64_t axis = 0;
    const auto axis_const = op::Constant::create(element::i64, {}, {axis});

    const auto concat = make_shared<op::Concat>(NodeVector{A, B, C}, axis);

    const vector<int64_t> indices{0, 1};
    const auto indices_const = op::Constant::create(element::i64, {indices.size()}, indices);
    const auto gather = make_shared<op::v1::Gather>(concat, indices_const, axis_const);
    auto f = make_shared<Function>(gather, ParameterVector{A, B, C});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::Concat>(f), 1);
    ASSERT_EQ(count_ops_of_type<op::v1::Gather>(f), 1);
}

TEST(constant_folding, const_gather_v1_subgraph_skip_if_concat_output_shape_dynamic)
{
    const auto A = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    const auto B = make_shared<op::Parameter>(element::f32, Shape{1});
    const auto C = make_shared<op::Parameter>(element::f32, Shape{1});
    const int64_t axis = 0;
    const auto axis_const = op::Constant::create(element::i64, {}, {axis});

    const auto concat = make_shared<op::Concat>(NodeVector{A, B, C}, axis);

    const vector<int64_t> indices{1};
    const auto indices_const = op::Constant::create(element::i64, {indices.size()}, indices);
    const auto gather = make_shared<op::v1::Gather>(concat, indices_const, axis_const);
    auto f = make_shared<Function>(gather, ParameterVector{A, B, C});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::Concat>(f), 1);
    ASSERT_EQ(count_ops_of_type<op::v1::Gather>(f), 1);
}

TEST(constant_folding, const_gather_v1_subgraph_skip_if_not_single_input)
{
    const auto A = make_shared<op::Parameter>(element::f32, Shape{2});
    const auto B = make_shared<op::Parameter>(element::f32, Shape{1});
    const auto C = make_shared<op::Parameter>(element::f32, Shape{1});
    const int64_t axis = 0;
    const auto axis_const = op::Constant::create(element::i64, {}, {axis});

    const auto concat = make_shared<op::Concat>(NodeVector{A, B, C}, axis);

    const vector<int64_t> indices{1};
    const auto indices_const = op::Constant::create(element::i64, {indices.size()}, indices);
    const auto gather = make_shared<op::v1::Gather>(concat, indices_const, axis_const);
    auto f = make_shared<Function>(gather, ParameterVector{A, B, C});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::Concat>(f), 1);
    ASSERT_EQ(count_ops_of_type<op::v1::Gather>(f), 1);
}

TEST(constant_folding, const_strided_slice)
{
    Shape shape_in{16};

    vector<int> values_in{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    auto constant = make_shared<op::Constant>(element::i32, shape_in, values_in);
    auto begin = op::Constant::create(element::i64, {1}, {2});
    auto end = op::Constant::create(element::i64, {1}, {15});
    auto stride = op::Constant::create(element::i64, {1}, {3});
    auto slice = make_shared<op::v1::StridedSlice>(
        constant, begin, end, stride, std::vector<int64_t>{0}, std::vector<int64_t>{0});
    slice->set_friendly_name("test");

    auto f = make_shared<Function>(slice, ParameterVector{});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::v1::StridedSlice>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::Constant>(f), 1);

    auto new_const =
        as_type_ptr<op::Constant>(f->get_results().at(0)->input_value(0).get_node_shared_ptr());
    ASSERT_TRUE(new_const);
    ASSERT_EQ(new_const->get_friendly_name(), "test");
    auto values_out = new_const->get_vector<int>();

    vector<int> sliced_values{3, 6, 9, 12, 15};
    ASSERT_EQ(sliced_values, values_out);
}

TEST(constant_folding, constant_dyn_reshape)
{
    Shape shape_in{2, 4};
    vector<float> values_in{0, 1, 2, 3, 4, 5, 6, 7};

    Shape shape_shape{3};
    vector<int64_t> values_shape{2, 4, 1};

    auto constant_in = make_shared<op::Constant>(element::f32, shape_in, values_in);
    auto constant_shape = make_shared<op::Constant>(element::i64, shape_shape, values_shape);
    auto dyn_reshape = make_shared<op::v1::Reshape>(constant_in, constant_shape, false);
    dyn_reshape->set_friendly_name("test");
    auto f = make_shared<Function>(dyn_reshape, ParameterVector{});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::v1::Reshape>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::Constant>(f), 1);

    auto new_const =
        as_type_ptr<op::Constant>(f->get_results().at(0)->input_value(0).get_node_shared_ptr());
    ASSERT_TRUE(new_const);
    ASSERT_EQ(new_const->get_friendly_name(), "test");
    auto values_out = new_const->get_vector<float>();

    ASSERT_TRUE(test::all_close_f(values_in, values_out, MIN_FLOAT_TOLERANCE_BITS));
}

TEST(constant_folding, constant_dyn_reshape_shape_not_originally_constant)
{
    Shape shape_in{2, 4};
    vector<float> values_in{0, 1, 2, 3, 4, 5, 6, 7};

    Shape shape_shape{3};
    // We're going to add these two together elementwise to get {2, 4, 1}.
    // This means that when ConstantFolding starts, v1::Reshape will not yet
    // have static output shape. But by the time the Add op is folded, the
    // v1::Reshape's shape should be inferrable.
    vector<int64_t> values_shape_a{1, 3, 0};
    vector<int64_t> values_shape_b{1, 1, 1};

    auto constant_in = make_shared<op::Constant>(element::f32, shape_in, values_in);
    auto constant_shape_a = make_shared<op::Constant>(element::i64, shape_shape, values_shape_a);
    auto constant_shape_b = make_shared<op::Constant>(element::i64, shape_shape, values_shape_b);
    auto dyn_reshape = make_shared<op::v1::Reshape>(
        constant_in, std::make_shared<op::v1::Add>(constant_shape_a, constant_shape_b), false);
    dyn_reshape->set_friendly_name("test");
    auto f = make_shared<Function>(dyn_reshape, ParameterVector{});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::v1::Reshape>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::Constant>(f), 1);

    auto new_const =
        as_type_ptr<op::Constant>(f->get_results().at(0)->input_value(0).get_node_shared_ptr());
    ASSERT_TRUE(new_const);
    ASSERT_EQ(new_const->get_friendly_name(), "test");
    auto values_out = new_const->get_vector<float>();

    ASSERT_TRUE(test::all_close_f(values_in, values_out, MIN_FLOAT_TOLERANCE_BITS));
}

TEST(constant_folding, constant_transpose)
{
    Shape shape_in{2, 4};
    vector<double> values_in{0, 1, 2, 3, 4, 5, 6, 7};

    Shape shape_perm{2};
    vector<int64_t> values_perm{1, 0};

    auto constant_in = make_shared<op::Constant>(element::f64, shape_in, values_in);
    auto constant_perm = make_shared<op::Constant>(element::i64, shape_perm, values_perm);
    auto transpose = make_shared<op::Transpose>(constant_in, constant_perm);
    transpose->set_friendly_name("test");
    auto f = make_shared<Function>(transpose, ParameterVector{});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::Transpose>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::Constant>(f), 1);

    auto new_const =
        as_type_ptr<op::Constant>(f->get_results().at(0)->input_value(0).get_node_shared_ptr());
    ASSERT_TRUE(new_const);
    ASSERT_EQ(new_const->get_friendly_name(), "test");
    auto values_out = new_const->get_vector<double>();

    vector<double> values_permute{0, 4, 1, 5, 2, 6, 3, 7};
    ASSERT_TRUE(test::all_close_f(values_permute, values_out, MIN_FLOAT_TOLERANCE_BITS));
}

template <typename T>
void range_test(T start, T stop, T step, const vector<T>& values_expected)
{
    vector<T> values_start{start};
    vector<T> values_stop{stop};
    vector<T> values_step{step};

    auto constant_start = make_shared<op::Constant>(element::from<T>(), Shape{}, values_start);
    auto constant_stop = make_shared<op::Constant>(element::from<T>(), Shape{}, values_stop);
    auto constant_step = make_shared<op::Constant>(element::from<T>(), Shape{}, values_step);
    auto range = make_shared<op::Range>(constant_start, constant_stop, constant_step);
    range->set_friendly_name("test");
    auto f = make_shared<Function>(range, ParameterVector{});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::Range>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::Constant>(f), 1);

    auto new_const =
        as_type_ptr<op::Constant>(f->get_results().at(0)->input_value(0).get_node_shared_ptr());
    ASSERT_TRUE(new_const);
    ASSERT_EQ(new_const->get_friendly_name(), "test");

    auto values_out = new_const->template get_vector<T>();

    range_test_check(values_out, values_expected);
}

TEST(constant_folding, constant_range)
{
    range_test<int8_t>(5, 12, 2, {5, 7, 9, 11});
    range_test<int32_t>(5, 12, 2, {5, 7, 9, 11});
    range_test<int64_t>(5, 12, 2, {5, 7, 9, 11});
    range_test<uint64_t>(5, 12, 2, {5, 7, 9, 11});
    range_test<double>(5, 12, 2, {5, 7, 9, 11});
    range_test<float>(5, 12, 2, {5, 7, 9, 11});

    range_test<int32_t>(5, 12, -2, {});
    range_test<float>(12, 4, -2, {12, 10, 8, 6});
}

TEST(constant_folding, constant_v1_select)
{
    Shape shape{2, 4};
    vector<char> values_selection{0, 1, 1, 0};
    vector<int64_t> values_t{1, 2, 3, 4};
    vector<int64_t> values_f{11, 12, 13, 14, 15, 16, 17, 18};

    auto constant_selection =
        make_shared<op::Constant>(element::boolean, Shape{4}, values_selection);
    auto constant_t = make_shared<op::Constant>(element::i64, Shape{4}, values_t);
    auto constant_f = make_shared<op::Constant>(element::i64, Shape{2, 4}, values_f);
    auto select = make_shared<op::v1::Select>(constant_selection, constant_t, constant_f);
    select->set_friendly_name("test");
    auto f = make_shared<Function>(select, ParameterVector{});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::v1::Select>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::Constant>(f), 1);

    auto new_const =
        as_type_ptr<op::Constant>(f->get_results().at(0)->input_value(0).get_node_shared_ptr());
    ASSERT_TRUE(new_const);
    ASSERT_EQ(new_const->get_friendly_name(), "test");
    auto values_out = new_const->get_vector<int64_t>();

    vector<int64_t> values_expected{11, 2, 3, 14, 15, 2, 3, 18};
    ASSERT_EQ(values_expected, values_out);
}

TEST(constant_folding, constant_v1_split)
{
    vector<float> data{.1f, .2f, .3f, .4f, .5f, .6f};
    const auto const_data = op::Constant::create(element::f32, Shape{data.size()}, data);
    const auto const_axis = op::Constant::create(element::i64, Shape{}, {0});
    const auto num_splits = 3;

    auto split_v1 = make_shared<op::v1::Split>(const_data, const_axis, num_splits);
    auto f = make_shared<Function>(split_v1->outputs(), ParameterVector{});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::v1::Split>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::Constant>(f), num_splits);

    auto res1 =
        as_type_ptr<op::Constant>(f->get_results().at(0)->input_value(0).get_node_shared_ptr());
    auto res2 =
        as_type_ptr<op::Constant>(f->get_results().at(1)->input_value(0).get_node_shared_ptr());
    auto res3 =
        as_type_ptr<op::Constant>(f->get_results().at(2)->input_value(0).get_node_shared_ptr());
    ASSERT_TRUE(res1);
    ASSERT_TRUE(res2);
    ASSERT_TRUE(res3);

    auto res1_values = res1->get_vector<float>();
    ASSERT_TRUE(test::all_close_f(vector<float>(data.begin(), data.begin() + 2), res1_values));
    auto res2_values = res2->get_vector<float>();
    ASSERT_TRUE(test::all_close_f(vector<float>(data.begin() + 2, data.begin() + 4), res2_values));
    auto res3_values = res3->get_vector<float>();
    ASSERT_TRUE(test::all_close_f(vector<float>(data.begin() + 4, data.end()), res3_values));
}

TEST(constant_folding, constant_v1_split_specialized)
{
    vector<float> data{.1f, .2f, .3f, .4f, .5f, .6f};
    const auto const_data = op::Constant::create(element::f32, Shape{data.size()}, data);
    const auto const_axis = op::Constant::create(element::i64, Shape{}, {0});
    const auto num_splits = 3;

    auto split_v1 = make_shared<op::v1::Split>(const_data, const_axis, num_splits);
    auto f = make_shared<Function>(split_v1->outputs(), ParameterVector{});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::v1::Split>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::Constant>(f), num_splits);

    auto res1 =
        as_type_ptr<op::Constant>(f->get_results().at(0)->input_value(0).get_node_shared_ptr());
    auto res2 =
        as_type_ptr<op::Constant>(f->get_results().at(1)->input_value(0).get_node_shared_ptr());
    auto res3 =
        as_type_ptr<op::Constant>(f->get_results().at(2)->input_value(0).get_node_shared_ptr());
    ASSERT_TRUE(res1);
    ASSERT_TRUE(res2);
    ASSERT_TRUE(res3);

    auto res1_values = res1->get_vector<float>();
    ASSERT_TRUE(test::all_close_f(vector<float>(data.begin(), data.begin() + 2), res1_values));
    auto res2_values = res2->get_vector<float>();
    ASSERT_TRUE(test::all_close_f(vector<float>(data.begin() + 2, data.begin() + 4), res2_values));
    auto res3_values = res3->get_vector<float>();
    ASSERT_TRUE(test::all_close_f(vector<float>(data.begin() + 4, data.end()), res3_values));
}

TEST(constant_folding, constant_v1_split_axis_1_4_splits)
{
    vector<int64_t> data{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,

                         16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,

                         32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,

                         48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63};

    const auto const_data = op::Constant::create(element::i64, Shape{4, 4, 4}, data);
    const auto const_axis = op::Constant::create(element::i64, Shape{}, {1});
    const auto num_splits = 4;

    auto split_v1 = make_shared<op::v1::Split>(const_data, const_axis, num_splits);
    split_v1->set_friendly_name("test");
    auto f = make_shared<Function>(split_v1->outputs(), ParameterVector{});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::v1::Split>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::Constant>(f), num_splits);

    auto res1 =
        as_type_ptr<op::Constant>(f->get_results().at(0)->input_value(0).get_node_shared_ptr());
    auto res2 =
        as_type_ptr<op::Constant>(f->get_results().at(1)->input_value(0).get_node_shared_ptr());
    auto res3 =
        as_type_ptr<op::Constant>(f->get_results().at(2)->input_value(0).get_node_shared_ptr());
    auto res4 =
        as_type_ptr<op::Constant>(f->get_results().at(3)->input_value(0).get_node_shared_ptr());
    ASSERT_TRUE(res1);
    ASSERT_EQ(res1->get_friendly_name(), "test.0");
    ASSERT_TRUE(res2);
    ASSERT_EQ(res2->get_friendly_name(), "test.1");
    ASSERT_TRUE(res3);
    ASSERT_EQ(res3->get_friendly_name(), "test.2");
    ASSERT_TRUE(res4);
    ASSERT_EQ(res4->get_friendly_name(), "test.3");

    auto res1_values = res1->get_vector<int64_t>();
    ASSERT_EQ(vector<int64_t>({0, 1, 2, 3, 16, 17, 18, 19, 32, 33, 34, 35, 48, 49, 50, 51}),
              res1_values);
    auto res2_values = res2->get_vector<int64_t>();
    ASSERT_EQ(vector<int64_t>({4, 5, 6, 7, 20, 21, 22, 23, 36, 37, 38, 39, 52, 53, 54, 55}),
              res2_values);
    auto res3_values = res3->get_vector<int64_t>();
    ASSERT_EQ(vector<int64_t>({8, 9, 10, 11, 24, 25, 26, 27, 40, 41, 42, 43, 56, 57, 58, 59}),
              res3_values);
    auto res4_values = res4->get_vector<int64_t>();
    ASSERT_EQ(vector<int64_t>({12, 13, 14, 15, 28, 29, 30, 31, 44, 45, 46, 47, 60, 61, 62, 63}),
              res4_values);
}

TEST(constant_folding, constant_v1_split_axis_1_2_splits)
{
    vector<int64_t> data{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,

                         16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,

                         32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,

                         48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63};

    const auto const_data = op::Constant::create(element::i64, Shape{4, 4, 4}, data);
    const auto const_axis = op::Constant::create(element::i64, Shape{}, {1});
    const auto num_splits = 2;

    auto split_v1 = make_shared<op::v1::Split>(const_data, const_axis, num_splits);
    auto f = make_shared<Function>(split_v1->outputs(), ParameterVector{});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::v1::Split>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::Constant>(f), num_splits);

    auto res1 =
        as_type_ptr<op::Constant>(f->get_results().at(0)->input_value(0).get_node_shared_ptr());
    auto res2 =
        as_type_ptr<op::Constant>(f->get_results().at(1)->input_value(0).get_node_shared_ptr());
    ASSERT_TRUE(res1);
    ASSERT_TRUE(res2);

    auto res1_values = res1->get_vector<int64_t>();
    ASSERT_EQ(vector<int64_t>({0,  1,  2,  3,  4,  5,  6,  7,  16, 17, 18, 19, 20, 21, 22, 23,
                               32, 33, 34, 35, 36, 37, 38, 39, 48, 49, 50, 51, 52, 53, 54, 55}),
              res1_values);
    auto res2_values = res2->get_vector<int64_t>();
    ASSERT_EQ(vector<int64_t>({8,  9,  10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31,
                               40, 41, 42, 43, 44, 45, 46, 47, 56, 57, 58, 59, 60, 61, 62, 63}),
              res2_values);
}

TEST(constant_folding, constant_v1_variadic_split_axis_1_2_splits)
{
    vector<int64_t> data{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,

                         16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,

                         32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,

                         48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63};

    const auto const_data = op::Constant::create(element::i64, Shape{4, 4, 4}, data);
    const auto const_axis = op::Constant::create(element::i16, Shape{}, {1});
    vector<int64_t> values_lengths{3, 1};
    auto constant_lengths =
        make_shared<op::Constant>(element::i64, Shape{values_lengths.size()}, values_lengths);

    auto variadic_split_v1 =
        make_shared<op::v1::VariadicSplit>(const_data, const_axis, constant_lengths);
    auto f = make_shared<Function>(variadic_split_v1->outputs(), ParameterVector{});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::v1::VariadicSplit>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::Constant>(f), values_lengths.size());

    auto res1 =
        as_type_ptr<op::Constant>(f->get_results().at(0)->input_value(0).get_node_shared_ptr());
    auto res2 =
        as_type_ptr<op::Constant>(f->get_results().at(1)->input_value(0).get_node_shared_ptr());
    ASSERT_TRUE(res1);
    ASSERT_TRUE(res2);

    auto res1_values = res1->get_vector<int64_t>();
    ASSERT_EQ(vector<int64_t>({0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 16, 17, 18, 19,
                               20, 21, 22, 23, 24, 25, 26, 27, 32, 33, 34, 35, 36, 37, 38, 39,
                               40, 41, 42, 43, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59}),
              res1_values);
    auto res2_values = res2->get_vector<int64_t>();
    ASSERT_EQ(vector<int64_t>({12, 13, 14, 15, 28, 29, 30, 31, 44, 45, 46, 47, 60, 61, 62, 63}),
              res2_values);
}

TEST(constant_folding, constant_v1_variadic_split_axis_1_3_splits_neg_length)
{
    vector<int64_t> data{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,

                         16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,

                         32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,

                         48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63};

    const auto const_data = op::Constant::create(element::i64, Shape{4, 4, 4}, data);
    const auto const_axis = op::Constant::create(element::i32, Shape{}, {1});
    vector<int64_t> values_lengths{1, 1, -1};
    auto constant_lengths =
        make_shared<op::Constant>(element::i64, Shape{values_lengths.size()}, values_lengths);

    auto variadic_split_v1 =
        make_shared<op::v1::VariadicSplit>(const_data, const_axis, constant_lengths);
    auto f = make_shared<Function>(variadic_split_v1->outputs(), ParameterVector{});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::v1::VariadicSplit>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::Constant>(f), values_lengths.size());

    auto res1 =
        as_type_ptr<op::Constant>(f->get_results().at(0)->input_value(0).get_node_shared_ptr());
    auto res2 =
        as_type_ptr<op::Constant>(f->get_results().at(1)->input_value(0).get_node_shared_ptr());
    auto res3 =
        as_type_ptr<op::Constant>(f->get_results().at(2)->input_value(0).get_node_shared_ptr());
    ASSERT_TRUE(res1);
    ASSERT_TRUE(res2);
    ASSERT_TRUE(res3);

    auto res1_values = res1->get_vector<int64_t>();
    ASSERT_EQ(vector<int64_t>({0, 1, 2, 3, 16, 17, 18, 19, 32, 33, 34, 35, 48, 49, 50, 51}),
              res1_values);
    auto res2_values = res2->get_vector<int64_t>();
    ASSERT_EQ(vector<int64_t>({4, 5, 6, 7, 20, 21, 22, 23, 36, 37, 38, 39, 52, 53, 54, 55}),
              res2_values);
    auto res3_values = res3->get_vector<int64_t>();
    ASSERT_EQ(vector<int64_t>({8,  9,  10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31,
                               40, 41, 42, 43, 44, 45, 46, 47, 56, 57, 58, 59, 60, 61, 62, 63}),
              res3_values);
}

TEST(constant_folding, constant_v1_one_hot)
{
    const vector<int64_t> indices{0, 1, 2};
    const float on_value = 1.123f;
    const float off_value = 0.321f;

    const auto indices_const = op::Constant::create(element::i64, Shape{3}, indices);
    const auto depth_const = op::Constant::create(element::i64, Shape{}, {3});
    const auto on_const = op::Constant::create(element::f32, Shape{}, {on_value});
    const auto off_const = op::Constant::create(element::f32, Shape{}, {off_value});
    int64_t axis = 1;

    auto one_hot_v1 =
        make_shared<op::v1::OneHot>(indices_const, depth_const, on_const, off_const, axis);
    auto f = make_shared<Function>(one_hot_v1, ParameterVector{});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::v1::OneHot>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::Constant>(f), 1);

    auto res =
        as_type_ptr<op::Constant>(f->get_results().at(0)->input_value(0).get_node_shared_ptr());
    ASSERT_TRUE(res);

    ASSERT_EQ((Shape{3, 3}), res->get_output_shape(0));
    ASSERT_EQ(vector<float>({on_value,
                             off_value,
                             off_value,
                             off_value,
                             on_value,
                             off_value,
                             off_value,
                             off_value,
                             on_value}),
              res->get_vector<float>());
}

TEST(constant_folding, constant_v1_one_hot_negative_axes)
{
    const vector<int64_t> indices{0, 2, 3, 1};
    const int32_t on_value = 4;
    const int32_t off_value = 1;

    const auto indices_const = op::Constant::create(element::i64, Shape{4}, indices);
    const auto depth_const = op::Constant::create(element::i64, Shape{}, {3});
    const auto on_const = op::Constant::create(element::i32, Shape{}, {on_value});
    const auto off_const = op::Constant::create(element::i32, Shape{}, {off_value});
    int64_t axis = -1;

    auto one_hot_v1 =
        make_shared<op::v1::OneHot>(indices_const, depth_const, on_const, off_const, axis);
    auto f = make_shared<Function>(one_hot_v1, ParameterVector{});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::v1::OneHot>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::Constant>(f), 1);

    auto res =
        as_type_ptr<op::Constant>(f->get_results().at(0)->input_value(0).get_node_shared_ptr());
    ASSERT_TRUE(res);

    ASSERT_EQ((Shape{4, 3}), res->get_output_shape(0));
    ASSERT_EQ(vector<int32_t>({on_value,
                               off_value,
                               off_value,
                               off_value,
                               off_value,
                               on_value,
                               off_value,
                               off_value,
                               off_value,
                               off_value,
                               on_value,
                               off_value}),
              res->get_vector<int32_t>());
}

TEST(constant_folding, constant_v1_one_hot_negative_axes_2)
{
    vector<int64_t> indices{0, 2, 1, 3};
    auto on_value = true;
    auto off_value = false;

    const auto indices_const = op::Constant::create(element::i64, Shape{2, 2}, indices);
    const auto depth_const = op::Constant::create(element::i64, Shape{}, {3});
    const auto on_const = op::Constant::create(element::boolean, Shape{}, {on_value});
    const auto off_const = op::Constant::create(element::boolean, Shape{}, {off_value});
    int64_t axis = -1;

    auto one_hot_v1 =
        make_shared<op::v1::OneHot>(indices_const, depth_const, on_const, off_const, axis);
    one_hot_v1->set_friendly_name("test");
    auto f = make_shared<Function>(one_hot_v1, ParameterVector{});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::v1::OneHot>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::Constant>(f), 1);

    auto res =
        as_type_ptr<op::Constant>(f->get_results().at(0)->input_value(0).get_node_shared_ptr());
    ASSERT_TRUE(res);
    ASSERT_EQ(res->get_friendly_name(), "test");

    ASSERT_EQ((Shape{2, 2, 3}), res->get_output_shape(0));
    ASSERT_EQ(vector<bool>({on_value,
                            off_value,
                            off_value,
                            off_value,
                            off_value,
                            on_value,
                            off_value,
                            on_value,
                            off_value,
                            off_value,
                            off_value,
                            off_value}),
              res->get_vector<bool>());
}

TEST(constant_folding, constant_tile_1d)
{
    Shape shape_in{2};
    Shape shape_repeats{1};
    Shape shape_out{4};

    vector<int> values_in{0, 1};
    auto data = make_shared<op::Constant>(element::i32, shape_in, values_in);
    vector<int> values_repeats{2};
    auto repeats = make_shared<op::Constant>(element::i64, shape_repeats, values_repeats);
    auto tile = make_shared<op::v0::Tile>(data, repeats);
    tile->set_friendly_name("test");
    auto f = make_shared<Function>(tile, ParameterVector{});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::v0::Tile>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::Constant>(f), 1);

    auto new_const =
        as_type_ptr<op::Constant>(f->get_results().at(0)->input_value(0).get_node_shared_ptr());
    ASSERT_TRUE(new_const);
    ASSERT_EQ(new_const->get_friendly_name(), "test");
    auto values_out = new_const->get_vector<int>();

    vector<int> values_expected{0, 1, 0, 1};
    ASSERT_EQ(values_expected, values_out);
}

TEST(constant_folding, constant_tile_3d_small_data_rank)
{
    Shape shape_in{2};
    Shape shape_repeats{3};
    Shape shape_out{2, 2, 4};

    vector<int> values_in{0, 1};
    auto data = make_shared<op::Constant>(element::i32, shape_in, values_in);
    vector<int> values_repeats{2, 2, 2};
    auto repeats = make_shared<op::Constant>(element::i64, shape_repeats, values_repeats);
    auto tile = make_shared<op::v0::Tile>(data, repeats);
    tile->set_friendly_name("test");
    auto f = make_shared<Function>(tile, ParameterVector{});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::v0::Tile>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::Constant>(f), 1);

    auto new_const =
        as_type_ptr<op::Constant>(f->get_results().at(0)->input_value(0).get_node_shared_ptr());
    ASSERT_TRUE(new_const);
    ASSERT_EQ(new_const->get_friendly_name(), "test");
    auto values_out = new_const->get_vector<int>();

    vector<int> values_expected{0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1};
    ASSERT_EQ(values_expected, values_out);
}

TEST(constant_folding, constant_tile_3d_few_repeats)
{
    Shape shape_in{2, 1, 3};
    Shape shape_repeats{2};
    Shape shape_out{2, 2, 3};

    vector<int> values_in{1, 2, 3, 4, 5, 6};
    auto data = make_shared<op::Constant>(element::i32, shape_in, values_in);
    vector<int> values_repeats{2, 1};
    auto repeats = make_shared<op::Constant>(element::i64, shape_repeats, values_repeats);
    auto tile = make_shared<op::v0::Tile>(data, repeats);
    tile->set_friendly_name("test");
    auto f = make_shared<Function>(tile, ParameterVector{});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::v0::Tile>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::Constant>(f), 1);

    auto new_const =
        as_type_ptr<op::Constant>(f->get_results().at(0)->input_value(0).get_node_shared_ptr());
    ASSERT_TRUE(new_const);
    ASSERT_EQ(new_const->get_friendly_name(), "test");
    auto values_out = new_const->get_vector<int>();

    vector<int> values_expected{1, 2, 3, 1, 2, 3, 4, 5, 6, 4, 5, 6};
    ASSERT_EQ(values_expected, values_out);
}

TEST(constant_folding, constant_tile_1d_0_repeats)
{
    Shape shape_in{2};
    Shape shape_repeats{1};
    Shape shape_out{};

    vector<int> values_in{0, 1};
    auto data = make_shared<op::Constant>(element::i32, shape_in, values_in);
    vector<int> values_repeats{0};
    auto repeats = make_shared<op::Constant>(element::i64, shape_repeats, values_repeats);
    auto tile = make_shared<op::v0::Tile>(data, repeats);
    tile->set_friendly_name("test");
    auto f = make_shared<Function>(tile, ParameterVector{});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::v0::Tile>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::Constant>(f), 1);

    auto new_const =
        as_type_ptr<op::Constant>(f->get_results().at(0)->input_value(0).get_node_shared_ptr());
    ASSERT_TRUE(new_const);
    ASSERT_EQ(new_const->get_friendly_name(), "test");
    auto values_out = new_const->get_vector<int>();

    vector<int> values_expected{};
    ASSERT_EQ(values_expected, values_out);
}

TEST(constant_folding, constant_tile_0_rank_data)
{
    Shape shape_in{};
    Shape shape_repeats{1};
    Shape shape_out{4};

    vector<int> values_in{1};
    auto data = make_shared<op::Constant>(element::i32, shape_in, values_in);
    vector<int> values_repeats{4};
    auto repeats = make_shared<op::Constant>(element::i64, shape_repeats, values_repeats);
    auto tile = make_shared<op::v0::Tile>(data, repeats);
    tile->set_friendly_name("test");
    auto f = make_shared<Function>(tile, ParameterVector{});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::v0::Tile>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::Constant>(f), 1);

    auto new_const =
        as_type_ptr<op::Constant>(f->get_results().at(0)->input_value(0).get_node_shared_ptr());
    ASSERT_TRUE(new_const);
    ASSERT_EQ(new_const->get_friendly_name(), "test");
    auto values_out = new_const->get_vector<int>();

    vector<int> values_expected{1, 1, 1, 1};
    ASSERT_EQ(values_expected, values_out);
}

TEST(constant_folding, constant_non_zero_0D)
{
    auto data = op::Constant::create(element::i32, Shape{}, {1});
    auto non_zero = make_shared<op::v3::NonZero>(data);
    non_zero->set_friendly_name("test");
    auto f = make_shared<Function>(non_zero, ParameterVector{});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(f);

    // Fold into constant with shape of {1, 1} for scalar input with
    // non-zero value
    ASSERT_EQ(count_ops_of_type<op::v3::NonZero>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::Constant>(f), 1);

    const auto new_const =
        as_type_ptr<op::Constant>(f->get_results().at(0)->input_value(0).get_node_shared_ptr());
    ASSERT_TRUE(new_const);
    ASSERT_EQ(new_const->get_friendly_name(), "test");
    const auto values_out = new_const->get_vector<int64_t>();

    const vector<int64_t> values_expected{0};
    ASSERT_EQ(values_expected, values_out);
    ASSERT_EQ((Shape{1, 1}), new_const->get_shape());
}

TEST(constant_folding, constant_non_zero_1D)
{
    vector<int> values_in{0, 1, 0, 1};
    auto data = make_shared<op::Constant>(element::i32, Shape{4}, values_in);
    auto non_zero = make_shared<op::v3::NonZero>(data);
    non_zero->set_friendly_name("test");
    auto f = make_shared<Function>(non_zero, ParameterVector{});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::v3::NonZero>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::Constant>(f), 1);

    const auto new_const =
        as_type_ptr<op::Constant>(f->get_results().at(0)->input_value(0).get_node_shared_ptr());
    ASSERT_TRUE(new_const);
    ASSERT_EQ(new_const->get_friendly_name(), "test");
    const auto values_out = new_const->get_vector<int64_t>();

    const vector<int64_t> values_expected{1, 3};
    ASSERT_EQ(values_expected, values_out);
    ASSERT_EQ((Shape{1, 2}), new_const->get_shape());
}

TEST(constant_folding, constant_non_zero_int32_output_type)
{
    vector<int> values_in{0, 1, 0, 1};
    auto data = make_shared<op::Constant>(element::i32, Shape{4}, values_in);
    auto non_zero = make_shared<op::v3::NonZero>(data, element::i32);
    non_zero->set_friendly_name("test");
    auto f = make_shared<Function>(non_zero, ParameterVector{});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::v3::NonZero>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::Constant>(f), 1);

    const auto new_const =
        as_type_ptr<op::Constant>(f->get_results().at(0)->input_value(0).get_node_shared_ptr());
    ASSERT_TRUE(new_const);
    ASSERT_EQ(new_const->get_friendly_name(), "test");
    ASSERT_EQ(element::i32, new_const->get_element_type());
    const auto values_out = new_const->get_vector<int32_t>();

    const vector<int32_t> values_expected{1, 3};
    ASSERT_EQ(values_expected, values_out);
    ASSERT_EQ((Shape{1, 2}), new_const->get_shape());
}

TEST(constant_folding, constant_non_zero_1D_all_indices)
{
    const vector<float> values_in{1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    const auto data = make_shared<op::Constant>(element::f32, Shape{values_in.size()}, values_in);
    const auto non_zero = make_shared<op::v3::NonZero>(data);
    non_zero->set_friendly_name("test");
    auto f = make_shared<Function>(non_zero, ParameterVector{});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::v3::NonZero>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::Constant>(f), 1);

    const auto new_const =
        as_type_ptr<op::Constant>(f->get_results().at(0)->input_value(0).get_node_shared_ptr());
    ASSERT_TRUE(new_const);
    ASSERT_EQ(new_const->get_friendly_name(), "test");
    const auto values_out = new_const->get_vector<int64_t>();

    const vector<int64_t> values_expected{0, 1, 2, 3, 4, 5, 6, 7};
    ASSERT_EQ(values_expected, values_out);
    ASSERT_EQ((Shape{1, values_in.size()}), new_const->get_shape());
}

TEST(constant_folding, constant_non_zero_2D)
{
    vector<int> values_in{1, 0, 0, 0, 1, 0, 1, 1, 0};
    auto data = make_shared<op::Constant>(element::i32, Shape{3, 3}, values_in);
    auto non_zero = make_shared<op::v3::NonZero>(data);
    non_zero->set_friendly_name("test");
    auto f = make_shared<Function>(non_zero, ParameterVector{});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::v3::NonZero>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::Constant>(f), 1);

    const auto new_const =
        as_type_ptr<op::Constant>(f->get_results().at(0)->input_value(0).get_node_shared_ptr());
    ASSERT_TRUE(new_const);
    ASSERT_EQ(new_const->get_friendly_name(), "test");
    const auto values_out = new_const->get_vector<int64_t>();

    const vector<int64_t> values_expected{0, 1, 2, 2, 0, 1, 0, 1};
    ASSERT_EQ(values_expected, values_out);
    ASSERT_EQ((Shape{2, 4}), new_const->get_shape());
}

TEST(constant_folding, DISABLED_constant_non_zero_2D_all_indices)
{
    const vector<int8_t> values_in{1, 1, 1, 1, 1, 1, 1, 1, 1};
    const auto data = make_shared<op::Constant>(element::i8, Shape{3, 3}, values_in);
    const auto non_zero = make_shared<op::v3::NonZero>(data);
    non_zero->set_friendly_name("test");
    auto f = make_shared<Function>(non_zero, ParameterVector{});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::v3::NonZero>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::Constant>(f), 1);

    const auto new_const =
        as_type_ptr<op::Constant>(f->get_results().at(0)->input_value(0).get_node_shared_ptr());
    ASSERT_TRUE(new_const);
    ASSERT_EQ(new_const->get_friendly_name(), "test");
    const auto values_out = new_const->get_vector<int64_t>();

    const vector<int64_t> values_expected{0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2};
    ASSERT_EQ(values_expected, values_out);
    ASSERT_EQ((Shape{2, values_in.size()}), new_const->get_shape());
}

TEST(constant_folding, DISABLED_constant_non_zero_2D_all_zeros)
{
    const vector<uint8_t> values_in{0, 0, 0, 0, 0, 0};
    const auto data = make_shared<op::Constant>(element::u8, Shape{2, 3}, values_in);
    const auto non_zero = make_shared<op::v3::NonZero>(data);
    non_zero->set_friendly_name("test");
    auto f = make_shared<Function>(non_zero, ParameterVector{});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(f);

    // fold into Constant with shape of {0}
    ASSERT_EQ(count_ops_of_type<op::v3::NonZero>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::Constant>(f), 1);

    const auto new_const =
        as_type_ptr<op::Constant>(f->get_results().at(0)->input_value(0).get_node_shared_ptr());
    ASSERT_TRUE(new_const);
    ASSERT_EQ(new_const->get_friendly_name(), "test");
    ASSERT_EQ(shape_size(new_const->get_shape()), 0);
}

TEST(constant_folding, constant_non_zero_3D)
{
    vector<int> values_in{1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0};
    auto data = make_shared<op::Constant>(element::i32, Shape{2, 3, 3}, values_in);
    auto non_zero = make_shared<op::v3::NonZero>(data);
    non_zero->set_friendly_name("test");
    auto f = make_shared<Function>(non_zero, ParameterVector{});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::v3::NonZero>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::Constant>(f), 1);

    const auto new_const =
        as_type_ptr<op::Constant>(f->get_results().at(0)->input_value(0).get_node_shared_ptr());
    ASSERT_TRUE(new_const);
    ASSERT_EQ(new_const->get_friendly_name(), "test");
    const auto values_out = new_const->get_vector<int64_t>();

    const vector<int64_t> values_expected{0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 2, 2, 2,
                                          0, 0, 0, 1, 1, 2, 0, 2, 1, 0, 1, 2, 0, 1, 2, 0, 2, 1};
    ASSERT_EQ(values_expected, values_out);
    ASSERT_EQ((Shape{3, 12}), new_const->get_shape());
}

TEST(constant_folding, constant_scatter_elements_update_basic)
{
    const Shape data_shape{3, 3};
    const Shape indices_shape{2, 3};

    const auto data_const = op::Constant::create(
        element::f32, data_shape, std::vector<float>(shape_size(data_shape), 0.f));
    const auto indices_const =
        op::Constant::create(element::i32, indices_shape, {1, 0, 2, 0, 2, 1});
    const auto updates_const =
        op::Constant::create(element::f32, indices_shape, {1.0f, 1.1f, 1.2f, 2.0f, 2.1f, 2.2f});
    const auto axis_const = op::Constant::create(element::i64, Shape{}, {0});

    auto scatter_elem_updt = make_shared<op::v3::ScatterElementsUpdate>(
        data_const, indices_const, updates_const, axis_const);
    scatter_elem_updt->set_friendly_name("test");
    auto f = make_shared<Function>(scatter_elem_updt, ParameterVector{});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::v3::ScatterElementsUpdate>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::Constant>(f), 1);

    auto result_node =
        as_type_ptr<op::Constant>(f->get_results().at(0)->input_value(0).get_node_shared_ptr());
    ASSERT_TRUE(result_node);
    ASSERT_EQ(result_node->get_friendly_name(), "test");
    ASSERT_EQ(data_shape, result_node->get_output_shape(0));
    std::vector<float> expected{2.f, 1.1f, 0.0f, 1.f, 0.0f, 2.2f, 0.f, 2.1f, 1.2f};
    range_test_check(result_node->cast_vector<float>(), expected);
}

TEST(constant_folding, constant_scatter_elements_update_negative_axis)
{
    const Shape data_shape{3, 3};
    const Shape indices_shape{2, 3};

    const auto data_const = op::Constant::create(
        element::f32, data_shape, std::vector<float>(shape_size(data_shape), 0.f));
    const auto indices_const =
        op::Constant::create(element::i32, indices_shape, {1, 0, 2, 0, 2, 1});
    const auto updates_const =
        op::Constant::create(element::f32, indices_shape, {1.0f, 1.1f, 1.2f, 2.0f, 2.1f, 2.2f});
    const auto axis_const = op::Constant::create(element::i64, Shape{}, {-1});

    auto scatter_elem_updt = make_shared<op::v3::ScatterElementsUpdate>(
        data_const, indices_const, updates_const, axis_const);
    auto f = make_shared<Function>(scatter_elem_updt, ParameterVector{});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::v3::ScatterElementsUpdate>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::Constant>(f), 1);

    auto result_node =
        as_type_ptr<op::Constant>(f->get_results().at(0)->input_value(0).get_node_shared_ptr());
    ASSERT_TRUE(result_node);
    ASSERT_EQ(data_shape, result_node->get_output_shape(0));
    std::vector<float> expected{1.1f, 1.0f, 1.2f, 2.0f, 2.2f, 2.1f, 0.0f, 0.0f, 0.0f};
    range_test_check(result_node->cast_vector<float>(), expected);
}

TEST(constant_folding, constant_scatter_elements_update_1d_axis)
{
    const Shape data_shape{3, 3};
    const Shape indices_shape{2, 3};

    const auto data_const = op::Constant::create(
        element::f32, data_shape, std::vector<float>(shape_size(data_shape), 0.f));
    const auto indices_const =
        op::Constant::create(element::i32, indices_shape, {1, 0, 2, 0, 2, 1});
    const auto updates_const =
        op::Constant::create(element::f32, indices_shape, {1.0f, 1.1f, 1.2f, 2.0f, 2.1f, 2.2f});
    const auto axis_const = op::Constant::create(element::i64, Shape{1}, {0});

    auto scatter_elem_updt = make_shared<op::v3::ScatterElementsUpdate>(
        data_const, indices_const, updates_const, axis_const);
    auto f = make_shared<Function>(scatter_elem_updt, ParameterVector{});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::v3::ScatterElementsUpdate>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::Constant>(f), 1);

    auto result_node =
        as_type_ptr<op::Constant>(f->get_results().at(0)->input_value(0).get_node_shared_ptr());
    ASSERT_TRUE(result_node);
    ASSERT_EQ(data_shape, result_node->get_output_shape(0));
    std::vector<float> expected{2.f, 1.1f, 0.0f, 1.f, 0.0f, 2.2f, 0.f, 2.1f, 1.2f};
    range_test_check(result_node->cast_vector<float>(), expected);
}

TEST(constant_folding, constant_scatter_elements_update_3d_i16)
{
    const Shape data_shape{3, 3, 3};
    const Shape indices_shape{2, 2, 3};

    const auto data_const = op::Constant::create(
        element::i16, data_shape, std::vector<int16_t>(shape_size(data_shape), 0));
    const auto indices_const =
        op::Constant::create(element::i16, indices_shape, {1, 0, 2, 0, 2, 1, 2, 2, 2, 0, 1, 0});
    const auto updates_const =
        op::Constant::create(element::i16, indices_shape, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    const auto axis_const = op::Constant::create(element::i64, Shape{}, {1});

    auto scatter_elem_updt = make_shared<op::v3::ScatterElementsUpdate>(
        data_const, indices_const, updates_const, axis_const);
    auto f = make_shared<Function>(scatter_elem_updt, ParameterVector{});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::v3::ScatterElementsUpdate>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::Constant>(f), 1);

    auto result_node =
        as_type_ptr<op::Constant>(f->get_results().at(0)->input_value(0).get_node_shared_ptr());
    ASSERT_TRUE(result_node);
    ASSERT_EQ(data_shape, result_node->get_output_shape(0));
    std::vector<int16_t> expected{4, 2, 0, 1, 0, 6, 0, 5, 3, 10, 0, 12, 0, 11,
                                  0, 7, 8, 9, 0, 0, 0, 0, 0, 0,  0, 0,  0};
    range_test_check(result_node->cast_vector<int16_t>(), expected);
}

TEST(constant_folding, constant_scatter_elements_update_one_elem)
{
    const Shape data_shape{3, 3, 3};
    const Shape indices_shape{1, 1, 1};
    const auto input_data = std::vector<int32_t>(shape_size(data_shape), 0);

    const auto data_const = op::Constant::create(element::i32, data_shape, input_data);
    const auto indices_const = op::Constant::create(element::i32, indices_shape, {1});
    const auto updates_const = op::Constant::create(element::i32, indices_shape, {2});
    const auto axis_const = op::Constant::create(element::i64, Shape{}, {0});

    auto scatter_elem_updt = make_shared<op::v3::ScatterElementsUpdate>(
        data_const, indices_const, updates_const, axis_const);
    auto f = make_shared<Function>(scatter_elem_updt, ParameterVector{});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::v3::ScatterElementsUpdate>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::Constant>(f), 1);

    auto result_node =
        as_type_ptr<op::Constant>(f->get_results().at(0)->input_value(0).get_node_shared_ptr());
    ASSERT_TRUE(result_node);
    ASSERT_EQ(data_shape, result_node->get_output_shape(0));
    std::vector<int32_t> expected{input_data};
    // we have updated coordinate (1, 0, 0)
    expected.at(9) = 2;
    range_test_check(result_node->cast_vector<int32_t>(), expected);
}

void test_constant_folding_reshape_v1(Shape& shape_in,
                                      vector<float>& values_in,
                                      Shape shape_shape,
                                      vector<int32_t> values_shape,
                                      bool zero_flag = false)
{
    auto constant_in = make_shared<op::Constant>(element::f32, shape_in, values_in);
    auto constant_shape = make_shared<op::Constant>(element::i64, shape_shape, values_shape);
    auto dyn_reshape = make_shared<op::v1::Reshape>(constant_in, constant_shape, zero_flag);
    dyn_reshape->set_friendly_name("test");
    auto f = make_shared<Function>(dyn_reshape, ParameterVector{});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::v1::Reshape>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::Constant>(f), 1);

    auto new_const =
        as_type_ptr<op::Constant>(f->get_results().at(0)->input_value(0).get_node_shared_ptr());
    ASSERT_TRUE(new_const);
    ASSERT_EQ(new_const->get_friendly_name(), "test");
    auto values_out = new_const->get_vector<float>();

    ASSERT_TRUE(test::all_close_f(values_in, values_out, MIN_FLOAT_TOLERANCE_BITS));
}
TEST(constant_folding, constant_dyn_reshape_v1_2d)
{
    Shape shape_in{2, 5};
    vector<float> values_in{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

    test_constant_folding_reshape_v1(shape_in, values_in, {4}, {1, 1, 1, 10});
    test_constant_folding_reshape_v1(shape_in, values_in, {4}, {1, 1, 2, 5});
    test_constant_folding_reshape_v1(shape_in, values_in, {3}, {1, 2, 5});
    test_constant_folding_reshape_v1(shape_in, values_in, {3}, {5, 2, 1});
}

TEST(constant_folding, constant_dyn_reshape_v1_pattern_with_negative_indices)
{
    Shape shape_in{2, 2, 2, 2};
    vector<float> values_in{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};

    test_constant_folding_reshape_v1(shape_in, values_in, {3}, {4, -1, 2});
    test_constant_folding_reshape_v1(shape_in, values_in, {2}, {4, -1});
    test_constant_folding_reshape_v1(shape_in, values_in, {1}, {-1});
}

TEST(constant_folding, constant_dyn_reshape_v1_pattern_with_zero_dims)
{
    Shape shape_in{2, 2, 2, 2};
    vector<float> values_in{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};

    test_constant_folding_reshape_v1(shape_in, values_in, {4}, {2, -1, 2, 0}, true);
    test_constant_folding_reshape_v1(shape_in, values_in, {4}, {4, 1, 0, 2}, true);
}

TEST(constant_folding, disable_constant_folding)
{
    auto input = make_shared<op::Parameter>(element::f32, Shape{1, 3});
    auto constant_shape = op::Constant::create(element::i64, Shape{1}, {3});
    auto dyn_reshape = make_shared<op::v1::Reshape>(input, constant_shape, true);
    auto& rt_info = dyn_reshape->get_rt_info();
    rt_info["DISABLED_CONSTANT_FOLDING"];
    auto f = make_shared<Function>(dyn_reshape, ParameterVector{input});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::v1::Reshape>(f), 1);
    ASSERT_EQ(count_ops_of_type<op::Constant>(f), 1);
}

TEST(constant_folding, constant_loop)
{
    auto X = make_shared<opset5::Constant>(
        element::f32, Shape{2, 1, 3}, std::vector<int64_t>{0, 1, 2, 3, 4, 5});
    auto Y =
        make_shared<opset5::Constant>(element::f32, Shape{1, 1, 3}, std::vector<int64_t>{1, 2, 3});

    // Body parameters
    auto Xi = make_shared<opset5::Parameter>(element::f32, PartialShape::dynamic());
    auto Yi = make_shared<opset5::Parameter>(element::f32, PartialShape::dynamic());
    auto body_condition = std::make_shared<ngraph::opset5::Constant>(
        ngraph::element::boolean, ngraph::Shape{1}, true);

    auto trip_count =
        std::make_shared<ngraph::opset5::Constant>(ngraph::element::i64, ngraph::Shape{1}, 2);
    auto exec_condition = std::make_shared<ngraph::opset5::Constant>(
        ngraph::element::boolean, ngraph::Shape{1}, true);
    // Body
    auto sum = make_shared<ngraph::opset5::Add>(Xi, Yi);
    auto body =
        make_shared<ngraph::Function>(OutputVector{body_condition, sum}, ParameterVector{Xi, Yi});
    auto loop = make_shared<opset5::Loop>(trip_count, exec_condition);
    loop->set_function(body);
    loop->set_special_body_ports(ngraph::opset5::Loop::SpecialBodyPorts{-1, 0});

    loop->set_sliced_input(Xi, X, 0, 1, 1, -1, 0);
    loop->set_invariant_input(Yi, Y);

    auto out0 = loop->get_iter_value(sum, -1);
    auto out1 = loop->get_concatenated_slices(sum, 0, 1, 1, -1, 0);

    auto result0 = make_shared<opset5::Result>(out0);
    auto result1 = make_shared<opset5::Result>(out1);

    auto results = ResultVector{result0, result1};
    auto f = make_shared<Function>(results, ParameterVector{});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<ngraph::opset5::Loop>(f), 0);
    ASSERT_EQ(count_ops_of_type<ngraph::opset5::Constant>(f), 2);

    auto result_node_0 =
        as_type_ptr<op::Constant>(f->get_results().at(0)->input_value(0).get_node_shared_ptr());
    auto result_node_1 =
        as_type_ptr<op::Constant>(f->get_results().at(1)->input_value(0).get_node_shared_ptr());
    ASSERT_TRUE(result_node_0);
    ASSERT_TRUE(result_node_1);

    const ngraph::Shape shape_0{1, 1, 3};
    const ngraph::Shape shape_1{2, 1, 3};

    ASSERT_EQ(shape_0, result_node_0->get_output_shape(0));
    ASSERT_EQ(shape_1, result_node_1->get_output_shape(0));
    std::vector<float> expected_0{4, 6, 8};
    std::vector<float> expected_1{1, 3, 5, 4, 6, 8};
    range_test_check(result_node_0->cast_vector<float>(), expected_0);
    range_test_check(result_node_1->cast_vector<float>(), expected_1);
}
