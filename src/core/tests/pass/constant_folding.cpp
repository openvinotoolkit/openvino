// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/pass/constant_folding.hpp"

#include <gmock/gmock.h>

#include "common_test_utils/all_close_f.hpp"
#include "common_test_utils/ov_test_utils.hpp"
#include "common_test_utils/test_tools.hpp"
#include "openvino/core/constant_fold_utils.hpp"
#include "openvino/op/ops.hpp"
#include "ov_ops/type_relaxed.hpp"
#include "transformations/common_optimizations/disable_shapeof_constant_folding.hpp"
#include "transformations/utils/utils.hpp"

using namespace ov;
using namespace std;

namespace {

std::shared_ptr<op::v0::Constant> get_result_constant(std::shared_ptr<Model> m, size_t pos = 0) {
    return ov::as_type_ptr<op::v0::Constant>(m->get_results().at(pos)->input_value(0).get_node_shared_ptr());
}

template <typename T>
std::vector<T> get_result_constant_data(std::shared_ptr<Model> m, size_t pos) {
    auto new_const = get_result_constant(m, pos);
    return new_const->cast_vector<T>();
}

void range_test_check(const vector<double>& values_out, const vector<double>& values_expected) {
    ASSERT_TRUE(ov::test::utils::all_close_f(values_out, values_expected, MIN_FLOAT_TOLERANCE_BITS));
}

void range_test_check(const vector<float>& values_out, const vector<float>& values_expected) {
    ASSERT_TRUE(ov::test::utils::all_close_f(values_out, values_expected, MIN_FLOAT_TOLERANCE_BITS));
}

template <typename T>
typename std::enable_if<std::is_integral<T>::value>::type range_test_check(const vector<T>& values_out,
                                                                           const vector<T>& values_expected) {
    ASSERT_EQ(values_out, values_expected);
}

std::ostream& operator<<(std::ostream& os, const std::vector<std::string>& s) {
    os << "[";
    for (auto it = s.begin(); it != s.end(); ++it) {
        if (it != s.begin()) {
            os << ", " << *it;
        } else {
            os << *it;
        }
    }
    os << "]";
    return os;
}

void run_constant_folding(std::shared_ptr<ov::Model>& model) {
    pass::Manager pass_manager;
    pass_manager.register_pass<ov::pass::InitNodeInfo>();
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(model);
}

void check_names(const std::shared_ptr<ov::Node>& node,
                 const std::vector<std::string>& expected_fused_names,
                 const std::string expected_name = "test",
                 bool exact = true) {
    EXPECT_TRUE(node);

    // Check node name
    ASSERT_EQ(node->get_friendly_name(), expected_name);

    // Check fused name
    ASSERT_TRUE(!expected_fused_names.empty());
    std::vector<std::string> fused_names = ov::getFusedNamesVector(node);
    if (exact) {
        std::vector<std::string> expected_sorted = expected_fused_names;
        std::sort(fused_names.begin(), fused_names.end());
        std::sort(expected_sorted.begin(), expected_sorted.end());
        bool is_equal = std::equal(fused_names.begin(), fused_names.end(), expected_sorted.begin());
        std::stringstream ss;
        if (!is_equal) {
            ss << "Expected names are not matched to the fused names. Expected '" << expected_fused_names
               << "' but actually received '" << fused_names << "'";
        }
        ASSERT_TRUE(is_equal) << ss.str();
    } else {
        bool is_expected_name_missed = false;
        for (auto& name : expected_fused_names) {
            if (std::find(fused_names.begin(), fused_names.end(), name) == fused_names.end()) {
                is_expected_name_missed = true;
                break;
            }
        }
        std::stringstream ss;
        if (is_expected_name_missed) {
            ss << "Not all expected names are found in fused names. Expected '" << expected_fused_names
               << "' but actually received '" << fused_names << "'";
        }
        ASSERT_FALSE(is_expected_name_missed) << ss.str();
    }
}

}  // namespace

TEST(constant_folding, acosh) {
    Shape shape_in{2, 4, 1};

    vector<float> values_in{0, 1, 2, 3, 4, 5, 6, 7};
    vector<float> expected;
    for (float f : values_in) {
        expected.push_back(std::acosh(f));
    }
    auto constant = make_shared<op::v0::Constant>(element::f32, shape_in, values_in);
    constant->set_friendly_name("constant");
    auto acosh = make_shared<op::v3::Acosh>(constant);
    acosh->set_friendly_name("test");
    auto m = make_shared<Model>(acosh, ParameterVector{});

    run_constant_folding(m);

    EXPECT_EQ(count_ops_of_type<op::v3::Acosh>(m), 0);
    EXPECT_EQ(count_ops_of_type<op::v0::Constant>(m), 1);
    ASSERT_EQ(m->get_results().size(), 1);

    auto new_const = get_result_constant(m);
    EXPECT_TRUE(new_const);

    check_names(new_const, {"constant", "test"});

    auto values_out = new_const->get_vector<float>();
    EXPECT_TRUE(ov::test::utils::all_close_f(expected, values_out, MIN_FLOAT_TOLERANCE_BITS));
}

TEST(constant_folding, asinh) {
    Shape shape_in{2, 4, 1};

    vector<float> values_in{0, 1, 2, 3, 4, 5, 6, 7};
    vector<float> expected;
    for (float f : values_in) {
        expected.push_back(std::asinh(f));
    }
    auto constant = make_shared<op::v0::Constant>(element::f32, shape_in, values_in);
    constant->set_friendly_name("constant");
    auto asinh = make_shared<op::v3::Asinh>(constant);
    asinh->set_friendly_name("test");
    auto m = make_shared<Model>(asinh, ParameterVector{});

    run_constant_folding(m);

    EXPECT_EQ(count_ops_of_type<op::v3::Asinh>(m), 0);
    EXPECT_EQ(count_ops_of_type<op::v0::Constant>(m), 1);
    ASSERT_EQ(m->get_results().size(), 1);

    auto new_const = get_result_constant(m);
    EXPECT_TRUE(new_const);
    check_names(new_const, {"constant", "test"});

    auto values_out = new_const->get_vector<float>();
    EXPECT_TRUE(ov::test::utils::all_close_f(expected, values_out, MIN_FLOAT_TOLERANCE_BITS));
}

TEST(constant_folding, atanh) {
    Shape shape_in{2, 4, 1};

    vector<float> values_in{0, 1, 2, 3, 4, 5, 6, 7};
    vector<float> expected;
    for (float f : values_in) {
        expected.push_back(std::atanh(f));
    }
    auto constant = make_shared<op::v0::Constant>(element::f32, shape_in, values_in);
    constant->set_friendly_name("constant");
    auto atanh = make_shared<op::v3::Atanh>(constant);
    atanh->set_friendly_name("test");
    auto m = make_shared<Model>(atanh, ParameterVector{});

    run_constant_folding(m);

    EXPECT_EQ(count_ops_of_type<op::v3::Atanh>(m), 0);
    EXPECT_EQ(count_ops_of_type<op::v0::Constant>(m), 1);
    ASSERT_EQ(m->get_results().size(), 1);

    auto new_const = get_result_constant(m);
    EXPECT_TRUE(new_const);
    check_names(new_const, {"constant", "test"});

    auto values_out = new_const->get_vector<float>();
    EXPECT_TRUE(ov::test::utils::all_close_f(expected, values_out, MIN_FLOAT_TOLERANCE_BITS));
}

TEST(constant_folding, constant_squeeze) {
    Shape shape_in{2, 4, 1};
    Shape shape_out{2, 4};
    Shape axes_shape{1};

    vector<float> values_in{0, 1, 2, 3, 4, 5, 6, 7};
    auto constant = make_shared<op::v0::Constant>(element::f32, shape_in, values_in);
    constant->set_friendly_name("constant");
    vector<int64_t> values_axes{2};
    auto constant_axes = op::v0::Constant::create(element::i64, axes_shape, values_axes);
    constant_axes->set_friendly_name("constant_axes");
    auto squeeze = make_shared<op::v0::Squeeze>(constant, constant_axes);
    squeeze->set_friendly_name("test");
    auto m = make_shared<Model>(squeeze, ParameterVector{});

    run_constant_folding(m);

    ASSERT_EQ(count_ops_of_type<op::v0::Squeeze>(m), 0);
    ASSERT_EQ(count_ops_of_type<op::v0::Constant>(m), 1);

    auto new_const = get_result_constant(m);
    EXPECT_TRUE(new_const);
    check_names(new_const, {"constant", "constant_axes", "test"});
    ASSERT_EQ(new_const->get_shape(), shape_out);

    auto values_out = new_const->get_vector<float>();
    ASSERT_TRUE(ov::test::utils::all_close_f(values_in, values_out, MIN_FLOAT_TOLERANCE_BITS));
}

TEST(constant_folding, constant_unsqueeze) {
    Shape shape_in{2, 4};
    Shape shape_out{2, 4, 1, 1};
    Shape axes_shape{2};

    vector<float> values_in{0, 1, 2, 3, 4, 5, 6, 7};
    auto constant = make_shared<ov::op::v0::Constant>(element::f32, shape_in, values_in);
    constant->set_friendly_name("constant");
    vector<int64_t> values_axes{2, 3};
    auto constant_axes = ov::op::v0::Constant::create(element::i64, axes_shape, values_axes);
    constant_axes->set_friendly_name("constant_axes");
    auto unsqueeze = make_shared<op::v0::Unsqueeze>(constant, constant_axes);
    unsqueeze->set_friendly_name("test");
    auto f = make_shared<Model>(unsqueeze, ParameterVector{});

    run_constant_folding(f);

    ASSERT_EQ(count_ops_of_type<op::v0::Unsqueeze>(f), 0);
    ASSERT_EQ(count_ops_of_type<ov::op::v0::Constant>(f), 1);

    auto new_const = get_result_constant(f);
    ASSERT_TRUE(new_const);
    check_names(new_const, {"constant", "constant_axes", "test"});
    ASSERT_EQ(new_const->get_shape(), shape_out);

    auto values_out = new_const->get_vector<float>();
    ASSERT_TRUE(ov::test::utils::all_close_f(values_in, values_out, MIN_FLOAT_TOLERANCE_BITS));
}

TEST(constant_folding, constant_broadcast_v1) {
    vector<int32_t> values_in{0, 1};
    auto constant_in = make_shared<ov::op::v0::Constant>(element::i32, Shape{2}, values_in);
    constant_in->set_friendly_name("constant_in");
    vector<int64_t> shape_in{2, 4};
    auto constant_shape = make_shared<ov::op::v0::Constant>(element::i64, Shape{2}, shape_in);
    constant_shape->set_friendly_name("constant_shape");
    vector<int64_t> axes_in{0};
    auto constant_axes = make_shared<ov::op::v0::Constant>(element::i64, Shape{1}, axes_in);
    constant_axes->set_friendly_name("constant_axes");
    auto broadcast_v1 = make_shared<op::v1::Broadcast>(constant_in, constant_shape, constant_axes);
    broadcast_v1->set_friendly_name("test");
    auto f = make_shared<Model>(broadcast_v1, ParameterVector{});

    run_constant_folding(f);

    ASSERT_EQ(count_ops_of_type<op::v1::Broadcast>(f), 0);
    ASSERT_EQ(count_ops_of_type<ov::op::v0::Constant>(f), 1);

    auto new_const = get_result_constant(f);
    ASSERT_TRUE(new_const);
    check_names(new_const, {"constant_in", "constant_shape", "constant_axes", "test"});
    auto values_out = new_const->get_vector<int32_t>();

    vector<int32_t> values_expected{0, 0, 0, 0, 1, 1, 1, 1};
    ASSERT_EQ(values_expected, values_out);
}

TEST(constant_folding, constant_broadcast_v1_with_target_shape) {
    vector<int32_t> values_in{1};
    auto constant_in = make_shared<ov::op::v0::Constant>(element::i32, Shape{1, 1, 1, 1}, values_in);
    constant_in->set_friendly_name("constant_in");
    vector<int64_t> shape_in{1, 3, 1, 1};
    auto target_shape = make_shared<ov::op::v0::Constant>(element::i64, Shape{4}, shape_in);
    target_shape->set_friendly_name("target_shape");
    auto broadcast_v1 = make_shared<op::v1::Broadcast>(constant_in, target_shape);
    broadcast_v1->set_friendly_name("test");
    auto f = make_shared<Model>(broadcast_v1, ParameterVector{});

    run_constant_folding(f);

    ASSERT_EQ(count_ops_of_type<op::v1::Broadcast>(f), 0);
    ASSERT_EQ(count_ops_of_type<ov::op::v0::Constant>(f), 1);

    auto new_const = get_result_constant(f);
    ASSERT_TRUE(new_const);
    check_names(new_const, {"constant_in", "target_shape", "test"}, "test", false);
    auto values_out = new_const->get_vector<int32_t>();

    vector<int32_t> values_expected{1, 1, 1};
    ASSERT_EQ(values_expected, values_out);
}

TEST(constant_folding, constant_broadcast_v1_numpy) {
    vector<int32_t> values_in{0, 1};
    auto constant_in = make_shared<ov::op::v0::Constant>(element::i32, Shape{2}, values_in);
    constant_in->set_friendly_name("constant_in");
    vector<int64_t> shape_in{4, 2};
    auto constant_shape = make_shared<ov::op::v0::Constant>(element::i64, Shape{2}, shape_in);
    constant_shape->set_friendly_name("constant_shape");
    auto broadcast_v1 = make_shared<op::v1::Broadcast>(constant_in, constant_shape);
    broadcast_v1->set_friendly_name("test");
    auto f = make_shared<Model>(broadcast_v1, ParameterVector{});

    run_constant_folding(f);

    ASSERT_EQ(count_ops_of_type<op::v1::Broadcast>(f), 0);
    ASSERT_EQ(count_ops_of_type<ov::op::v0::Constant>(f), 1);

    auto new_const = get_result_constant(f);
    ASSERT_TRUE(new_const);
    check_names(new_const, {"constant_in", "constant_shape", "test"}, "test", false);
    auto values_out = new_const->get_vector<int32_t>();

    vector<int32_t> values_expected{0, 1, 0, 1, 0, 1, 0, 1};
    ASSERT_EQ(values_expected, values_out);
}

TEST(constant_folding, constant_unary_binary) {
    vector<int> values_a{1, 2, 3, 4};
    vector<int> values_b{1, 2, 3, 4};
    vector<int> values_c{-1, -1, -1, -1};
    vector<int> values_d{1, 4, 9, 16};
    vector<int> values_e{5, 6};
    vector<int> values_f{0, 10};
    vector<int> values_g{1, 4};
    vector<char> values_h{0, 0, 1, 1};
    vector<char> values_i{0, 1};
    vector<int8_t> values_j{-3, 5};
    vector<uint8_t> values_k{3, 5};
    auto a = make_shared<ov::op::v0::Constant>(element::i32, Shape{2, 2}, values_a);
    a->set_friendly_name("a");
    auto b = make_shared<ov::op::v0::Constant>(element::i32, Shape{2, 2}, values_b);
    b->set_friendly_name("b");
    auto c = make_shared<ov::op::v0::Constant>(element::i32, Shape{2, 2}, values_c);
    c->set_friendly_name("c");
    auto d = make_shared<ov::op::v0::Constant>(element::i32, Shape{2, 2}, values_d);
    d->set_friendly_name("d");
    auto e = make_shared<ov::op::v0::Constant>(element::i32, Shape{2}, values_e);
    e->set_friendly_name("e");
    auto f = make_shared<ov::op::v0::Constant>(element::i32, Shape{2}, values_f);
    f->set_friendly_name("f");
    auto g = make_shared<ov::op::v0::Constant>(element::i32, Shape{2}, values_g);
    g->set_friendly_name("g");
    auto h = make_shared<ov::op::v0::Constant>(element::boolean, Shape{2, 2}, values_h);
    h->set_friendly_name("h");
    auto i = make_shared<ov::op::v0::Constant>(element::boolean, Shape{2}, values_i);
    i->set_friendly_name("i");
    auto j = make_shared<ov::op::v0::Constant>(element::i8, Shape{2}, values_j);
    j->set_friendly_name("j");
    auto k = make_shared<ov::op::v0::Constant>(element::u8, Shape{2}, values_k);
    k->set_friendly_name("k");
    auto doubles = make_shared<ov::op::v0::Constant>(element::f64, Shape{2}, std::vector<double>{4.0, 9.0});
    doubles->set_friendly_name("doubles");
    auto doubles2 = make_shared<ov::op::v0::Constant>(element::f64, Shape{2}, std::vector<double>{4.0, 1.0});
    doubles2->set_friendly_name("doubles2");
    auto shorts = make_shared<ov::op::v0::Constant>(element::i16, Shape{3}, std::vector<int16_t>{14, -3, -3});
    shorts->set_friendly_name("shorts");
    auto shorts2 = make_shared<ov::op::v0::Constant>(element::i16, Shape{1}, std::vector<int16_t>{-3});
    shorts2->set_friendly_name("shorts2");
    auto unsigned_shorts =
        make_shared<ov::op::v0::Constant>(element::u16, Shape{3}, std::vector<uint16_t>{14, 300, 14});
    unsigned_shorts->set_friendly_name("unsigned_shorts");
    auto unsigned_shorts2 = make_shared<ov::op::v0::Constant>(element::u16, Shape{1}, std::vector<uint16_t>{300});
    unsigned_shorts2->set_friendly_name("unsigned_shorts2");

    auto add = make_shared<op::v1::Add>(a, b);
    add->set_friendly_name("add");
    auto sub = make_shared<op::v1::Subtract>(a, b);
    sub->set_friendly_name("sub");
    auto mul = make_shared<op::v1::Multiply>(a, b);
    mul->set_friendly_name("mul");
    auto divn = make_shared<op::v1::Divide>(a, b);
    divn->set_friendly_name("divn");
    auto pow = make_shared<op::v1::Power>(a, b);
    pow->set_friendly_name("pow");
    auto min = make_shared<op::v1::Minimum>(c, a);
    min->set_friendly_name("min");
    auto max = make_shared<op::v1::Maximum>(a, c);
    max->set_friendly_name("max");
    auto absn = make_shared<op::v0::Abs>(c);
    absn->set_friendly_name("absn");
    auto neg = make_shared<op::v0::Negative>(c);
    neg->set_friendly_name("neg");
    auto sqrt = make_shared<op::v0::Sqrt>(d);
    sqrt->set_friendly_name("sqrt");
    auto add_autob_numpy = make_shared<op::v1::Add>(a, e, op::AutoBroadcastType::NUMPY);
    add_autob_numpy->set_friendly_name("add_autob_numpy");
    auto sub_autob_numpy = make_shared<op::v1::Subtract>(a, e, op::AutoBroadcastType::NUMPY);
    sub_autob_numpy->set_friendly_name("sub_autob_numpy");
    auto mul_autob_numpy = make_shared<op::v1::Multiply>(a, e, op::AutoBroadcastType::NUMPY);
    mul_autob_numpy->set_friendly_name("mul_autob_numpy");
    auto div_autob_numpy = make_shared<op::v1::Divide>(a, g, op::AutoBroadcastType::NUMPY);
    div_autob_numpy->set_friendly_name("div_autob_numpy");
    auto pow_autob_numpy = make_shared<op::v1::Power>(a, g, op::AutoBroadcastType::NUMPY);
    pow_autob_numpy->set_friendly_name("pow_autob_numpy");
    auto min_autob_numpy = make_shared<op::v1::Minimum>(a, f, op::AutoBroadcastType::NUMPY);
    min_autob_numpy->set_friendly_name("min_autob_numpy");
    auto max_autob_numpy = make_shared<op::v1::Maximum>(a, f, op::AutoBroadcastType::NUMPY);
    max_autob_numpy->set_friendly_name("max_autob_numpy");
    auto equal_autob_numpy = make_shared<op::v1::Equal>(a, g, op::AutoBroadcastType::NUMPY);
    equal_autob_numpy->set_friendly_name("equal_autob_numpy");
    auto not_equal_autob_numpy = make_shared<op::v1::NotEqual>(a, g, op::AutoBroadcastType::NUMPY);
    not_equal_autob_numpy->set_friendly_name("not_equal_autob_numpy");
    auto greater_autob_numpy = make_shared<op::v1::Greater>(a, g, op::AutoBroadcastType::NUMPY);
    greater_autob_numpy->set_friendly_name("greater_autob_numpy");
    auto greater_eq_autob_numpy = make_shared<op::v1::GreaterEqual>(a, g, op::AutoBroadcastType::NUMPY);
    greater_eq_autob_numpy->set_friendly_name("greater_eq_autob_numpy");
    auto less_autob_numpy = make_shared<op::v1::Less>(a, g, op::AutoBroadcastType::NUMPY);
    less_autob_numpy->set_friendly_name("less_autob_numpy");
    auto less_eq_autob_numpy = make_shared<op::v1::LessEqual>(a, g, op::AutoBroadcastType::NUMPY);
    less_eq_autob_numpy->set_friendly_name("less_eq_autob_numpy");
    auto logical_or_autob_numpy = make_shared<op::v1::LogicalOr>(h, i, op::AutoBroadcastType::NUMPY);
    logical_or_autob_numpy->set_friendly_name("logical_or_autob_numpy");
    auto logical_xor_autob_numpy = make_shared<op::v0::Xor>(h, i, op::AutoBroadcastType::NUMPY);
    logical_xor_autob_numpy->set_friendly_name("logical_xor_autob_numpy");
    auto doubles_sqrt = make_shared<op::v0::Sqrt>(doubles);
    doubles_sqrt->set_friendly_name("doubles_sqrt");
    auto sub_int8 = make_shared<op::v1::Subtract>(j, j);
    sub_int8->set_friendly_name("sub_int8");
    auto sub_uint8 = make_shared<op::v1::Subtract>(k, k);
    sub_uint8->set_friendly_name("sub_uint8");
    auto equal_doubles = make_shared<op::v1::Equal>(doubles, doubles2, op::AutoBroadcastType::NUMPY);
    equal_doubles->set_friendly_name("equal_doubles");
    auto equal_shorts = make_shared<op::v1::Equal>(shorts, shorts2, op::AutoBroadcastType::NUMPY);
    equal_shorts->set_friendly_name("equal_shorts");
    auto equal_unsigned_shorts =
        make_shared<op::v1::Equal>(unsigned_shorts, unsigned_shorts2, op::AutoBroadcastType::NUMPY);
    equal_unsigned_shorts->set_friendly_name("equal_unsigned_shorts");
    auto neg_sqrt = make_shared<op::v0::Sqrt>(c);
    neg_sqrt->set_friendly_name("neg_sqrt");

    auto func = make_shared<Model>(NodeVector{add,
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
                                              logical_xor_autob_numpy,
                                              doubles_sqrt,
                                              sub_int8,
                                              sub_uint8,
                                              equal_doubles,
                                              equal_shorts,
                                              equal_unsigned_shorts},
                                   ParameterVector{});
    auto func_error = make_shared<Model>(NodeVector{neg_sqrt}, ParameterVector{});

    run_constant_folding(func);

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
    vector<double> doubles_sqrt_expected{2.0, 3.0};
    vector<int8_t> sub_int8_expected{0, 0};
    vector<uint8_t> sub_uint8_expected{0, 0};
    vector<char> equal_doubles_expected{1, 0};
    vector<char> equal_shorts_expected{0, 1, 1};
    vector<char> equal_unsigned_shorts_expected{0, 1, 0};

    size_t index = 0;
    ASSERT_EQ(get_result_constant_data<int>(func, index), add_expected);
    check_names(get_result_constant(func, index++), {"a", "b", "add"}, "add");
    ASSERT_EQ(get_result_constant_data<int>(func, index), sub_expected);
    check_names(get_result_constant(func, index++), {"a", "b", "sub"}, "sub");
    ASSERT_EQ(get_result_constant_data<int>(func, index), mul_expected);
    check_names(get_result_constant(func, index++), {"a", "b", "mul"}, "mul");
    ASSERT_EQ(get_result_constant_data<int>(func, index), div_expected);
    check_names(get_result_constant(func, index++), {"a", "b", "divn"}, "divn");
    ASSERT_EQ(get_result_constant_data<int>(func, index), pow_expected);
    check_names(get_result_constant(func, index++), {"a", "b", "pow"}, "pow");
    ASSERT_EQ(get_result_constant_data<int>(func, index), min_expected);
    check_names(get_result_constant(func, index++), {"c", "a", "min"}, "min");
    ASSERT_EQ(get_result_constant_data<int>(func, index), max_expected);
    check_names(get_result_constant(func, index++), {"a", "c", "max"}, "max");
    ASSERT_EQ(get_result_constant_data<int>(func, index), abs_neg_expected);
    check_names(get_result_constant(func, index++), {"c", "absn"}, "absn");
    ASSERT_EQ(get_result_constant_data<int>(func, index), abs_neg_expected);
    check_names(get_result_constant(func, index++), {"c", "neg"}, "neg");
    ASSERT_EQ(get_result_constant_data<int>(func, index), sqrt_expected);
    check_names(get_result_constant(func, index++), {"d", "sqrt"}, "sqrt");
    ASSERT_EQ(get_result_constant_data<int>(func, index), add_autob_numpy_expected);
    check_names(get_result_constant(func, index++), {"a", "e", "add_autob_numpy"}, "add_autob_numpy");
    ASSERT_EQ(get_result_constant_data<int>(func, index), sub_autob_numpy_expected);
    check_names(get_result_constant(func, index++), {"a", "e", "sub_autob_numpy"}, "sub_autob_numpy");
    ASSERT_EQ(get_result_constant_data<int>(func, index), mul_autob_numpy_expected);
    check_names(get_result_constant(func, index++), {"a", "e", "mul_autob_numpy"}, "mul_autob_numpy");
    ASSERT_EQ(get_result_constant_data<int>(func, index), div_autob_numpy_expected);
    check_names(get_result_constant(func, index++), {"a", "g", "div_autob_numpy"}, "div_autob_numpy");
    ASSERT_EQ(get_result_constant_data<int>(func, index), pow_autob_numpy_expected);
    check_names(get_result_constant(func, index++), {"a", "g", "pow_autob_numpy"}, "pow_autob_numpy");
    ASSERT_EQ(get_result_constant_data<int>(func, index), min_autob_numpy_expected);
    check_names(get_result_constant(func, index++), {"a", "f", "min_autob_numpy"}, "min_autob_numpy");
    ASSERT_EQ(get_result_constant_data<int>(func, index), max_autob_numpy_expected);
    check_names(get_result_constant(func, index++), {"a", "f", "max_autob_numpy"}, "max_autob_numpy");
    ASSERT_EQ(get_result_constant_data<char>(func, index), equal_autob_numpy_expected);
    check_names(get_result_constant(func, index++), {"a", "g", "equal_autob_numpy"}, "equal_autob_numpy");
    ASSERT_EQ(get_result_constant_data<char>(func, index), not_equal_autob_numpy_expected);
    check_names(get_result_constant(func, index++), {"a", "g", "not_equal_autob_numpy"}, "not_equal_autob_numpy");
    ASSERT_EQ(get_result_constant_data<char>(func, index), greater_autob_numpy_expected);
    check_names(get_result_constant(func, index++), {"a", "g", "greater_autob_numpy"}, "greater_autob_numpy");
    ASSERT_EQ(get_result_constant_data<char>(func, index), greater_eq_autob_numpy_expected);
    check_names(get_result_constant(func, index++), {"a", "g", "greater_eq_autob_numpy"}, "greater_eq_autob_numpy");
    ASSERT_EQ(get_result_constant_data<char>(func, index), less_autob_numpy_expected);
    check_names(get_result_constant(func, index++), {"a", "g", "less_autob_numpy"}, "less_autob_numpy");
    ASSERT_EQ(get_result_constant_data<char>(func, index), less_eq_autob_numpy_expected);
    check_names(get_result_constant(func, index++), {"a", "g", "less_eq_autob_numpy"}, "less_eq_autob_numpy");
    ASSERT_EQ(get_result_constant_data<char>(func, index), logical_or_autob_numpy_expected);
    check_names(get_result_constant(func, index++), {"h", "i", "logical_or_autob_numpy"}, "logical_or_autob_numpy");
    ASSERT_EQ(get_result_constant_data<char>(func, index), logical_xor_autob_numpy_expected);
    check_names(get_result_constant(func, index++), {"h", "i", "logical_xor_autob_numpy"}, "logical_xor_autob_numpy");
    ASSERT_EQ(get_result_constant_data<double>(func, index), doubles_sqrt_expected);
    check_names(get_result_constant(func, index++), {"doubles", "doubles_sqrt"}, "doubles_sqrt");
    ASSERT_EQ(get_result_constant_data<int8_t>(func, index), sub_int8_expected);
    check_names(get_result_constant(func, index++), {"j", "sub_int8"}, "sub_int8");
    ASSERT_EQ(get_result_constant_data<uint8_t>(func, index), sub_uint8_expected);
    check_names(get_result_constant(func, index++), {"k", "sub_uint8"}, "sub_uint8");
    ASSERT_EQ(get_result_constant_data<char>(func, index), equal_doubles_expected);
    check_names(get_result_constant(func, index++), {"doubles", "doubles2", "equal_doubles"}, "equal_doubles");
    ASSERT_EQ(get_result_constant_data<char>(func, index), equal_shorts_expected);
    check_names(get_result_constant(func, index++), {"shorts", "shorts2", "equal_shorts"}, "equal_shorts");
    ASSERT_EQ(get_result_constant_data<char>(func, index), equal_unsigned_shorts_expected);
    check_names(get_result_constant(func, index++),
                {"unsigned_shorts", "unsigned_shorts2", "equal_unsigned_shorts"},
                "equal_unsigned_shorts");

    pass::Manager pass_manager;
    OV_ASSERT_NO_THROW(pass_manager.run_passes(func_error));
}

template <element::Type_t from, element::Type_t to, typename T, typename U>
static void test_const_convert(const vector<T>& values_in, const vector<U>& values_expected) {
    auto constant = ov::op::v0::Constant::create(from, Shape{values_in.size()}, values_in);
    constant->set_friendly_name("constant");
    auto convert = make_shared<op::v0::Convert>(constant, to);
    convert->set_friendly_name("test");
    auto f = make_shared<Model>(convert, ParameterVector{});

    run_constant_folding(f);

    ASSERT_EQ(count_ops_of_type<op::v0::Convert>(f), 0);
    ASSERT_EQ(count_ops_of_type<ov::op::v0::Constant>(f), 1);

    auto new_const = get_result_constant(f);
    ASSERT_TRUE(new_const);
    check_names(new_const, {"constant", "test"});
    ASSERT_EQ(new_const->get_output_element_type(0), to);
    auto values_out = new_const->template cast_vector<U>();

    ASSERT_EQ(values_expected, values_out);
}

TEST(constant_folding, const_convert) {
    {
        vector<float> in{1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7};
        vector<uint64_t> expected{1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7};
        test_const_convert<element::f32, element::u64>(in, expected);
    }
    {
        vector<bool> in{false, true, true, false, false, false, true};
        vector<float> expected{0.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f};
        test_const_convert<element::boolean, element::f32>(in, expected);
    }
    {
        vector<float> in{1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f};
        vector<bool> expected{true, false, true, false, true, false, true};
        test_const_convert<element::f32, element::boolean>(in, expected);
    }
    {
        vector<int64_t> in{1, 2, 3, 4, 5};
        vector<double> expected{1.0, 2.0, 3.0, 4.0, 5.0};
        test_const_convert<element::i64, element::f64>(in, expected);
    }
    {
        vector<double> in{1.2, 2.1, 3.3, 4.45, 5.02};
        vector<int64_t> expected{1, 2, 3, 4, 5};
        test_const_convert<element::f64, element::i64>(in, expected);
    }
    {
        vector<int8_t> in{7, 0, 1, 2, 3, 4, 5, -1, -2, -8};
        vector<float> expected{7, 0, 1, 2, 3, 4, 5, -1, -2, -8};
        test_const_convert<element::i4, element::f32>(in, expected);
    }
    {
        vector<float> in{9, 0, 1, 2, 3, 4, 5, -1, -2, -10};
        vector<int8_t> expected{-7, 0, 1, 2, 3, 4, 5, -1, -2, 6};
        test_const_convert<element::f32, element::i4>(in, expected);
    }
    {
        vector<int8_t> in{-128, -2, 0, 1, 3, 127};
        vector<float> expected{-128, -2, 0, 1, 3, 127};
        test_const_convert<element::i8, element::f32>(in, expected);
    }
    {
        vector<uint8_t> in{0, 1, 3, 127, 255};
        vector<float> expected{0, 1, 3, 127, 255};
        test_const_convert<element::u8, element::f32>(in, expected);
    }
    {
        vector<float> in{-300, -128, -1, 0, 33, 127, 128};
        vector<int8_t> expected{-44, -128, -1, 0, 33, 127, -128};
        test_const_convert<element::f32, element::i8>(in, expected);
    }
    {
        vector<float> in{0, 33, 127, 255, 256};
        vector<uint8_t> expected{0, 33, 127, 255, 0};
        test_const_convert<element::f32, element::u8>(in, expected);
    }
}

TEST(constant_folding, shape_of_v0) {
    Shape input_shape{3, 4, 0, 22, 608, 909, 3};

    auto param = make_shared<ov::op::v0::Parameter>(element::boolean, input_shape);
    param->set_friendly_name("param");
    auto shape_of = make_shared<op::v0::ShapeOf>(param);
    shape_of->set_friendly_name("test");
    auto f = make_shared<Model>(shape_of, ParameterVector{param});

    run_constant_folding(f);

    ASSERT_EQ(count_ops_of_type<op::v0::ShapeOf>(f), 0);
    ASSERT_EQ(count_ops_of_type<ov::op::v0::Constant>(f), 1);

    auto new_const = get_result_constant(f);
    ASSERT_TRUE(new_const);
    check_names(new_const, {"test"});
    ASSERT_EQ(new_const->get_output_element_type(0), element::i64);
    auto values_out = new_const->get_vector<int64_t>();

    ASSERT_EQ((vector<int64_t>{3, 4, 0, 22, 608, 909, 3}), values_out);
}

TEST(constant_folding, shape_of_v3) {
    Shape input_shape{3, 4, 0, 22, 608, 909, 3};

    auto param = make_shared<ov::op::v0::Parameter>(element::boolean, input_shape);
    param->set_friendly_name("param");
    auto shape_of = make_shared<op::v3::ShapeOf>(param);
    shape_of->set_friendly_name("test");
    auto f = make_shared<Model>(shape_of, ParameterVector{param});

    run_constant_folding(f);

    ASSERT_EQ(count_ops_of_type<op::v3::ShapeOf>(f), 0);
    ASSERT_EQ(count_ops_of_type<ov::op::v0::Constant>(f), 1);

    auto new_const = get_result_constant(f);
    ASSERT_TRUE(new_const);
    check_names(new_const, {"test"});
    ASSERT_EQ(new_const->get_output_element_type(0), element::i64);
    auto values_out = new_const->get_vector<int64_t>();

    ASSERT_EQ((vector<int64_t>{3, 4, 0, 22, 608, 909, 3}), values_out);
}

TEST(constant_folding, shape_of_i32_v3) {
    Shape input_shape{3, 4, 0, 22, 608, 909, 3};

    auto param = make_shared<ov::op::v0::Parameter>(element::boolean, input_shape);
    param->set_friendly_name("param");
    auto shape_of = make_shared<op::v3::ShapeOf>(param, element::i32);
    shape_of->set_friendly_name("test");
    auto f = make_shared<Model>(shape_of, ParameterVector{param});

    run_constant_folding(f);

    ASSERT_EQ(count_ops_of_type<op::v3::ShapeOf>(f), 0);
    ASSERT_EQ(count_ops_of_type<ov::op::v0::Constant>(f), 1);

    auto new_const = get_result_constant(f);
    ASSERT_TRUE(new_const);
    check_names(new_const, {"test"});
    ASSERT_EQ(new_const->get_output_element_type(0), element::i32);
    auto values_out = new_const->get_vector<int32_t>();

    ASSERT_EQ((vector<int32_t>{3, 4, 0, 22, 608, 909, 3}), values_out);
}

TEST(constant_folding, shape_of_dynamic_v0) {
    PartialShape input_shape{3, 4, Dimension::dynamic(), 22, 608, 909, 3};

    auto param = make_shared<ov::op::v0::Parameter>(element::boolean, input_shape);
    auto shape_of = make_shared<op::v0::ShapeOf>(param);
    shape_of->set_friendly_name("test");
    auto f = make_shared<Model>(shape_of, ParameterVector{param});

    run_constant_folding(f);

    ASSERT_EQ(f->get_ops().size(), 3);

    auto result_shape_of = f->get_results().at(0)->get_input_node_shared_ptr(0);
    ASSERT_EQ(result_shape_of, shape_of);
    check_names(result_shape_of, {"test"});
}

TEST(constant_folding, shape_of_dynamic_v3) {
    PartialShape input_shape{3, 4, Dimension::dynamic(), 22, 608, 909, 3};

    auto param = make_shared<ov::op::v0::Parameter>(element::boolean, input_shape);
    auto shape_of = make_shared<op::v3::ShapeOf>(param);
    shape_of->set_friendly_name("test");
    auto f = make_shared<Model>(shape_of, ParameterVector{param});

    run_constant_folding(f);

    ASSERT_EQ(f->get_ops().size(), 3);

    auto result_shape_of = f->get_results().at(0)->get_input_node_shared_ptr(0);
    ASSERT_EQ(result_shape_of, shape_of);
    check_names(result_shape_of, {"test"});
}

TEST(constant_folding, shape_of_dynamic_i32_v3) {
    PartialShape input_shape{3, 4, Dimension::dynamic(), 22, 608, 909, 3};

    auto param = make_shared<ov::op::v0::Parameter>(element::boolean, input_shape);
    auto shape_of = make_shared<op::v3::ShapeOf>(param, element::i32);
    shape_of->set_friendly_name("test");
    auto f = make_shared<Model>(shape_of, ParameterVector{param});

    run_constant_folding(f);

    ASSERT_EQ(f->get_ops().size(), 3);

    auto result_shape_of = f->get_results().at(0)->get_input_node_shared_ptr(0);
    ASSERT_EQ(result_shape_of, shape_of);
    check_names(result_shape_of, {"test"});
}

// We need to be sure that constant folding won't be calculated endlessly.
TEST(constant_folding, shape_of_dynamic_double_folding_v0) {
    PartialShape input_shape{3, 4, Dimension::dynamic(), 22, 608, 909, 3};

    auto param = make_shared<ov::op::v0::Parameter>(element::boolean, input_shape);
    auto shape_of = make_shared<op::v0::ShapeOf>(param);
    shape_of->set_friendly_name("test");
    auto f = make_shared<Model>(shape_of, ParameterVector{param});

    run_constant_folding(f);

    ASSERT_EQ(f->get_ops().size(), 3);

    auto result_shape_of = f->get_results().at(0)->get_input_node_shared_ptr(0);
    ASSERT_EQ(result_shape_of, shape_of);
    check_names(result_shape_of, {"test"});
}

TEST(constant_folding, shape_of_dynamic_double_folding_v3) {
    PartialShape input_shape{3, 4, Dimension::dynamic(), 22, 608, 909, 3};

    auto param = make_shared<ov::op::v0::Parameter>(element::boolean, input_shape);
    auto shape_of = make_shared<op::v3::ShapeOf>(param);
    shape_of->set_friendly_name("test");
    auto f = make_shared<Model>(shape_of, ParameterVector{param});

    run_constant_folding(f);

    ASSERT_EQ(f->get_ops().size(), 3);

    auto result_shape_of = f->get_results().at(0)->get_input_node_shared_ptr(0);
    ASSERT_EQ(result_shape_of, shape_of);
    check_names(result_shape_of, {"test"});
}

// Constant folding will not succeed on ShapeOf if the argument rank is dynamic.
// We want to make sure it fails gracefully, leaving the ShapeOf op in place.
TEST(constant_folding, shape_of_rank_dynamic_v0) {
    PartialShape input_shape{PartialShape::dynamic()};

    auto param = make_shared<ov::op::v0::Parameter>(element::boolean, input_shape);
    auto shape_of = make_shared<op::v0::ShapeOf>(param);
    shape_of->set_friendly_name("test");
    auto f = make_shared<Model>(shape_of, ParameterVector{param});

    run_constant_folding(f);

    ASSERT_EQ(count_ops_of_type<op::v0::ShapeOf>(f), 1);
    ASSERT_EQ(count_ops_of_type<ov::op::v0::Constant>(f), 0);

    auto result_shape_of = f->get_results().at(0)->get_input_node_shared_ptr(0);
    ASSERT_EQ(result_shape_of, shape_of);
    check_names(result_shape_of, {"test"});
}

TEST(constant_folding, shape_of_rank_dynamic_v3) {
    PartialShape input_shape{PartialShape::dynamic()};

    auto param = make_shared<ov::op::v0::Parameter>(element::boolean, input_shape);
    auto shape_of = make_shared<op::v3::ShapeOf>(param);
    shape_of->set_friendly_name("test");
    auto f = make_shared<Model>(shape_of, ParameterVector{param});

    run_constant_folding(f);

    ASSERT_EQ(count_ops_of_type<op::v3::ShapeOf>(f), 1);
    ASSERT_EQ(count_ops_of_type<ov::op::v0::Constant>(f), 0);

    auto result_shape_of = f->get_results().at(0)->get_input_node_shared_ptr(0);
    ASSERT_EQ(result_shape_of, shape_of);
    check_names(result_shape_of, {"test"});
}

static void const_reverse(const element::Type& axes_elem_type) {
    Shape input_shape{3, 3};

    vector<int32_t> values_in{1, 2, 3, 4, 5, 6, 7, 8, 9};
    auto constant = ov::op::v0::Constant::create(element::i32, input_shape, values_in);
    constant->set_friendly_name("constant");
    auto axes = ov::op::v0::Constant::create(axes_elem_type, {1}, {1});
    axes->set_friendly_name("axes");
    auto convert = make_shared<op::v1::Reverse>(constant, axes, op::v1::Reverse::Mode::INDEX);
    convert->set_friendly_name("test");
    auto f = make_shared<Model>(convert, ParameterVector{});

    run_constant_folding(f);

    ASSERT_EQ(count_ops_of_type<op::v1::Reverse>(f), 0);
    ASSERT_EQ(count_ops_of_type<ov::op::v0::Constant>(f), 1);

    auto new_const = get_result_constant(f);
    ASSERT_TRUE(new_const);
    check_names(new_const, {"constant", "axes", "test"});
    auto values_out = new_const->get_vector<int32_t>();

    vector<int32_t> values_expected{3, 2, 1, 6, 5, 4, 9, 8, 7};
    ASSERT_EQ(values_expected, values_out);
}

TEST(constant_folding, const_reverse) {
    for (auto&& axes_elem_type : {element::i8,
                                  element::u8,
                                  element::i16,
                                  element::u16,
                                  element::i32,
                                  element::u32,
                                  element::i64,
                                  element::u64}) {
        const_reverse(axes_elem_type);
    }
}

TEST(constant_folding, const_reduceprod) {
    Shape input_shape{3, 3};
    Shape output_shape{3};

    vector<int32_t> values_in{1, 2, 3, 4, 5, 6, 7, 8, 9};
    auto constant = ov::op::v0::Constant::create(element::i32, input_shape, values_in);
    constant->set_friendly_name("constant");
    Shape axes_shape{1};
    vector<int32_t> values_axes{1};
    auto constant_axes = ov::op::v0::Constant::create(element::i64, axes_shape, values_axes);
    constant_axes->set_friendly_name("constant_axes");
    auto convert = make_shared<op::v1::ReduceProd>(constant, constant_axes);
    convert->set_friendly_name("test");
    auto f = make_shared<Model>(convert, ParameterVector{});

    run_constant_folding(f);

    ASSERT_EQ(count_ops_of_type<op::v1::ReduceProd>(f), 0);
    ASSERT_EQ(count_ops_of_type<ov::op::v0::Constant>(f), 1);

    auto new_const = get_result_constant(f);
    ASSERT_TRUE(new_const);
    check_names(new_const, {"constant", "constant_axes", "test"});
    ASSERT_EQ(new_const->get_shape(), output_shape);

    auto values_out = new_const->get_vector<int32_t>();

    vector<int32_t> values_expected{6, 120, 504};

    ASSERT_EQ(values_expected, values_out);
}

TEST(constant_folding, const_reduceprod_keepdims) {
    Shape input_shape{3, 3};
    Shape output_shape{3, 1};

    vector<int32_t> values_in{1, 2, 3, 4, 5, 6, 7, 8, 9};
    auto constant = ov::op::v0::Constant::create(element::i32, input_shape, values_in);
    constant->set_friendly_name("constant");
    Shape axes_shape{1};
    vector<int32_t> values_axes{1};
    auto constant_axes = ov::op::v0::Constant::create(element::i64, axes_shape, values_axes);
    constant_axes->set_friendly_name("constant_axes");
    auto convert = make_shared<op::v1::ReduceProd>(constant, constant_axes, true);
    convert->set_friendly_name("test");
    auto f = make_shared<Model>(convert, ParameterVector{});

    run_constant_folding(f);

    ASSERT_EQ(count_ops_of_type<op::v1::ReduceProd>(f), 0);
    ASSERT_EQ(count_ops_of_type<ov::op::v0::Constant>(f), 1);

    auto new_const = get_result_constant(f);
    ASSERT_TRUE(new_const);
    check_names(new_const, {"constant", "constant_axes", "test"});
    ASSERT_EQ(new_const->get_shape(), output_shape);

    auto values_out = new_const->get_vector<int32_t>();

    vector<int32_t> values_expected{6, 120, 504};

    ASSERT_EQ(values_expected, values_out);
}

TEST(constant_folding, const_reducesum) {
    Shape input_shape{3, 3};
    Shape output_shape{3};

    vector<int32_t> values_in{1, 2, 3, 4, 5, 6, 7, 8, 9};
    auto constant = ov::op::v0::Constant::create(element::i32, input_shape, values_in);
    constant->set_friendly_name("constant");
    Shape axes_shape{1};
    vector<int32_t> values_axes{1};
    auto constant_axes = ov::op::v0::Constant::create(element::i64, axes_shape, values_axes);
    constant_axes->set_friendly_name("constant_axes");
    auto convert = make_shared<op::v1::ReduceSum>(constant, constant_axes);
    convert->set_friendly_name("test");
    auto f = make_shared<Model>(convert, ParameterVector{});

    run_constant_folding(f);
    ASSERT_EQ(count_ops_of_type<op::v1::ReduceSum>(f), 0);
    ASSERT_EQ(count_ops_of_type<ov::op::v0::Constant>(f), 1);

    auto new_const = get_result_constant(f);
    ASSERT_TRUE(new_const);
    check_names(new_const, {"constant", "constant_axes", "test"});
    ASSERT_EQ(new_const->get_shape(), output_shape);

    auto values_out = new_const->get_vector<int32_t>();

    vector<int32_t> values_expected{6, 15, 24};

    ASSERT_EQ(values_expected, values_out);
}

TEST(constant_folding, const_reducesum_keepdims) {
    Shape input_shape{3, 3};
    Shape output_shape{3, 1};

    vector<int32_t> values_in{1, 2, 3, 4, 5, 6, 7, 8, 9};
    auto constant = ov::op::v0::Constant::create(element::i32, input_shape, values_in);
    constant->set_friendly_name("constant");
    Shape axes_shape{1};
    vector<int32_t> values_axes{1};
    auto constant_axes = ov::op::v0::Constant::create(element::i64, axes_shape, values_axes);
    constant_axes->set_friendly_name("constant_axes");
    auto convert = make_shared<op::v1::ReduceSum>(constant, constant_axes, true);
    convert->set_friendly_name("test");
    auto f = make_shared<Model>(convert, ParameterVector{});

    run_constant_folding(f);

    ASSERT_EQ(count_ops_of_type<op::v1::ReduceSum>(f), 0);
    ASSERT_EQ(count_ops_of_type<ov::op::v0::Constant>(f), 1);

    auto new_const = get_result_constant(f);
    ASSERT_TRUE(new_const);
    check_names(new_const, {"constant", "constant_axes", "test"});
    ASSERT_EQ(new_const->get_shape(), output_shape);

    auto values_out = new_const->get_vector<int32_t>();

    vector<int32_t> values_expected{6, 15, 24};

    ASSERT_EQ(values_expected, values_out);
}

TEST(constant_folding, const_reducemax) {
    Shape input_shape{3, 2};
    Shape output_shape{3};

    vector<int32_t> values_in{1, 2, 3, 4, 5, 6};
    auto constant = ov::op::v0::Constant::create(element::i32, input_shape, values_in);
    constant->set_friendly_name("constant");
    Shape axes_shape{1};
    vector<int32_t> values_axes{1};
    auto constant_axes = ov::op::v0::Constant::create(element::i64, axes_shape, values_axes);
    constant_axes->set_friendly_name("constant_axes");
    auto convert = make_shared<op::v1::ReduceMax>(constant, constant_axes);
    convert->set_friendly_name("test");
    auto f = make_shared<Model>(convert, ParameterVector{});

    run_constant_folding(f);

    ASSERT_EQ(count_ops_of_type<op::v1::ReduceMax>(f), 0);
    ASSERT_EQ(count_ops_of_type<ov::op::v0::Constant>(f), 1);

    auto new_const = get_result_constant(f);
    ASSERT_TRUE(new_const);
    check_names(new_const, {"constant", "constant_axes", "test"});
    ASSERT_EQ(new_const->get_shape(), output_shape);

    auto values_out = new_const->get_vector<int32_t>();

    vector<int32_t> values_expected{2, 4, 6};

    ASSERT_EQ(values_expected, values_out);
}

TEST(constant_folding, const_reducemax_keepdims) {
    Shape input_shape{3, 2};
    Shape output_shape{3, 1};

    vector<int32_t> values_in{1, 2, 3, 4, 5, 6};
    auto constant = ov::op::v0::Constant::create(element::i32, input_shape, values_in);
    constant->set_friendly_name("constant");
    Shape axes_shape{1};
    vector<int32_t> values_axes{1};
    auto constant_axes = ov::op::v0::Constant::create(element::i64, axes_shape, values_axes);
    constant_axes->set_friendly_name("constant_axes");
    auto convert = make_shared<op::v1::ReduceMax>(constant, constant_axes, true);
    convert->set_friendly_name("test");
    auto f = make_shared<Model>(convert, ParameterVector{});

    run_constant_folding(f);

    ASSERT_EQ(count_ops_of_type<op::v1::ReduceMax>(f), 0);
    ASSERT_EQ(count_ops_of_type<ov::op::v0::Constant>(f), 1);

    auto new_const = get_result_constant(f);
    ASSERT_TRUE(new_const);
    check_names(new_const, {"constant", "constant_axes", "test"});
    ASSERT_EQ(new_const->get_shape(), output_shape);

    auto values_out = new_const->get_vector<int32_t>();

    vector<int32_t> values_expected{2, 4, 6};

    ASSERT_EQ(values_expected, values_out);
}

TEST(constant_folding, const_reducemin) {
    Shape input_shape{3, 2};
    Shape output_shape{3};

    vector<int32_t> values_in{1, 2, 3, 4, 5, 6};
    auto constant = ov::op::v0::Constant::create(element::i32, input_shape, values_in);
    constant->set_friendly_name("constant");
    Shape axes_shape{1};
    vector<int32_t> values_axes{1};
    auto constant_axes = ov::op::v0::Constant::create(element::i64, axes_shape, values_axes);
    constant_axes->set_friendly_name("constant_axes");
    auto convert = make_shared<op::v1::ReduceMin>(constant, constant_axes);
    convert->set_friendly_name("test");
    auto f = make_shared<Model>(convert, ParameterVector{});

    run_constant_folding(f);

    ASSERT_EQ(count_ops_of_type<op::v1::ReduceMin>(f), 0);
    ASSERT_EQ(count_ops_of_type<ov::op::v0::Constant>(f), 1);

    auto new_const = get_result_constant(f);
    ASSERT_TRUE(new_const);
    check_names(new_const, {"constant", "constant_axes", "test"});
    ASSERT_EQ(new_const->get_shape(), output_shape);

    auto values_out = new_const->get_vector<int32_t>();

    vector<int32_t> values_expected{1, 3, 5};

    ASSERT_EQ(values_expected, values_out);
}

TEST(constant_folding, const_reducemin_keepdims) {
    Shape input_shape{3, 2};
    Shape output_shape{3, 1};

    vector<int32_t> values_in{1, 2, 3, 4, 5, 6};
    auto constant = ov::op::v0::Constant::create(element::i32, input_shape, values_in);
    constant->set_friendly_name("constant");
    Shape axes_shape{1};
    vector<int32_t> values_axes{1};
    auto constant_axes = ov::op::v0::Constant::create(element::i64, axes_shape, values_axes);
    constant_axes->set_friendly_name("constant_axes");
    auto convert = make_shared<op::v1::ReduceMin>(constant, constant_axes, true);
    convert->set_friendly_name("test");
    auto f = make_shared<Model>(convert, ParameterVector{});

    run_constant_folding(f);

    ASSERT_EQ(count_ops_of_type<op::v1::ReduceMin>(f), 0);
    ASSERT_EQ(count_ops_of_type<ov::op::v0::Constant>(f), 1);

    auto new_const = get_result_constant(f);
    ASSERT_TRUE(new_const);
    check_names(new_const, {"constant", "constant_axes", "test"});
    ASSERT_EQ(new_const->get_shape(), output_shape);

    auto values_out = new_const->get_vector<int32_t>();

    vector<int32_t> values_expected{1, 3, 5};

    ASSERT_EQ(values_expected, values_out);
}

TEST(constant_folding, const_reducemean) {
    Shape input_shape{3, 3};
    Shape output_shape{3};

    vector<int32_t> values_in{1, 2, 3, 4, 5, 6, 7, 8, 9};
    auto constant = ov::op::v0::Constant::create(element::i32, input_shape, values_in);
    constant->set_friendly_name("constant");
    Shape axes_shape{1};
    vector<int32_t> values_axes{1};
    auto constant_axes = ov::op::v0::Constant::create(element::i64, axes_shape, values_axes);
    constant_axes->set_friendly_name("constant_axes");
    auto convert = make_shared<op::v1::ReduceMean>(constant, constant_axes);
    convert->set_friendly_name("test");
    auto f = make_shared<Model>(convert, ParameterVector{});

    run_constant_folding(f);

    ASSERT_EQ(count_ops_of_type<op::v1::ReduceMean>(f), 0);
    ASSERT_EQ(count_ops_of_type<ov::op::v0::Constant>(f), 1);

    auto new_const = get_result_constant(f);
    ASSERT_TRUE(new_const);
    check_names(new_const, {"constant", "constant_axes", "test"});
    ASSERT_EQ(new_const->get_shape(), output_shape);

    auto values_out = new_const->get_vector<int32_t>();

    vector<int32_t> values_expected{2, 5, 8};

    ASSERT_EQ(values_expected, values_out);
}

TEST(constant_folding, const_reducemean_keepdims) {
    Shape input_shape{3, 3};
    Shape output_shape{3, 1};

    vector<int32_t> values_in{1, 2, 3, 4, 5, 6, 7, 8, 9};
    auto constant = ov::op::v0::Constant::create(element::i32, input_shape, values_in);
    constant->set_friendly_name("constant");
    Shape axes_shape{1};
    vector<int32_t> values_axes{1};
    auto constant_axes = ov::op::v0::Constant::create(element::i64, axes_shape, values_axes);
    constant_axes->set_friendly_name("constant_axes");
    auto convert = make_shared<op::v1::ReduceMean>(constant, constant_axes, true);
    convert->set_friendly_name("test");
    auto f = make_shared<Model>(convert, ParameterVector{});

    run_constant_folding(f);

    ASSERT_EQ(count_ops_of_type<op::v1::ReduceMean>(f), 0);
    ASSERT_EQ(count_ops_of_type<ov::op::v0::Constant>(f), 1);

    auto new_const = get_result_constant(f);
    ASSERT_TRUE(new_const);
    check_names(new_const, {"constant", "constant_axes", "test"});
    ASSERT_EQ(new_const->get_shape(), output_shape);

    auto values_out = new_const->get_vector<int32_t>();

    vector<int32_t> values_expected{2, 5, 8};

    ASSERT_EQ(values_expected, values_out);
}

TEST(constant_folding, const_reduce_logical_and__no_keepdims) {
    const Shape input_shape{3, 3};

    const vector<char> values_in{0, 1, 1, 0, 1, 0, 1, 1, 1};
    const auto data = ov::op::v0::Constant::create(element::boolean, input_shape, values_in);
    data->set_friendly_name("data");
    const auto axes = ov::op::v0::Constant::create(element::i64, {1}, {1});
    axes->set_friendly_name("axes");
    const auto convert = make_shared<op::v1::ReduceLogicalAnd>(data, axes, false);
    convert->set_friendly_name("test");
    auto f = make_shared<Model>(convert, ParameterVector{});

    run_constant_folding(f);

    ASSERT_EQ(count_ops_of_type<op::v1::ReduceLogicalAnd>(f), 0);
    ASSERT_EQ(count_ops_of_type<ov::op::v0::Constant>(f), 1);

    const auto new_const = get_result_constant(f);
    ASSERT_TRUE(new_const);
    check_names(new_const, {"data", "axes", "test"});

    const Shape expected_out_shape{3};
    ASSERT_EQ(new_const->get_shape(), expected_out_shape);

    const auto values_out = new_const->get_vector<char>();

    const vector<char> values_expected{0, 0, 1};

    ASSERT_EQ(values_expected, values_out);
}

TEST(constant_folding, const_reduce_logical_and__keepdims) {
    const Shape input_shape{3, 3};

    const vector<char> values_in{0, 1, 1, 0, 1, 0, 1, 1, 1};
    const auto data = ov::op::v0::Constant::create(element::boolean, input_shape, values_in);
    data->set_friendly_name("data");
    const auto axes = ov::op::v0::Constant::create(element::i64, {1}, {1});
    axes->set_friendly_name("axes");
    const auto convert = make_shared<op::v1::ReduceLogicalAnd>(data, axes, true);
    convert->set_friendly_name("test");
    auto f = make_shared<Model>(convert, ParameterVector{});

    run_constant_folding(f);

    ASSERT_EQ(count_ops_of_type<op::v1::ReduceLogicalAnd>(f), 0);
    ASSERT_EQ(count_ops_of_type<ov::op::v0::Constant>(f), 1);

    const auto new_const = get_result_constant(f);
    ASSERT_TRUE(new_const);
    check_names(new_const, {"data", "axes", "test"});

    // the output shape is expected to have 'ones' at the positions specified in the reduction axes
    // in case the keep_dims attribute of ReduceLogicalAnd is set to true
    const Shape expected_out_shape{3, 1};
    ASSERT_EQ(new_const->get_shape(), expected_out_shape);

    const auto values_out = new_const->get_vector<char>();

    const vector<char> values_expected{0, 0, 1};

    ASSERT_EQ(values_expected, values_out);
}

TEST(constant_folding, const_reduce_logical_and__keepdims_3d) {
    const Shape input_shape{2, 2, 2};

    const vector<char> values_in{1, 1, 0, 0, 1, 0, 0, 1};
    const auto data = ov::op::v0::Constant::create(element::boolean, input_shape, values_in);
    data->set_friendly_name("data");
    const auto axes = ov::op::v0::Constant::create(element::i64, {2}, {0, 2});
    axes->set_friendly_name("axes");
    const auto convert = make_shared<op::v1::ReduceLogicalAnd>(data, axes, true);
    convert->set_friendly_name("test");
    auto f = make_shared<Model>(convert, ParameterVector{});

    run_constant_folding(f);

    ASSERT_EQ(count_ops_of_type<op::v1::ReduceLogicalAnd>(f), 0);
    ASSERT_EQ(count_ops_of_type<ov::op::v0::Constant>(f), 1);

    const auto new_const = get_result_constant(f);
    ASSERT_TRUE(new_const);
    check_names(new_const, {"data", "axes", "test"});

    const Shape expected_out_shape{1, 2, 1};
    ASSERT_EQ(new_const->get_shape(), expected_out_shape);

    const auto values_out = new_const->get_vector<char>();

    const vector<char> values_expected{0, 0};

    ASSERT_EQ(values_expected, values_out);
}

TEST(constant_folding, const_reduce_logical_or__no_keepdims) {
    const Shape input_shape{3, 3};

    const vector<char> values_in{1, 0, 0, 1, 0, 1, 0, 0, 0};
    const auto data = ov::op::v0::Constant::create(element::boolean, input_shape, values_in);
    data->set_friendly_name("data");
    const auto axes = ov::op::v0::Constant::create(element::i64, {1}, {1});
    axes->set_friendly_name("axes");
    const auto convert = make_shared<op::v1::ReduceLogicalOr>(data, axes, false);
    convert->set_friendly_name("test");
    auto f = make_shared<Model>(convert, ParameterVector{});

    run_constant_folding(f);

    ASSERT_EQ(count_ops_of_type<op::v1::ReduceLogicalAnd>(f), 0);
    ASSERT_EQ(count_ops_of_type<ov::op::v0::Constant>(f), 1);

    const auto new_const = get_result_constant(f);
    ASSERT_TRUE(new_const);
    check_names(new_const, {"data", "axes", "test"});

    const Shape expected_out_shape{3};
    ASSERT_EQ(new_const->get_shape(), expected_out_shape);

    const auto values_out = new_const->get_vector<char>();

    const vector<char> values_expected{1, 1, 0};

    ASSERT_EQ(values_expected, values_out);
}

TEST(constant_folding, const_concat) {
    auto constant0 = ov::op::v0::Constant::create(element::i32, Shape{2, 3}, vector<int32_t>{1, 2, 3, 4, 5, 6});
    constant0->set_friendly_name("constant0");
    auto constant1 = ov::op::v0::Constant::create(element::i32, Shape{2, 1}, vector<int32_t>{7, 8});
    constant1->set_friendly_name("constant1");
    auto concat = make_shared<op::v0::Concat>(NodeVector{constant0, constant1}, 1);
    concat->set_friendly_name("test");
    auto f = make_shared<Model>(concat, ParameterVector{});

    run_constant_folding(f);

    ASSERT_EQ(count_ops_of_type<op::v0::Concat>(f), 0);
    ASSERT_EQ(count_ops_of_type<ov::op::v0::Constant>(f), 1);

    auto new_const = get_result_constant(f);
    ASSERT_TRUE(new_const);
    check_names(new_const, {"constant0", "constant1", "test"});
    auto values_out = new_const->get_vector<int32_t>();

    vector<int32_t> values_expected{1, 2, 3, 7, 4, 5, 6, 8};

    ASSERT_EQ(values_expected, values_out);
}

TEST(constant_folding, const_concat_3d_single_elem) {
    auto constant_1 = ov::op::v0::Constant::create(element::i32, Shape{1, 1, 1}, vector<int32_t>{1});
    constant_1->set_friendly_name("constant_1");
    auto constant_2 = ov::op::v0::Constant::create(element::i32, Shape{1, 1, 1}, vector<int32_t>{2});
    constant_2->set_friendly_name("constant_2");
    auto concat = make_shared<op::v0::Concat>(NodeVector{constant_1, constant_2}, 0);
    concat->set_friendly_name("test");
    auto f = make_shared<Model>(concat, ParameterVector{});

    run_constant_folding(f);

    ASSERT_EQ(count_ops_of_type<op::v0::Concat>(f), 0);
    ASSERT_EQ(count_ops_of_type<ov::op::v0::Constant>(f), 1);

    auto new_const = get_result_constant(f);

    ASSERT_TRUE(new_const);
    check_names(new_const, {"constant_1", "constant_2", "test"});
    ASSERT_EQ(new_const->get_output_shape(0), (Shape{2, 1, 1}));

    auto values_out = new_const->get_vector<int32_t>();
    vector<int32_t> values_expected{1, 2};
    ASSERT_EQ(values_expected, values_out);
}

TEST(constant_folding, const_concat_axis_2) {
    auto constant_1 = ov::op::v0::Constant::create(element::i32, Shape{3, 1, 2}, vector<int32_t>{1, 2, 3, 4, 5, 6});
    constant_1->set_friendly_name("constant_1");
    auto constant_2 = ov::op::v0::Constant::create(element::i32,
                                                   Shape{3, 1, 4},
                                                   vector<int32_t>{7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18});
    constant_2->set_friendly_name("constant_2");
    auto concat = make_shared<op::v0::Concat>(NodeVector{constant_1, constant_2}, 2);
    concat->set_friendly_name("test");
    auto f = make_shared<Model>(concat, ParameterVector{});

    run_constant_folding(f);

    ASSERT_EQ(count_ops_of_type<op::v0::Concat>(f), 0);
    ASSERT_EQ(count_ops_of_type<ov::op::v0::Constant>(f), 1);

    auto new_const = get_result_constant(f);

    ASSERT_TRUE(new_const);
    check_names(new_const, {"constant_1", "constant_2", "test"});
    ASSERT_EQ(new_const->get_output_shape(0), (Shape{3, 1, 6}));

    auto values_out = new_const->get_vector<int32_t>();
    vector<int32_t> values_expected{1, 2, 7, 8, 9, 10, 3, 4, 11, 12, 13, 14, 5, 6, 15, 16, 17, 18};
    ASSERT_EQ(values_expected, values_out);
}

TEST(constant_folding, const_concat_axis_1_bool_type) {
    auto constant_1 = ov::op::v0::Constant::create(element::boolean, Shape{1, 1, 2}, vector<int32_t>{true, true});
    constant_1->set_friendly_name("constant_1");
    auto constant_2 =
        ov::op::v0::Constant::create(element::boolean, Shape{1, 2, 2}, vector<char>{true, false, true, false});
    constant_2->set_friendly_name("constant_2");
    auto constant_3 = ov::op::v0::Constant::create(element::boolean,
                                                   Shape{1, 3, 2},
                                                   vector<char>{true, false, true, false, true, false});
    constant_3->set_friendly_name("constant_3");
    auto concat = make_shared<op::v0::Concat>(NodeVector{constant_1, constant_2, constant_3}, 1);
    concat->set_friendly_name("test");
    auto f = make_shared<Model>(concat, ParameterVector{});

    run_constant_folding(f);

    ASSERT_EQ(count_ops_of_type<op::v0::Concat>(f), 0);
    ASSERT_EQ(count_ops_of_type<ov::op::v0::Constant>(f), 1);

    auto new_const = get_result_constant(f);

    ASSERT_TRUE(new_const);
    check_names(new_const, {"constant_1", "constant_2", "constant_3", "test"});
    ASSERT_EQ(new_const->get_output_shape(0), (Shape{1, 6, 2}));

    auto values_out = new_const->get_vector<char>();
    vector<char> values_expected{true, true, true, false, true, false, true, false, true, false, true, false};
    ASSERT_EQ(values_expected, values_out);
}

TEST(constant_folding, const_logical_not) {
    auto constant = ov::op::v0::Constant::create(element::boolean, Shape{2, 3}, vector<char>{0, 1, 0, 0, 1, 1});
    constant->set_friendly_name("constant");
    auto logical_not = make_shared<op::v1::LogicalNot>(constant);
    logical_not->set_friendly_name("test");
    auto f = make_shared<Model>(logical_not, ParameterVector{});

    run_constant_folding(f);

    ASSERT_EQ(count_ops_of_type<op::v1::LogicalNot>(f), 0);
    ASSERT_EQ(count_ops_of_type<ov::op::v0::Constant>(f), 1);

    auto new_const = get_result_constant(f);
    ASSERT_TRUE(new_const);
    check_names(new_const, {"constant", "test"});
    auto values_out = new_const->get_vector<char>();

    vector<char> values_expected{1, 0, 1, 1, 0, 0};

    ASSERT_EQ(values_expected, values_out);
}

TEST(constant_folding, const_equal) {
    auto constant0 = ov::op::v0::Constant::create(element::i32, Shape{2, 3}, vector<int32_t>{1, 2, 3, 4, 5, 6});
    constant0->set_friendly_name("constant0");
    auto constant1 = ov::op::v0::Constant::create(element::i32, Shape{2, 3}, vector<int32_t>{1, 2, 2, 3, 5, 6});
    constant1->set_friendly_name("constant1");
    auto eq = make_shared<op::v1::Equal>(constant0, constant1);
    eq->set_friendly_name("test");
    auto f = make_shared<Model>(eq, ParameterVector{});

    run_constant_folding(f);

    ASSERT_EQ(count_ops_of_type<op::v1::Equal>(f), 0);
    ASSERT_EQ(count_ops_of_type<ov::op::v0::Constant>(f), 1);

    auto new_const = get_result_constant(f);
    ASSERT_TRUE(new_const);
    check_names(new_const, {"constant0", "constant1", "test"});
    auto values_out = new_const->get_vector<char>();

    vector<char> values_expected{1, 1, 0, 0, 1, 1};

    ASSERT_EQ(values_expected, values_out);
}

TEST(constant_folding, const_not_equal) {
    auto constant0 = ov::op::v0::Constant::create(element::i32, Shape{2, 3}, vector<int32_t>{1, 2, 3, 4, 5, 6});
    constant0->set_friendly_name("constant0");
    auto constant1 = ov::op::v0::Constant::create(element::i32, Shape{2, 3}, vector<int32_t>{1, 2, 2, 3, 5, 6});
    constant1->set_friendly_name("constant1");
    auto eq = make_shared<op::v1::NotEqual>(constant0, constant1);
    eq->set_friendly_name("test");
    auto f = make_shared<Model>(eq, ParameterVector{});

    run_constant_folding(f);

    ASSERT_EQ(count_ops_of_type<op::v1::NotEqual>(f), 0);
    ASSERT_EQ(count_ops_of_type<ov::op::v0::Constant>(f), 1);

    auto new_const = get_result_constant(f);
    ASSERT_TRUE(new_const);
    check_names(new_const, {"constant0", "constant1", "test"});
    auto values_out = new_const->get_vector<char>();

    vector<char> values_expected{0, 0, 1, 1, 0, 0};

    ASSERT_EQ(values_expected, values_out);
}

TEST(constant_folding, const_greater) {
    auto constant0 = ov::op::v0::Constant::create(element::i32, Shape{2, 3}, vector<int32_t>{1, 2, 3, 4, 5, 6});
    constant0->set_friendly_name("constant0");
    auto constant1 = ov::op::v0::Constant::create(element::i32, Shape{2, 3}, vector<int32_t>{2, 2, 2, 5, 5, 5});
    constant1->set_friendly_name("constant1");
    auto eq = make_shared<op::v1::Greater>(constant0, constant1);
    eq->set_friendly_name("test");
    auto f = make_shared<Model>(eq, ParameterVector{});

    run_constant_folding(f);

    ASSERT_EQ(count_ops_of_type<op::v1::Greater>(f), 0);
    ASSERT_EQ(count_ops_of_type<ov::op::v0::Constant>(f), 1);

    auto new_const = get_result_constant(f);
    ASSERT_TRUE(new_const);
    check_names(new_const, {"constant0", "constant1", "test"});
    auto values_out = new_const->get_vector<char>();

    vector<char> values_expected{0, 0, 1, 0, 0, 1};

    ASSERT_EQ(values_expected, values_out);
}

TEST(constant_folding, const_greater_eq) {
    auto constant0 = ov::op::v0::Constant::create(element::i32, Shape{2, 3}, vector<int32_t>{1, 2, 3, 4, 5, 6});
    constant0->set_friendly_name("constant0");
    auto constant1 = ov::op::v0::Constant::create(element::i32, Shape{2, 3}, vector<int32_t>{2, 2, 2, 5, 5, 5});
    constant1->set_friendly_name("constant1");
    auto eq = make_shared<op::v1::GreaterEqual>(constant0, constant1);
    eq->set_friendly_name("test");
    auto f = make_shared<Model>(eq, ParameterVector{});

    run_constant_folding(f);

    ASSERT_EQ(count_ops_of_type<op::v1::GreaterEqual>(f), 0);
    ASSERT_EQ(count_ops_of_type<ov::op::v0::Constant>(f), 1);

    auto new_const = get_result_constant(f);
    ASSERT_TRUE(new_const);
    check_names(new_const, {"constant0", "constant1", "test"});
    auto values_out = new_const->get_vector<char>();

    vector<char> values_expected{0, 1, 1, 0, 1, 1};

    ASSERT_EQ(values_expected, values_out);
}

TEST(constant_folding, const_less) {
    auto constant0 = ov::op::v0::Constant::create(element::i32, Shape{2, 3}, vector<int32_t>{1, 2, 3, 4, 5, 6});
    constant0->set_friendly_name("constant0");
    auto constant1 = ov::op::v0::Constant::create(element::i32, Shape{2, 3}, vector<int32_t>{2, 2, 2, 5, 5, 5});
    constant1->set_friendly_name("constant1");
    auto eq = make_shared<op::v1::Less>(constant0, constant1);
    eq->set_friendly_name("test");
    auto f = make_shared<Model>(eq, ParameterVector{});

    run_constant_folding(f);

    ASSERT_EQ(count_ops_of_type<op::v1::Less>(f), 0);
    ASSERT_EQ(count_ops_of_type<ov::op::v0::Constant>(f), 1);

    auto new_const = get_result_constant(f);
    ASSERT_TRUE(new_const);
    check_names(new_const, {"constant0", "constant1", "test"});
    auto values_out = new_const->get_vector<char>();

    vector<char> values_expected{1, 0, 0, 1, 0, 0};

    ASSERT_EQ(values_expected, values_out);
}

TEST(constant_folding, const_less_eq) {
    auto constant0 = ov::op::v0::Constant::create(element::i32, Shape{2, 3}, vector<int32_t>{1, 2, 3, 4, 5, 6});
    constant0->set_friendly_name("constant0");
    auto constant1 = ov::op::v0::Constant::create(element::i32, Shape{2, 3}, vector<int32_t>{2, 2, 2, 5, 5, 5});
    constant1->set_friendly_name("constant1");
    auto eq = make_shared<op::v1::LessEqual>(constant0, constant1);
    eq->set_friendly_name("test");
    auto f = make_shared<Model>(eq, ParameterVector{});

    run_constant_folding(f);

    ASSERT_EQ(count_ops_of_type<op::v1::LessEqual>(f), 0);
    ASSERT_EQ(count_ops_of_type<ov::op::v0::Constant>(f), 1);

    auto new_const = get_result_constant(f);
    ASSERT_TRUE(new_const);
    check_names(new_const, {"constant0", "constant1", "test"});
    auto values_out = new_const->get_vector<char>();

    vector<char> values_expected{1, 1, 0, 1, 1, 0};

    ASSERT_EQ(values_expected, values_out);
}

TEST(constant_folding, const_or) {
    auto constant0 = ov::op::v0::Constant::create(element::boolean, Shape{2, 3}, vector<int32_t>{0, 0, 1, 0, 1, 1});
    constant0->set_friendly_name("constant0");
    auto constant1 = ov::op::v0::Constant::create(element::boolean, Shape{2, 3}, vector<int32_t>{0, 1, 1, 1, 0, 1});
    constant1->set_friendly_name("constant1");
    auto eq = make_shared<op::v1::LogicalOr>(constant0, constant1);
    eq->set_friendly_name("test");
    auto f = make_shared<Model>(eq, ParameterVector{});

    run_constant_folding(f);

    ASSERT_EQ(count_ops_of_type<op::v1::LogicalOr>(f), 0);
    ASSERT_EQ(count_ops_of_type<ov::op::v0::Constant>(f), 1);

    auto new_const = get_result_constant(f);
    ASSERT_TRUE(new_const);
    check_names(new_const, {"constant0", "constant1", "test"});
    auto values_out = new_const->get_vector<char>();

    vector<char> values_expected{0, 1, 1, 1, 1, 1};

    ASSERT_EQ(values_expected, values_out);
}

TEST(constant_folding, const_xor) {
    auto constant0 = ov::op::v0::Constant::create(element::boolean, Shape{2, 3}, vector<int32_t>{0, 0, 1, 0, 1, 1});
    constant0->set_friendly_name("constant0");
    auto constant1 = ov::op::v0::Constant::create(element::boolean, Shape{2, 3}, vector<int32_t>{0, 1, 1, 1, 0, 1});
    constant1->set_friendly_name("constant1");
    auto eq = make_shared<op::v0::Xor>(constant0, constant1);
    eq->set_friendly_name("test");
    auto f = make_shared<Model>(eq, ParameterVector{});

    run_constant_folding(f);

    ASSERT_EQ(count_ops_of_type<op::v0::Xor>(f), 0);
    ASSERT_EQ(count_ops_of_type<ov::op::v0::Constant>(f), 1);

    auto new_const = get_result_constant(f);
    ASSERT_TRUE(new_const);
    check_names(new_const, {"constant0", "constant1", "test"});
    auto values_out = new_const->get_vector<char>();

    vector<char> values_expected{0, 1, 0, 1, 1, 0};

    ASSERT_EQ(values_expected, values_out);
}

TEST(constant_folding, const_ceiling) {
    auto constant =
        ov::op::v0::Constant::create(element::f32, Shape{2, 3}, vector<float>{0.0f, 0.1f, -0.1f, -2.5f, 2.5f, 3.0f});
    constant->set_friendly_name("constant");
    auto ceil = make_shared<op::v0::Ceiling>(constant);
    ceil->set_friendly_name("test");
    auto f = make_shared<Model>(ceil, ParameterVector{});

    run_constant_folding(f);

    ASSERT_EQ(count_ops_of_type<op::v0::Ceiling>(f), 0);
    ASSERT_EQ(count_ops_of_type<ov::op::v0::Constant>(f), 1);

    auto new_const = get_result_constant(f);
    ASSERT_TRUE(new_const);
    check_names(new_const, {"constant", "test"});
    auto values_out = new_const->get_vector<float>();

    vector<float> values_expected{0.0f, 1.0f, 0.0f, -2.0f, 3.0f, 3.0f};

    ASSERT_TRUE(ov::test::utils::all_close_f(values_out, values_expected, MIN_FLOAT_TOLERANCE_BITS));
}

TEST(constant_folding, const_floor) {
    auto constant =
        ov::op::v0::Constant::create(element::f32, Shape{2, 3}, vector<float>{0.0f, 0.1f, -0.1f, -2.5f, 2.5f, 3.0f});
    constant->set_friendly_name("constant");
    auto floor = make_shared<op::v0::Floor>(constant);
    floor->set_friendly_name("test");
    auto f = make_shared<Model>(floor, ParameterVector{});

    run_constant_folding(f);

    ASSERT_EQ(count_ops_of_type<op::v0::Floor>(f), 0);
    ASSERT_EQ(count_ops_of_type<ov::op::v0::Constant>(f), 1);

    auto new_const = get_result_constant(f);
    ASSERT_TRUE(new_const);
    check_names(new_const, {"constant", "test"});
    auto values_out = new_const->get_vector<float>();

    vector<float> values_expected{0.0f, 0.0f, -1.0f, -3.0f, 2.0f, 3.0f};

    ASSERT_TRUE(ov::test::utils::all_close_f(values_out, values_expected, MIN_FLOAT_TOLERANCE_BITS));
}

TEST(constant_folding, const_gather_v1) {
    auto constant_data =
        ov::op::v0::Constant::create(element::f32,
                                     Shape{2, 5},
                                     vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f});
    constant_data->set_friendly_name("constant_data");
    auto constant_indices = ov::op::v0::Constant::create(element::i64, Shape{4}, vector<int64_t>{0, 3, 2, 2});
    constant_indices->set_friendly_name("constant_indices");
    auto constant_axis = ov::op::v0::Constant::create(element::i64, Shape{1}, vector<int64_t>{1});
    constant_axis->set_friendly_name("constant_axis");
    auto gather = make_shared<op::v1::Gather>(constant_data, constant_indices, constant_axis);
    gather->set_friendly_name("test");
    auto f = make_shared<Model>(gather, ParameterVector{});

    run_constant_folding(f);

    ASSERT_EQ(count_ops_of_type<op::v1::Gather>(f), 0);
    ASSERT_EQ(count_ops_of_type<ov::op::v0::Constant>(f), 1);

    auto new_const = get_result_constant(f);
    ASSERT_TRUE(new_const);
    check_names(new_const, {"constant_data", "constant_indices", "constant_axis", "test"});
    auto values_out = new_const->get_vector<float>();

    vector<float> values_expected{1.0f, 4.0f, 3.0f, 3.0f, 6.0f, 9.0f, 8.0f, 8.0f};

    ASSERT_TRUE(ov::test::utils::all_close_f(values_out, values_expected, MIN_FLOAT_TOLERANCE_BITS));
}

TEST(constant_folding, const_gather_v1_scalar) {
    auto constant_data =
        ov::op::v0::Constant::create(element::f32,
                                     Shape{2, 5},
                                     vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f});
    constant_data->set_friendly_name("constant_data");
    auto constant_indices = ov::op::v0::Constant::create(element::i64, Shape{4}, vector<int64_t>{0, 3, 2, 2});
    constant_indices->set_friendly_name("constant_indices");
    auto constant_axis = ov::op::v0::Constant::create(element::i64, Shape{}, vector<int64_t>{1});
    constant_axis->set_friendly_name("constant_axis");
    auto gather = make_shared<op::v1::Gather>(constant_data, constant_indices, constant_axis);
    gather->set_friendly_name("test");
    auto f = make_shared<Model>(gather, ParameterVector{});

    run_constant_folding(f);

    ASSERT_EQ(count_ops_of_type<op::v1::Gather>(f), 0);
    ASSERT_EQ(count_ops_of_type<ov::op::v0::Constant>(f), 1);

    auto new_const = get_result_constant(f);
    ASSERT_TRUE(new_const);
    check_names(new_const, {"constant_data", "constant_indices", "constant_axis", "test"});
    auto values_out = new_const->get_vector<float>();

    vector<float> values_expected{1.0f, 4.0f, 3.0f, 3.0f, 6.0f, 9.0f, 8.0f, 8.0f};

    ASSERT_TRUE(ov::test::utils::all_close_f(values_out, values_expected, MIN_FLOAT_TOLERANCE_BITS));
}

TEST(constant_folding, const_gather_v1_subgraph) {
    const auto A = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1});
    const float b_value = 3.21f;
    const auto B_const = ov::op::v0::Constant::create(element::f32, {1}, {b_value});
    const auto C = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1});
    const int64_t axis = 0;
    const auto axis_const = ov::op::v0::Constant::create(element::i64, {}, {axis});
    axis_const->set_friendly_name("axis_const");

    const auto concat = make_shared<op::v0::Concat>(NodeVector{A, B_const, C}, axis);
    concat->set_friendly_name("concat");

    const vector<int64_t> indices{1};
    const auto indices_const = ov::op::v0::Constant::create(element::i64, {indices.size()}, indices);
    indices_const->set_friendly_name("indices_const");
    const auto gather = make_shared<op::v1::Gather>(concat, indices_const, axis_const);
    gather->set_friendly_name("test");
    auto f = make_shared<Model>(gather, ParameterVector{A, C});

    run_constant_folding(f);

    ASSERT_EQ(count_ops_of_type<op::v0::Concat>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::v1::Gather>(f), 0);
    ASSERT_EQ(count_ops_of_type<ov::op::v0::Constant>(f), 1);

    const auto new_const = get_result_constant(f);
    ASSERT_TRUE(new_const);
    check_names(new_const, {"axis_const", "concat", "indices_const", "test"});

    const auto values_out = new_const->get_vector<float>();
    ASSERT_TRUE(ov::test::utils::all_close_f(values_out, {b_value}, MIN_FLOAT_TOLERANCE_BITS));
}

TEST(constant_folding, const_gather_v1_subgraph_neg_axis) {
    const auto A = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1});
    const float b_value = 1.23f;
    const auto B = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1});
    const auto C_const = ov::op::v0::Constant::create(element::f32, {1}, {b_value});
    const int64_t axis = 0;
    const auto axis_const = ov::op::v0::Constant::create(element::i64, {}, {axis});
    axis_const->set_friendly_name("axis_const");

    const auto concat = make_shared<ov::op::v0::Concat>(NodeVector{A, B, C_const}, axis);
    concat->set_friendly_name("concat");

    const vector<int64_t> indices{-1};
    const auto indices_const = ov::op::v0::Constant::create(element::i64, {indices.size()}, indices);
    indices_const->set_friendly_name("indices_const");
    const auto gather = make_shared<op::v1::Gather>(concat, indices_const, axis_const);
    gather->set_friendly_name("test");
    auto f = make_shared<Model>(gather, ParameterVector{A, B});

    run_constant_folding(f);

    ASSERT_EQ(count_ops_of_type<ov::op::v0::Concat>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::v1::Gather>(f), 0);
    ASSERT_EQ(count_ops_of_type<ov::op::v0::Constant>(f), 1);

    const auto new_const = get_result_constant(f);
    ASSERT_TRUE(new_const);
    check_names(new_const, {"axis_const", "concat", "indices_const", "test"});

    const auto values_out = new_const->get_vector<float>();
    ASSERT_TRUE(ov::test::utils::all_close_f(values_out, {b_value}, MIN_FLOAT_TOLERANCE_BITS));
}

TEST(constant_folding, const_gather_v1_subgraph_no_constant_input) {
    const auto A = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1});
    const auto B = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1});
    const auto C = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1});
    const int64_t axis = 0;
    const auto axis_const = ov::op::v0::Constant::create(element::i64, {}, {axis});

    const auto concat = make_shared<ov::op::v0::Concat>(NodeVector{A, B, C}, axis);

    const vector<int64_t> indices{1};
    const auto indices_const = ov::op::v0::Constant::create(element::i64, {indices.size()}, indices);
    const auto gather = make_shared<op::v1::Gather>(concat, indices_const, axis_const);
    gather->set_friendly_name("test");
    auto f = make_shared<Model>(gather, ParameterVector{A, B, C});

    run_constant_folding(f);

    ASSERT_EQ(count_ops_of_type<ov::op::v0::Concat>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::v1::Gather>(f), 0);
}

TEST(constant_folding, const_gather_v1_subgraph_no_constant_input_scalar) {
    const auto A = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1});
    const auto B = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1});
    const auto C = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1});
    const int64_t axis = 0;
    const auto axis_const = ov::op::v0::Constant::create(element::i64, {}, {axis});

    const auto concat = make_shared<ov::op::v0::Concat>(NodeVector{A, B, C}, axis);

    const vector<int64_t> indices{1};
    const auto indices_const = ov::op::v0::Constant::create(element::i64, {}, indices);
    const auto gather = make_shared<op::v1::Gather>(concat, indices_const, axis_const);
    auto f = make_shared<Model>(gather, ParameterVector{A, B, C});

    run_constant_folding(f);

    ASSERT_EQ(count_ops_of_type<ov::op::v0::Concat>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::v1::Gather>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::v0::Squeeze>(f), 1);
}

TEST(constant_folding, const_gather_v1_subgraph_skip_if_non_zero_axis) {
    const auto A = make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 2});
    const auto B = make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 2});
    const auto C = make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 2});
    const int64_t axis = 1;
    const auto axis_const = ov::op::v0::Constant::create(element::i64, {}, {axis});

    const auto concat = make_shared<ov::op::v0::Concat>(NodeVector{A, B, C}, axis);

    const vector<int64_t> indices{1};
    const auto indices_const = ov::op::v0::Constant::create(element::i64, {indices.size()}, indices);
    const auto gather = make_shared<op::v1::Gather>(concat, indices_const, axis_const);
    auto f = make_shared<Model>(gather, ParameterVector{A, B, C});

    run_constant_folding(f);

    ASSERT_EQ(count_ops_of_type<ov::op::v0::Concat>(f), 1);
    ASSERT_EQ(count_ops_of_type<op::v1::Gather>(f), 1);
}

TEST(constant_folding, const_gather_v1_subgraph_skip_if_non_single_indices) {
    const auto A = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1});
    const auto B = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1});
    const auto C = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1});
    const int64_t axis = 0;
    const auto axis_const = ov::op::v0::Constant::create(element::i64, {}, {axis});

    const auto concat = make_shared<ov::op::v0::Concat>(NodeVector{A, B, C}, axis);

    const vector<int64_t> indices{0, 1};
    const auto indices_const = ov::op::v0::Constant::create(element::i64, {indices.size()}, indices);
    const auto gather = make_shared<op::v1::Gather>(concat, indices_const, axis_const);
    auto f = make_shared<Model>(gather, ParameterVector{A, B, C});

    run_constant_folding(f);

    ASSERT_EQ(count_ops_of_type<ov::op::v0::Concat>(f), 1);
    ASSERT_EQ(count_ops_of_type<op::v1::Gather>(f), 1);
}

TEST(constant_folding, const_gather_v1_subgraph_skip_if_concat_output_shape_dynamic) {
    const auto A = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic());
    const auto B = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1});
    const auto C = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1});
    const int64_t axis = 0;
    const auto axis_const = ov::op::v0::Constant::create(element::i64, {}, {axis});

    const auto concat = make_shared<ov::op::v0::Concat>(NodeVector{A, B, C}, axis);

    const vector<int64_t> indices{1};
    const auto indices_const = ov::op::v0::Constant::create(element::i64, {indices.size()}, indices);
    const auto gather = make_shared<op::v1::Gather>(concat, indices_const, axis_const);
    auto f = make_shared<Model>(gather, ParameterVector{A, B, C});

    run_constant_folding(f);

    ASSERT_EQ(count_ops_of_type<ov::op::v0::Concat>(f), 1);
    ASSERT_EQ(count_ops_of_type<op::v1::Gather>(f), 1);
}

TEST(constant_folding, const_gather_v1_subgraph_skip_if_not_single_input) {
    const auto A = make_shared<ov::op::v0::Parameter>(element::f32, Shape{2});
    const auto B = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1});
    const auto C = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1});
    const int64_t axis = 0;
    const auto axis_const = ov::op::v0::Constant::create(element::i64, {}, {axis});

    const auto concat = make_shared<ov::op::v0::Concat>(NodeVector{A, B, C}, axis);

    const vector<int64_t> indices{1};
    const auto indices_const = ov::op::v0::Constant::create(element::i64, {indices.size()}, indices);
    const auto gather = make_shared<op::v1::Gather>(concat, indices_const, axis_const);
    auto f = make_shared<Model>(gather, ParameterVector{A, B, C});

    run_constant_folding(f);

    ASSERT_EQ(count_ops_of_type<ov::op::v0::Concat>(f), 1);
    ASSERT_EQ(count_ops_of_type<op::v1::Gather>(f), 1);
}

TEST(constant_folding, const_gather_v7) {
    auto constant_data =
        ov::op::v0::Constant::create(element::f32,
                                     Shape{2, 5},
                                     vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f});
    constant_data->set_friendly_name("constant_data");
    auto constant_indices = ov::op::v0::Constant::create(element::i64, Shape{4}, vector<int64_t>{0, 3, 2, 2});
    constant_indices->set_friendly_name("constant_indices");
    auto constant_axis = ov::op::v0::Constant::create(element::i64, Shape{1}, vector<int64_t>{1});
    constant_axis->set_friendly_name("constant_axis");
    auto gather = make_shared<op::v7::Gather>(constant_data, constant_indices, constant_axis);
    gather->set_friendly_name("test");
    auto f = make_shared<Model>(gather, ParameterVector{});

    run_constant_folding(f);

    ASSERT_EQ(count_ops_of_type<op::v7::Gather>(f), 0);
    ASSERT_EQ(count_ops_of_type<ov::op::v0::Constant>(f), 1);

    auto new_const = get_result_constant(f);
    ASSERT_TRUE(new_const);
    check_names(new_const, {"constant_data", "constant_indices", "constant_axis", "test"});
    auto values_out = new_const->get_vector<float>();

    vector<float> values_expected{1.0f, 4.0f, 3.0f, 3.0f, 6.0f, 9.0f, 8.0f, 8.0f};

    ASSERT_TRUE(ov::test::utils::all_close_f(values_out, values_expected, MIN_FLOAT_TOLERANCE_BITS));
}

TEST(constant_folding, const_gather_v7_scalar) {
    auto constant_data =
        ov::op::v0::Constant::create(element::f32,
                                     Shape{2, 5},
                                     vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f});
    constant_data->set_friendly_name("constant_data");
    auto constant_indices = ov::op::v0::Constant::create(element::i64, Shape{4}, vector<int64_t>{0, 3, 2, 2});
    constant_indices->set_friendly_name("constant_indices");
    auto constant_axis = ov::op::v0::Constant::create(element::i64, Shape{}, vector<int64_t>{1});
    constant_axis->set_friendly_name("constant_axis");
    auto gather = make_shared<op::v7::Gather>(constant_data, constant_indices, constant_axis);
    gather->set_friendly_name("test");
    auto f = make_shared<Model>(gather, ParameterVector{});

    run_constant_folding(f);

    ASSERT_EQ(count_ops_of_type<op::v7::Gather>(f), 0);
    ASSERT_EQ(count_ops_of_type<ov::op::v0::Constant>(f), 1);

    auto new_const = get_result_constant(f);
    ASSERT_TRUE(new_const);
    check_names(new_const, {"constant_data", "constant_indices", "constant_axis", "test"});
    auto values_out = new_const->get_vector<float>();

    vector<float> values_expected{1.0f, 4.0f, 3.0f, 3.0f, 6.0f, 9.0f, 8.0f, 8.0f};

    ASSERT_TRUE(ov::test::utils::all_close_f(values_out, values_expected, MIN_FLOAT_TOLERANCE_BITS));
}

TEST(constant_folding, const_gather_v7_subgraph) {
    const auto A = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1});
    const float b_value = 3.21f;
    const auto B_const = ov::op::v0::Constant::create(element::f32, {1}, {b_value});
    const auto C = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1});
    const int64_t axis = 0;
    const auto axis_const = ov::op::v0::Constant::create(element::i64, {}, {axis});
    axis_const->set_friendly_name("axis_const");

    const auto concat = make_shared<ov::op::v0::Concat>(NodeVector{A, B_const, C}, axis);
    concat->set_friendly_name("concat");

    const vector<int64_t> indices{1};
    const auto indices_const = ov::op::v0::Constant::create(element::i64, {indices.size()}, indices);
    indices_const->set_friendly_name("indices_const");
    const auto gather = make_shared<op::v7::Gather>(concat, indices_const, axis_const);
    gather->set_friendly_name("test");
    auto f = make_shared<Model>(gather, ParameterVector{A, C});

    run_constant_folding(f);

    ASSERT_EQ(count_ops_of_type<ov::op::v0::Concat>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::v7::Gather>(f), 0);
    ASSERT_EQ(count_ops_of_type<ov::op::v0::Constant>(f), 1);

    const auto new_const = get_result_constant(f);
    ASSERT_TRUE(new_const);
    check_names(new_const, {"axis_const", "concat", "indices_const", "test"});

    const auto values_out = new_const->get_vector<float>();
    ASSERT_TRUE(ov::test::utils::all_close_f(values_out, {b_value}, MIN_FLOAT_TOLERANCE_BITS));
}

TEST(constant_folding, const_gather_v7_subgraph_neg_axis) {
    const auto A = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1});
    const float b_value = 1.23f;
    const auto B = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1});
    const auto C_const = ov::op::v0::Constant::create(element::f32, {1}, {b_value});
    const int64_t axis = 0;
    const auto axis_const = ov::op::v0::Constant::create(element::i64, {}, {axis});
    axis_const->set_friendly_name("axis_const");

    const auto concat = make_shared<ov::op::v0::Concat>(NodeVector{A, B, C_const}, axis);
    concat->set_friendly_name("concat");

    const vector<int64_t> indices{-1};
    const auto indices_const = ov::op::v0::Constant::create(element::i64, {indices.size()}, indices);
    indices_const->set_friendly_name("indices_const");
    const auto gather = make_shared<op::v7::Gather>(concat, indices_const, axis_const);
    gather->set_friendly_name("test");
    auto f = make_shared<Model>(gather, ParameterVector{A, B});

    run_constant_folding(f);

    ASSERT_EQ(count_ops_of_type<ov::op::v0::Concat>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::v7::Gather>(f), 0);
    ASSERT_EQ(count_ops_of_type<ov::op::v0::Constant>(f), 1);

    const auto new_const = get_result_constant(f);
    ASSERT_TRUE(new_const);
    check_names(new_const, {"axis_const", "concat", "indices_const", "test"});

    const auto values_out = new_const->get_vector<float>();
    ASSERT_TRUE(ov::test::utils::all_close_f(values_out, {b_value}, MIN_FLOAT_TOLERANCE_BITS));
}

TEST(constant_folding, const_gather_v7_subgraph_no_constant_input) {
    const auto A = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1});
    const auto B = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1});
    const auto C = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1});
    const int64_t axis = 0;
    const auto axis_const = ov::op::v0::Constant::create(element::i64, {}, {axis});

    const auto concat = make_shared<ov::op::v0::Concat>(NodeVector{A, B, C}, axis);

    const vector<int64_t> indices{1};
    const auto indices_const = ov::op::v0::Constant::create(element::i64, {indices.size()}, indices);
    const auto gather = make_shared<op::v7::Gather>(concat, indices_const, axis_const);
    gather->set_friendly_name("test");
    auto f = make_shared<Model>(gather, ParameterVector{A, B, C});

    run_constant_folding(f);

    ASSERT_EQ(count_ops_of_type<ov::op::v0::Concat>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::v7::Gather>(f), 0);
}

TEST(constant_folding, const_gather_v7_subgraph_no_constant_input_scalar) {
    const auto A = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1});
    const auto B = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1});
    const auto C = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1});
    const int64_t axis = 0;
    const auto axis_const = ov::op::v0::Constant::create(element::i64, {}, {axis});

    const auto concat = make_shared<ov::op::v0::Concat>(NodeVector{A, B, C}, axis);

    const vector<int64_t> indices{1};
    const auto indices_const = ov::op::v0::Constant::create(element::i64, {}, indices);
    const auto gather = make_shared<op::v7::Gather>(concat, indices_const, axis_const);
    auto f = make_shared<Model>(gather, ParameterVector{A, B, C});

    run_constant_folding(f);

    ASSERT_EQ(count_ops_of_type<ov::op::v0::Concat>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::v7::Gather>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::v0::Squeeze>(f), 1);
}

TEST(constant_folding, const_gather_v7_subgraph_skip_if_non_zero_axis) {
    const auto A = make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 2});
    const auto B = make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 2});
    const auto C = make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 2});
    const int64_t axis = 1;
    const auto axis_const = ov::op::v0::Constant::create(element::i64, {}, {axis});

    const auto concat = make_shared<ov::op::v0::Concat>(NodeVector{A, B, C}, axis);

    const vector<int64_t> indices{1};
    const auto indices_const = ov::op::v0::Constant::create(element::i64, {indices.size()}, indices);
    const auto gather = make_shared<op::v7::Gather>(concat, indices_const, axis_const);
    auto f = make_shared<Model>(gather, ParameterVector{A, B, C});

    run_constant_folding(f);

    ASSERT_EQ(count_ops_of_type<ov::op::v0::Concat>(f), 1);
    ASSERT_EQ(count_ops_of_type<op::v7::Gather>(f), 1);
}

TEST(constant_folding, const_gather_v7_subgraph_skip_if_non_single_indices) {
    const auto A = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1});
    const auto B = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1});
    const auto C = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1});
    const int64_t axis = 0;
    const auto axis_const = ov::op::v0::Constant::create(element::i64, {}, {axis});

    const auto concat = make_shared<ov::op::v0::Concat>(NodeVector{A, B, C}, axis);

    const vector<int64_t> indices{0, 1};
    const auto indices_const = ov::op::v0::Constant::create(element::i64, {indices.size()}, indices);
    const auto gather = make_shared<op::v7::Gather>(concat, indices_const, axis_const);
    auto f = make_shared<Model>(gather, ParameterVector{A, B, C});

    run_constant_folding(f);

    ASSERT_EQ(count_ops_of_type<ov::op::v0::Concat>(f), 1);
    ASSERT_EQ(count_ops_of_type<op::v7::Gather>(f), 1);
}

TEST(constant_folding, const_gather_v7_subgraph_skip_if_concat_output_shape_dynamic) {
    const auto A = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic());
    const auto B = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1});
    const auto C = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1});
    const int64_t axis = 0;
    const auto axis_const = ov::op::v0::Constant::create(element::i64, {}, {axis});

    const auto concat = make_shared<ov::op::v0::Concat>(NodeVector{A, B, C}, axis);

    const vector<int64_t> indices{1};
    const auto indices_const = ov::op::v0::Constant::create(element::i64, {indices.size()}, indices);
    const auto gather = make_shared<op::v7::Gather>(concat, indices_const, axis_const);
    auto f = make_shared<Model>(gather, ParameterVector{A, B, C});

    run_constant_folding(f);

    ASSERT_EQ(count_ops_of_type<ov::op::v0::Concat>(f), 1);
    ASSERT_EQ(count_ops_of_type<op::v7::Gather>(f), 1);
}

TEST(constant_folding, const_gather_v7_subgraph_skip_if_not_single_input) {
    const auto A = make_shared<ov::op::v0::Parameter>(element::f32, Shape{2});
    const auto B = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1});
    const auto C = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1});
    const int64_t axis = 0;
    const auto axis_const = ov::op::v0::Constant::create(element::i64, {}, {axis});

    const auto concat = make_shared<ov::op::v0::Concat>(NodeVector{A, B, C}, axis);

    const vector<int64_t> indices{1};
    const auto indices_const = ov::op::v0::Constant::create(element::i64, {indices.size()}, indices);
    const auto gather = make_shared<op::v7::Gather>(concat, indices_const, axis_const);
    auto f = make_shared<Model>(gather, ParameterVector{A, B, C});

    run_constant_folding(f);

    ASSERT_EQ(count_ops_of_type<ov::op::v0::Concat>(f), 1);
    ASSERT_EQ(count_ops_of_type<op::v7::Gather>(f), 1);
}

TEST(constant_folding, const_strided_slice) {
    Shape shape_in{16};

    vector<int> values_in{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    auto constant = make_shared<ov::op::v0::Constant>(element::i32, shape_in, values_in);
    constant->set_friendly_name("constant");
    auto begin = ov::op::v0::Constant::create(element::i64, {1}, {2});
    begin->set_friendly_name("begin");
    auto end = ov::op::v0::Constant::create(element::i64, {1}, {15});
    end->set_friendly_name("end");
    auto stride = ov::op::v0::Constant::create(element::i64, {1}, {3});
    stride->set_friendly_name("stride");
    auto slice = make_shared<op::v1::StridedSlice>(constant,
                                                   begin,
                                                   end,
                                                   stride,
                                                   std::vector<int64_t>{0},
                                                   std::vector<int64_t>{0});
    slice->set_friendly_name("test");

    auto f = make_shared<Model>(slice, ParameterVector{});

    run_constant_folding(f);

    ASSERT_EQ(count_ops_of_type<op::v1::StridedSlice>(f), 0);
    ASSERT_EQ(count_ops_of_type<ov::op::v0::Constant>(f), 1);

    auto new_const = get_result_constant(f);
    ASSERT_TRUE(new_const);
    check_names(new_const, {"constant", "begin", "end", "stride", "test"});
    auto values_out = new_const->get_vector<int>();

    vector<int> sliced_values{3, 6, 9, 12, 15};
    ASSERT_EQ(sliced_values, values_out);
}

TEST(constant_folding, strided_slice_ignored_dynamic_begin_end_values_from_shape_of) {
    const auto constant =
        make_shared<ov::op::v0::Constant>(element::i32,
                                          Shape{1, 1, 2, 4, 2},
                                          std::vector<int>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    constant->set_friendly_name("constant");

    const auto begin_shape = PartialShape{0, -1, 0, 0, 0};
    const auto p_begin = std::make_shared<ov::op::v0::Parameter>(element::i64, begin_shape);
    const auto shape_of_begin = std::make_shared<ov::op::v0::ShapeOf>(p_begin);
    shape_of_begin->set_friendly_name("begin");

    const auto end_shape = PartialShape{-1, 512, 2, 2, 16};
    const auto p_end = std::make_shared<ov::op::v0::Parameter>(element::i64, end_shape);
    const auto shape_of_end = std::make_shared<ov::op::v0::ShapeOf>(p_end);
    shape_of_end->set_friendly_name("end");

    const auto stride = ov::op::v0::Constant::create(element::i64, {5}, {1, 1, 1, 1, 1});
    stride->set_friendly_name("stride");

    const auto slice = make_shared<op::v1::StridedSlice>(constant,
                                                         shape_of_begin,
                                                         shape_of_end,
                                                         stride,
                                                         std::vector<int64_t>{0, 1, 0, 0, 0},
                                                         std::vector<int64_t>{1, 1, 0, 0, 1});
    slice->set_friendly_name("test");

    auto model = make_shared<ov::Model>(slice, ParameterVector{p_begin, p_end});

    run_constant_folding(model);

    ASSERT_EQ(count_ops_of_type<op::v1::StridedSlice>(model), 0);
    ASSERT_EQ(count_ops_of_type<ov::op::v0::Constant>(model), 1);

    const auto new_const = get_result_constant(model);
    ASSERT_TRUE(new_const);
    check_names(new_const, {"constant", "begin", "end", "stride", "test"});
    const auto values_out = new_const->get_vector<int>();

    vector<int> sliced_values{1, 2, 3, 4, 9, 10, 11, 12};
    ASSERT_EQ(sliced_values, values_out);
}

TEST(constant_folding, strided_slice_all_ignore_mask_set_for_non_parameter_begin_end) {
    const auto constant =
        make_shared<ov::op::v0::Constant>(element::i32,
                                          Shape{1, 1, 2, 4, 2},
                                          std::vector<int>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    constant->set_friendly_name("constant");

    const auto begin_shape = PartialShape{{2, 5}, -1, 10, 1, {0, 200}};
    const auto p_begin = std::make_shared<ov::op::v0::Parameter>(element::i64, begin_shape);
    const auto shape_of_begin = std::make_shared<ov::op::v0::ShapeOf>(p_begin);
    shape_of_begin->set_friendly_name("begin");

    const auto end_shape = PartialShape{-1, 1, {2, 3}, 0, 0};
    const auto p_end = std::make_shared<ov::op::v0::Parameter>(element::i64, end_shape);
    const auto shape_of_end = std::make_shared<ov::op::v0::ShapeOf>(p_end);
    shape_of_end->set_friendly_name("end");

    const auto stride = ov::op::v0::Constant::create(element::i64, {5}, {1, 1, 1, 2, 2});
    stride->set_friendly_name("stride");

    const auto slice = make_shared<op::v1::StridedSlice>(constant,
                                                         shape_of_begin,
                                                         shape_of_end,
                                                         stride,
                                                         std::vector<int64_t>{1, 1, 1, 1, 1},
                                                         std::vector<int64_t>{1, 1, 1, 1, 1});
    slice->set_friendly_name("test");

    auto model = make_shared<ov::Model>(slice, ParameterVector{p_begin, p_end});

    run_constant_folding(model);

    ASSERT_EQ(count_ops_of_type<op::v1::StridedSlice>(model), 0);
    ASSERT_EQ(count_ops_of_type<ov::op::v0::Constant>(model), 1);

    const auto new_const = get_result_constant(model);
    ASSERT_TRUE(new_const);
    check_names(new_const, {"constant", "begin", "end", "stride", "test"});
    const auto values_out = new_const->get_vector<int>();

    vector<int> sliced_values{1, 5, 9, 13};
    ASSERT_EQ(sliced_values, values_out);
}

TEST(constant_folding, strided_slice_all_ignore_mask_set_for_parameter_begin_end) {
    const auto constant =
        make_shared<ov::op::v0::Constant>(element::i32,
                                          Shape{1, 1, 2, 4, 2},
                                          std::vector<int>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    constant->set_friendly_name("constant");

    const auto begin_shape = PartialShape{{1, 5}};
    const auto p_begin = std::make_shared<ov::op::v0::Parameter>(element::i64, begin_shape);
    p_begin->set_friendly_name("begin");

    const auto end_shape = PartialShape{5};
    const auto p_end = std::make_shared<ov::op::v0::Parameter>(element::i64, end_shape);
    p_end->set_friendly_name("end");

    const auto stride = ov::op::v0::Constant::create(element::i64, {5}, {1, 1, 1, 2, 2});
    stride->set_friendly_name("stride");

    const auto slice = make_shared<op::v1::StridedSlice>(constant,
                                                         p_begin,
                                                         p_end,
                                                         stride,
                                                         std::vector<int64_t>{1, 1, 1, 1, 1},
                                                         std::vector<int64_t>{1, 1, 1, 1, 1});
    slice->set_friendly_name("test");

    auto model = make_shared<ov::Model>(slice, ParameterVector{p_begin, p_end});

    run_constant_folding(model);

    ASSERT_EQ(count_ops_of_type<op::v1::StridedSlice>(model), 0);
    ASSERT_EQ(count_ops_of_type<ov::op::v0::Constant>(model), 1);

    const auto new_const = get_result_constant(model);
    ASSERT_TRUE(new_const);
    check_names(new_const, {"constant", "begin", "end", "stride", "test"});
    const auto values_out = new_const->get_vector<int>();

    vector<int> sliced_values{1, 5, 9, 13};
    ASSERT_EQ(sliced_values, values_out);
}

TEST(constant_folding, strided_slice_not_all_ignore_mask_set_for_parameter_begin_end) {
    const auto constant =
        make_shared<ov::op::v0::Constant>(element::i32,
                                          Shape{1, 1, 2, 4, 2},
                                          std::vector<int>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    constant->set_friendly_name("constant");

    const auto begin_shape = PartialShape::dynamic();
    const auto p_begin = std::make_shared<ov::op::v0::Parameter>(element::i64, begin_shape);
    p_begin->set_friendly_name("begin");

    const auto end_shape = PartialShape{5};
    const auto p_end = std::make_shared<ov::op::v0::Parameter>(element::i64, end_shape);
    p_end->set_friendly_name("end");

    const auto stride = ov::op::v0::Constant::create(element::i64, {5}, {1, 1, 1, 2, 2});
    stride->set_friendly_name("stride");

    const auto slice = make_shared<op::v1::StridedSlice>(constant,
                                                         p_begin,
                                                         p_end,
                                                         stride,
                                                         std::vector<int64_t>{1, 1, 1, 1},
                                                         std::vector<int64_t>{1, 1, 1, 1});
    slice->set_friendly_name("test");

    auto model = make_shared<ov::Model>(slice, ParameterVector{p_begin, p_end});

    run_constant_folding(model);

    ASSERT_EQ(count_ops_of_type<op::v1::StridedSlice>(model), 1);
    ASSERT_EQ(count_ops_of_type<ov::op::v0::Constant>(model), 2);
}

TEST(constant_folding, strided_slice_not_ignored_dynamic_begin_from_shape_of) {
    const auto constant =
        make_shared<ov::op::v0::Constant>(element::i32,
                                          Shape{1, 1, 2, 4, 2},
                                          std::vector<int>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    constant->set_friendly_name("constant");

    const auto begin_shape = PartialShape{0, -1, 0, 0, 0};
    const auto p_begin = std::make_shared<ov::op::v0::Parameter>(element::i64, begin_shape);
    const auto shape_of_begin = std::make_shared<ov::op::v0::ShapeOf>(p_begin);
    shape_of_begin->set_friendly_name("begin");

    const auto end_shape = PartialShape{-1, 512, 2, 2, 16};
    const auto p_end = std::make_shared<ov::op::v0::Parameter>(element::i64, end_shape);
    const auto shape_of_end = std::make_shared<ov::op::v0::ShapeOf>(p_end);
    shape_of_end->set_friendly_name("end");

    const auto stride = ov::op::v0::Constant::create(element::i64, {5}, {1, 1, 1, 1, 1});
    stride->set_friendly_name("stride");

    const auto slice = make_shared<op::v1::StridedSlice>(constant,
                                                         shape_of_begin,
                                                         shape_of_end,
                                                         stride,
                                                         std::vector<int64_t>{0, 0, 0, 0, 0},
                                                         std::vector<int64_t>{1, 1, 0, 0, 1});
    slice->set_friendly_name("test");

    auto model = make_shared<ov::Model>(slice, ParameterVector{p_begin, p_end});

    run_constant_folding(model);

    ASSERT_EQ(count_ops_of_type<op::v1::StridedSlice>(model), 1);
    ASSERT_EQ(count_ops_of_type<ov::op::v0::Constant>(model), 2);
}

TEST(constant_folding, strided_slice_can_be_folded_but_is_blocked_by_shape_of_which_got_folding_disabled) {
    const auto constant =
        make_shared<ov::op::v0::Constant>(element::i32,
                                          Shape{1, 1, 2, 4, 2},
                                          std::vector<int>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    constant->set_friendly_name("constant");

    const auto begin_shape = PartialShape{0, -1, 0, 0, 0};
    const auto p_begin = std::make_shared<ov::op::v0::Parameter>(element::i64, begin_shape);
    const auto shape_of_begin = std::make_shared<ov::op::v0::ShapeOf>(p_begin);
    shape_of_begin->set_friendly_name("begin");

    const auto end_shape = PartialShape{-1, 512, 2, 2, 16};
    const auto p_end = std::make_shared<ov::op::v0::Parameter>(element::i64, end_shape);
    const auto shape_of_end = std::make_shared<ov::op::v0::ShapeOf>(p_end);
    shape_of_end->set_friendly_name("end");

    const auto stride = ov::op::v0::Constant::create(element::i64, {5}, {1, 1, 1, 1, 1});
    stride->set_friendly_name("stride");

    const auto slice = make_shared<op::v1::StridedSlice>(constant,
                                                         shape_of_begin,
                                                         shape_of_end,
                                                         stride,
                                                         std::vector<int64_t>{0, 1, 0, 0, 0},
                                                         std::vector<int64_t>{1, 1, 0, 0, 1});
    slice->set_friendly_name("test");

    auto model = make_shared<ov::Model>(slice, ParameterVector{p_begin, p_end});

    pass::Manager pass_manager;
    pass_manager.register_pass<ov::pass::InitNodeInfo>();
    pass_manager.register_pass<ov::pass::DisableShapeOfConstantFolding>();
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(model);

    ASSERT_EQ(count_ops_of_type<op::v1::StridedSlice>(model), 1);
    ASSERT_EQ(count_ops_of_type<ov::op::v0::Constant>(model), 2);
}

TEST(constant_folding, strided_slice_is_foldable_but_got_set_disable_constant_fold) {
    const auto constant =
        make_shared<ov::op::v0::Constant>(element::i32,
                                          Shape{1, 1, 2, 4, 2},
                                          std::vector<int>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    constant->set_friendly_name("constant");

    const auto begin_shape = PartialShape{0, -1, 0, 0, 0};
    const auto p_begin = std::make_shared<ov::op::v0::Parameter>(element::i64, begin_shape);
    const auto shape_of_begin = std::make_shared<ov::op::v0::ShapeOf>(p_begin);
    shape_of_begin->set_friendly_name("begin");

    const auto end_shape = PartialShape{-1, 512, 2, 2, 16};
    const auto p_end = std::make_shared<ov::op::v0::Parameter>(element::i64, end_shape);
    const auto shape_of_end = std::make_shared<ov::op::v0::ShapeOf>(p_end);
    shape_of_end->set_friendly_name("end");

    const auto stride = ov::op::v0::Constant::create(element::i64, {5}, {1, 1, 1, 1, 1});
    stride->set_friendly_name("stride");

    const auto slice = make_shared<op::v1::StridedSlice>(constant,
                                                         shape_of_begin,
                                                         shape_of_end,
                                                         stride,
                                                         std::vector<int64_t>{0, 1, 0, 0, 0},
                                                         std::vector<int64_t>{1, 1, 0, 0, 1});
    slice->set_friendly_name("test");

    auto model = make_shared<ov::Model>(slice, ParameterVector{p_begin, p_end});

    ov::disable_constant_folding(slice);

    run_constant_folding(model);

    ASSERT_EQ(count_ops_of_type<op::v1::StridedSlice>(model), 1);
    ASSERT_EQ(count_ops_of_type<ov::op::v0::Constant>(model), 2);
}

TEST(constant_folding, constant_dyn_reshape) {
    Shape shape_in{2, 4};
    vector<float> values_in{0, 1, 2, 3, 4, 5, 6, 7};

    Shape shape_shape{3};
    vector<int64_t> values_shape{2, 4, 1};

    auto constant_in = make_shared<ov::op::v0::Constant>(element::f32, shape_in, values_in);
    constant_in->set_friendly_name("constant_in");
    auto constant_shape = make_shared<ov::op::v0::Constant>(element::i64, shape_shape, values_shape);
    constant_shape->set_friendly_name("constant_shape");
    auto dyn_reshape = make_shared<op::v1::Reshape>(constant_in, constant_shape, false);
    dyn_reshape->set_friendly_name("test");
    auto f = make_shared<Model>(dyn_reshape, ParameterVector{});

    run_constant_folding(f);

    ASSERT_EQ(count_ops_of_type<op::v1::Reshape>(f), 0);
    ASSERT_EQ(count_ops_of_type<ov::op::v0::Constant>(f), 1);

    auto new_const = get_result_constant(f);
    ASSERT_TRUE(new_const);
    check_names(new_const, {"constant_in", "constant_shape", "test"});
    auto values_out = new_const->get_vector<float>();

    ASSERT_TRUE(ov::test::utils::all_close_f(values_in, values_out, MIN_FLOAT_TOLERANCE_BITS));
}

TEST(constant_folding, constant_dyn_reshape_shape_not_originally_constant) {
    Shape shape_in{2, 4};
    vector<float> values_in{0, 1, 2, 3, 4, 5, 6, 7};

    Shape shape_shape{3};
    // We're going to add these two together elementwise to get {2, 4, 1}.
    // This means that when ConstantFolding starts, v1::Reshape will not yet
    // have static output shape. But by the time the Add op is folded, the
    // v1::Reshape's shape should be inferrable.
    vector<int64_t> values_shape_a{1, 3, 0};
    vector<int64_t> values_shape_b{1, 1, 1};

    auto constant_in = make_shared<ov::op::v0::Constant>(element::f32, shape_in, values_in);
    constant_in->set_friendly_name("constant_in");
    auto constant_shape_a = make_shared<ov::op::v0::Constant>(element::i64, shape_shape, values_shape_a);
    constant_shape_a->set_friendly_name("constant_shape_a");
    auto constant_shape_b = make_shared<ov::op::v0::Constant>(element::i64, shape_shape, values_shape_b);
    constant_shape_b->set_friendly_name("constant_shape_b");
    auto add = std::make_shared<op::v1::Add>(constant_shape_a, constant_shape_b);
    add->set_friendly_name("add");
    auto dyn_reshape = make_shared<op::v1::Reshape>(constant_in, add, false);
    dyn_reshape->set_friendly_name("test");
    auto f = make_shared<Model>(dyn_reshape, ParameterVector{});

    run_constant_folding(f);

    ASSERT_EQ(count_ops_of_type<op::v1::Reshape>(f), 0);
    ASSERT_EQ(count_ops_of_type<ov::op::v0::Constant>(f), 1);

    auto new_const = get_result_constant(f);
    ASSERT_TRUE(new_const);
    check_names(new_const, {"constant_in", "constant_shape_a", "constant_shape_b", "add", "test"});
    auto values_out = new_const->get_vector<float>();

    ASSERT_TRUE(ov::test::utils::all_close_f(values_in, values_out, MIN_FLOAT_TOLERANCE_BITS));
}

TEST(constant_folding, const_reshape_no_data_copy) {
    auto const_data = ov::op::v0::Constant::create(element::f32, Shape{1, 64}, {1});
    auto const_reshape = ov::op::v0::Constant::create(element::i64, Shape{2}, {2, 32});
    auto reshape = std::make_shared<op::v1::Reshape>(const_data, const_reshape, false);
    auto consumer1 = std::make_shared<ov::op::v0::Relu>(reshape);
    auto consumer2 = std::make_shared<ov::op::v0::Relu>(reshape);

    auto f = std::make_shared<Model>(NodeVector{consumer1, consumer2}, ParameterVector{});

    run_constant_folding(f);

    auto const1 = std::dynamic_pointer_cast<ov::op::v0::Constant>(consumer1->input_value(0).get_node_shared_ptr());
    auto const2 = std::dynamic_pointer_cast<ov::op::v0::Constant>(consumer2->input_value(0).get_node_shared_ptr());

    ASSERT_TRUE(const1);
    ASSERT_TRUE(const2);
    ASSERT_EQ(const1, const2);
    ASSERT_EQ(const1->get_data_ptr(), const2->get_data_ptr());
}

TEST(constant_folding, const_squeeze_no_data_copy) {
    auto const_data = ov::op::v0::Constant::create(element::f32, Shape{1, 64}, {1});
    auto const_reshape = ov::op::v0::Constant::create(element::i64, Shape{1}, {0});
    auto reshape = std::make_shared<op::v0::Squeeze>(const_data, const_reshape);
    auto consumer1 = std::make_shared<ov::op::v0::Relu>(reshape);
    auto consumer2 = std::make_shared<ov::op::v0::Relu>(reshape);

    auto f = std::make_shared<Model>(NodeVector{consumer1, consumer2}, ParameterVector{});

    run_constant_folding(f);

    auto const1 = std::dynamic_pointer_cast<ov::op::v0::Constant>(consumer1->input_value(0).get_node_shared_ptr());
    auto const2 = std::dynamic_pointer_cast<ov::op::v0::Constant>(consumer2->input_value(0).get_node_shared_ptr());

    ASSERT_TRUE(const1);
    ASSERT_TRUE(const2);
    ASSERT_EQ(const1, const2);
    ASSERT_EQ(const1->get_data_ptr(), const2->get_data_ptr());
}

TEST(constant_folding, const_unsqueeze_no_data_copy) {
    auto const_data = ov::op::v0::Constant::create(element::f32, Shape{1, 64}, {1});
    auto const_reshape = ov::op::v0::Constant::create(element::i64, Shape{1}, {0});
    auto reshape = std::make_shared<op::v0::Unsqueeze>(const_data, const_reshape);
    auto consumer1 = std::make_shared<ov::op::v0::Relu>(reshape);
    auto consumer2 = std::make_shared<ov::op::v0::Relu>(reshape);

    auto f = std::make_shared<Model>(NodeVector{consumer1, consumer2}, ParameterVector{});

    run_constant_folding(f);

    auto const1 = std::dynamic_pointer_cast<ov::op::v0::Constant>(consumer1->input_value(0).get_node_shared_ptr());
    auto const2 = std::dynamic_pointer_cast<ov::op::v0::Constant>(consumer2->input_value(0).get_node_shared_ptr());

    ASSERT_TRUE(const1);
    ASSERT_TRUE(const2);
    ASSERT_EQ(const1, const2);
    ASSERT_EQ(const1->get_data_ptr(), const2->get_data_ptr());
}

TEST(constant_folding, constant_transpose) {
    Shape shape_in{2, 4};
    vector<double> values_in{0, 1, 2, 3, 4, 5, 6, 7};

    Shape shape_perm{2};
    vector<int64_t> values_perm{1, 0};

    auto constant_in = make_shared<ov::op::v0::Constant>(element::f64, shape_in, values_in);
    constant_in->set_friendly_name("constant_in");
    auto constant_perm = make_shared<ov::op::v0::Constant>(element::i64, shape_perm, values_perm);
    constant_perm->set_friendly_name("constant_perm");
    auto transpose = make_shared<ov::op::v1::Transpose>(constant_in, constant_perm);
    transpose->set_friendly_name("test");
    auto f = make_shared<Model>(transpose, ParameterVector{});

    run_constant_folding(f);

    ASSERT_EQ(count_ops_of_type<ov::op::v1::Transpose>(f), 0);
    ASSERT_EQ(count_ops_of_type<ov::op::v0::Constant>(f), 1);

    auto new_const = get_result_constant(f);
    ASSERT_TRUE(new_const);
    check_names(new_const, {"constant_in", "constant_perm", "test"});
    auto values_out = new_const->get_vector<double>();

    vector<double> values_permute{0, 4, 1, 5, 2, 6, 3, 7};
    ASSERT_TRUE(ov::test::utils::all_close_f(values_permute, values_out, MIN_FLOAT_TOLERANCE_BITS));
}

template <typename T>
void range_test(T start, T stop, T step, const vector<T>& values_expected) {
    vector<T> values_start{start};
    vector<T> values_stop{stop};
    vector<T> values_step{step};

    auto constant_start = make_shared<ov::op::v0::Constant>(ov::element::from<T>(), Shape{}, values_start);
    constant_start->set_friendly_name("constant_start");
    auto constant_stop = make_shared<ov::op::v0::Constant>(ov::element::from<T>(), Shape{}, values_stop);
    constant_stop->set_friendly_name("constant_stop");
    auto constant_step = make_shared<ov::op::v0::Constant>(ov::element::from<T>(), Shape{}, values_step);
    constant_step->set_friendly_name("constant_step");
    auto range = make_shared<ov::op::v0::Range>(constant_start, constant_stop, constant_step);
    range->set_friendly_name("test");
    auto f = make_shared<Model>(range, ParameterVector{});

    run_constant_folding(f);

    ASSERT_EQ(count_ops_of_type<ov::op::v0::Range>(f), 0);
    ASSERT_EQ(count_ops_of_type<ov::op::v0::Constant>(f), 1);

    auto new_const = get_result_constant(f);
    ASSERT_TRUE(new_const);
    check_names(new_const, {"constant_start", "constant_stop", "constant_step", "test"});

    auto values_out = new_const->template get_vector<T>();

    range_test_check(values_out, values_expected);
}

TEST(constant_folding, constant_range) {
    range_test<int8_t>(5, 12, 2, {5, 7, 9, 11});
    range_test<int32_t>(5, 12, 2, {5, 7, 9, 11});
    range_test<int64_t>(5, 12, 2, {5, 7, 9, 11});
    range_test<uint64_t>(5, 12, 2, {5, 7, 9, 11});
    range_test<double>(5, 12, 2, {5, 7, 9, 11});
    range_test<float>(5, 12, 2, {5, 7, 9, 11});

    range_test<int32_t>(5, 12, -2, {});
    range_test<float>(12, 4, -2, {12, 10, 8, 6});
}

TEST(constant_folding, constant_v1_select) {
    Shape shape{2, 4};
    vector<char> values_selection{0, 1, 1, 0};
    vector<int64_t> values_t{1, 2, 3, 4};
    vector<int64_t> values_f{11, 12, 13, 14, 15, 16, 17, 18};

    auto constant_selection = make_shared<ov::op::v0::Constant>(element::boolean, Shape{4}, values_selection);
    constant_selection->set_friendly_name("constant_selection");
    auto constant_t = make_shared<ov::op::v0::Constant>(element::i64, Shape{4}, values_t);
    constant_t->set_friendly_name("constant_t");
    auto constant_f = make_shared<ov::op::v0::Constant>(element::i64, Shape{2, 4}, values_f);
    constant_f->set_friendly_name("constant_f");
    auto select = make_shared<op::v1::Select>(constant_selection, constant_t, constant_f);
    select->set_friendly_name("test");
    auto f = make_shared<Model>(select, ParameterVector{});

    run_constant_folding(f);

    ASSERT_EQ(count_ops_of_type<op::v1::Select>(f), 0);
    ASSERT_EQ(count_ops_of_type<ov::op::v0::Constant>(f), 1);

    auto new_const = get_result_constant(f);
    ASSERT_TRUE(new_const);
    check_names(new_const, {"constant_selection", "constant_t", "constant_f", "test"});
    auto values_out = new_const->get_vector<int64_t>();

    vector<int64_t> values_expected{11, 2, 3, 14, 15, 2, 3, 18};
    ASSERT_EQ(values_expected, values_out);
}

TEST(constant_folding, constant_v1_split) {
    vector<float> data{.1f, .2f, .3f, .4f, .5f, .6f};
    const auto const_data = ov::op::v0::Constant::create(element::f32, Shape{data.size()}, data);
    const auto const_axis = ov::op::v0::Constant::create(element::i64, Shape{}, {0});
    const auto num_splits = 3;

    auto split_v1 = make_shared<op::v1::Split>(const_data, const_axis, num_splits);
    auto f = make_shared<Model>(split_v1->outputs(), ParameterVector{});

    run_constant_folding(f);

    ASSERT_EQ(count_ops_of_type<op::v1::Split>(f), 0);
    ASSERT_EQ(count_ops_of_type<ov::op::v0::Constant>(f), num_splits);

    auto res1 = get_result_constant(f);
    auto res2 = get_result_constant(f, 1);
    auto res3 = get_result_constant(f, 2);
    ASSERT_TRUE(res1);
    ASSERT_TRUE(res2);
    ASSERT_TRUE(res3);

    auto res1_values = res1->get_vector<float>();
    ASSERT_TRUE(ov::test::utils::all_close_f(vector<float>(data.begin(), data.begin() + 2), res1_values));
    auto res2_values = res2->get_vector<float>();
    ASSERT_TRUE(ov::test::utils::all_close_f(vector<float>(data.begin() + 2, data.begin() + 4), res2_values));
    auto res3_values = res3->get_vector<float>();
    ASSERT_TRUE(ov::test::utils::all_close_f(vector<float>(data.begin() + 4, data.end()), res3_values));
}

TEST(constant_folding, constant_v1_split_specialized) {
    vector<float> data{.1f, .2f, .3f, .4f, .5f, .6f};
    const auto const_data = ov::op::v0::Constant::create(element::f32, Shape{data.size()}, data);
    const auto const_axis = ov::op::v0::Constant::create(element::i64, Shape{}, {0});
    const auto num_splits = 3;

    auto split_v1 = make_shared<op::v1::Split>(const_data, const_axis, num_splits);
    auto f = make_shared<Model>(split_v1->outputs(), ParameterVector{});

    run_constant_folding(f);

    ASSERT_EQ(count_ops_of_type<op::v1::Split>(f), 0);
    ASSERT_EQ(count_ops_of_type<ov::op::v0::Constant>(f), num_splits);

    auto res1 = get_result_constant(f);
    auto res2 = get_result_constant(f, 1);
    auto res3 = get_result_constant(f, 2);
    ASSERT_TRUE(res1);
    ASSERT_TRUE(res2);
    ASSERT_TRUE(res3);

    auto res1_values = res1->get_vector<float>();
    ASSERT_TRUE(ov::test::utils::all_close_f(vector<float>(data.begin(), data.begin() + 2), res1_values));
    auto res2_values = res2->get_vector<float>();
    ASSERT_TRUE(ov::test::utils::all_close_f(vector<float>(data.begin() + 2, data.begin() + 4), res2_values));
    auto res3_values = res3->get_vector<float>();
    ASSERT_TRUE(ov::test::utils::all_close_f(vector<float>(data.begin() + 4, data.end()), res3_values));
}

TEST(constant_folding, constant_v1_split_axis_1_4_splits) {
    vector<int64_t> data{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,

                         16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,

                         32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,

                         48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63};

    const auto const_data = ov::op::v0::Constant::create(element::i64, Shape{4, 4, 4}, data);
    const_data->set_friendly_name("const_data");
    const auto const_axis = ov::op::v0::Constant::create(element::i64, Shape{}, {1});
    const_axis->set_friendly_name("const_axis");
    const auto num_splits = 4;

    auto split_v1 = make_shared<op::v1::Split>(const_data, const_axis, num_splits);
    split_v1->set_friendly_name("test");
    auto f = make_shared<Model>(split_v1->outputs(), ParameterVector{});

    run_constant_folding(f);

    ASSERT_EQ(count_ops_of_type<op::v1::Split>(f), 0);
    ASSERT_EQ(count_ops_of_type<ov::op::v0::Constant>(f), num_splits);

    auto res1 = get_result_constant(f);
    auto res2 = get_result_constant(f, 1);
    auto res3 = get_result_constant(f, 2);
    auto res4 = get_result_constant(f, 3);
    ASSERT_TRUE(res1);
    check_names(res1, {"const_data", "const_axis", "test"}, "test.0");
    ASSERT_TRUE(res2);
    check_names(res2, {"const_data", "const_axis", "test"}, "test.1");
    ASSERT_TRUE(res3);
    check_names(res3, {"const_data", "const_axis", "test"}, "test.2");
    ASSERT_TRUE(res4);
    check_names(res4, {"const_data", "const_axis", "test"}, "test.3");

    auto res1_values = res1->get_vector<int64_t>();
    ASSERT_EQ(vector<int64_t>({0, 1, 2, 3, 16, 17, 18, 19, 32, 33, 34, 35, 48, 49, 50, 51}), res1_values);
    auto res2_values = res2->get_vector<int64_t>();
    ASSERT_EQ(vector<int64_t>({4, 5, 6, 7, 20, 21, 22, 23, 36, 37, 38, 39, 52, 53, 54, 55}), res2_values);
    auto res3_values = res3->get_vector<int64_t>();
    ASSERT_EQ(vector<int64_t>({8, 9, 10, 11, 24, 25, 26, 27, 40, 41, 42, 43, 56, 57, 58, 59}), res3_values);
    auto res4_values = res4->get_vector<int64_t>();
    ASSERT_EQ(vector<int64_t>({12, 13, 14, 15, 28, 29, 30, 31, 44, 45, 46, 47, 60, 61, 62, 63}), res4_values);
}

TEST(constant_folding, constant_v1_split_axis_1_2_splits) {
    vector<int64_t> data{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,

                         16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,

                         32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,

                         48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63};

    const auto const_data = ov::op::v0::Constant::create(element::i64, Shape{4, 4, 4}, data);
    const auto const_axis = ov::op::v0::Constant::create(element::i64, Shape{}, {1});
    const auto num_splits = 2;

    auto split_v1 = make_shared<op::v1::Split>(const_data, const_axis, num_splits);
    auto f = make_shared<Model>(split_v1->outputs(), ParameterVector{});

    run_constant_folding(f);

    ASSERT_EQ(count_ops_of_type<op::v1::Split>(f), 0);
    ASSERT_EQ(count_ops_of_type<ov::op::v0::Constant>(f), num_splits);

    auto res1 = get_result_constant(f);
    auto res2 = get_result_constant(f, 1);
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

TEST(constant_folding, constant_v1_variadic_split_axis_1_2_splits) {
    vector<int64_t> data{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,

                         16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,

                         32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,

                         48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63};

    const auto const_data = ov::op::v0::Constant::create(element::i64, Shape{4, 4, 4}, data);
    const auto const_axis = ov::op::v0::Constant::create(element::i16, Shape{}, {1});
    vector<int64_t> values_lengths{3, 1};
    auto constant_lengths =
        make_shared<ov::op::v0::Constant>(element::i64, Shape{values_lengths.size()}, values_lengths);

    auto variadic_split_v1 = make_shared<op::v1::VariadicSplit>(const_data, const_axis, constant_lengths);
    auto f = make_shared<Model>(variadic_split_v1->outputs(), ParameterVector{});

    run_constant_folding(f);

    ASSERT_EQ(count_ops_of_type<op::v1::VariadicSplit>(f), 0);
    ASSERT_EQ(count_ops_of_type<ov::op::v0::Constant>(f), values_lengths.size());

    auto res1 = get_result_constant(f);
    auto res2 = get_result_constant(f, 1);
    ASSERT_TRUE(res1);
    ASSERT_TRUE(res2);

    auto res1_values = res1->get_vector<int64_t>();
    ASSERT_EQ(vector<int64_t>({0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 16, 17, 18, 19,
                               20, 21, 22, 23, 24, 25, 26, 27, 32, 33, 34, 35, 36, 37, 38, 39,
                               40, 41, 42, 43, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59}),
              res1_values);
    auto res2_values = res2->get_vector<int64_t>();
    ASSERT_EQ(vector<int64_t>({12, 13, 14, 15, 28, 29, 30, 31, 44, 45, 46, 47, 60, 61, 62, 63}), res2_values);
}

TEST(constant_folding, constant_v1_variadic_split_axis_1_3_splits_neg_length) {
    vector<int64_t> data{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,

                         16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,

                         32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,

                         48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63};

    const auto const_data = ov::op::v0::Constant::create(element::i64, Shape{4, 4, 4}, data);
    const auto const_axis = ov::op::v0::Constant::create(element::i32, Shape{}, {1});
    vector<int64_t> values_lengths{1, 1, -1};
    auto constant_lengths =
        make_shared<ov::op::v0::Constant>(element::i64, Shape{values_lengths.size()}, values_lengths);

    auto variadic_split_v1 = make_shared<op::v1::VariadicSplit>(const_data, const_axis, constant_lengths);
    auto f = make_shared<Model>(variadic_split_v1->outputs(), ParameterVector{});

    run_constant_folding(f);

    ASSERT_EQ(count_ops_of_type<op::v1::VariadicSplit>(f), 0);
    ASSERT_EQ(count_ops_of_type<ov::op::v0::Constant>(f), values_lengths.size());

    auto res1 = get_result_constant(f);
    auto res2 = get_result_constant(f, 1);
    auto res3 = get_result_constant(f, 2);
    ASSERT_TRUE(res1);
    ASSERT_TRUE(res2);
    ASSERT_TRUE(res3);

    auto res1_values = res1->get_vector<int64_t>();
    ASSERT_EQ(vector<int64_t>({0, 1, 2, 3, 16, 17, 18, 19, 32, 33, 34, 35, 48, 49, 50, 51}), res1_values);
    auto res2_values = res2->get_vector<int64_t>();
    ASSERT_EQ(vector<int64_t>({4, 5, 6, 7, 20, 21, 22, 23, 36, 37, 38, 39, 52, 53, 54, 55}), res2_values);
    auto res3_values = res3->get_vector<int64_t>();
    ASSERT_EQ(vector<int64_t>({8,  9,  10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31,
                               40, 41, 42, 43, 44, 45, 46, 47, 56, 57, 58, 59, 60, 61, 62, 63}),
              res3_values);
}

TEST(constant_folding, constant_v1_one_hot) {
    const vector<int64_t> indices{0, 1, 2};
    const float on_value = 1.123f;
    const float off_value = 0.321f;

    const auto indices_const = ov::op::v0::Constant::create(element::i64, Shape{3}, indices);
    const auto depth_const = ov::op::v0::Constant::create(element::i64, Shape{}, {3});
    const auto on_const = ov::op::v0::Constant::create(element::f32, Shape{}, {on_value});
    const auto off_const = ov::op::v0::Constant::create(element::f32, Shape{}, {off_value});
    int64_t axis = 1;

    auto one_hot_v1 = make_shared<op::v1::OneHot>(indices_const, depth_const, on_const, off_const, axis);
    auto f = make_shared<Model>(one_hot_v1, ParameterVector{});

    run_constant_folding(f);

    ASSERT_EQ(count_ops_of_type<op::v1::OneHot>(f), 0);
    ASSERT_EQ(count_ops_of_type<ov::op::v0::Constant>(f), 1);

    auto res = get_result_constant(f);
    ASSERT_TRUE(res);

    ASSERT_EQ((Shape{3, 3}), res->get_output_shape(0));
    ASSERT_EQ(
        vector<float>({on_value, off_value, off_value, off_value, on_value, off_value, off_value, off_value, on_value}),
        res->get_vector<float>());
}

TEST(constant_folding, constant_v1_one_hot_negative_axes) {
    const vector<int64_t> indices{0, 2, 3, 1};
    const int32_t on_value = 4;
    const int32_t off_value = 1;

    const auto indices_const = ov::op::v0::Constant::create(element::i64, Shape{4}, indices);
    const auto depth_const = ov::op::v0::Constant::create(element::i64, Shape{}, {3});
    const auto on_const = ov::op::v0::Constant::create(element::i32, Shape{}, {on_value});
    const auto off_const = ov::op::v0::Constant::create(element::i32, Shape{}, {off_value});
    int64_t axis = -1;

    auto one_hot_v1 = make_shared<op::v1::OneHot>(indices_const, depth_const, on_const, off_const, axis);
    auto f = make_shared<Model>(one_hot_v1, ParameterVector{});

    run_constant_folding(f);

    ASSERT_EQ(count_ops_of_type<op::v1::OneHot>(f), 0);
    ASSERT_EQ(count_ops_of_type<ov::op::v0::Constant>(f), 1);

    auto res = get_result_constant(f);
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

TEST(constant_folding, constant_v1_one_hot_negative_axes_2) {
    vector<int64_t> indices{0, 2, 1, 3};
    auto on_value = true;
    auto off_value = false;

    const auto indices_const = ov::op::v0::Constant::create(element::i64, Shape{2, 2}, indices);
    indices_const->set_friendly_name("indices_const");
    const auto depth_const = ov::op::v0::Constant::create(element::i64, Shape{}, {3});
    depth_const->set_friendly_name("depth_const");
    const auto on_const = ov::op::v0::Constant::create(element::boolean, Shape{}, {on_value});
    on_const->set_friendly_name("on_const");
    const auto off_const = ov::op::v0::Constant::create(element::boolean, Shape{}, {off_value});
    off_const->set_friendly_name("off_const");
    int64_t axis = -1;

    auto one_hot_v1 = make_shared<op::v1::OneHot>(indices_const, depth_const, on_const, off_const, axis);
    one_hot_v1->set_friendly_name("test");
    auto f = make_shared<Model>(one_hot_v1, ParameterVector{});

    run_constant_folding(f);

    ASSERT_EQ(count_ops_of_type<op::v1::OneHot>(f), 0);
    ASSERT_EQ(count_ops_of_type<ov::op::v0::Constant>(f), 1);

    auto res = get_result_constant(f);
    ASSERT_TRUE(res);
    check_names(res, {"indices_const", "depth_const", "on_const", "off_const", "test"});

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

TEST(constant_folding, constant_tile_1d) {
    Shape shape_in{2};
    Shape shape_repeats{1};
    Shape shape_out{4};

    vector<int> values_in{0, 1};
    auto data = make_shared<ov::op::v0::Constant>(element::i32, shape_in, values_in);
    data->set_friendly_name("data");
    vector<int> values_repeats{2};
    auto repeats = make_shared<ov::op::v0::Constant>(element::i64, shape_repeats, values_repeats);
    repeats->set_friendly_name("repeats");
    auto tile = make_shared<op::v0::Tile>(data, repeats);
    tile->set_friendly_name("test");
    auto f = make_shared<Model>(tile, ParameterVector{});

    run_constant_folding(f);

    ASSERT_EQ(count_ops_of_type<op::v0::Tile>(f), 0);
    ASSERT_EQ(count_ops_of_type<ov::op::v0::Constant>(f), 1);

    auto new_const = get_result_constant(f);
    ASSERT_TRUE(new_const);
    check_names(new_const, {"data", "repeats", "test"});
    auto values_out = new_const->get_vector<int>();

    vector<int> values_expected{0, 1, 0, 1};
    ASSERT_EQ(values_expected, values_out);
}

TEST(constant_folding, constant_tile_3d_small_data_rank) {
    Shape shape_in{2};
    Shape shape_repeats{3};
    Shape shape_out{2, 2, 4};

    vector<int> values_in{0, 1};
    auto data = make_shared<ov::op::v0::Constant>(element::i32, shape_in, values_in);
    data->set_friendly_name("data");
    vector<int> values_repeats{2, 2, 2};
    auto repeats = make_shared<ov::op::v0::Constant>(element::i64, shape_repeats, values_repeats);
    repeats->set_friendly_name("repeats");
    auto tile = make_shared<op::v0::Tile>(data, repeats);
    tile->set_friendly_name("test");
    auto f = make_shared<Model>(tile, ParameterVector{});

    run_constant_folding(f);

    ASSERT_EQ(count_ops_of_type<op::v0::Tile>(f), 0);
    ASSERT_EQ(count_ops_of_type<ov::op::v0::Constant>(f), 1);

    auto new_const = get_result_constant(f);
    ASSERT_TRUE(new_const);
    check_names(new_const, {"data", "repeats", "test"});
    auto values_out = new_const->get_vector<int>();

    vector<int> values_expected{0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1};
    ASSERT_EQ(values_expected, values_out);
}

TEST(constant_folding, constant_tile_3d_few_repeats) {
    Shape shape_in{2, 1, 3};
    Shape shape_repeats{2};
    Shape shape_out{2, 2, 3};

    vector<int> values_in{1, 2, 3, 4, 5, 6};
    auto data = make_shared<ov::op::v0::Constant>(element::i32, shape_in, values_in);
    data->set_friendly_name("data");
    vector<int> values_repeats{2, 1};
    auto repeats = make_shared<ov::op::v0::Constant>(element::i64, shape_repeats, values_repeats);
    repeats->set_friendly_name("repeats");
    auto tile = make_shared<op::v0::Tile>(data, repeats);
    tile->set_friendly_name("test");
    auto f = make_shared<Model>(tile, ParameterVector{});

    run_constant_folding(f);

    ASSERT_EQ(count_ops_of_type<op::v0::Tile>(f), 0);
    ASSERT_EQ(count_ops_of_type<ov::op::v0::Constant>(f), 1);

    auto new_const = get_result_constant(f);
    ASSERT_TRUE(new_const);
    check_names(new_const, {"data", "repeats", "test"});
    auto values_out = new_const->get_vector<int>();

    vector<int> values_expected{1, 2, 3, 1, 2, 3, 4, 5, 6, 4, 5, 6};
    ASSERT_EQ(values_expected, values_out);
}

TEST(constant_folding, constant_tile_1d_0_repeats) {
    Shape shape_in{2};
    Shape shape_repeats{1};
    Shape shape_out{};

    vector<int> values_in{0, 1};
    auto data = make_shared<ov::op::v0::Constant>(element::i32, shape_in, values_in);
    data->set_friendly_name("data");
    vector<int> values_repeats{0};
    auto repeats = make_shared<ov::op::v0::Constant>(element::i64, shape_repeats, values_repeats);
    repeats->set_friendly_name("repeats");
    auto tile = make_shared<op::v0::Tile>(data, repeats);
    tile->set_friendly_name("test");
    auto f = make_shared<Model>(tile, ParameterVector{});

    run_constant_folding(f);

    ASSERT_EQ(count_ops_of_type<op::v0::Tile>(f), 0);
    ASSERT_EQ(count_ops_of_type<ov::op::v0::Constant>(f), 1);

    auto new_const = get_result_constant(f);
    ASSERT_TRUE(new_const);
    check_names(new_const, {"data", "repeats", "test"});
    auto values_out = new_const->get_vector<int>();

    vector<int> values_expected{};
    ASSERT_EQ(values_expected, values_out);
}

TEST(constant_folding, constant_tile_2d_0_repeats) {
    Shape shape_in{2, 2};
    Shape shape_repeats{2};
    Shape shape_out{};

    vector<int> values_in{0, 1, 2, 3};
    auto data = make_shared<ov::op::v0::Constant>(element::i32, shape_in, values_in);
    data->set_friendly_name("data");
    vector<int> values_repeats{0, 0};
    auto repeats = make_shared<ov::op::v0::Constant>(element::i64, shape_repeats, values_repeats);
    repeats->set_friendly_name("repeats");
    auto tile = make_shared<op::v0::Tile>(data, repeats);
    tile->set_friendly_name("test");
    auto f = make_shared<Model>(tile, ParameterVector{});

    run_constant_folding(f);

    ASSERT_EQ(count_ops_of_type<op::v0::Tile>(f), 0);
    ASSERT_EQ(count_ops_of_type<ov::op::v0::Constant>(f), 1);

    auto new_const = get_result_constant(f);
    ASSERT_TRUE(new_const);
    check_names(new_const, {"data", "repeats", "test"});
    auto values_out = new_const->get_vector<int>();

    vector<int> values_expected{};
    ASSERT_EQ(values_expected, values_out);
}

TEST(constant_folding, constant_tile_0_rank_data) {
    Shape shape_in{};
    Shape shape_repeats{1};
    Shape shape_out{4};

    vector<int> values_in{1};
    auto data = make_shared<ov::op::v0::Constant>(element::i32, shape_in, values_in);
    data->set_friendly_name("data");
    vector<int> values_repeats{4};
    auto repeats = make_shared<ov::op::v0::Constant>(element::i64, shape_repeats, values_repeats);
    repeats->set_friendly_name("repeats");
    auto tile = make_shared<op::v0::Tile>(data, repeats);
    tile->set_friendly_name("test");
    auto f = make_shared<Model>(tile, ParameterVector{});

    run_constant_folding(f);

    ASSERT_EQ(count_ops_of_type<op::v0::Tile>(f), 0);
    ASSERT_EQ(count_ops_of_type<ov::op::v0::Constant>(f), 1);

    auto new_const = get_result_constant(f);
    ASSERT_TRUE(new_const);
    check_names(new_const, {"data", "repeats", "test"});
    auto values_out = new_const->get_vector<int>();

    vector<int> values_expected{1, 1, 1, 1};
    ASSERT_EQ(values_expected, values_out);
}

TEST(constant_folding, constant_non_zero_0D) {
    auto data = ov::op::v0::Constant::create(element::i32, Shape{}, {1});
    data->set_friendly_name("data");
    auto non_zero = make_shared<op::v3::NonZero>(data);
    non_zero->set_friendly_name("test");
    auto f = make_shared<Model>(non_zero, ParameterVector{});

    run_constant_folding(f);

    // Fold into constant with shape of {1, 1} for scalar input with
    // non-zero value
    ASSERT_EQ(count_ops_of_type<op::v3::NonZero>(f), 0);
    ASSERT_EQ(count_ops_of_type<ov::op::v0::Constant>(f), 1);

    const auto new_const = get_result_constant(f);
    ASSERT_TRUE(new_const);
    check_names(new_const, {"data", "test"});
    const auto values_out = new_const->get_vector<int64_t>();

    const vector<int64_t> values_expected{0};
    ASSERT_EQ(values_expected, values_out);
    ASSERT_EQ((Shape{1, 1}), new_const->get_shape());
}

TEST(constant_folding, constant_non_zero_1D) {
    vector<int> values_in{0, 1, 0, 1};
    auto data = make_shared<ov::op::v0::Constant>(element::i32, Shape{4}, values_in);
    data->set_friendly_name("data");
    auto non_zero = make_shared<op::v3::NonZero>(data);
    non_zero->set_friendly_name("test");
    auto f = make_shared<Model>(non_zero, ParameterVector{});

    run_constant_folding(f);

    ASSERT_EQ(count_ops_of_type<op::v3::NonZero>(f), 0);
    ASSERT_EQ(count_ops_of_type<ov::op::v0::Constant>(f), 1);

    const auto new_const = get_result_constant(f);
    ASSERT_TRUE(new_const);
    check_names(new_const, {"data", "test"});
    const auto values_out = new_const->get_vector<int64_t>();

    const vector<int64_t> values_expected{1, 3};
    ASSERT_EQ(values_expected, values_out);
    ASSERT_EQ((Shape{1, 2}), new_const->get_shape());
}

TEST(constant_folding, constant_non_zero_int32_output_type) {
    vector<int> values_in{0, 1, 0, 1};
    auto data = make_shared<ov::op::v0::Constant>(element::i32, Shape{4}, values_in);
    data->set_friendly_name("data");
    auto non_zero = make_shared<op::v3::NonZero>(data, element::i32);
    non_zero->set_friendly_name("test");
    auto f = make_shared<Model>(non_zero, ParameterVector{});

    run_constant_folding(f);

    ASSERT_EQ(count_ops_of_type<op::v3::NonZero>(f), 0);
    ASSERT_EQ(count_ops_of_type<ov::op::v0::Constant>(f), 1);

    const auto new_const = get_result_constant(f);
    ASSERT_TRUE(new_const);
    check_names(new_const, {"data", "test"});
    ASSERT_EQ(element::i32, new_const->get_element_type());
    const auto values_out = new_const->get_vector<int32_t>();

    const vector<int32_t> values_expected{1, 3};
    ASSERT_EQ(values_expected, values_out);
    ASSERT_EQ((Shape{1, 2}), new_const->get_shape());
}

TEST(constant_folding, constant_non_zero_1D_all_indices) {
    const vector<float> values_in{1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    const auto data = make_shared<ov::op::v0::Constant>(element::f32, Shape{values_in.size()}, values_in);
    data->set_friendly_name("data");
    const auto non_zero = make_shared<op::v3::NonZero>(data);
    non_zero->set_friendly_name("test");
    auto f = make_shared<Model>(non_zero, ParameterVector{});

    run_constant_folding(f);

    ASSERT_EQ(count_ops_of_type<op::v3::NonZero>(f), 0);
    ASSERT_EQ(count_ops_of_type<ov::op::v0::Constant>(f), 1);

    const auto new_const = get_result_constant(f);
    ASSERT_TRUE(new_const);
    check_names(new_const, {"data", "test"});
    const auto values_out = new_const->get_vector<int64_t>();

    const vector<int64_t> values_expected{0, 1, 2, 3, 4, 5, 6, 7};
    ASSERT_EQ(values_expected, values_out);
    ASSERT_EQ((Shape{1, values_in.size()}), new_const->get_shape());
}

TEST(constant_folding, constant_non_zero_2D) {
    vector<int> values_in{1, 0, 0, 0, 1, 0, 1, 1, 0};
    auto data = make_shared<ov::op::v0::Constant>(element::i32, Shape{3, 3}, values_in);
    data->set_friendly_name("data");
    auto non_zero = make_shared<op::v3::NonZero>(data);
    non_zero->set_friendly_name("test");
    auto f = make_shared<Model>(non_zero, ParameterVector{});

    run_constant_folding(f);

    ASSERT_EQ(count_ops_of_type<op::v3::NonZero>(f), 0);
    ASSERT_EQ(count_ops_of_type<ov::op::v0::Constant>(f), 1);

    const auto new_const = get_result_constant(f);
    ASSERT_TRUE(new_const);
    check_names(new_const, {"data", "test"});
    const auto values_out = new_const->get_vector<int64_t>();

    const vector<int64_t> values_expected{0, 1, 2, 2, 0, 1, 0, 1};
    ASSERT_EQ(values_expected, values_out);
    ASSERT_EQ((Shape{2, 4}), new_const->get_shape());
}

TEST(constant_folding, DISABLED_constant_non_zero_2D_all_indices) {
    const vector<int8_t> values_in{1, 1, 1, 1, 1, 1, 1, 1, 1};
    const auto data = make_shared<ov::op::v0::Constant>(element::i8, Shape{3, 3}, values_in);
    data->set_friendly_name("data");
    const auto non_zero = make_shared<op::v3::NonZero>(data);
    non_zero->set_friendly_name("test");
    auto f = make_shared<Model>(non_zero, ParameterVector{});

    run_constant_folding(f);

    ASSERT_EQ(count_ops_of_type<op::v3::NonZero>(f), 0);
    ASSERT_EQ(count_ops_of_type<ov::op::v0::Constant>(f), 1);

    const auto new_const = get_result_constant(f);
    ASSERT_TRUE(new_const);
    check_names(new_const, {"data", "test"});
    const auto values_out = new_const->get_vector<int64_t>();

    const vector<int64_t> values_expected{0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2};
    ASSERT_EQ(values_expected, values_out);
    ASSERT_EQ((Shape{2, values_in.size()}), new_const->get_shape());
}

TEST(constant_folding, DISABLED_constant_non_zero_2D_all_zeros) {
    const vector<uint8_t> values_in{0, 0, 0, 0, 0, 0};
    const auto data = make_shared<ov::op::v0::Constant>(element::u8, Shape{2, 3}, values_in);
    data->set_friendly_name("data");
    const auto non_zero = make_shared<op::v3::NonZero>(data);
    non_zero->set_friendly_name("test");
    auto f = make_shared<Model>(non_zero, ParameterVector{});

    run_constant_folding(f);

    // fold into Constant with shape of {0}
    ASSERT_EQ(count_ops_of_type<op::v3::NonZero>(f), 0);
    ASSERT_EQ(count_ops_of_type<ov::op::v0::Constant>(f), 1);

    const auto new_const = get_result_constant(f);
    ASSERT_TRUE(new_const);
    check_names(new_const, {"data", "test"});
    ASSERT_EQ(shape_size(new_const->get_shape()), 0);
}

TEST(constant_folding, constant_non_zero_3D) {
    vector<int> values_in{1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0};
    auto data = make_shared<ov::op::v0::Constant>(element::i32, Shape{2, 3, 3}, values_in);
    data->set_friendly_name("data");
    auto non_zero = make_shared<op::v3::NonZero>(data);
    non_zero->set_friendly_name("test");
    auto f = make_shared<Model>(non_zero, ParameterVector{});

    run_constant_folding(f);

    ASSERT_EQ(count_ops_of_type<op::v3::NonZero>(f), 0);
    ASSERT_EQ(count_ops_of_type<ov::op::v0::Constant>(f), 1);

    const auto new_const = get_result_constant(f);
    ASSERT_TRUE(new_const);
    check_names(new_const, {"data", "test"});
    const auto values_out = new_const->get_vector<int64_t>();

    const vector<int64_t> values_expected{0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 2, 2, 2,
                                          0, 0, 0, 1, 1, 2, 0, 2, 1, 0, 1, 2, 0, 1, 2, 0, 2, 1};
    ASSERT_EQ(values_expected, values_out);
    ASSERT_EQ((Shape{3, 12}), new_const->get_shape());
}

TEST(constant_folding, constant_scatter_elements_update_basic) {
    const Shape data_shape{3, 3};
    const Shape indices_shape{2, 3};

    const auto data_const =
        ov::op::v0::Constant::create(element::f32, data_shape, std::vector<float>(shape_size(data_shape), 0.f));
    data_const->set_friendly_name("data_const");
    const auto indices_const = ov::op::v0::Constant::create(element::i32, indices_shape, {1, 0, 2, 0, 2, 1});
    indices_const->set_friendly_name("indices_const");
    const auto updates_const =
        ov::op::v0::Constant::create(element::f32, indices_shape, {1.0f, 1.1f, 1.2f, 2.0f, 2.1f, 2.2f});
    updates_const->set_friendly_name("updates_const");
    const auto axis_const = ov::op::v0::Constant::create(element::i64, Shape{}, {0});
    axis_const->set_friendly_name("axis_const");

    auto scatter_elem_updt =
        make_shared<op::v3::ScatterElementsUpdate>(data_const, indices_const, updates_const, axis_const);
    scatter_elem_updt->set_friendly_name("test");
    auto f = make_shared<Model>(scatter_elem_updt, ParameterVector{});

    run_constant_folding(f);

    ASSERT_EQ(count_ops_of_type<op::v3::ScatterElementsUpdate>(f), 0);
    ASSERT_EQ(count_ops_of_type<ov::op::v0::Constant>(f), 1);

    auto result_node = get_result_constant(f);
    ASSERT_TRUE(result_node);
    check_names(result_node, {"data_const", "indices_const", "updates_const", "axis_const", "test"});
    ASSERT_EQ(data_shape, result_node->get_output_shape(0));
    std::vector<float> expected{2.f, 1.1f, 0.0f, 1.f, 0.0f, 2.2f, 0.f, 2.1f, 1.2f};
    range_test_check(result_node->cast_vector<float>(), expected);
}

TEST(constant_folding, constant_scatter_elements_update_negative_axis) {
    const Shape data_shape{3, 3};
    const Shape indices_shape{2, 3};

    const auto data_const =
        ov::op::v0::Constant::create(element::f32, data_shape, std::vector<float>(shape_size(data_shape), 0.f));
    const auto indices_const = ov::op::v0::Constant::create(element::i32, indices_shape, {1, 0, 2, 0, 2, 1});
    const auto updates_const =
        ov::op::v0::Constant::create(element::f32, indices_shape, {1.0f, 1.1f, 1.2f, 2.0f, 2.1f, 2.2f});
    const auto axis_const = ov::op::v0::Constant::create(element::i64, Shape{}, {-1});

    auto scatter_elem_updt =
        make_shared<op::v3::ScatterElementsUpdate>(data_const, indices_const, updates_const, axis_const);
    auto f = make_shared<Model>(scatter_elem_updt, ParameterVector{});

    run_constant_folding(f);

    ASSERT_EQ(count_ops_of_type<op::v3::ScatterElementsUpdate>(f), 0);
    ASSERT_EQ(count_ops_of_type<ov::op::v0::Constant>(f), 1);

    auto result_node = get_result_constant(f);
    ASSERT_TRUE(result_node);
    ASSERT_EQ(data_shape, result_node->get_output_shape(0));
    std::vector<float> expected{1.1f, 1.0f, 1.2f, 2.0f, 2.2f, 2.1f, 0.0f, 0.0f, 0.0f};
    range_test_check(result_node->cast_vector<float>(), expected);
}

TEST(constant_folding, constant_scatter_elements_update_1d_axis) {
    const Shape data_shape{3, 3};
    const Shape indices_shape{2, 3};

    const auto data_const =
        ov::op::v0::Constant::create(element::f32, data_shape, std::vector<float>(shape_size(data_shape), 0.f));
    const auto indices_const = ov::op::v0::Constant::create(element::i32, indices_shape, {1, 0, 2, 0, 2, 1});
    const auto updates_const =
        ov::op::v0::Constant::create(element::f32, indices_shape, {1.0f, 1.1f, 1.2f, 2.0f, 2.1f, 2.2f});
    const auto axis_const = ov::op::v0::Constant::create(element::i64, Shape{1}, {0});

    auto scatter_elem_updt =
        make_shared<op::v3::ScatterElementsUpdate>(data_const, indices_const, updates_const, axis_const);
    auto f = make_shared<Model>(scatter_elem_updt, ParameterVector{});

    run_constant_folding(f);

    ASSERT_EQ(count_ops_of_type<op::v3::ScatterElementsUpdate>(f), 0);
    ASSERT_EQ(count_ops_of_type<ov::op::v0::Constant>(f), 1);

    auto result_node = get_result_constant(f);
    ASSERT_TRUE(result_node);
    ASSERT_EQ(data_shape, result_node->get_output_shape(0));
    std::vector<float> expected{2.f, 1.1f, 0.0f, 1.f, 0.0f, 2.2f, 0.f, 2.1f, 1.2f};
    range_test_check(result_node->cast_vector<float>(), expected);
}

TEST(constant_folding, constant_scatter_elements_update_3d_i16) {
    const Shape data_shape{3, 3, 3};
    const Shape indices_shape{2, 2, 3};

    const auto data_const =
        ov::op::v0::Constant::create(element::i16, data_shape, std::vector<int16_t>(shape_size(data_shape), 0));
    const auto indices_const =
        ov::op::v0::Constant::create(element::i16, indices_shape, {1, 0, 2, 0, 2, 1, 2, 2, 2, 0, 1, 0});
    const auto updates_const =
        ov::op::v0::Constant::create(element::i16, indices_shape, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    const auto axis_const = ov::op::v0::Constant::create(element::i64, Shape{}, {1});

    auto scatter_elem_updt =
        make_shared<op::v3::ScatterElementsUpdate>(data_const, indices_const, updates_const, axis_const);
    auto f = make_shared<Model>(scatter_elem_updt, ParameterVector{});

    run_constant_folding(f);

    ASSERT_EQ(count_ops_of_type<op::v3::ScatterElementsUpdate>(f), 0);
    ASSERT_EQ(count_ops_of_type<ov::op::v0::Constant>(f), 1);

    auto result_node = get_result_constant(f);
    ASSERT_TRUE(result_node);
    ASSERT_EQ(data_shape, result_node->get_output_shape(0));
    std::vector<int16_t> expected{4, 2, 0, 1, 0, 6, 0, 5, 3, 10, 0, 12, 0, 11, 0, 7, 8, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    range_test_check(result_node->cast_vector<int16_t>(), expected);
}

TEST(constant_folding, constant_scatter_elements_update_one_elem) {
    const Shape data_shape{3, 3, 3};
    const Shape indices_shape{1, 1, 1};
    const auto input_data = std::vector<int32_t>(shape_size(data_shape), 0);

    const auto data_const = ov::op::v0::Constant::create(element::i32, data_shape, input_data);
    const auto indices_const = ov::op::v0::Constant::create(element::i32, indices_shape, {1});
    const auto updates_const = ov::op::v0::Constant::create(element::i32, indices_shape, {2});
    const auto axis_const = ov::op::v0::Constant::create(element::i64, Shape{}, {0});

    auto scatter_elem_updt =
        make_shared<op::v3::ScatterElementsUpdate>(data_const, indices_const, updates_const, axis_const);
    auto f = make_shared<Model>(scatter_elem_updt, ParameterVector{});

    run_constant_folding(f);

    ASSERT_EQ(count_ops_of_type<op::v3::ScatterElementsUpdate>(f), 0);
    ASSERT_EQ(count_ops_of_type<ov::op::v0::Constant>(f), 1);

    auto result_node = get_result_constant(f);
    ASSERT_TRUE(result_node);
    ASSERT_EQ(data_shape, result_node->get_output_shape(0));
    std::vector<int32_t> expected{input_data};
    // we have updated coordinate (1, 0, 0)
    expected.at(9) = 2;
    range_test_check(result_node->cast_vector<int32_t>(), expected);
}

static void test_constant_folding_reshape_v1(Shape& shape_in,
                                             vector<float>& values_in,
                                             Shape shape_shape,
                                             vector<int32_t> values_shape,
                                             bool zero_flag = false) {
    auto constant_in = make_shared<ov::op::v0::Constant>(element::f32, shape_in, values_in);
    constant_in->set_friendly_name("constant_in");
    auto constant_shape = make_shared<ov::op::v0::Constant>(element::i64, shape_shape, values_shape);
    constant_shape->set_friendly_name("constant_shape");
    auto dyn_reshape = make_shared<op::v1::Reshape>(constant_in, constant_shape, zero_flag);
    dyn_reshape->set_friendly_name("test");
    auto f = make_shared<Model>(dyn_reshape, ParameterVector{});

    run_constant_folding(f);

    ASSERT_EQ(count_ops_of_type<op::v1::Reshape>(f), 0);
    ASSERT_EQ(count_ops_of_type<ov::op::v0::Constant>(f), 1);

    auto new_const = get_result_constant(f);
    ASSERT_TRUE(new_const);
    check_names(new_const, {"constant_in", "constant_shape", "test"});
    auto values_out = new_const->get_vector<float>();

    ASSERT_TRUE(ov::test::utils::all_close_f(values_in, values_out, MIN_FLOAT_TOLERANCE_BITS));
}

TEST(constant_folding, constant_dyn_reshape_v1_2d) {
    Shape shape_in{2, 5};
    vector<float> values_in{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

    test_constant_folding_reshape_v1(shape_in, values_in, {4}, {1, 1, 1, 10});
    test_constant_folding_reshape_v1(shape_in, values_in, {4}, {1, 1, 2, 5});
    test_constant_folding_reshape_v1(shape_in, values_in, {3}, {1, 2, 5});
    test_constant_folding_reshape_v1(shape_in, values_in, {3}, {5, 2, 1});
}

TEST(constant_folding, constant_dyn_reshape_v1_pattern_with_negative_indices) {
    Shape shape_in{2, 2, 2, 2};
    vector<float> values_in{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};

    test_constant_folding_reshape_v1(shape_in, values_in, {3}, {4, -1, 2});
    test_constant_folding_reshape_v1(shape_in, values_in, {2}, {4, -1});
    test_constant_folding_reshape_v1(shape_in, values_in, {1}, {-1});
}

TEST(constant_folding, constant_dyn_reshape_v1_pattern_with_zero_dims) {
    Shape shape_in{2, 2, 2, 2};
    vector<float> values_in{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};

    test_constant_folding_reshape_v1(shape_in, values_in, {4}, {2, -1, 2, 0}, true);
    test_constant_folding_reshape_v1(shape_in, values_in, {4}, {4, 1, 0, 2}, true);
}

TEST(constant_folding, disable_constant_folding) {
    auto data = std::make_shared<ov::op::v0::Parameter>(element::f16, Shape{1, 3, 22, 22});

    // In this test case following sub-graph will be consumed by Interpolate, so during shape inference Interpolate
    // will request values from this sub-graph and ConstantFolding pass will try to use this pre-calculated values
    // to fold it. But in our case we are disabling CF for this sub-graph first and then enable CF to check that all
    // checks inside ConstantFolding transformation are working and doesn't cache anytihng.
    auto gather = ov::op::util::node_to_get_shape_value_of_indices_from_shape_source(data, {2, 3});
    auto convert = std::make_shared<op::v0::Convert>(gather, element::f16);
    auto divide_constant = ov::op::v0::Constant::create(element::f16, Shape{1}, {0.5});
    auto divide = std::make_shared<op::v1::Divide>(convert, divide_constant);
    auto convert_after = std::make_shared<op::v0::Convert>(divide, element::i32);

    op::v0::Interpolate::Attributes interp_attr;
    interp_attr.antialias = false;
    interp_attr.axes = {2, 3};
    interp_attr.mode = "nearest";
    interp_attr.pads_begin = {0, 0, 0, 0};
    interp_attr.pads_end = {0, 0, 0, 0};

    auto interpolate = std::make_shared<op::v0::Interpolate>(data, convert_after, interp_attr);
    auto f = std::make_shared<Model>(NodeVector{interpolate}, ParameterVector{data});

    ov::disable_constant_folding(convert);

    run_constant_folding(f);
    // Check that sub-graph on second Interpolate input wasn't folded
    ASSERT_EQ(interpolate->input_value(1), convert_after->output(0));

    ov::enable_constant_folding(convert);

    run_constant_folding(f);

    // After we enabled CF the sub-graph will be folded to Constant
    ASSERT_TRUE(ov::is_type<ov::op::v0::Constant>(interpolate->get_input_node_shared_ptr(1)));

    // Check that DisableConstantFolding attribute wasn't propagated to some other nodes during CF
    for (auto node : f->get_ordered_ops()) {
        ASSERT_FALSE(ov::pass::constant_folding_is_disabled(node));
    }
}

TEST(constant_folding, disable_constant_folding_simple) {
    // This test case checks the behaviour of CF pass when output values are not precalculated
    // so CF triggers another branch where it goes through nodes and trying to fold one by one.
    auto data = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 3, 22, 22});
    auto reshape = std::make_shared<op::v1::Reshape>(ov::op::v0::Constant::create(element::f32, Shape{3}, {1, 2, 3}),
                                                     ov::op::v0::Constant::create(element::i64, Shape{3}, {3, 1, 1}),
                                                     true);
    auto divide = std::make_shared<op::v1::Divide>(data, reshape);
    auto f = std::make_shared<Model>(NodeVector{divide}, ParameterVector{data});

    ov::disable_constant_folding(reshape);

    run_constant_folding(f);

    // Check that Reshape is not folded
    ASSERT_EQ(divide->input_value(1), reshape->output(0));

    ov::enable_constant_folding(reshape);

    run_constant_folding(f);

    // After we enabled CF the sub-graph will be folded to Constant
    ASSERT_TRUE(ov::is_type<ov::op::v0::Constant>(divide->get_input_node_shared_ptr(1)));

    // Check that DisableConstantFolding attribute wasn't propagated to some other nodes during CF
    for (auto node : f->get_ordered_ops()) {
        ASSERT_FALSE(ov::pass::constant_folding_is_disabled(node));
    }
}

TEST(constant_folding, disable_constant_folding_check) {
    auto data = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 3, 22, 22});
    auto shapeof1 = std::make_shared<op::v0::ShapeOf>(data);
    auto reshape1 = std::make_shared<op::v1::Reshape>(data, shapeof1, true);
    auto shapeof2 = std::make_shared<op::v0::ShapeOf>(reshape1);
    auto reshape2 = std::make_shared<op::v1::Reshape>(reshape1, shapeof2, true);
    auto f = std::make_shared<Model>(NodeVector{reshape2}, ParameterVector{data});

    ov::disable_constant_folding(shapeof1);

    class ConstantFoldingAccessor : public pass::ConstantFolding {
    public:
        ConstantFoldingAccessor() = default;
        using ConstantFolding::pre_calculated_values_folding;
    };

    ConstantFoldingAccessor().pre_calculated_values_folding(f);

    ASSERT_TRUE(shapeof1->get_rt_info().count("can_be_folded"));
    ASSERT_FALSE(shapeof1->get_rt_info().at("can_be_folded").as<bool>());

    ASSERT_TRUE(shapeof2->get_rt_info().count("can_be_folded"));
    ASSERT_TRUE(shapeof2->get_rt_info().at("can_be_folded").as<bool>());

    ASSERT_TRUE(ov::is_type<ov::op::v0::Constant>(reshape2->get_input_node_shared_ptr(1)));
}

TEST(constant_folding, constant_loop) {
    auto X = make_shared<op::v0::Constant>(element::f32, Shape{2, 1, 3}, std::vector<int64_t>{0, 1, 2, 3, 4, 5});
    auto Y = make_shared<op::v0::Constant>(element::f32, Shape{1, 1, 3}, std::vector<int64_t>{1, 2, 3});

    // Body parameters
    auto Xi = make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());
    auto Yi = make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());
    auto body_condition = std::make_shared<op::v0::Constant>(ov::element::boolean, ov::Shape{1}, true);

    auto trip_count = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, 2);
    auto exec_condition = std::make_shared<ov::op::v0::Constant>(ov::element::boolean, ov::Shape{1}, true);
    // Body
    auto sum = make_shared<ov::op::v1::Add>(Xi, Yi);
    auto body = make_shared<ov::Model>(OutputVector{body_condition, sum}, ParameterVector{Xi, Yi});
    auto loop = make_shared<op::v5::Loop>(trip_count, exec_condition);
    loop->set_function(body);
    loop->set_special_body_ports(ov::op::v5::Loop::SpecialBodyPorts{-1, 0});

    loop->set_sliced_input(Xi, X, 0, 1, 1, -1, 0);
    loop->set_invariant_input(Yi, Y);

    auto out0 = loop->get_iter_value(sum, -1);
    auto out1 = loop->get_concatenated_slices(sum, 0, 1, 1, -1, 0);

    auto result0 = make_shared<op::v0::Result>(out0);
    auto result1 = make_shared<op::v0::Result>(out1);

    auto results = ResultVector{result0, result1};
    auto f = make_shared<Model>(results, ParameterVector{});

    run_constant_folding(f);

    ASSERT_EQ(count_ops_of_type<ov::op::v5::Loop>(f), 0);
    ASSERT_EQ(count_ops_of_type<ov::op::v0::Constant>(f), 2);

    auto result_node_0 = get_result_constant(f);
    auto result_node_1 = get_result_constant(f, 1);
    ASSERT_TRUE(result_node_0);
    ASSERT_TRUE(result_node_1);

    const ov::Shape shape_0{1, 1, 3};
    const ov::Shape shape_1{2, 1, 3};

    ASSERT_EQ(shape_0, result_node_0->get_output_shape(0));
    ASSERT_EQ(shape_1, result_node_1->get_output_shape(0));
    std::vector<float> expected_0{4, 6, 8};
    std::vector<float> expected_1{1, 3, 5, 4, 6, 8};
    range_test_check(result_node_0->cast_vector<float>(), expected_0);
    range_test_check(result_node_1->cast_vector<float>(), expected_1);
}

TEST(constant_folding, disable_constant_folding_for_shapeof) {
    auto data = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 3, 22, 22});
    auto shapeof = std::make_shared<op::v3::ShapeOf>(data);
    auto reshape = std::make_shared<op::v1::Reshape>(data, shapeof, true);
    auto model = std::make_shared<ov::Model>(NodeVector{reshape}, ParameterVector{data});

    ov::disable_constant_folding(shapeof);

    run_constant_folding(model);

    ASSERT_EQ(reshape->input_value(1), shapeof->output(0));
}

TEST(constant_folding, disable_constant_folding_for_squeeze_unsqueeze) {
    auto const_data = ov::op::v0::Constant::create(element::f32, Shape{1, 64}, {1});
    auto const_axes = ov::op::v0::Constant::create(element::i64, Shape{1}, {0});
    auto squeeze = std::make_shared<op::v0::Squeeze>(const_data, const_axes);
    auto unsqueeze = make_shared<op::v0::Unsqueeze>(const_data, const_axes);
    auto consumer1 = std::make_shared<ov::op::v0::Relu>(squeeze);
    auto consumer2 = std::make_shared<ov::op::v0::Relu>(unsqueeze);

    auto model = std::make_shared<ov::Model>(NodeVector{consumer1, consumer2}, ParameterVector{});

    ov::disable_constant_folding(squeeze);
    ov::disable_constant_folding(unsqueeze);

    run_constant_folding(model);

    ASSERT_EQ(count_ops_of_type<op::v0::Squeeze>(model), 1);
    ASSERT_EQ(count_ops_of_type<op::v0::Unsqueeze>(model), 1);
}

TEST(constant_folding, disable_constant_folding_for_convert_like) {
    auto data = ov::op::v0::Constant::create(element::f32, Shape{1, 64}, {1});
    auto like = ov::op::v0::Constant::create(element::i64, Shape{1, 64}, {1});
    auto convert_like = std::make_shared<op::v1::ConvertLike>(data, like);
    auto consumer1 = std::make_shared<ov::op::v0::Relu>(convert_like);

    auto model = std::make_shared<ov::Model>(NodeVector{consumer1}, ParameterVector{});

    ov::disable_constant_folding(convert_like);

    run_constant_folding(model);

    ASSERT_EQ(count_ops_of_type<op::v1::ConvertLike>(model), 1);
}

TEST(constant_folding, fold_convert_like_node) {
    auto data = ov::op::v0::Constant::create(element::f32, Shape{1, 64}, {1});
    auto like = ov::op::v0::Constant::create(element::i64, Shape{1, 64}, {1});
    auto convert_like = std::make_shared<op::v1::ConvertLike>(data, like);
    auto consumer1 = std::make_shared<ov::op::v0::Relu>(convert_like);

    auto model = std::make_shared<ov::Model>(NodeVector{consumer1}, ParameterVector{});

    run_constant_folding(model);

    ASSERT_EQ(count_ops_of_type<op::v1::ConvertLike>(model), 0);
}

TEST(constant_folding, fold_convert_like_but_node_is_not_foldable) {
    auto data = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 64});
    auto like = ov::op::v0::Constant::create(element::i64, Shape{1, 64}, {1});
    auto convert_like = std::make_shared<op::v1::ConvertLike>(data, like);
    auto consumer1 = std::make_shared<ov::op::v0::Relu>(convert_like);

    auto model = std::make_shared<ov::Model>(NodeVector{consumer1}, ParameterVector{data});

    run_constant_folding(model);

    ASSERT_EQ(count_ops_of_type<op::v1::ConvertLike>(model), 1);
}

class MockAddOp : public ov::op::v1::Add {
public:
    MockAddOp(
        const Output<Node>& arg0,
        const Output<Node>& arg1,
        const ov::op::AutoBroadcastSpec& auto_broadcast = ov::op::AutoBroadcastSpec(ov::op::AutoBroadcastType::NUMPY))
        : ov::op::v1::Add(arg0, arg1, auto_broadcast) {
        ON_CALL(*this, evaluate).WillByDefault([this](ov::TensorVector& outputs, const ov::TensorVector& inputs) {
            return ov::op::v1::Add::evaluate(outputs, inputs);
        });
    }
    MOCK_METHOD(bool,
                evaluate,
                (ov::TensorVector & output_values, const ov::TensorVector& input_values),
                (const, override));
};

TEST(constant_folding, evaluate_on_tensor_vector) {
    vector<int> values_a{1, 2, 3, 4};
    vector<int> values_b{1, 2, 3, 4};
    auto data_shape = Shape{2, 2};
    auto a = make_shared<ov::op::v0::Constant>(element::i32, data_shape, values_a);
    auto b = make_shared<ov::op::v0::Constant>(element::i32, data_shape, values_b);

    auto mock = std::make_shared<::testing::StrictMock<MockAddOp>>(a, b);
    EXPECT_CALL(*mock, evaluate).Times(1);

    auto model = std::make_shared<ov::Model>(NodeVector{mock}, ParameterVector{});

    run_constant_folding(model);

    vector<int> add_expected{2, 4, 6, 8};
    auto result_node = get_result_constant(model);
    ASSERT_TRUE(result_node);
    ASSERT_EQ(data_shape, result_node->get_output_shape(0));
    ASSERT_EQ(add_expected, result_node->cast_vector<int>());
}

TEST(constant_folding, gather_with_dynamic_shapes_in_data_input) {
    auto in_0 = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{30});

    // dynamic input to Gather
    auto in_1 = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{-1, 2});
    in_1->set_friendly_name("in_1");
    auto shape_of = std::make_shared<ov::op::v3::ShapeOf>(in_1);
    shape_of->set_friendly_name("shape_of");
    auto indices = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{1}, std::vector<int>{1});
    indices->set_friendly_name("indices");
    auto axis = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{1}, std::vector<int>{0});
    axis->set_friendly_name("axis");
    auto gather = std::make_shared<ov::op::v8::Gather>(shape_of, indices, axis);
    gather->set_friendly_name("test");
    auto in_2 = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{1}, std::vector<int>{10});
    in_2->set_friendly_name("in_2");
    auto in_3 = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{1}, std::vector<int>{1});
    in_3->set_friendly_name("in_3");
    auto strided_slice = std::make_shared<op::v1::StridedSlice>(in_0,
                                                                gather,
                                                                in_2,
                                                                in_3,
                                                                std::vector<int64_t>{0, 0},
                                                                std::vector<int64_t>{0, 0},
                                                                std::vector<int64_t>{0, 0},
                                                                std::vector<int64_t>{0, 1});
    strided_slice->set_friendly_name("strided_slice");
    auto res = std::make_shared<ov::op::v0::Result>(strided_slice);
    res->set_friendly_name("result");

    auto model = std::make_shared<ov::Model>(ov::ResultVector{res}, ov::ParameterVector{in_0, in_1});

    run_constant_folding(model);

    ASSERT_EQ(count_ops_of_type<ov::op::v8::Gather>(model), 0);
    ASSERT_EQ(count_ops_of_type<ov::op::v1::StridedSlice>(model), 1);

    auto new_const = dynamic_pointer_cast<ov::op::v0::Constant>(strided_slice->input_value(1).get_node_shared_ptr());
    EXPECT_NE(new_const, nullptr);

    check_names(new_const, {"shape_of", "indices", "axis", "test"});

    // check that we are not copying unnecessary values
    check_names(strided_slice, {"strided_slice"}, "strided_slice");
    check_names(res, {"result"}, "result");
}

TEST(constant_folding, parameter_with_unspecified_type_from_host_tensor) {
    auto param = std::make_shared<ov::op::v0::Parameter>(element::undefined, ov::PartialShape{});
    auto res = std::make_shared<ov::op::v0::Result>(param);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{res}, ov::ParameterVector{param});
    EXPECT_NO_THROW(run_constant_folding(model));
}

TEST(constant_folding, sq_diff) {
    auto const_0 = std::make_shared<ov::op::v0::Constant>(element::f32, ov::Shape{1}, std::vector<float>{4});
    auto const_1 = std::make_shared<ov::op::v0::Constant>(element::f32, ov::Shape{1}, std::vector<float>{2});
    auto sq_diff = std::make_shared<ov::op::v0::SquaredDifference>(const_0, const_1);
    auto res = std::make_shared<ov::op::v0::Result>(sq_diff);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{res}, ov::ParameterVector{});
    auto ops = model->get_ops();
    ASSERT_GT(ops.size(), 2);
    EXPECT_NO_THROW(run_constant_folding(model));
    ops = model->get_ordered_ops();
    // constant + result
    ASSERT_EQ(ops.size(), 2);
    auto const_node = std::dynamic_pointer_cast<ov::op::v0::Constant>(ops.front());
    ASSERT_NE(const_node, nullptr);
    auto res_node = std::dynamic_pointer_cast<ov::op::v0::Result>(ops.back());
    ASSERT_NE(res_node, nullptr);
}

class UnsupportedTypesTest : public testing::TestWithParam<element::Type> {};

TEST_P(UnsupportedTypesTest, add_multiply) {
    Shape shape_in{2, 4, 1};

    const auto& type = GetParam();
    auto param = make_shared<op::v0::Parameter>(type, shape_in);
    auto c1 = op::v0::Constant::create(type, shape_in, {1});
    auto c2 = op::v0::Constant::create(type, shape_in, {1});
    auto add = make_shared<op::v1::Add>(c1, c2);
    auto mul = make_shared<op::v1::Multiply>(param, add);
    auto m = make_shared<Model>(mul, ParameterVector{param});

    run_constant_folding(m);

    EXPECT_EQ(m->get_ops().size(), 4);
    EXPECT_EQ(count_ops_of_type<op::v1::Add>(m), 0);
    EXPECT_EQ(count_ops_of_type<op::v1::Multiply>(m), 1);
    EXPECT_EQ(count_ops_of_type<op::v0::Constant>(m), 1);
    ASSERT_EQ(m->get_results().size(), 1);
}

TEST_P(UnsupportedTypesTest, convert_like) {
    Shape shape_in{2, 4, 1};

    const auto& type = GetParam();
    auto param = make_shared<op::v0::Parameter>(type, shape_in);
    auto param2 = make_shared<op::v0::Parameter>(element::f32, shape_in);
    auto c1 = op::v0::Constant::create(type, shape_in, {1});
    auto c2 = op::v0::Constant::create(type, shape_in, {1});
    auto c3 = op::v0::Constant::create(element::i32, shape_in, {1});
    auto add = make_shared<op::v1::Add>(c1, c2);
    auto convert_like = make_shared<op::v1::ConvertLike>(c3, add);
    auto convert_like2 = make_shared<op::v1::ConvertLike>(param2, add);
    auto mul = make_shared<op::v1::Multiply>(convert_like, convert_like2);
    auto m = make_shared<Model>(mul, ParameterVector{param, param2});

    run_constant_folding(m);

    EXPECT_EQ(m->get_ops().size(), 7);
    EXPECT_EQ(count_ops_of_type<op::v1::Add>(m), 0);
    EXPECT_EQ(count_ops_of_type<op::v1::ConvertLike>(m), 1);
    EXPECT_EQ(count_ops_of_type<op::v1::Multiply>(m), 1);
    EXPECT_EQ(count_ops_of_type<op::v0::Constant>(m), 2);
    ASSERT_EQ(m->get_results().size(), 1);
}

TEST_P(UnsupportedTypesTest, type_relaxed) {
    Shape shape_in{2, 4, 1};

    const auto& type = GetParam();
    auto cond = op::v0::Constant::create(element::boolean, shape_in, {1});
    auto param = std::make_shared<op::v0::Parameter>(type, shape_in);
    auto constant1 = op::v0::Constant::create(type, shape_in, {2});
    auto then_value = std::make_shared<op::v0::Concat>(OutputVector{param, constant1}, 2);
    auto constant2 = op::v0::Constant::create(type, shape_in, {3});
    auto else_value = std::make_shared<op::v3::Broadcast>(
        constant2,
        op::v0::Constant::create(element::u64, Shape{shape_in.size()}, Shape{shape_in[0], shape_in[1], 2}));
    auto select = make_shared<op::v1::Select>(cond, then_value, else_value);
    auto type_relaxed = make_shared<op::TypeRelaxed<op::v1::Select>>(*select,
                                                                     element::TypeVector{element::boolean},
                                                                     element::TypeVector{});
    auto m = make_shared<Model>(type_relaxed, ParameterVector{param});

    run_constant_folding(m);

    EXPECT_EQ(m->get_ops().size(), 7);
    EXPECT_EQ(count_ops_of_type<op::v1::Select>(m), 1);
    EXPECT_EQ(count_ops_of_type<op::v0::Constant>(m), 3);
    EXPECT_EQ(count_ops_of_type<op::v3::Broadcast>(m), 0);
    EXPECT_EQ(count_ops_of_type<op::v0::Concat>(m), 1);
    ASSERT_EQ(m->get_results().size(), 1);
}

TEST_P(UnsupportedTypesTest, random_uniform) {
    // Make sure that ConstantFolding with RandomUniform doesn't throw
    const auto& type = GetParam();
    auto shape = op::v0::Constant::create(element::i32, Shape{2}, {2, 3});
    auto min_val = op::v0::Constant::create(type, Shape{}, {-1});
    auto max_val = op::v0::Constant::create(type, Shape{}, {3});
    auto random = std::make_shared<op::v8::RandomUniform>(shape, min_val, max_val, type);
    auto m = make_shared<Model>(random, ParameterVector{});

    EXPECT_NO_THROW(run_constant_folding(m));

    EXPECT_EQ(m->get_ops().size(), 5);
    // RandomUniform is not constantfolded
    EXPECT_EQ(count_ops_of_type<op::v8::RandomUniform>(m), 1);
    EXPECT_EQ(count_ops_of_type<op::v0::Constant>(m), 3);
    ASSERT_EQ(m->get_results().size(), 1);
}

static std::string unsupported_types_test_case_name(const testing::TestParamInfo<element::Type>& info) {
    return info.param.get_type_name();
}

INSTANTIATE_TEST_SUITE_P(constant_folding,
                         UnsupportedTypesTest,
                         testing::ValuesIn(ov::util::unsupported_types()),
                         unsupported_types_test_case_name);
