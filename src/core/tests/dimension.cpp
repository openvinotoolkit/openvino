// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/type_prop.hpp"
#include "gtest/gtest.h"
#include "openvino/core/partial_shape.hpp"
#include "openvino/core/symbol.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/parameter.hpp"

using namespace std;
using namespace ov;

TEST(dimension, broadcast_merge_static_1_and_10) {
    Dimension result;
    Dimension one(1), ten(10);
    bool success = Dimension::broadcast_merge(result, one, ten);

    EXPECT_TRUE(success);
    EXPECT_EQ(result, ten);
}

TEST(dimension, broadcast_merge_static_1_5_and_10_15) {
    Dimension result;
    Dimension one(1, 5), ten(10, 15);
    bool success = Dimension::broadcast_merge(result, one, ten);

    EXPECT_TRUE(success);
    EXPECT_EQ(result, ten);
}

TEST(dimension, broadcast_merge_static_1_12_and_10_15) {
    Dimension result;
    Dimension one(1, 12), ten(10, 15);
    bool success = Dimension::broadcast_merge(result, one, ten);

    EXPECT_TRUE(success);
    EXPECT_EQ(result, ten);
}

TEST(dimension, broadcast_merge_static_7_12_and_10_15) {
    Dimension result;
    Dimension one(7, 12), ten(10, 15);
    bool success = Dimension::broadcast_merge(result, one, ten);

    EXPECT_TRUE(success);
    EXPECT_EQ(result, Dimension(10, 12));
}

TEST(dimension, broadcast_merge_static_0_12_and_1_15) {
    Dimension result;
    Dimension one(0, 12), ten(1, 15);
    bool success = Dimension::broadcast_merge(result, one, ten);

    EXPECT_TRUE(success);
    EXPECT_EQ(result, Dimension(0, 15));
}

TEST(dimension, dimension_mul_operator_ordinary_intervals) {
    Dimension interval_1(0, 10);
    Dimension interval_2(2, 100);
    Dimension ref_value(0, 1000);
    EXPECT_EQ(ref_value, interval_1 * interval_2);
}

TEST(dimension, dimension_mul_operator_1) {
    Dimension fully_dynamic_dim(-1);
    Dimension two(2);
    Dimension ref_value(-1);
    EXPECT_EQ(ref_value, fully_dynamic_dim * two);
}

TEST(dimension, dimension_mul_operator_2) {
    // overflow happens and clip_times keeps result within in64 limits
    // (Interval::s_max - 1) * 2 = 9223372036854775806 * 2 = 18446744073709551612
    // arithmetical result does not fit into int64, is clipped into int64_max
    Dimension large_interval(2, Interval::s_max - 1);
    Dimension two(2);
    Dimension ref_value(4, Interval::s_max);
    EXPECT_EQ(ref_value, large_interval * two);
}

TEST(dimension, dimension_mul_operator_3) {
    // no overflow
    // (int64_max / 2) * 2= 4611686018427387903 * 2 = 9223372036854775806 = int64_max - 1
    Dimension large_interval(2, ov::Interval::s_max / 2);
    Dimension two(2);
    Dimension ref_value(4, ov::Interval::s_max - 1);
    EXPECT_EQ(ref_value, large_interval * two);
}

TEST(dimension, dimension_mul_operator_4) {
    // overflow happens and clip_times keeps result within in64 limits
    // (int64_max / 2 + 1) * 2 = 4611686018427387904 * 2 = 9223372036854775808 = int64_max + 1
    // 9223372036854775808 does not fit into int64, is clipped into int64_max
    Dimension large_interval(2, ov::Interval::s_max / 2 + 1);
    Dimension two(2);
    Dimension ref_value(4, ov::Interval::s_max);
    EXPECT_EQ(ref_value, large_interval * two);
}

TEST(dimension, dimension_mul_operator_5) {
    // (int64_max / 3 + 2) = 3074457345618258604 * 3 = 9223372036854775812 = int64_max + 5
    // overflow happens and clip_times keeps result within in64 limits
    // 9223372036854775812 does not fit into int64, is clipped into int64_max
    Dimension large_interval(2, ov::Interval::s_max / 3 + 2);
    Dimension three(3);
    Dimension ref_value(6, ov::Interval::s_max);
    EXPECT_EQ(ref_value, large_interval * three);
}

TEST(dimension, division_of_static_dims_twenty_three_div_three_eq_seven) {
    Dimension twenty_three(23);
    Dimension::value_type three(3);
    Dimension empty(8, 7);
    EXPECT_EQ(empty, twenty_three / three);
}

TEST(dimension, division_of_static_dims) {
    Dimension seven(7);
    Dimension::value_type four(4);
    Dimension empty(2, 1);
    EXPECT_EQ(seven / four, empty);
}

TEST(dimension, dimension_equality) {
    // labeling dimensions
    PartialShape dimensions = PartialShape::dynamic(5);  // A, B, C, D, E
    auto symbols = set_shape_symbols(dimensions);

    // checking symbols are unique
    for (const auto& dimension : dimensions)
        EXPECT_NE(dimension.get_symbol(), nullptr);

    for (const auto& lhs : dimensions) {
        for (const auto& rhs : dimensions) {
            if (&lhs == &rhs)
                continue;
            EXPECT_NE(lhs.get_symbol(), rhs.get_symbol());
            EXPECT_FALSE(ov::symbol::are_equal(lhs.get_symbol(), rhs.get_symbol()));
        }
    }

    ov::symbol::set_equal(dimensions[0].get_symbol(), dimensions[1].get_symbol());  // A == B
    ov::symbol::set_equal(dimensions[3].get_symbol(), dimensions[4].get_symbol());  // D == E
    ov::symbol::set_equal(dimensions[2].get_symbol(), dimensions[3].get_symbol());  // C == D
    ov::symbol::set_equal(dimensions[1].get_symbol(), dimensions[2].get_symbol());  // B == C

    // expected to see A == B == C == D == E
    for (const auto& lhs : dimensions)
        for (const auto& rhs : dimensions)
            EXPECT_TRUE(ov::symbol::are_equal(lhs.get_symbol(), rhs.get_symbol()));

    // clear up all the tracking info
    for (auto& dimension : dimensions)
        dimension.set_symbol(nullptr);

    // checking labels are nullified
    for (const auto& dimension : dimensions)
        EXPECT_EQ(dimension.get_symbol(), nullptr);
}

TEST(dimension, dimension_symbolic_equality) {
    auto A = std::make_shared<ov::Symbol>(), B = std::make_shared<ov::Symbol>();
    auto C = std::make_shared<ov::Symbol>(), D = std::make_shared<ov::Symbol>();
    ov::symbol::set_equal(A, B);
    ov::symbol::set_equal(D, C);
    ov::symbol::set_equal(A, D);
    EXPECT_TRUE(ov::symbol::are_equal(B, C));
}

// --- Compound symbol tests ---

TEST(symbol, leaf_properties) {
    auto s = std::make_shared<ov::Symbol>();
    EXPECT_TRUE(s->is_leaf());
    EXPECT_FALSE(s->is_compound());
    EXPECT_EQ(s->get_kind(), ov::SymbolKind::LEAF);
    EXPECT_EQ(s->get_lhs(), nullptr);
    EXPECT_EQ(s->get_rhs(), nullptr);
}

TEST(symbol, add_creates_compound) {
    auto a = std::make_shared<ov::Symbol>();
    auto b = std::make_shared<ov::Symbol>();
    auto c = ov::symbol::add(a, b);

    ASSERT_NE(c, nullptr);
    EXPECT_TRUE(c->is_compound());
    EXPECT_FALSE(c->is_leaf());
    EXPECT_EQ(c->get_kind(), ov::SymbolKind::ADD);
    EXPECT_EQ(c->get_lhs(), a);
    EXPECT_EQ(c->get_rhs(), b);
}

TEST(symbol, add_null_identity) {
    auto a = std::make_shared<ov::Symbol>();

    // null + a = a
    EXPECT_EQ(ov::symbol::add(nullptr, a), a);
    // a + null = a
    EXPECT_EQ(ov::symbol::add(a, nullptr), a);
    // null + null = null
    EXPECT_EQ(ov::symbol::add(nullptr, nullptr), nullptr);
}

TEST(symbol, structural_equality_same_operands) {
    auto a = std::make_shared<ov::Symbol>();
    auto b = std::make_shared<ov::Symbol>();
    auto c1 = ov::symbol::add(a, b);
    auto c2 = ov::symbol::add(a, b);

    EXPECT_TRUE(ov::symbol::structurally_equal(c1, c2));
    EXPECT_TRUE(ov::symbol::are_equal(c1, c2));
}

TEST(symbol, structural_equality_commutativity) {
    auto a = std::make_shared<ov::Symbol>();
    auto b = std::make_shared<ov::Symbol>();
    auto ab = ov::symbol::add(a, b);
    auto ba = ov::symbol::add(b, a);

    // ADD is commutative: A+B == B+A
    EXPECT_TRUE(ov::symbol::structurally_equal(ab, ba));
    EXPECT_TRUE(ov::symbol::are_equal(ab, ba));
}

TEST(symbol, structural_equality_with_union_find) {
    auto a = std::make_shared<ov::Symbol>();
    auto b = std::make_shared<ov::Symbol>();
    auto c = std::make_shared<ov::Symbol>();
    auto d = std::make_shared<ov::Symbol>();

    ov::symbol::set_equal(a, c);  // a == c via union-find
    ov::symbol::set_equal(b, d);  // b == d via union-find

    auto ab = ov::symbol::add(a, b);
    auto cd = ov::symbol::add(c, d);

    // (a+b) structurally equals (c+d) because a==c and b==d
    EXPECT_TRUE(ov::symbol::structurally_equal(ab, cd));
}

TEST(symbol, structural_inequality_different_operands) {
    auto a = std::make_shared<ov::Symbol>();
    auto b = std::make_shared<ov::Symbol>();
    auto c = std::make_shared<ov::Symbol>();
    auto ab = ov::symbol::add(a, b);
    auto ac = ov::symbol::add(a, c);

    EXPECT_FALSE(ov::symbol::structurally_equal(ab, ac));
    EXPECT_FALSE(ov::symbol::are_equal(ab, ac));
}

TEST(symbol, set_equal_noop_for_compound) {
    auto a = std::make_shared<ov::Symbol>();
    auto b = std::make_shared<ov::Symbol>();
    auto compound = ov::symbol::add(a, b);
    auto leaf = std::make_shared<ov::Symbol>();

    // set_equal should silently do nothing when either operand is compound
    ov::symbol::set_equal(compound, leaf);
    EXPECT_FALSE(ov::symbol::are_equal(compound, leaf));
}

TEST(symbol, ancestor_of_compound_returns_self) {
    auto a = std::make_shared<ov::Symbol>();
    auto b = std::make_shared<ov::Symbol>();
    auto c = ov::symbol::add(a, b);

    EXPECT_EQ(ov::symbol::ancestor_of(c), c);
}

TEST(symbol, structural_equality_null_handling) {
    auto a = std::make_shared<ov::Symbol>();
    EXPECT_FALSE(ov::symbol::structurally_equal(nullptr, a));
    EXPECT_FALSE(ov::symbol::structurally_equal(a, nullptr));
    EXPECT_FALSE(ov::symbol::structurally_equal(nullptr, nullptr));
}

TEST(symbol, nested_compound_equality) {
    auto a = std::make_shared<ov::Symbol>();
    auto b = std::make_shared<ov::Symbol>();
    auto c = std::make_shared<ov::Symbol>();

    auto ab = ov::symbol::add(a, b);
    auto abc1 = ov::symbol::add(ab, c);
    auto abc2 = ov::symbol::add(ov::symbol::add(a, b), c);

    EXPECT_TRUE(ov::symbol::structurally_equal(abc1, abc2));
}

TEST(dimension, addition_propagates_compound_symbol) {
    ov::Dimension d1(3);
    ov::Dimension d2(5);

    auto s1 = std::make_shared<ov::Symbol>();
    auto s2 = std::make_shared<ov::Symbol>();
    d1.set_symbol(s1);
    d2.set_symbol(s2);

    auto result = d1 + d2;
    EXPECT_EQ(result.get_length(), 8);
    ASSERT_NE(result.get_symbol(), nullptr);
    EXPECT_TRUE(result.get_symbol()->is_compound());
    EXPECT_EQ(result.get_symbol()->get_kind(), ov::SymbolKind::ADD);
    EXPECT_EQ(result.get_symbol()->get_lhs(), s1);
    EXPECT_EQ(result.get_symbol()->get_rhs(), s2);
}

TEST(dimension, addition_one_symbol_returns_it) {
    ov::Dimension d1(3);
    ov::Dimension d2(5);

    auto s1 = std::make_shared<ov::Symbol>();
    d1.set_symbol(s1);
    // d2 has no symbol

    auto result = d1 + d2;
    EXPECT_EQ(result.get_length(), 8);
    // symbol::add(s1, nullptr) returns s1 directly
    EXPECT_EQ(result.get_symbol(), s1);
}

TEST(dimension, addition_no_symbols_no_symbol) {
    ov::Dimension d1(3);
    ov::Dimension d2(5);

    auto result = d1 + d2;
    EXPECT_EQ(result.get_length(), 8);
    EXPECT_EQ(result.get_symbol(), nullptr);
}

TEST(symbol, add_op_evaluate_symbol) {
    // Create two 1D parameters with shape {3}
    auto param0 = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{3});
    auto param1 = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{3});

    // Set up value symbols on both inputs via ShapeOf-like mechanism
    auto s0 = std::make_shared<ov::Symbol>();
    auto s1 = std::make_shared<ov::Symbol>();
    auto s2 = std::make_shared<ov::Symbol>();
    auto s3 = std::make_shared<ov::Symbol>();
    auto s4 = std::make_shared<ov::Symbol>();
    auto s5 = std::make_shared<ov::Symbol>();

    param0->get_output_tensor(0).set_value_symbol({s0, s1, s2});
    param1->get_output_tensor(0).set_value_symbol({s3, s4, s5});

    auto add_node = std::make_shared<ov::op::v1::Add>(param0, param1);
    add_node->validate_and_infer_types();

    ov::TensorSymbolVector output_symbols;
    ASSERT_TRUE(add_node->evaluate_symbol(output_symbols));
    ASSERT_EQ(output_symbols.size(), 1u);
    ASSERT_EQ(output_symbols[0].size(), 3u);

    // Each output symbol should be compound ADD of corresponding inputs
    for (size_t i = 0; i < 3; ++i) {
        ASSERT_NE(output_symbols[0][i], nullptr);
        EXPECT_TRUE(output_symbols[0][i]->is_compound());
        EXPECT_EQ(output_symbols[0][i]->get_kind(), ov::SymbolKind::ADD);
    }
    // Verify specific operand linkage
    EXPECT_EQ(output_symbols[0][0]->get_lhs(), s0);
    EXPECT_EQ(output_symbols[0][0]->get_rhs(), s3);
    EXPECT_EQ(output_symbols[0][1]->get_lhs(), s1);
    EXPECT_EQ(output_symbols[0][1]->get_rhs(), s4);
    EXPECT_EQ(output_symbols[0][2]->get_lhs(), s2);
    EXPECT_EQ(output_symbols[0][2]->get_rhs(), s5);
}

TEST(symbol, add_op_evaluate_symbol_one_input_no_symbols) {
    auto param0 = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{2});
    auto param1 = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{2});

    auto s0 = std::make_shared<ov::Symbol>();
    auto s1 = std::make_shared<ov::Symbol>();
    param0->get_output_tensor(0).set_value_symbol({s0, s1});
    // param1 has no value symbols

    auto add_node = std::make_shared<ov::op::v1::Add>(param0, param1);
    add_node->validate_and_infer_types();

    ov::TensorSymbolVector output_symbols;
    ASSERT_TRUE(add_node->evaluate_symbol(output_symbols));
    ASSERT_EQ(output_symbols.size(), 1u);
    ASSERT_EQ(output_symbols[0].size(), 2u);

    // symbol::add(s, nullptr) returns s directly (identity)
    EXPECT_EQ(output_symbols[0][0], s0);
    EXPECT_EQ(output_symbols[0][1], s1);
}
