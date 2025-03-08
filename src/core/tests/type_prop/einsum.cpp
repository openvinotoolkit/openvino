// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/einsum.hpp"

#include <gmock/gmock.h>

#include "common_test_utils/test_assertions.hpp"
#include "common_test_utils/type_prop.hpp"
#include "openvino/pass/graph_rewrite.hpp"

using namespace std;
using namespace ov;
using namespace testing;

namespace {
constexpr size_t exp_einsum_outputs_count = 1;
}

class TypePropEinsumTest : public TypePropOpTest<ov::op::v7::Einsum> {
protected:
    template <class ShapeContainer>
    OutputVector make_inputs(const element::Type dtype, ShapeContainer&& input_shapes) const {
        OutputVector inputs;
        inputs.reserve(input_shapes.size());
        for (auto&& shape : input_shapes) {
            inputs.push_back(std::make_shared<ov::op::v0::Parameter>(dtype, shape));
        }
        return inputs;
    }
};

TEST_F(TypePropEinsumTest, static_shape_dot_product) {
    const auto equation = std::string("i,i->");
    constexpr auto et = element::f32;
    const auto inputs = make_inputs(et, Shapes{{3}, {3}});
    const auto o = make_op(inputs, equation);

    EXPECT_EQ(o->get_equation(), equation);
    EXPECT_EQ(o->get_element_type(), et);
    EXPECT_EQ(o->get_output_size(), exp_einsum_outputs_count);
    EXPECT_EQ(o->get_shape(), Shape({}));
}

TEST_F(TypePropEinsumTest, static_shape_matmul) {
    const std::string equation = "ab,bc->ac";
    constexpr auto et = element::f32;
    const auto inputs = make_inputs(et, Shapes{{2, 3}, {3, 4}});
    const auto o = make_op(inputs, equation);

    EXPECT_EQ(o->get_equation(), equation);
    EXPECT_EQ(o->get_element_type(), et);
    EXPECT_EQ(o->get_output_size(), exp_einsum_outputs_count);
    EXPECT_EQ(o->get_shape(), Shape({2, 4}));
}

TEST_F(TypePropEinsumTest, static_shape_trace) {
    const std::string equation = "kii->k";
    constexpr auto et = element::f32;

    const auto input = make_shared<ov::op::v0::Parameter>(et, Shape{2, 3, 3});
    const auto o = make_op(OutputVector{input}, equation);

    EXPECT_EQ(o->get_equation(), equation);
    EXPECT_EQ(o->get_element_type(), et);
    EXPECT_EQ(o->get_output_size(), exp_einsum_outputs_count);
    EXPECT_EQ(o->get_output_partial_shape(0), Shape({2}));
}

TEST_F(TypePropEinsumTest, static_shape_diag_extraction) {
    const std::string equation = "kii->ki";
    constexpr auto et = element::f32;

    const auto input = make_shared<ov::op::v0::Parameter>(et, Shape{2, 3, 3});
    const auto o = make_op(OutputVector{input}, equation);

    EXPECT_EQ(o->get_equation(), equation);
    EXPECT_EQ(o->get_element_type(), et);
    EXPECT_EQ(o->get_output_size(), exp_einsum_outputs_count);
    EXPECT_EQ(o->get_output_partial_shape(0), Shape({2, 3}));
}

TEST_F(TypePropEinsumTest, static_shape_transpose) {
    const std::string equation = "ijk->kij";
    constexpr auto et = element::f32;

    const auto input = make_shared<ov::op::v0::Parameter>(et, Shape{1, 2, 3});
    const auto o = make_op(OutputVector{input}, equation);

    EXPECT_EQ(o->get_equation(), equation);
    EXPECT_EQ(o->get_element_type(), et);
    EXPECT_EQ(o->get_output_size(), exp_einsum_outputs_count);
    EXPECT_EQ(o->get_output_partial_shape(0), Shape({3, 1, 2}));
}

TEST_F(TypePropEinsumTest, static_shape_multi_matmul) {
    const std::string equation = "ab,bcd,bc->ca";
    constexpr auto et = element::i32;
    const auto inputs = make_inputs(et, Shapes{{2, 5}, {5, 3, 6}, {5, 3}});
    const auto o = make_op(inputs, equation);

    EXPECT_EQ(o->get_equation(), equation);
    EXPECT_EQ(o->get_element_type(), et);
    EXPECT_EQ(o->get_output_size(), exp_einsum_outputs_count);
    EXPECT_EQ(o->get_shape(), Shape({3, 2}));
}

TEST_F(TypePropEinsumTest, static_shape_ellipsis_one_input) {
    const std::string equation = "a...->...";
    constexpr auto et = element::f32;

    const auto input = make_shared<ov::op::v0::Parameter>(et, Shape{5, 3});
    const auto o = make_op(OutputVector{input}, equation);

    EXPECT_EQ(o->get_equation(), equation);
    EXPECT_EQ(o->get_element_type(), et);
    EXPECT_EQ(o->get_output_size(), exp_einsum_outputs_count);
    EXPECT_EQ(o->get_output_partial_shape(0), Shape({3}));
}

TEST_F(TypePropEinsumTest, static_shape_ellipsis_with_1d) {
    const std::string equation = "a...,...->a...";
    constexpr auto et = element::f32;
    const auto inputs = make_inputs(et, Shapes{{3, 5}, {1}});
    const auto o = make_op(inputs, equation);

    EXPECT_EQ(o->get_equation(), equation);
    EXPECT_EQ(o->get_element_type(), et);
    EXPECT_EQ(o->get_output_size(), exp_einsum_outputs_count);
    EXPECT_EQ(o->get_shape(), Shape({3, 5}));
}

TEST_F(TypePropEinsumTest, static_shape_ellipsis) {
    const std::string equation = "a...b,b...->a...";
    constexpr auto et = element::i32;
    const auto inputs = make_inputs(et, Shapes{{11, 1, 4, 3}, {3, 11, 7, 1}});
    const auto o = make_op(inputs, equation);

    EXPECT_EQ(o->get_equation(), equation);
    EXPECT_EQ(o->get_element_type(), et);
    EXPECT_EQ(o->get_output_size(), exp_einsum_outputs_count);
    EXPECT_EQ(o->get_shape(), Shape({11, 11, 7, 4}));
}

TEST_F(TypePropEinsumTest, static_shape_multi_input_broadcasting) {
    const std::string equation = "a...b,b...,aacd,,...dd->da...c";
    constexpr auto et = element::i32;
    const auto input_shapes = Shapes{
        {1, 5},
        {1, 1, 2, 1},
        {2, 2, 1, 4},
        {},
        {2, 1, 1},
    };
    const auto inputs = make_inputs(et, input_shapes);
    const auto o = make_op(inputs, equation);

    EXPECT_EQ(o->get_equation(), equation);
    EXPECT_EQ(o->get_element_type(), et);
    EXPECT_EQ(o->get_output_size(), exp_einsum_outputs_count);
    EXPECT_EQ(o->get_shape(), Shape({4, 2, 1, 2, 2, 1}));
}

// Dynamic shapes test also symbol propagation as each Einsum tests different equation

TEST_F(TypePropEinsumTest, dynamic_shape_dot_product) {
    constexpr auto et = element::f64;
    const auto equation = std::string("a,ab->ab");
    auto input_shapes = PartialShapes{{{2, 7}}, {{3, 10}, 3}};

    auto A = std::make_shared<ov::Symbol>(), B = std::make_shared<ov::Symbol>();
    input_shapes[0][0].set_symbol(A);
    input_shapes[1][1].set_symbol(B);

    const auto inputs = make_inputs(et, input_shapes);
    const auto o = make_op(inputs, equation);

    EXPECT_EQ(o->get_equation(), equation);
    EXPECT_EQ(o->get_element_type(), et);
    EXPECT_EQ(o->get_output_size(), exp_einsum_outputs_count);
    EXPECT_EQ(o->get_output_partial_shape(0), PartialShape({{3, 7}, 3}));
    EXPECT_THAT(get_shape_symbols(o->get_output_partial_shape(0)), ElementsAre(A, B));
}

TEST_F(TypePropEinsumTest, dynamic_shape_diag_extraction) {
    const std::string equation = "xyzxy->xyz";
    constexpr auto et = element::i32;
    auto input_shape = PartialShape{{2, 7}, {1, 5}, 4, {3, 5}, 3};
    auto symbols = set_shape_symbols(input_shape);

    const auto input = make_shared<ov::op::v0::Parameter>(et, input_shape);
    const auto o = make_op(OutputVector{input}, equation);

    EXPECT_EQ(o->get_equation(), equation);
    EXPECT_EQ(o->get_element_type(), et);
    EXPECT_EQ(o->get_output_size(), exp_einsum_outputs_count);
    EXPECT_EQ(o->get_output_partial_shape(0), PartialShape({{3, 5}, 3, 4}));
    EXPECT_THAT(get_shape_symbols(o->get_output_partial_shape(0)), ElementsAre(symbols[0], symbols[1], symbols[2]));
}

TEST_F(TypePropEinsumTest, dynamic_shape_ellipsis) {
    const std::string equation = "a...b,b...->a...";
    constexpr auto et = element::f32;
    auto input_shapes = PartialShapes{{11, 1, {3, 5}, 3}, {3, 11, 7, {1, 3}}};

    auto symbols_a = set_shape_symbols(input_shapes[0]);
    auto symbols_b = set_shape_symbols(input_shapes[1]);

    const auto inputs = make_inputs(et, input_shapes);
    const auto o = make_op(inputs, equation);

    EXPECT_EQ(o->get_equation(), equation);
    EXPECT_EQ(o->get_element_type(), et);
    EXPECT_EQ(o->get_output_size(), exp_einsum_outputs_count);
    EXPECT_EQ(o->get_output_partial_shape(0), PartialShape({11, 11, 7, {3, 5}}));
    EXPECT_THAT(get_shape_symbols(o->get_output_partial_shape(0)),
                ElementsAre(symbols_a[0], symbols_b[1], symbols_b[2], symbols_a[2]));
}

TEST_F(TypePropEinsumTest, implicit_mode_mixed_case_letters) {
    // the following equation is equivalent to "AbC->ACb"
    const std::string equation = "AbC";
    constexpr auto et = element::i32;
    auto input_shape = PartialShape{1, {2, 3}, {4, 5}};
    auto symbols = set_shape_symbols(input_shape);

    const auto input = make_shared<ov::op::v0::Parameter>(et, input_shape);
    const auto o = make_op(OutputVector{input}, equation);

    EXPECT_EQ(o->get_equation(), equation);
    EXPECT_EQ(o->get_element_type(), et);
    EXPECT_EQ(o->get_output_size(), exp_einsum_outputs_count);
    EXPECT_EQ(o->get_output_partial_shape(0), PartialShape({1, {4, 5}, {2, 3}}));
    EXPECT_THAT(get_shape_symbols(o->get_output_partial_shape(0)), ElementsAre(symbols[0], symbols[2], symbols[1]));
}

TEST_F(TypePropEinsumTest, implicit_mode_mixed_case_letters_ellipsis) {
    // the following equation is equivalent to "a...b,B...->...Bab"
    const std::string equation = "a...b,B...";
    constexpr auto et = element::f32;
    auto input_shapes = PartialShapes{{{3, 5}, 11, 1, 3}, {{1, 3}, 3, 1, 7}};
    auto symbols_a = set_shape_symbols(input_shapes[0]);
    auto symbols_b = set_shape_symbols(input_shapes[1]);

    const auto inputs = make_inputs(et, input_shapes);
    const auto o = make_op(inputs, equation);

    EXPECT_EQ(o->get_equation(), equation);
    EXPECT_EQ(o->get_element_type(), et);
    EXPECT_EQ(o->get_output_size(), exp_einsum_outputs_count);
    EXPECT_EQ(o->get_output_partial_shape(0), PartialShape({3, 11, 7, {1, 3}, {3, 5}, 3}));
    EXPECT_THAT(get_shape_symbols(o->get_output_partial_shape(0)),
                ElementsAre(symbols_b[1], symbols_a[1], symbols_b[3], symbols_b[0], symbols_a[0], symbols_a[3]));
}

TEST_F(TypePropEinsumTest, implicit_mode_repeated_symbols) {
    // the following equation is equivalent to "a...b,b...->...a"
    const std::string equation = "a...b,b...";
    constexpr auto et = element::f32;
    auto input_shapes = PartialShapes{{{3, 5}, 11, 1, 3}, {{1, 3}, 3, 1, 7}};
    auto symbols_a = set_shape_symbols(input_shapes[0]);
    auto symbols_b = set_shape_symbols(input_shapes[1]);

    const auto inputs = make_inputs(et, input_shapes);
    const auto o = make_op(inputs, equation);

    EXPECT_EQ(o->get_equation(), equation);
    EXPECT_EQ(o->get_element_type(), et);
    EXPECT_EQ(o->get_output_size(), exp_einsum_outputs_count);
    EXPECT_EQ(o->get_output_partial_shape(0), PartialShape({3, 11, 7, {3, 5}}));
    EXPECT_THAT(get_shape_symbols(o->get_output_partial_shape(0)),
                ElementsAre(symbols_b[1], symbols_a[1], symbols_b[3], symbols_a[0]));
}

TEST_F(TypePropEinsumTest, dynamic_shape_implict_mode_inner_prod) {
    // the following equation is equivalent to "i,i->"
    const std::string equation = "i,i";
    constexpr auto et = element::f16;

    auto input_shapes = PartialShapes{{11}, {{1, 20}}};
    auto symbols_a = set_shape_symbols(input_shapes[0]);
    auto symbols_b = set_shape_symbols(input_shapes[1]);

    const auto inputs = make_inputs(et, input_shapes);
    const auto o = make_op(inputs, equation);

    EXPECT_EQ(o->get_equation(), equation);
    EXPECT_EQ(o->get_element_type(), et);
    EXPECT_EQ(o->get_output_size(), exp_einsum_outputs_count);
    EXPECT_EQ(o->get_output_partial_shape(0), PartialShape({}));
    EXPECT_TRUE(get_shape_symbols(o->get_output_partial_shape(0)).empty());
}

TEST_F(TypePropEinsumTest, dynamic_rank_multi_matmul) {
    const std::string equation = "ab,bcd,bc->ca";
    constexpr auto et = element::i32;

    auto input_shapes = PartialShapes{{2, 5}, PartialShape::dynamic(), {5, 3}};
    auto symbols_a = set_shape_symbols(input_shapes[0]);
    auto symbols_b = set_shape_symbols(input_shapes[2]);

    const auto inputs = make_inputs(et, input_shapes);
    const auto o = make_op(inputs, equation);

    EXPECT_EQ(o->get_equation(), equation);
    EXPECT_EQ(o->get_element_type(), et);
    EXPECT_EQ(o->get_output_size(), exp_einsum_outputs_count);
    EXPECT_EQ(o->get_output_partial_shape(0), PartialShape({3, 2}));
    EXPECT_THAT(get_shape_symbols(o->get_output_partial_shape(0)), ElementsAre(symbols_b[1], symbols_a[0]));
}

TEST_F(TypePropEinsumTest, all_dynamic_rank_multi_matmul) {
    const std::string equation = "ab,bcd,bc->ca";
    constexpr auto et = element::i32;

    auto input_shapes = PartialShapes(3, PartialShape::dynamic());
    const auto inputs = make_inputs(et, input_shapes);
    const auto o = make_op(inputs, equation);

    EXPECT_EQ(o->get_equation(), equation);
    EXPECT_EQ(o->get_element_type(), et);
    EXPECT_EQ(o->get_output_size(), exp_einsum_outputs_count);
    EXPECT_EQ(o->get_output_partial_shape(0), PartialShape({-1, -1}));
    EXPECT_THAT(get_shape_symbols(o->get_output_partial_shape(0)), Each(nullptr));
}

TEST_F(TypePropEinsumTest, default_ctor) {
    const std::string equation = "ab,bcd,bc->ca";
    constexpr auto et = element::i32;

    auto input_shapes = PartialShapes(3, PartialShape::dynamic());
    const auto inputs = make_inputs(et, input_shapes);
    auto o = make_op();

    o->set_arguments(inputs);
    o->set_equation(equation);
    o->validate_and_infer_types();

    EXPECT_EQ(o->get_equation(), equation);
    EXPECT_EQ(o->get_element_type(), et);
    EXPECT_EQ(o->get_output_size(), exp_einsum_outputs_count);
    EXPECT_EQ(o->get_output_partial_shape(0), PartialShape({-1, -1}));
    EXPECT_THAT(get_shape_symbols(o->get_output_partial_shape(0)), Each(nullptr));
}

TEST_F(TypePropEinsumTest, incorrect_equation_subscript_number) {
    const std::string equation = "ab,bc,cd->ac";

    const auto input_shapes = PartialShapes{{2, 3}, {3, 4}};
    const auto inputs = make_inputs(element::f32, input_shapes);

    OV_EXPECT_THROW(auto o = make_op(inputs, equation),
                    NodeValidationFailure,
                    HasSubstr("Equation must contain a number of subscripts equal to a "
                              "number of Einsum inputs."));
}

TEST_F(TypePropEinsumTest, incorrect_equation_invalid_symbols) {
    const std::string equation = "a$,Bc->ac";

    const auto input_shapes = Shapes{{2, 3}, {3, 4}};
    const auto inputs = make_inputs(element::f32, input_shapes);

    OV_EXPECT_THROW(auto o = make_op(inputs, equation),
                    AssertFailure,
                    HasSubstr("Input subscript of Einsum equation must consist of either only alphabetic "
                              "letters or alphabetic letters with one ellipsis."));
}

TEST_F(TypePropEinsumTest, incorrect_equation_incompatible_repeated_label_shape) {
    const std::string equation = "aa->a";

    const auto input_shapes = Shapes{{1, 10}};
    const auto inputs = make_inputs(element::f32, input_shapes);

    OV_EXPECT_THROW(auto o = make_op(inputs, equation),
                    AssertFailure,
                    HasSubstr("Different input dimensions indicated by the repeated labels within single "
                              "input node for Einsum must be compatible."));
}

TEST_F(TypePropEinsumTest, incorrect_equation_not_broadcastable_label_shape) {
    const std::string equation = "ab,bc->ac";

    const auto input_shapes = Shapes{{2, 10}, {3, 4}};
    const auto inputs = make_inputs(element::f32, input_shapes);

    OV_EXPECT_THROW(auto o = make_op(inputs, equation),
                    AssertFailure,
                    HasSubstr("Different input dimensions indicated by the same labels "
                              "for Einsum must be broadcastable."));
}

TEST_F(TypePropEinsumTest, incorrect_equation_not_broadcastable_ellipsis_shapes) {
    const std::string equation = "a...b,b...->a...";

    const auto input_shapes = Shapes{{11, 1, 4, 3}, {3, 11, 7, 5}};
    const auto inputs = make_inputs(element::f32, input_shapes);

    OV_EXPECT_THROW(auto o = make_op(inputs, equation),
                    NodeValidationFailure,
                    HasSubstr("Input dimensions labeled with ellipsis for Einsum must be broadcastable."));
}

TEST_F(TypePropEinsumTest, missed_out_ellipsis) {
    const std::string equation = "a...b,b...->a";

    auto input_shapes = PartialShapes{{11, 1, 4, 3}, {3, 11, 7, 4}};
    auto symbols_a = set_shape_symbols(input_shapes[0]);
    auto symbols_b = set_shape_symbols(input_shapes[1]);
    const auto inputs = make_inputs(element::f32, input_shapes);
    const auto o = make_op(inputs, equation);
    EXPECT_EQ(o->get_equation(), equation);
    EXPECT_EQ(o->get_element_type(), element::f32);
    EXPECT_EQ(o->get_output_size(), exp_einsum_outputs_count);
    EXPECT_EQ(o->get_output_partial_shape(0), PartialShape({11}));
    EXPECT_THAT(get_shape_symbols(o->get_output_partial_shape(0)), ElementsAre(symbols_a[0]));
}

TEST_F(TypePropEinsumTest, missed_rhs_out_ellipsis) {
    const std::string equation = "a...b,b->a";

    auto input_shapes = PartialShapes{{11, 1, 4, 3}, {3}};
    auto symbols_a = set_shape_symbols(input_shapes[0]);
    auto symbols_b = set_shape_symbols(input_shapes[1]);
    const auto inputs = make_inputs(element::f32, input_shapes);
    const auto o = make_op(inputs, equation);
    EXPECT_EQ(o->get_equation(), equation);
    EXPECT_EQ(o->get_element_type(), element::f32);
    EXPECT_EQ(o->get_output_size(), exp_einsum_outputs_count);
    EXPECT_EQ(o->get_output_partial_shape(0), PartialShape({11}));
    EXPECT_THAT(get_shape_symbols(o->get_output_partial_shape(0)), ElementsAre(symbols_a[0]));
}

TEST_F(TypePropEinsumTest, missed_lhs_out_ellipsis) {
    const std::string equation = "ab,b...->a";

    auto input_shapes = PartialShapes{{11, 3}, {3, 11, 7, 4}};
    auto symbols_a = set_shape_symbols(input_shapes[0]);
    auto symbols_b = set_shape_symbols(input_shapes[1]);
    const auto inputs = make_inputs(element::f32, input_shapes);
    const auto o = make_op(inputs, equation);
    EXPECT_EQ(o->get_equation(), equation);
    EXPECT_EQ(o->get_element_type(), element::f32);
    EXPECT_EQ(o->get_output_size(), exp_einsum_outputs_count);
    EXPECT_EQ(o->get_output_partial_shape(0), PartialShape({11}));
    EXPECT_THAT(get_shape_symbols(o->get_output_partial_shape(0)), ElementsAre(symbols_a[0]));
}

TEST_F(TypePropEinsumTest, missed_rhs_ellipsis) {
    const std::string equation = "a...b,b->a...";

    auto input_shapes = PartialShapes{{11, 1, 4, 3}, {3}};
    auto symbols_a = set_shape_symbols(input_shapes[0]);
    auto symbols_b = set_shape_symbols(input_shapes[1]);
    const auto inputs = make_inputs(element::f32, input_shapes);
    const auto o = make_op(inputs, equation);
    EXPECT_EQ(o->get_equation(), equation);
    EXPECT_EQ(o->get_element_type(), element::f32);
    EXPECT_EQ(o->get_output_size(), exp_einsum_outputs_count);
    EXPECT_EQ(o->get_output_partial_shape(0), PartialShape({11, 1, 4}));
    EXPECT_THAT(get_shape_symbols(o->get_output_partial_shape(0)),
                ElementsAre(symbols_a[0], symbols_a[1], symbols_a[2]));
}

TEST_F(TypePropEinsumTest, missed_lhs_ellipsis) {
    const std::string equation = "ab,b...->a...";

    auto input_shapes = PartialShapes{{11, 3}, {3, 11, 7, 4}};
    auto symbols_a = set_shape_symbols(input_shapes[0]);
    auto symbols_b = set_shape_symbols(input_shapes[1]);
    const auto inputs = make_inputs(element::f32, input_shapes);
    const auto o = make_op(inputs, equation);
    EXPECT_EQ(o->get_equation(), equation);
    EXPECT_EQ(o->get_element_type(), element::f32);
    EXPECT_EQ(o->get_output_size(), exp_einsum_outputs_count);
    EXPECT_EQ(o->get_output_partial_shape(0), PartialShape({11, 11, 7, 4}));
    EXPECT_THAT(get_shape_symbols(o->get_output_partial_shape(0)),
                ElementsAre(symbols_a[0], symbols_b[1], symbols_b[2], symbols_b[3]));
}

TEST_F(TypePropEinsumTest, missed_rhs_ellipsis_implicit) {
    const std::string equation = "a...b,b";

    auto input_shapes = PartialShapes{{11, 1, 4, 3}, {3}};
    auto symbols_a = set_shape_symbols(input_shapes[0]);
    auto symbols_b = set_shape_symbols(input_shapes[1]);
    const auto inputs = make_inputs(element::f32, input_shapes);
    const auto o = make_op(inputs, equation);
    EXPECT_EQ(o->get_equation(), equation);
    EXPECT_EQ(o->get_element_type(), element::f32);
    EXPECT_EQ(o->get_output_size(), exp_einsum_outputs_count);
    EXPECT_EQ(o->get_output_partial_shape(0), PartialShape({1, 4, 11}));
    EXPECT_THAT(get_shape_symbols(o->get_output_partial_shape(0)),
                ElementsAre(symbols_a[1], symbols_a[2], symbols_a[0]));
}

TEST_F(TypePropEinsumTest, missed_lhs_ellipsis_implicit) {
    const std::string equation = "ab,b...";

    auto input_shapes = PartialShapes{{11, 3}, {3, 11, 7, 4}};
    auto symbols_a = set_shape_symbols(input_shapes[0]);
    auto symbols_b = set_shape_symbols(input_shapes[1]);
    const auto inputs = make_inputs(element::f32, input_shapes);
    const auto o = make_op(inputs, equation);
    EXPECT_EQ(o->get_equation(), equation);
    EXPECT_EQ(o->get_element_type(), element::f32);
    EXPECT_EQ(o->get_output_size(), exp_einsum_outputs_count);
    EXPECT_EQ(o->get_output_partial_shape(0), PartialShape({11, 7, 4, 11}));
    EXPECT_THAT(get_shape_symbols(o->get_output_partial_shape(0)),
                ElementsAre(symbols_b[1], symbols_b[2], symbols_b[3], symbols_a[0]));
}

TEST_F(TypePropEinsumTest, all_dynamic_rank_ellipsis) {
    const std::string equation = "a...b,b...->...a";
    constexpr auto et = element::i32;

    auto input_shapes = PartialShapes(2, PartialShape::dynamic());
    auto symbols_a = set_shape_symbols(input_shapes[0]);
    auto symbols_b = set_shape_symbols(input_shapes[1]);
    const auto inputs = make_inputs(et, input_shapes);
    const auto o = make_op(inputs, equation);

    EXPECT_EQ(o->get_equation(), equation);
    EXPECT_EQ(o->get_element_type(), et);
    EXPECT_EQ(o->get_output_size(), exp_einsum_outputs_count);
    EXPECT_EQ(o->get_output_partial_shape(0), PartialShape::dynamic());
    EXPECT_THAT(get_shape_symbols(o->get_output_partial_shape(0)), Each(nullptr));
}

TEST_F(TypePropEinsumTest, lhs_dynamic_rank_ellipsis) {
    const std::string equation = "a...b,b...->...a";
    constexpr auto et = element::i32;

    auto input_shapes = PartialShapes{PartialShape::dynamic(), {3, 11, 7, 4}};
    auto symbols_a = set_shape_symbols(input_shapes[0]);
    auto symbols_b = set_shape_symbols(input_shapes[1]);
    const auto inputs = make_inputs(et, input_shapes);
    const auto o = make_op(inputs, equation);

    EXPECT_EQ(o->get_equation(), equation);
    EXPECT_EQ(o->get_element_type(), et);
    EXPECT_EQ(o->get_output_size(), exp_einsum_outputs_count);
    EXPECT_EQ(o->get_output_partial_shape(0), PartialShape::dynamic());
    EXPECT_THAT(get_shape_symbols(o->get_output_partial_shape(0)), Each(nullptr));
}

TEST_F(TypePropEinsumTest, rhs_dynamic_rank_ellipsis) {
    const std::string equation = "a...b,b...->...a";
    constexpr auto et = element::i32;

    auto input_shapes = PartialShapes{{3, 11, 7, 4}, PartialShape::dynamic()};
    auto symbols_a = set_shape_symbols(input_shapes[0]);
    auto symbols_b = set_shape_symbols(input_shapes[1]);
    const auto inputs = make_inputs(et, input_shapes);
    const auto o = make_op(inputs, equation);

    EXPECT_EQ(o->get_equation(), equation);
    EXPECT_EQ(o->get_element_type(), et);
    EXPECT_EQ(o->get_output_size(), exp_einsum_outputs_count);
    EXPECT_EQ(o->get_output_partial_shape(0), PartialShape::dynamic());
    EXPECT_THAT(get_shape_symbols(o->get_output_partial_shape(0)), Each(nullptr));
}

TEST_F(TypePropEinsumTest, lhs_dynamic_rank_ellipsis_reduced_out_ellipsis) {
    const std::string equation = "a...b,b...->a";
    constexpr auto et = element::i32;

    auto input_shapes = PartialShapes{PartialShape::dynamic(), {3, 11, 7, 4}};
    auto symbols_a = set_shape_symbols(input_shapes[0]);
    auto symbols_b = set_shape_symbols(input_shapes[1]);
    const auto inputs = make_inputs(et, input_shapes);
    const auto o = make_op(inputs, equation);

    EXPECT_EQ(o->get_equation(), equation);
    EXPECT_EQ(o->get_element_type(), et);
    EXPECT_EQ(o->get_output_size(), exp_einsum_outputs_count);
    EXPECT_EQ(o->get_output_partial_shape(0), PartialShape({ov::Dimension::dynamic()}));
    EXPECT_THAT(get_shape_symbols(o->get_output_partial_shape(0)), Each(nullptr));
}

TEST_F(TypePropEinsumTest, rhs_dynamic_rank_ellipsis_reduced_out_ellipsis) {
    const std::string equation = "a...b,b...->a";
    constexpr auto et = element::i32;

    auto input_shapes = PartialShapes{{3, 11, 7, 4}, PartialShape::dynamic()};
    auto symbols_a = set_shape_symbols(input_shapes[0]);
    auto symbols_b = set_shape_symbols(input_shapes[1]);
    const auto inputs = make_inputs(et, input_shapes);
    const auto o = make_op(inputs, equation);

    EXPECT_EQ(o->get_equation(), equation);
    EXPECT_EQ(o->get_element_type(), et);
    EXPECT_EQ(o->get_output_size(), exp_einsum_outputs_count);
    EXPECT_EQ(o->get_output_partial_shape(0), PartialShape({3}));
    EXPECT_THAT(get_shape_symbols(o->get_output_partial_shape(0)), ElementsAre(symbols_a[0]));
}

TEST_F(TypePropEinsumTest, broadcasting_same_symbol_common) {
    const std::string equation = "ab,ba->b";
    constexpr auto et = element::i32;

    auto input_shapes = PartialShapes{{7, 5}, {1, 7}};
    auto symbols_a = set_shape_symbols(input_shapes[0]);
    auto symbols_b = set_shape_symbols(input_shapes[1]);
    const auto inputs = make_inputs(et, input_shapes);
    const auto o = make_op(inputs, equation);

    EXPECT_EQ(o->get_equation(), equation);
    EXPECT_EQ(o->get_element_type(), et);
    EXPECT_EQ(o->get_output_size(), exp_einsum_outputs_count);
    EXPECT_EQ(o->get_output_partial_shape(0), PartialShape({5}));
    EXPECT_THAT(get_shape_symbols(o->get_output_partial_shape(0)), ElementsAre(symbols_a[1]));
}

TEST_F(TypePropEinsumTest, broadcasting_same_symbol_reduced) {
    const std::string equation = "ab,ba->b";
    constexpr auto et = element::i32;

    auto input_shapes = PartialShapes{{1, 5}, {5, 7}};
    auto symbols_a = set_shape_symbols(input_shapes[0]);
    auto symbols_b = set_shape_symbols(input_shapes[1]);
    const auto inputs = make_inputs(et, input_shapes);
    const auto o = make_op(inputs, equation);

    EXPECT_EQ(o->get_equation(), equation);
    EXPECT_EQ(o->get_element_type(), et);
    EXPECT_EQ(o->get_output_size(), exp_einsum_outputs_count);
    EXPECT_EQ(o->get_output_partial_shape(0), PartialShape({5}));
    EXPECT_THAT(get_shape_symbols(o->get_output_partial_shape(0)), ElementsAre(symbols_a[1]));
}

TEST_F(TypePropEinsumTest, broadcasting_same_symbol) {
    const std::string equation = "ab,ba->b";
    constexpr auto et = element::i32;

    auto input_shapes = PartialShapes{{7, 1}, {5, 1}};
    auto symbols_a = set_shape_symbols(input_shapes[0]);
    auto symbols_b = set_shape_symbols(input_shapes[1]);
    const auto inputs = make_inputs(et, input_shapes);
    const auto o = make_op(inputs, equation);

    EXPECT_EQ(o->get_equation(), equation);
    EXPECT_EQ(o->get_element_type(), et);
    EXPECT_EQ(o->get_output_size(), exp_einsum_outputs_count);
    EXPECT_EQ(o->get_output_partial_shape(0), PartialShape({5}));
    EXPECT_THAT(get_shape_symbols(o->get_output_partial_shape(0)), ElementsAre(symbols_b[0]));
}

TEST_F(TypePropEinsumTest, ellipsis_no_dimension) {
    const std::string equation = "...ab,ba...->b...";
    constexpr auto et = element::i32;

    auto input_shapes = PartialShapes{{5, 1}, {5, 5}};
    auto symbols_a = set_shape_symbols(input_shapes[0]);
    auto symbols_b = set_shape_symbols(input_shapes[1]);
    const auto inputs = make_inputs(et, input_shapes);
    const auto o = make_op(inputs, equation);

    EXPECT_EQ(o->get_equation(), equation);
    EXPECT_EQ(o->get_element_type(), et);
    EXPECT_EQ(o->get_output_size(), exp_einsum_outputs_count);
    EXPECT_EQ(o->get_output_partial_shape(0), PartialShape({5}));
    EXPECT_THAT(get_shape_symbols(o->get_output_partial_shape(0)), ElementsAre(symbols_b[0]));
}

TEST_F(TypePropEinsumTest, ellipsis_dynamic_shape) {
    const std::string equation = "...ab,ba...->b...";
    constexpr auto et = element::i32;

    auto input_shapes = PartialShapes{{-1, 57, 5, 5}, {5, 5}};
    auto symbols_a = set_shape_symbols(input_shapes[0]);
    auto symbols_b = set_shape_symbols(input_shapes[1]);
    const auto inputs = make_inputs(et, input_shapes);
    const auto o = make_op(inputs, equation);

    EXPECT_EQ(o->get_equation(), equation);
    EXPECT_EQ(o->get_element_type(), et);
    EXPECT_EQ(o->get_output_size(), exp_einsum_outputs_count);
    EXPECT_EQ(o->get_output_partial_shape(0), PartialShape({5, -1, 57}));
    EXPECT_THAT(get_shape_symbols(o->get_output_partial_shape(0)),
                ElementsAre(symbols_a[3], symbols_a[0], symbols_a[1]));
}

TEST_F(TypePropEinsumTest, dynamic_shape_repeated_label_2_input_broadcasting) {
    const std::string equation = "aabb,aabb->ab";
    constexpr auto et = element::i32;
    auto input_shapes = PartialShapes{
        {{1, 2}, {1, 4}, {1, 5}, {2, 4}},
        {{1, 3}, {2, 4}, {1, 3}, {3, 4}},
    };
    auto symbols_0 = set_shape_symbols(input_shapes[0]);
    auto symbols_1 = set_shape_symbols(input_shapes[1]);
    const auto inputs = make_inputs(et, input_shapes);
    const auto o = make_op(inputs, equation);

    EXPECT_EQ(o->get_equation(), equation);
    EXPECT_EQ(o->get_element_type(), et);
    EXPECT_EQ(o->get_output_size(), exp_einsum_outputs_count);
    EXPECT_EQ(o->get_output_partial_shape(0), PartialShape({{2, 3}, {3}}));
    EXPECT_THAT(get_shape_symbols(o->get_output_partial_shape(0)), ElementsAre(symbols_1[0], symbols_0[2]));
}

TEST_F(TypePropEinsumTest, dynamic_shape_multi_input_broadcasting) {
    const std::string equation = "a...b,b...,aacd,,...dd->da...c";
    constexpr auto et = element::i32;
    auto input_shapes = PartialShapes{
        {{1, 2}, {5, 6}},
        {{1, 3}, 1, {3, 4}, 1},
        {{1, 2}, {2, 4}, 1, 4},
        {},
        {2, 1, 1},
    };
    auto symbols_0 = set_shape_symbols(input_shapes[0]);
    auto symbols_1 = set_shape_symbols(input_shapes[1]);
    auto symbols_2 = set_shape_symbols(input_shapes[2]);
    auto symbols_3 = set_shape_symbols(input_shapes[3]);
    auto symbols_4 = set_shape_symbols(input_shapes[4]);
    const auto inputs = make_inputs(et, input_shapes);
    const auto o = make_op(inputs, equation);

    EXPECT_EQ(o->get_equation(), equation);
    EXPECT_EQ(o->get_element_type(), et);
    EXPECT_EQ(o->get_output_size(), exp_einsum_outputs_count);
    EXPECT_EQ(o->get_output_partial_shape(0), PartialShape({4, 2, 1, {3, 4}, 2, 1}));
    EXPECT_THAT(get_shape_symbols(o->get_output_partial_shape(0)),
                ElementsAre(symbols_2[3], symbols_2[0], symbols_1[1], symbols_1[2], symbols_4[0], symbols_2[2]));
}

TEST_F(TypePropEinsumTest, input_rank_incompatible_with_equation) {
    const std::string equation = "ab,bc->ac";

    const auto input_shapes = Shapes{{2, 2, 10}, {3, 4}};
    const auto inputs = make_inputs(element::f32, input_shapes);

    OV_EXPECT_THROW(auto o = make_op(inputs, equation),
                    AssertFailure,
                    HasSubstr("Input rank must be greater or equal to a number of labels in the "
                              "corresponding input subscript."));
}

TEST_F(TypePropEinsumTest, input_rank_incompatible_with_equation_single_input) {
    const std::string equation = "ab->ba";

    const auto input_shapes = Shapes{{3, 5, 7}};
    const auto inputs = make_inputs(element::f32, input_shapes);

    OV_EXPECT_THROW(auto o = make_op(inputs, equation),
                    AssertFailure,
                    HasSubstr("Input rank must be greater or equal to a number of labels in the "
                              "corresponding input subscript."));
}
