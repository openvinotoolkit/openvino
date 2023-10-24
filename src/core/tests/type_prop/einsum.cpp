// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/einsum.hpp"

#include <gmock/gmock.h>

#include "common_test_utils/test_assertions.hpp"
#include "common_test_utils/type_prop.hpp"
#include "openvino/core/dimension_tracker.hpp"
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
    EXPECT_EQ(o->get_output_partial_shape(0).to_shape(), Shape({}));
}

TEST_F(TypePropEinsumTest, static_shape_matmul) {
    const std::string equation = "ab,bc->ac";
    constexpr auto et = element::f32;
    const auto inputs = make_inputs(et, Shapes{{2, 3}, {3, 4}});
    const auto o = make_op(inputs, equation);

    EXPECT_EQ(o->get_equation(), equation);
    EXPECT_EQ(o->get_element_type(), et);
    EXPECT_EQ(o->get_output_size(), exp_einsum_outputs_count);
    EXPECT_EQ(o->get_output_partial_shape(0).to_shape(), Shape({2, 4}));
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
    EXPECT_EQ(o->get_output_partial_shape(0).to_shape(), Shape({3, 2}));
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
    EXPECT_EQ(o->get_output_partial_shape(0).to_shape(), Shape({3, 5}));
}

TEST_F(TypePropEinsumTest, static_shape_ellipsis) {
    const std::string equation = "a...b,b...->a...";
    constexpr auto et = element::i32;
    const auto inputs = make_inputs(et, Shapes{{11, 1, 4, 3}, {3, 11, 7, 1}});
    const auto o = make_op(inputs, equation);

    EXPECT_EQ(o->get_equation(), equation);
    EXPECT_EQ(o->get_element_type(), et);
    EXPECT_EQ(o->get_output_size(), exp_einsum_outputs_count);
    EXPECT_EQ(o->get_output_partial_shape(0).to_shape(), Shape({11, 11, 7, 4}));
}

// Dynamic shapes test also label propagation as each Einsum tests different equation

TEST_F(TypePropEinsumTest, dynamic_shape_dot_product) {
    constexpr auto et = element::f64;
    const auto equation = std::string("a,ab->ab");
    auto input_shapes = PartialShapes{{{2, 7}}, {{3, 10}, 3}};
    set_shape_labels(input_shapes[0], 10);
    set_shape_labels(input_shapes[1], {ov::no_label, 21});

    const auto inputs = make_inputs(et, input_shapes);
    const auto o = make_op(inputs, equation);

    EXPECT_EQ(o->get_equation(), equation);
    EXPECT_EQ(o->get_element_type(), et);
    EXPECT_EQ(o->get_output_size(), exp_einsum_outputs_count);
    EXPECT_EQ(o->get_output_partial_shape(0), PartialShape({{3, 7}, 3}));
    EXPECT_THAT(get_shape_labels(o->get_output_partial_shape(0)), ElementsAre(10, 21));
}

TEST_F(TypePropEinsumTest, dynamic_shape_diag_extraction) {
    const std::string equation = "xyzxy->xyz";
    constexpr auto et = element::i32;
    auto input_shape = PartialShape{{2, 7}, {1, 5}, 4, {3, 5}, 3};
    set_shape_labels(input_shape, 10);

    const auto input = make_shared<ov::op::v0::Parameter>(et, input_shape);
    const auto o = make_op(OutputVector{input}, equation);

    EXPECT_EQ(o->get_equation(), equation);
    EXPECT_EQ(o->get_element_type(), et);
    EXPECT_EQ(o->get_output_size(), exp_einsum_outputs_count);
    EXPECT_EQ(o->get_output_partial_shape(0), PartialShape({{3, 5}, 3, 4}));
    EXPECT_THAT(get_shape_labels(o->get_output_partial_shape(0)), ElementsAre(13, 14, 12));
}

TEST_F(TypePropEinsumTest, dynamic_shape_ellipsis) {
    const std::string equation = "a...b,b...->a...";
    constexpr auto et = element::f32;
    auto input_shapes = PartialShapes{{11, 1, {3, 5}, 3}, {3, 11, 7, {1, 3}}};
    set_shape_labels(input_shapes[0], 10);
    set_shape_labels(input_shapes[1], 20);

    const auto inputs = make_inputs(et, input_shapes);
    const auto o = make_op(inputs, equation);

    EXPECT_EQ(o->get_equation(), equation);
    EXPECT_EQ(o->get_element_type(), et);
    EXPECT_EQ(o->get_output_size(), exp_einsum_outputs_count);
    EXPECT_EQ(o->get_output_partial_shape(0), PartialShape({11, 11, 7, {3, 5}}));
    EXPECT_THAT(get_shape_labels(o->get_output_partial_shape(0)), ElementsAre(10, 21, 22, 12));
}

TEST_F(TypePropEinsumTest, implicit_mode_mixed_case_letters) {
    // the following equation is equivalent to "AbC->ACb"
    const std::string equation = "AbC";
    constexpr auto et = element::i32;
    auto input_shape = PartialShape{1, {2, 3}, {4, 5}};
    set_shape_labels(input_shape, 10);

    const auto input = make_shared<ov::op::v0::Parameter>(et, input_shape);
    const auto o = make_op(OutputVector{input}, equation);

    EXPECT_EQ(o->get_equation(), equation);
    EXPECT_EQ(o->get_element_type(), et);
    EXPECT_EQ(o->get_output_size(), exp_einsum_outputs_count);
    EXPECT_EQ(o->get_output_partial_shape(0), PartialShape({1, {4, 5}, {2, 3}}));
    EXPECT_THAT(get_shape_labels(o->get_output_partial_shape(0)), ElementsAre(10, 12, 11));
}

TEST_F(TypePropEinsumTest, implicit_mode_mixed_case_letters_ellipsis) {
    // the following equation is equivalent to "a...b,B...->...Bab"
    const std::string equation = "a...b,B...";
    constexpr auto et = element::f32;
    auto input_shapes = PartialShapes{{{3, 5}, 11, 1, 3}, {{1, 3}, 3, 1, 7}};
    set_shape_labels(input_shapes[0], 10);
    set_shape_labels(input_shapes[1], 20);

    const auto inputs = make_inputs(et, input_shapes);
    const auto o = make_op(inputs, equation);

    EXPECT_EQ(o->get_equation(), equation);
    EXPECT_EQ(o->get_element_type(), et);
    EXPECT_EQ(o->get_output_size(), exp_einsum_outputs_count);
    EXPECT_EQ(o->get_output_partial_shape(0), PartialShape({3, 11, 7, {1, 3}, {3, 5}, 3}));
    EXPECT_THAT(get_shape_labels(o->get_output_partial_shape(0)), ElementsAre(21, 11, 23, 20, 10, 13));
}

TEST_F(TypePropEinsumTest, implicit_mode_repeated_labels) {
    // the following equation is equivalent to "a...b,b...->...a"
    const std::string equation = "a...b,b...";
    constexpr auto et = element::f32;
    auto input_shapes = PartialShapes{{{3, 5}, 11, 1, 3}, {{1, 3}, 3, 1, 7}};
    set_shape_labels(input_shapes[0], 10);
    set_shape_labels(input_shapes[1], 20);

    const auto inputs = make_inputs(et, input_shapes);
    const auto o = make_op(inputs, equation);

    EXPECT_EQ(o->get_equation(), equation);
    EXPECT_EQ(o->get_element_type(), et);
    EXPECT_EQ(o->get_output_size(), exp_einsum_outputs_count);
    EXPECT_EQ(o->get_output_partial_shape(0), PartialShape({3, 11, 7, {3, 5}}));
    EXPECT_THAT(get_shape_labels(o->get_output_partial_shape(0)), ElementsAre(21, 11, 23, 10));
}

TEST_F(TypePropEinsumTest, dynamic_shape_implict_mode_inner_prod) {
    // the following equation is equivalent to "i,i->"
    const std::string equation = "i,i";
    constexpr auto et = element::f16;

    auto input_shapes = PartialShapes{{11}, {{1, 20}}};
    set_shape_labels(input_shapes[0], 10);
    set_shape_labels(input_shapes[1], 20);

    const auto inputs = make_inputs(et, input_shapes);
    const auto o = make_op(inputs, equation);

    EXPECT_EQ(o->get_equation(), equation);
    EXPECT_EQ(o->get_element_type(), et);
    EXPECT_EQ(o->get_output_size(), exp_einsum_outputs_count);
    EXPECT_EQ(o->get_output_partial_shape(0), PartialShape({}));
    EXPECT_TRUE(get_shape_labels(o->get_output_partial_shape(0)).empty());
}

TEST_F(TypePropEinsumTest, dynamic_rank_multi_matmul) {
    const std::string equation = "ab,bcd,bc->ca";
    constexpr auto et = element::i32;

    auto input_shapes = PartialShapes{{2, 5}, PartialShape::dynamic(), {5, 3}};
    set_shape_labels(input_shapes[0], 10);
    set_shape_labels(input_shapes[2], 30);

    const auto inputs = make_inputs(et, input_shapes);
    const auto o = make_op(inputs, equation);

    EXPECT_EQ(o->get_equation(), equation);
    EXPECT_EQ(o->get_element_type(), et);
    EXPECT_EQ(o->get_output_size(), exp_einsum_outputs_count);
    EXPECT_EQ(o->get_output_partial_shape(0), PartialShape({3, 2}));
    EXPECT_THAT(get_shape_labels(o->get_output_partial_shape(0)), ElementsAre(31, 10));
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
    EXPECT_THAT(get_shape_labels(o->get_output_partial_shape(0)), Each(ov::no_label));
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
    EXPECT_THAT(get_shape_labels(o->get_output_partial_shape(0)), Each(ov::no_label));
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

TEST_F(TypePropEinsumTest, incorrect_equation_invalid_labels) {
    const std::string equation = "a$,Bc->ac";

    const auto input_shapes = Shapes{{2, 3}, {3, 4}};
    const auto inputs = make_inputs(element::f32, input_shapes);

    OV_EXPECT_THROW(auto o = make_op(inputs, equation),
                    AssertFailure,
                    HasSubstr("Input subscript of Einsum equation must consist of either only alphabetic "
                              "letters or alphabetic letters with one ellipsis."));
}

TEST_F(TypePropEinsumTest, incorrect_equation_incompatible_shapes) {
    const std::string equation = "ab,bc->ac";

    const auto input_shapes = Shapes{{2, 10}, {3, 4}};
    const auto inputs = make_inputs(element::f32, input_shapes);

    OV_EXPECT_THROW(auto o = make_op(inputs, equation),
                    AssertFailure,
                    HasSubstr("Different input dimensions indicated by the same labels "
                              "for Einsum must be compatible."));
}

TEST_F(TypePropEinsumTest, incorrect_equation_not_broadcastable_shapes) {
    const std::string equation = "a...b,b...->a...";

    const auto input_shapes = Shapes{{11, 1, 4, 3}, {3, 11, 7, 5}};
    const auto inputs = make_inputs(element::f32, input_shapes);

    OV_EXPECT_THROW(auto o = make_op(inputs, equation),
                    NodeValidationFailure,
                    HasSubstr("Input dimensions labeled with ellipsis for Einsum must be broadcastable."));
}

TEST_F(TypePropEinsumTest, incorrect_equation_missed_ellipsis) {
    const std::string equation = "a...b,b...->a";

    const auto input_shapes = Shapes{{11, 1, 4, 3}, {3, 11, 7, 5}};
    const auto inputs = make_inputs(element::f32, input_shapes);

    OV_EXPECT_THROW(auto o = make_op(inputs, equation),
                    AssertFailure,
                    HasSubstr("Output subscript of Einsum equation must contain one "
                              "ellipsis if ellipsis is met in any input subscript."));
}
