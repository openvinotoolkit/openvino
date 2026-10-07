// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "bound_evaluate.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/test_assertions.hpp"
#include "common_test_utils/type_prop.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/reduce_prod.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/util/framework_node.hpp"

using namespace ov;
using ov::op::v0::Concat;
using ov::op::v0::Constant;
using ov::op::v0::Parameter;
using ov::op::v0::Result;
using ov::op::v1::ReduceProd;
using ov::op::v1::Subtract;
using ov::op::v3::ShapeOf;

class EvaluateBoundTest : public TypePropOpTest<op::util::FrameworkNode> {
protected:
    void SetUp() override {
        const auto c_inputs =
            ParameterVector(2, std::make_shared<Parameter>(element::f32, PartialShape{-1, 24, {1, -1}, {1, -1}}));
        const auto c = std::make_shared<Concat>(OutputVector(c_inputs.begin(), c_inputs.end()), 1);
        const auto s = std::make_shared<ShapeOf>(c);
        const auto s_res = std::make_shared<Result>(s);
        const auto body = std::make_shared<Model>(OutputVector{s_res}, c_inputs);

        fn_op = make_op(OutputVector{s}, 0, 2);

        auto attrs = op::util::FrameworkNodeAttrs();
        attrs.set_type_name("some_type");
        fn_op->set_attrs(attrs);
        fn_op->set_function(0, body);
        fn_op->set_function(1, body);
    }

    std::shared_ptr<op::util::FrameworkNode> fn_op;
};

// Simulate scenarios in pytorch when eval bounds for node PtFrameworkNode which inherits from FrameworkNode
TEST_F(EvaluateBoundTest, no_exception_when_node_has_output_with_dynamic_rank) {
    fn_op->set_output_type(0, element::i32, PartialShape::dynamic());
    fn_op->set_output_type(1, element::i32, PartialShape{{1, 4}});
    fn_op->validate_and_infer_types();

    EXPECT_NO_THROW(ov::util::evaluate_both_bounds(fn_op));
}

TEST_F(EvaluateBoundTest, no_exception_when_node_has_output_with_dynamic_element_type) {
    fn_op->set_output_type(0, element::dynamic, PartialShape{4});
    fn_op->set_output_type(1, element::dynamic, PartialShape{4});
    fn_op->validate_and_infer_types();

    EXPECT_NO_THROW(ov::util::evaluate_both_bounds(fn_op));
}

using BoundEvaluatorTest = ::testing::Test;
TEST(BoundEvaluatorTest, no_exception_on_single_bound) {
    constexpr auto et = element::i32;
    const auto s = Shape{1, 1};
    const auto a = std::make_shared<Parameter>(et, PartialShape{s});
    const auto b = Constant::create(et, s, {1});
    const auto sub = std::make_shared<Subtract>(a, b);

    int32_t a_l[1] = {1};
    a->get_output_tensor(0).set_lower_value(Tensor{et, s, a_l});

    int32_t o_[1] = {INT32_MIN};  // initial value of output tensor is not needed, it's set to check whether changed
    TensorVector output{{et, s, o_}};
    // evaluations won't be performed due to missing upper bound tensor of parameter a
    OV_ASSERT_NO_THROW(sub->evaluate_lower(output));
    EXPECT_EQ(o_[0], INT32_MIN);
    OV_ASSERT_NO_THROW(sub->evaluate_upper(output));
    EXPECT_EQ(o_[0], INT32_MIN);

    int32_t a_u[1] = {11};
    a->get_output_tensor(0).set_upper_value(Tensor{et, s, a_u});
    // now both bounds of sub node can be calculated
    OV_ASSERT_NO_THROW(sub->evaluate_lower(output));
    EXPECT_EQ(o_[0], 0);
    OV_ASSERT_NO_THROW(sub->evaluate_upper(output));
    EXPECT_EQ(o_[0], 10);
}

TEST(BoundEvaluatorTest, reduce_prod_no_bounds_for_possibly_negative_data) {
    const auto data = std::make_shared<Parameter>(element::f32, PartialShape{{1, 2}, {1, 3}});
    const auto shape = std::make_shared<ShapeOf>(data);
    // bounds of shape - 3 are [-2, -2] and [-1, 0], prod of them is not a valid interval
    const auto shifted = std::make_shared<Subtract>(shape, Constant::create(element::i64, Shape{1}, {3}));
    const auto axes = Constant::create(element::i64, Shape{1}, {0});
    const auto prod = std::make_shared<ReduceProd>(shifted, axes, false);

    const auto& [lower, upper] = ov::util::evaluate_both_bounds(prod->output(0));
    EXPECT_FALSE(lower);
    EXPECT_FALSE(upper);
}

TEST(BoundEvaluatorTest, reduce_prod_bounds_for_non_negative_data) {
    const auto data = std::make_shared<Parameter>(element::f32, PartialShape{{1, 2}, {1, 3}});
    const auto shape = std::make_shared<ShapeOf>(data);
    const auto axes = Constant::create(element::i64, Shape{1}, {0});
    const auto prod = std::make_shared<ReduceProd>(shape, axes, false);

    const auto& [lower, upper] = ov::util::evaluate_both_bounds(prod->output(0));
    ASSERT_TRUE(lower && upper);
    EXPECT_EQ(lower.data<int64_t>()[0], 1);
    EXPECT_EQ(upper.data<int64_t>()[0], 6);
}
