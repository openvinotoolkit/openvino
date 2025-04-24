// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/test_assertions.hpp"
#include "common_test_utils/type_prop.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/equal.hpp"
#include "openvino/op/greater.hpp"
#include "openvino/op/greater_eq.hpp"
#include "openvino/op/less.hpp"
#include "openvino/op/less_eq.hpp"
#include "openvino/op/maximum.hpp"
#include "openvino/op/minimum.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/not_equal.hpp"
#include "openvino/op/power.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/subtract.hpp"

using namespace std;
using namespace testing;

//
// Tests for binary elementwise ops.
//
static void test_binary(std::string /* node_type */,
                        shared_ptr<ov::Node>(f)(const shared_ptr<ov::Node>& x, const shared_ptr<ov::Node>& y)) {
    // Check for bad arguments
    auto tv0_2_4_param_0 = make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, 4});
    auto tv0_2_4_param_1 = make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, 4});
    auto tv0_2_4_param_2 = make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{2, 4});
    auto tv0_4_2_param = make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{4, 2});

    auto test_binary_bad_arguments_view_shapes = [&](const shared_ptr<ov::Node>& x, const shared_ptr<ov::Node>& y) {
        try {
            auto node = f(x, y);
            // Should have thrown, so fail if it didn't
            FAIL() << "Incompatible view arguments not detected.";
        } catch (const ov::NodeValidationFailure& error) {
            EXPECT_HAS_SUBSTRING(error.what(), std::string("Argument shapes are inconsistent"));
        } catch (...) {
            FAIL() << "Deduced type check failed for unexpected reason";
        }
    };
    test_binary_bad_arguments_view_shapes(tv0_2_4_param_0, tv0_4_2_param);

    auto test_binary_bad_arguments_view_element_types = [&](const shared_ptr<ov::Node>& x,
                                                            const shared_ptr<ov::Node>& y) {
        try {
            auto node = f(x, y);
            // Should have thrown, so fail if it didn't
            FAIL() << "Incompatible view arguments not detected.";
        } catch (const ov::NodeValidationFailure& error) {
            EXPECT_HAS_SUBSTRING(error.what(), std::string("Arguments do not have the same element type"));
        } catch (...) {
            FAIL() << "Deduced type check failed for unexpected reason";
        }
    };

    test_binary_bad_arguments_view_element_types(tv0_2_4_param_0, tv0_2_4_param_2);

    auto test_binary_good_arguments = [&](const shared_ptr<ov::Node>& x, const shared_ptr<ov::Node>& y) {
        auto node = f(x, y);
        EXPECT_TRUE(node->has_same_type(node->input_values()[0].get_node_shared_ptr()));
    };
    test_binary_good_arguments(tv0_2_4_param_0, tv0_2_4_param_1);
}

TEST(type_prop, add_bad_arguments) {
    test_binary("Add", [](const shared_ptr<ov::Node>& x, const shared_ptr<ov::Node>& y) -> shared_ptr<ov::Node> {
        return make_shared<ov::op::v1::Add>(x, y);
    });
}

namespace {
template <typename T>
void test_binary_eltwise_numpy(const ov::element::Type& et, const ov::op::AutoBroadcastSpec& autob) {
    auto param1 = make_shared<ov::op::v0::Parameter>(et, ov::Shape{1, 3, 6});
    auto param2 = make_shared<ov::op::v0::Parameter>(et, ov::Shape{3, 1});
    auto param3 = make_shared<ov::op::v0::Parameter>(et, ov::Shape{2, 3, 6});
    auto param4 = make_shared<ov::op::v0::Parameter>(et, ov::Shape{6});
    auto param5 = make_shared<ov::op::v0::Parameter>(et, ov::Shape{});

    EXPECT_EQ(make_shared<T>(param1, param2, autob)->get_shape(), (ov::Shape{1, 3, 6}));
    EXPECT_EQ(make_shared<T>(param1, param3, autob)->get_shape(), (ov::Shape{2, 3, 6}));
    EXPECT_EQ(make_shared<T>(param4, param3, autob)->get_shape(), (ov::Shape{2, 3, 6}));
    EXPECT_EQ(make_shared<T>(param5, param3, autob)->get_shape(), (ov::Shape{2, 3, 6}));
    EXPECT_EQ(make_shared<T>(param3, param5, autob)->get_shape(), (ov::Shape{2, 3, 6}));

    auto pp1 = make_shared<ov::op::v0::Parameter>(et, ov::PartialShape{1, ov::Dimension::dynamic(), 6});
    auto pp2 = make_shared<ov::op::v0::Parameter>(et, ov::PartialShape{3, 1});
    EXPECT_EQ(make_shared<T>(pp1, pp2, autob)->get_shape(), (ov::Shape{1, 3, 6}));
}

template <typename T>
void test_binary_eltwise_bad_argument_shape(const ov::element::Type& et) {
    auto input1 = make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, 4});
    auto input2 = make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 2, 4});

    OV_EXPECT_THROW(auto bc = make_shared<T>(input1, input2, ov::op::AutoBroadcastType::NONE),
                    ov::NodeValidationFailure,
                    HasSubstr("Argument shapes are inconsistent"));
}

template <class T>
shared_ptr<ov::op::v1::Reshape> createReshapeSubgraph(ov::PartialShape param_shape,
                                                      shared_ptr<ov::op::v0::Constant> constant_op,
                                                      bool const_rhs = true) {
    auto param = make_shared<ov::op::v0::Parameter>(ov::element::f32, param_shape);
    auto shape_of = make_shared<ov::op::v3::ShapeOf>(param);
    auto cast_fp = make_shared<ov::op::v0::Convert>(shape_of, ov::element::f32);

    ov::Output<ov::Node> op;
    if (const_rhs)
        op = make_shared<T>(cast_fp, constant_op);
    else
        op = make_shared<T>(constant_op, cast_fp);

    auto cast_int = make_shared<ov::op::v0::Convert>(op, ov::element::i32);
    return make_shared<ov::op::v1::Reshape>(param, cast_int, false);
}

}  // namespace

TEST(type_prop, eltwise_auto_bcast) {
    test_binary_eltwise_numpy<ov::op::v1::Add>(ov::element::f32, ov::op::AutoBroadcastType::NUMPY);
    test_binary_eltwise_numpy<ov::op::v1::Maximum>(ov::element::f32, ov::op::AutoBroadcastType::NUMPY);
}

// --- Binary elementwise comparision ops tests - start
/** \brief namespace to group binary elementwise comparision (BEC) tests */
namespace BEC {
template <class TOp>
class BinaryElementwiseCmpTest : public Test {
protected:
    template <class... Args>
    std::shared_ptr<TOp> make_op(Args&&... args) {
        return std::make_shared<TOp>(std::forward<Args>(args)...);
    }

    std::shared_ptr<TOp> make_op_with_types(ov::element::Type et0, ov::element::Type et1) {
        const auto a = std::make_shared<ov::op::v0::Parameter>(et0, ov::Shape{1, 2, 3});
        const auto b = std::make_shared<ov::op::v0::Parameter>(et1, ov::Shape{1, 2, 3});
        return make_op(a, b);
    }
};

TYPED_TEST_SUITE_P(BinaryElementwiseCmpTest);

TYPED_TEST_P(BinaryElementwiseCmpTest, argument_shapes_are_inconsistent) {
    test_binary_eltwise_bad_argument_shape<TypeParam>(ov::element::f64);
}

TYPED_TEST_P(BinaryElementwiseCmpTest, propagate_static_partial_shape_no_broadcast) {
    auto shape = ov::PartialShape{2, 4, 5};
    auto symbols = set_shape_symbols(shape);
    const auto a = make_shared<ov::op::v0::Parameter>(ov::element::f32, shape);
    const auto b = make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape({2, 4, 5}));

    const auto op = this->make_op(a, b, ov::op::AutoBroadcastType::NONE);

    EXPECT_EQ(op->get_output_size(), 1);
    EXPECT_EQ(op->get_element_type(), ov::element::boolean);
    EXPECT_EQ(op->get_output_partial_shape(0), shape);
    EXPECT_EQ(op->get_shape(), shape.get_shape());
    EXPECT_THAT(get_shape_symbols(op->get_output_partial_shape(0)), ElementsAre(symbols[0], symbols[1], symbols[2]));
    EXPECT_THAT(get_shape_symbols(a->get_output_partial_shape(0)), ElementsAre(symbols[0], symbols[1], symbols[2]));
    EXPECT_THAT(get_shape_symbols(b->get_output_partial_shape(0)), Each(nullptr));
}

TYPED_TEST_P(BinaryElementwiseCmpTest, propagate_static_partial_shape_numpy_broadcast) {
    test_binary_eltwise_numpy<TypeParam>(ov::element::f64, ov::op::AutoBroadcastType::NUMPY);
}

TYPED_TEST_P(BinaryElementwiseCmpTest, propagate_static_partial_shape_pdpd_broadcast) {
    auto a = make_shared<ov::op::v0::Parameter>(ov::element::f64, ov::PartialShape{1, 3, 6});
    auto b = make_shared<ov::op::v0::Parameter>(ov::element::f64, ov::PartialShape{1, 1, 1});

    const auto op = this->make_op(a, b, ov::op::AutoBroadcastType::PDPD);

    EXPECT_EQ(op->get_output_size(), 1);
    EXPECT_EQ(op->get_element_type(), ov::element::boolean);
    EXPECT_EQ(op->get_output_partial_shape(0), ov::PartialShape({1, 3, 6}));
    EXPECT_EQ(op->get_shape(), ov::Shape({1, 3, 6}));
}

TYPED_TEST_P(BinaryElementwiseCmpTest, propagate_dynamic_partial_shape_no_broadcast) {
    const auto shape = ov::PartialShape{2, {3, 4}, 8, {2, 5}, 10};
    auto a = make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{2, {3, 5}, -1, {-1, 5}, {6, -1}});
    auto b = make_shared<ov::op::v0::Parameter>(ov::element::i64, shape);

    auto op = this->make_op(a, b, ov::op::AutoBroadcastType::NONE);

    EXPECT_EQ(op->get_output_size(), 1);
    EXPECT_EQ(op->get_element_type(), ov::element::boolean);
    EXPECT_EQ(op->get_output_partial_shape(0), shape);
}

TYPED_TEST_P(BinaryElementwiseCmpTest, propagate_dynamic_partial_shape_numpy_broadcast) {
    auto a = make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{2, {3, 5}, -1, {-1, 5}, {6, -1}});
    auto b = make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{2, {3, 4}, 8});

    auto op = this->make_op(a, b, ov::op::AutoBroadcastType::NUMPY);

    EXPECT_EQ(op->get_output_size(), 1);
    EXPECT_EQ(op->get_element_type(), ov::element::boolean);
    EXPECT_EQ(op->get_output_partial_shape(0), ov::PartialShape({2, {3, 5}, 2, {3, 4}, 8}));
}

TYPED_TEST_P(BinaryElementwiseCmpTest, propagate_dynamic_rank_shape_no_broadcast) {
    const auto a = make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape::dynamic());
    const auto b = make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape::dynamic());

    const auto op = this->make_op(a, b, ov::op::AutoBroadcastType::NONE);

    EXPECT_EQ(op->get_output_size(), 1);
    EXPECT_EQ(op->get_element_type(), ov::element::boolean);
    EXPECT_EQ(op->get_output_partial_shape(0), ov::PartialShape::dynamic());
}

TYPED_TEST_P(BinaryElementwiseCmpTest, propagate_dynamic_rank_shape) {
    const auto a = make_shared<ov::op::v0::Parameter>(ov::element::i16, ov::PartialShape::dynamic());
    const auto b = make_shared<ov::op::v0::Parameter>(ov::element::i16, ov::PartialShape::dynamic());

    const auto op = this->make_op(a, b);

    EXPECT_EQ(op->get_output_size(), 1);
    EXPECT_EQ(op->get_element_type(), ov::element::boolean);
    EXPECT_EQ(op->get_output_partial_shape(0), ov::PartialShape::dynamic());
}

TYPED_TEST_P(BinaryElementwiseCmpTest, propagate_one_input_is_dynamic_rank_shape) {
    const auto a = make_shared<ov::op::v0::Parameter>(ov::element::i8, ov::PartialShape{3, 4, {1, 5}, -1});
    const auto b = make_shared<ov::op::v0::Parameter>(ov::element::i8, ov::PartialShape::dynamic());

    EXPECT_EQ(this->make_op(a, b)->get_output_partial_shape(0), ov::PartialShape::dynamic());
    EXPECT_EQ(this->make_op(b, a)->get_output_partial_shape(0), ov::PartialShape::dynamic());
}

TYPED_TEST_P(BinaryElementwiseCmpTest, allowed_mixed_input_types) {
    // Done as multiple assertion test because gtest not allow combine type param and data param combined fixture.
    ASSERT_EQ(this->make_op_with_types(ov::element::boolean, ov::element::boolean)->get_element_type(),
              ov::element::boolean);
    ASSERT_EQ(this->make_op_with_types(ov::element::boolean, ov::element::dynamic)->get_element_type(),
              ov::element::boolean);
    ASSERT_EQ(this->make_op_with_types(ov::element::dynamic, ov::element::i32)->get_element_type(),
              ov::element::boolean);
    ASSERT_EQ(this->make_op_with_types(ov::element::dynamic, ov::element::boolean)->get_element_type(),
              ov::element::boolean);
    ASSERT_EQ(this->make_op_with_types(ov::element::dynamic, ov::element::dynamic)->get_element_type(),
              ov::element::boolean);
}

TYPED_TEST_P(BinaryElementwiseCmpTest, not_allowed_mixed_input_types) {
    ASSERT_ANY_THROW({ this->make_op_with_types(ov::element::i32, ov::element::boolean); });
    ASSERT_ANY_THROW({ this->make_op_with_types(ov::element::boolean, ov::element::i32); });
}

TYPED_TEST_P(BinaryElementwiseCmpTest, propagate_symbols_from_one_input_only_no_broadcast) {
    constexpr auto et = ov::element::f64;

    auto symboled_shape = ov::PartialShape{2, 4, 5};
    const auto exp_symbols = set_shape_symbols(symboled_shape);

    const auto a = make_shared<ov::op::v0::Parameter>(et, symboled_shape);
    const auto b = make_shared<ov::op::v0::Parameter>(et, ov::PartialShape({2, 4, 5}));

    EXPECT_EQ(get_shape_symbols(this->make_op(a, b, ov::op::AutoBroadcastType::NONE)->get_output_partial_shape(0)),
              exp_symbols);
    EXPECT_EQ(get_shape_symbols(this->make_op(b, a, ov::op::AutoBroadcastType::NONE)->get_output_partial_shape(0)),
              exp_symbols);
}

TYPED_TEST_P(BinaryElementwiseCmpTest, propagate_symbols_from_both_inputs_no_broadcast) {
    constexpr auto et = ov::element::f64;

    auto A = std::make_shared<ov::Symbol>(), B = std::make_shared<ov::Symbol>();
    auto C = std::make_shared<ov::Symbol>(), D = std::make_shared<ov::Symbol>();
    auto E = std::make_shared<ov::Symbol>(), F = std::make_shared<ov::Symbol>();
    auto G = std::make_shared<ov::Symbol>(), H = std::make_shared<ov::Symbol>();
    auto I = std::make_shared<ov::Symbol>(), J = std::make_shared<ov::Symbol>();

    const auto symbols_a = ov::TensorSymbol{A, nullptr, B, C, D, E};
    auto shape_a = ov::PartialShape{2, 4, 5, -1, {4, 5}, {-1, 6}};
    set_shape_symbols(shape_a, symbols_a);
    const auto a = make_shared<ov::op::v0::Parameter>(et, shape_a);

    const auto symbols_b = ov::TensorSymbol{F, G, nullptr, H, I, J};
    auto shape_b = ov::PartialShape{2, 4, 5, 5, -1, {4, -1}};
    set_shape_symbols(shape_b, symbols_b);
    const auto b = make_shared<ov::op::v0::Parameter>(et, shape_b);

    EXPECT_THAT(this->make_op(a, b, ov::op::AutoBroadcastType::NONE)->get_output_partial_shape(0),
                AllOf(Eq(ov::PartialShape({2, 4, 5, 5, {4, 5}, {4, 6}})),
                      ResultOf(get_shape_symbols, ElementsAre(A, G, B, C, D, E))));

    EXPECT_THAT(this->make_op(b, a, ov::op::AutoBroadcastType::NONE)->get_output_partial_shape(0),
                AllOf(Eq(ov::PartialShape({2, 4, 5, 5, {4, 5}, {4, 6}})),
                      ResultOf(get_shape_symbols, ElementsAre(F, G, B, H, I, J))));
}

TYPED_TEST_P(BinaryElementwiseCmpTest, propagate_symbols_from_both_inputs_numpy_broadcast) {
    constexpr auto et = ov::element::f64;

    auto A = std::make_shared<ov::Symbol>(), B = std::make_shared<ov::Symbol>();
    auto C = std::make_shared<ov::Symbol>(), D = std::make_shared<ov::Symbol>();
    auto E = std::make_shared<ov::Symbol>(), F = std::make_shared<ov::Symbol>();
    auto G = std::make_shared<ov::Symbol>(), H = std::make_shared<ov::Symbol>();

    const auto symbols_a = ov::TensorSymbol{A, nullptr, B, C, nullptr, D};
    auto shape_a = ov::PartialShape{2, {2, 4}, -1, {4, 5}, {-1, 6}, 1};
    set_shape_symbols(shape_a, symbols_a);
    const auto a = make_shared<ov::op::v0::Parameter>(et, shape_a);

    const auto symbols_b = ov::TensorSymbol{E, F, nullptr, G};
    auto shape_b = ov::PartialShape{2, {4, -1}, 5, {4, -1}};
    set_shape_symbols(shape_b, symbols_b);
    const auto b = make_shared<ov::op::v0::Parameter>(et, shape_b);

    EXPECT_THAT(this->make_op(a, b, ov::op::AutoBroadcastType::NUMPY)->get_output_partial_shape(0),
                AllOf(Eq(ov::PartialShape({2, {2, 4}, 2, {4, 5}, 5, {4, -1}})),
                      ResultOf(get_shape_symbols, ElementsAre(A, nullptr, E, C, nullptr, G))));

    EXPECT_THAT(this->make_op(b, a, ov::op::AutoBroadcastType::NUMPY)->get_output_partial_shape(0),
                AllOf(Eq(ov::PartialShape({2, {2, 4}, 2, {4, 5}, 5, {4, -1}})),
                      ResultOf(get_shape_symbols, ElementsAre(A, nullptr, E, F, nullptr, G))));
}

TYPED_TEST_P(BinaryElementwiseCmpTest, use_default_ctor) {
    constexpr auto dtype = ov::element::f32;

    const auto a = make_shared<ov::op::v0::Parameter>(dtype, ov::PartialShape{2, 5, -1, {-1, 5}, {6, -1}});
    const auto b = make_shared<ov::op::v0::Parameter>(dtype, ov::PartialShape{2, 4, 8});

    const auto op = this->make_op();
    op->set_arguments(ov::NodeVector{a, b});
    op->set_autob(ov::op::AutoBroadcastType::NUMPY);
    op->validate_and_infer_types();

    EXPECT_EQ(op->get_autob(), ov::op::AutoBroadcastType::NUMPY);
    EXPECT_EQ(op->get_element_type(), ov::element::boolean);
    EXPECT_EQ(op->get_output_size(), 1);
    EXPECT_EQ(op->get_output_partial_shape(0), ov::PartialShape({2, 5, 2, 4, 8}));
}

REGISTER_TYPED_TEST_SUITE_P(BinaryElementwiseCmpTest,
                            argument_shapes_are_inconsistent,
                            propagate_static_partial_shape_no_broadcast,
                            propagate_static_partial_shape_numpy_broadcast,
                            propagate_static_partial_shape_pdpd_broadcast,
                            propagate_dynamic_partial_shape_no_broadcast,
                            propagate_dynamic_partial_shape_numpy_broadcast,
                            propagate_dynamic_rank_shape_no_broadcast,
                            propagate_dynamic_rank_shape,
                            propagate_one_input_is_dynamic_rank_shape,
                            allowed_mixed_input_types,
                            not_allowed_mixed_input_types,
                            propagate_symbols_from_one_input_only_no_broadcast,
                            propagate_symbols_from_both_inputs_no_broadcast,
                            propagate_symbols_from_both_inputs_numpy_broadcast,
                            use_default_ctor);

using BinaryOpTypes = Types<ov::op::v1::Equal,
                            ov::op::v1::NotEqual,
                            ov::op::v1::Greater,
                            ov::op::v1::GreaterEqual,
                            ov::op::v1::Less,
                            ov::op::v1::LessEqual>;
INSTANTIATE_TYPED_TEST_SUITE_P(type_prop, BinaryElementwiseCmpTest, BinaryOpTypes);
}  // namespace BEC

TEST(type_prop, binary_arithmetic_bool_argument_element_types) {
    auto param_0 = make_shared<ov::op::v0::Parameter>(ov::element::boolean, ov::Shape{2, 4});
    auto param_1 = make_shared<ov::op::v0::Parameter>(ov::element::boolean, ov::Shape{2, 4});

    OV_EXPECT_THROW(std::ignore = make_shared<ov::op::v1::Add>(param_0, param_1),
                    ov::NodeValidationFailure,
                    HasSubstr("This operation does not support inputs with element type: boolean"));
}

TEST(type_prop, binary_arithmetic_str_argument_element_types) {
    auto param_0 = make_shared<ov::op::v0::Parameter>(ov::element::string, ov::Shape{2, 4});
    auto param_1 = make_shared<ov::op::v0::Parameter>(ov::element::string, ov::Shape{2, 4});

    OV_EXPECT_THROW(std::ignore = make_shared<ov::op::v1::Add>(param_0, param_1),
                    ov::NodeValidationFailure,
                    HasSubstr("This operation does not support inputs with element type: string"));
}

TEST(type_prop, binary_arithmetic_bad_argument_shape_with_none_autobroadcast_attribute) {
    test_binary_eltwise_bad_argument_shape<ov::op::v1::Add>(ov::element::f32);
    test_binary_eltwise_bad_argument_shape<ov::op::v1::Maximum>(ov::element::f32);
}

TEST(type_prop, binary_elementwise_arithmetic_both_dynamic) {
    auto a = make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic());
    auto b = make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic());
    auto add = make_shared<ov::op::v1::Add>(a, b);

    ASSERT_TRUE(add->get_output_partial_shape(0).rank().is_dynamic());
}

TEST(type_prop, binary_elementwise_arithmetic_left_rank_static_dynamic_right_rank_static_dynamic_result_static) {
    auto a = make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, ov::Dimension::dynamic(), 3});
    auto b = make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 2, ov::Dimension::dynamic()});
    auto add = make_shared<ov::op::v1::Add>(a, b);

    ASSERT_TRUE(add->get_output_partial_shape(0).is_static());
    ASSERT_EQ(add->get_shape(), (ov::Shape{1, 2, 3}));
}

TEST(type_prop,
     binary_elementwise_arithmetic_left_rank_static_dynamic_right_rank_static_dynamic_result_rank_static_dynamic) {
    auto a =
        make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                           ov::PartialShape{1, ov::Dimension::dynamic(), ov::Dimension::dynamic()});
    auto b = make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 2, ov::Dimension::dynamic()});
    auto add = make_shared<ov::op::v1::Add>(a, b);

    ASSERT_TRUE(add->get_output_partial_shape(0).rank().is_static());
    ASSERT_TRUE(add->get_output_partial_shape(0).is_dynamic());
    ASSERT_TRUE(add->get_output_partial_shape(0).same_scheme(ov::PartialShape{1, 2, ov::Dimension::dynamic()}));
}

TEST(type_prop, binary_elementwise_arithmetic_left_static_right_rank_static_dynamic) {
    auto a = make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 2, 3});
    auto b = make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 2, ov::Dimension::dynamic()});
    auto add = make_shared<ov::op::v1::Add>(a, b);

    ASSERT_TRUE(add->get_output_partial_shape(0).is_static());
    ASSERT_EQ(add->get_shape(), (ov::Shape{1, 2, 3}));
}

TEST(type_prop, binary_elementwise_arithmetic_left_rank_static_dynamic_right_static) {
    auto a = make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 2, ov::Dimension::dynamic()});
    auto b = make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 2, 3});
    auto add = make_shared<ov::op::v1::Add>(a, b);

    ASSERT_TRUE(add->get_output_partial_shape(0).is_static());
    ASSERT_EQ(add->get_shape(), (ov::Shape{1, 2, 3}));
}

TEST(type_prop, binary_elementwise_arithmetic_left_rank_static_dynamic_inconsistent) {
    auto a = make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 2, ov::Dimension::dynamic()});
    auto b = make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 3, 3});

    try {
        auto add = make_shared<ov::op::v1::Add>(a, b);
        FAIL() << "Inconsistent partial shapes not detected";
    } catch (const ov::NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), "Argument shapes are inconsistent");
    } catch (...) {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, binary_elementwise_arithmetic_right_rank_static_dynamic_inconsistent) {
    auto a = make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 3, 3});
    auto b = make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 2, ov::Dimension::dynamic()});

    try {
        auto add = make_shared<ov::op::v1::Add>(a, b);
        FAIL() << "Inconsistent partial shapes not detected";
    } catch (const ov::NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), "Argument shapes are inconsistent");
    } catch (...) {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, binary_elementwise_arithmetic_both_rank_static_dynamic_inconsistent) {
    auto a = make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{ov::Dimension::dynamic(), 3, 3});
    auto b = make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 2, ov::Dimension::dynamic()});

    try {
        auto add = make_shared<ov::op::v1::Add>(a, b);
        FAIL() << "Inconsistent partial shapes not detected";
    } catch (const ov::NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), "Argument shapes are inconsistent");
    } catch (...) {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, binary_elementwise_arithmetic_left_rank_static_dynamic_different_rank) {
    auto a = make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 2, ov::Dimension::dynamic()});
    auto b = make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 2, 3, 4});

    try {
        auto add = make_shared<ov::op::v1::Add>(a, b);
        FAIL() << "Inconsistent partial shapes not detected";
    } catch (const ov::NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), "Argument shapes are inconsistent");
    } catch (...) {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, binary_elementwise_arithmetic_right_rank_static_dynamic_different_rank) {
    auto a = make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 2, 3, 4});
    auto b = make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 2, ov::Dimension::dynamic()});

    try {
        auto add = make_shared<ov::op::v1::Add>(a, b);
        FAIL() << "Inconsistent partial shapes not detected";
    } catch (const ov::NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), "Argument shapes are inconsistent");
    } catch (...) {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, binary_elementwise_arithmetic_both_rank_static_dynamic_different_rank) {
    auto a = make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, ov::Dimension::dynamic(), 3, 4});
    auto b = make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 2, ov::Dimension::dynamic()});

    try {
        auto add = make_shared<ov::op::v1::Add>(a, b);
        FAIL() << "Inconsistent partial shapes not detected";
    } catch (const ov::NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), "Argument shapes are inconsistent");
    } catch (...) {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, binary_elementwise_arithmetic_both_et_dynamic) {
    auto a = make_shared<ov::op::v0::Parameter>(ov::element::dynamic, ov::Shape{1, 2, 3, 4});
    auto b = make_shared<ov::op::v0::Parameter>(ov::element::dynamic, ov::Shape{1, 2, 3, 4});
    auto add = make_shared<ov::op::v1::Add>(a, b);

    ASSERT_TRUE(add->get_output_element_type(0).is_dynamic());
}

TEST(type_prop, binary_elementwise_arithmetic_left_et_dynamic) {
    auto a = make_shared<ov::op::v0::Parameter>(ov::element::dynamic, ov::Shape{1, 2, 3, 4});
    auto b = make_shared<ov::op::v0::Parameter>(ov::element::u32, ov::Shape{1, 2, 3, 4});
    auto add = make_shared<ov::op::v1::Add>(a, b);

    ASSERT_EQ(add->get_output_element_type(0), ov::element::u32);
}

TEST(type_prop, binary_elementwise_arithmetic_right_et_dynamic) {
    auto a = make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{1, 2, 3, 4});
    auto b = make_shared<ov::op::v0::Parameter>(ov::element::dynamic, ov::Shape{1, 2, 3, 4});
    auto add = make_shared<ov::op::v1::Add>(a, b);

    ASSERT_EQ(add->get_output_element_type(0), ov::element::i64);
}

TEST(type_prop, logic_arith_compare_partial_et) {
    auto test_arith = [](ov::element::Type et0, ov::element::Type et1) -> std::shared_ptr<ov::Node> {
        auto param0 = std::make_shared<ov::op::v0::Parameter>(et0, ov::Shape{1, 2, 3});
        auto param1 = std::make_shared<ov::op::v0::Parameter>(et1, ov::Shape{1, 2, 3});
        return std::make_shared<ov::op::v1::Add>(param0, param1);
    };

    // Arith ops:
    //
    // int int -> int
    // int boo -> !
    // int dyn -> int
    // boo int -> !
    // boo boo -> !
    // boo dyn -> !
    // dyn int -> int
    // dyn boo -> !
    // dyn dyn -> dyn
    ASSERT_EQ(test_arith(ov::element::i32, ov::element::i32)->get_element_type(), ov::element::i32);
    ASSERT_ANY_THROW({ test_arith(ov::element::i32, ov::element::boolean); });
    ASSERT_EQ(test_arith(ov::element::i32, ov::element::dynamic)->get_element_type(), ov::element::i32);
    ASSERT_ANY_THROW({ test_arith(ov::element::boolean, ov::element::i32); });
    ASSERT_ANY_THROW({ test_arith(ov::element::boolean, ov::element::boolean); });
    ASSERT_ANY_THROW({ test_arith(ov::element::boolean, ov::element::dynamic); });
    ASSERT_EQ(test_arith(ov::element::dynamic, ov::element::i32)->get_element_type(), ov::element::i32);
    ASSERT_ANY_THROW({ test_arith(ov::element::dynamic, ov::element::boolean); });
    ASSERT_EQ(test_arith(ov::element::dynamic, ov::element::dynamic)->get_element_type(), ov::element::dynamic);
}

TEST(type_prop, interval_value_propagation_add_rhs) {
    ov::PartialShape op_shape{ov::Dimension(-1),
                              ov::Dimension(2, -1),
                              ov::Dimension(-1, 6),
                              ov::Dimension(7, 10),
                              ov::Dimension(7, 10),
                              5};
    const auto const_op = ov::op::v0::Constant::create(ov::element::f32, {6}, {2, 3, 4, 5, -5, 6});
    // const rhs
    const auto reshape = createReshapeSubgraph<ov::op::v1::Add>(op_shape, const_op);
    EXPECT_EQ(reshape->get_element_type(), ov::element::f32);
    EXPECT_EQ(reshape->get_output_partial_shape(0),
              ov::PartialShape({-1, -1, ov::Dimension(4, 10), ov::Dimension(12, 15), ov::Dimension(2, 5), 11}));
}

TEST(type_prop, interval_value_propagation_add_lhs) {
    ov::PartialShape op_shape{ov::Dimension(-1),
                              ov::Dimension(2, -1),
                              ov::Dimension(-1, 6),
                              ov::Dimension(7, 10),
                              ov::Dimension(7, 10),
                              5};
    const auto const_op = ov::op::v0::Constant::create(ov::element::f32, {6}, {2, 3, 4, 5, -5, 6});
    // const lhs
    const auto reshape = createReshapeSubgraph<ov::op::v1::Add>(op_shape, const_op, false);
    EXPECT_EQ(reshape->get_element_type(), ov::element::f32);
    EXPECT_EQ(reshape->get_output_partial_shape(0),
              ov::PartialShape({-1, -1, ov::Dimension(4, 10), ov::Dimension(12, 15), ov::Dimension(2, 5), 11}));
}

TEST(type_prop, interval_value_propagation_add_incorrect_dim) {
    // const rhs - result lower than 0
    ov::PartialShape op_shape{ov::Dimension(5, 7)};
    const auto const_op = ov::op::v0::Constant::create(ov::element::f32, {1}, {-10});
    OV_EXPECT_THROW(createReshapeSubgraph<ov::op::v1::Add>(op_shape, const_op),
                    ov::NodeValidationFailure,
                    HasSubstr("dim[0] has invalid bounds"));
}

TEST(type_prop, interval_value_propagation_sub_rhs) {
    ov::PartialShape op_shape{ov::Dimension(-1),
                              ov::Dimension(24, -1),
                              ov::Dimension(4, 36),
                              ov::Dimension(13, 27),
                              ov::Dimension(13, 27),
                              15};
    // const rhs
    const auto const_op = ov::op::v0::Constant::create(ov::element::f32, {6}, {2, 3, 4, 5, -5, 6});
    const auto reshape = createReshapeSubgraph<ov::op::v1::Subtract>(op_shape, const_op);
    EXPECT_EQ(reshape->get_element_type(), ov::element::f32);
    EXPECT_EQ(reshape->get_output_partial_shape(0),
              ov::PartialShape({-1, -1, ov::Dimension(-1, 32), ov::Dimension(8, 22), ov::Dimension(18, 32), 9}));
}

TEST(type_prop, interval_value_propagation_sub_lhs) {
    ov::PartialShape op_shape{ov::Dimension(-1),
                              ov::Dimension(24, -1),
                              ov::Dimension(4, 36),
                              ov::Dimension(13, 27),
                              ov::Dimension(13, 27),
                              15};
    // const lhs
    const auto const_op = ov::op::v0::Constant::create(ov::element::f32, {6}, {12, 28, 36, 43, 27, 25});
    const auto reshape = createReshapeSubgraph<ov::op::v1::Subtract>(op_shape, const_op, false);
    EXPECT_EQ(reshape->get_element_type(), ov::element::f32);
    EXPECT_EQ(reshape->get_output_partial_shape(0),
              ov::PartialShape({-1, -1, ov::Dimension(0, 32), ov::Dimension(16, 30), ov::Dimension(0, 14), 10}));
}

TEST(type_prop, interval_value_propagation_sub_incorrect_dim) {
    // const lhs - result lower than 0
    ov::PartialShape op_shape{ov::Dimension(13, 27)};
    const auto const_op = ov::op::v0::Constant::create(ov::element::f32, {1}, {5});
    OV_EXPECT_THROW(createReshapeSubgraph<ov::op::v1::Subtract>(op_shape, const_op, false),
                    ov::NodeValidationFailure,
                    HasSubstr("dim[0] has invalid bounds"));
}

TEST(type_prop, interval_value_propagation_mul_rhs) {
    ov::PartialShape op_shape{ov::Dimension(-1),
                              ov::Dimension(4, -1),
                              ov::Dimension(-1, 6),
                              ov::Dimension(5, 7),
                              ov::Dimension(9, 10),
                              15};
    // const rhs
    const auto const_op = ov::op::v0::Constant::create(ov::element::f32, {6}, {7, 6, 5, 4, 3, 2});
    const auto reshape = createReshapeSubgraph<ov::op::v1::Multiply>(op_shape, const_op);
    EXPECT_EQ(reshape->get_element_type(), ov::element::f32);
    EXPECT_EQ(reshape->get_output_partial_shape(0),
              ov::PartialShape({-1, -1, ov::Dimension(-1, 30), ov::Dimension(20, 28), ov::Dimension(27, 30), 30}));
}

TEST(type_prop, interval_value_propagation_mul_lhs) {
    ov::PartialShape op_shape{ov::Dimension(-1),
                              ov::Dimension(4, -1),
                              ov::Dimension(-1, 6),
                              ov::Dimension(5, 7),
                              ov::Dimension(9, 10),
                              15};

    // const lhs
    const auto const_op = ov::op::v0::Constant::create(ov::element::f32, {6}, {7, 6, 5, 4, 3, 2});
    const auto reshape = createReshapeSubgraph<ov::op::v1::Multiply>(op_shape, const_op, false);
    EXPECT_EQ(reshape->get_element_type(), ov::element::f32);
    EXPECT_EQ(reshape->get_output_partial_shape(0),
              ov::PartialShape({-1, -1, ov::Dimension(-1, 30), ov::Dimension(20, 28), ov::Dimension(27, 30), 30}));
}

TEST(type_prop, interval_value_propagation_mul_incorrect_dim_rhs) {
    // const rhs - result lower than 0
    ov::PartialShape op_shape{ov::Dimension(5, 7)};
    const auto const_op = ov::op::v0::Constant::create(ov::element::f32, {1}, {-3});
    OV_EXPECT_THROW(createReshapeSubgraph<ov::op::v1::Multiply>(op_shape, const_op),
                    ov::NodeValidationFailure,
                    HasSubstr("dim[0] has invalid bounds"));
}

TEST(type_prop, interval_value_propagation_mul_incorrect_dim_lhs) {
    // const lhs - result lower than 0
    ov::PartialShape op_shape{ov::Dimension(5, 7)};
    const auto const_op = ov::op::v0::Constant::create(ov::element::f32, {1}, {-3});
    OV_EXPECT_THROW(createReshapeSubgraph<ov::op::v1::Multiply>(op_shape, const_op, false),
                    ov::NodeValidationFailure,
                    HasSubstr("dim[0] has invalid bounds"));
}

TEST(type_prop, interval_value_propagation_div_rhs) {
    // const rhs
    ov::PartialShape op_shape{ov::Dimension(8, 16), ov::Dimension(9, 30), 15};
    const auto const_op = ov::op::v0::Constant::create(ov::element::f32, {3}, {4, 3, 5});
    const auto reshape = createReshapeSubgraph<ov::op::v1::Divide>(op_shape, const_op);
    ov::PartialShape expected_shape{ov::Dimension(2, 4), ov::Dimension(3, 10), 3};
    EXPECT_EQ(reshape->get_element_type(), ov::element::f32);
    EXPECT_EQ(reshape->get_output_partial_shape(0), expected_shape);
}

TEST(type_prop, interval_value_propagation_div_rhs_full) {
    ov::PartialShape op_shape{ov::Dimension(-1),
                              ov::Dimension(4, -1),
                              ov::Dimension(-1, 6),
                              ov::Dimension(8, 16),
                              ov::Dimension(9, 30),
                              15};
    // const rhs
    const auto const_op = ov::op::v0::Constant::create(ov::element::f32, {6}, {8, 2, 2, 4, 3, 5});
    const auto reshape = createReshapeSubgraph<ov::op::v1::Divide>(op_shape, const_op);
    ov::PartialShape expected_shape{-1, -1, ov::Dimension(-1, 3), ov::Dimension(2, 4), ov::Dimension(3, 10), 3};
    EXPECT_EQ(reshape->get_element_type(), ov::element::f32);
    EXPECT_EQ(reshape->get_output_partial_shape(0), expected_shape);
}

TEST(type_prop, interval_value_propagation_div_lhs) {
    ov::PartialShape op_shape{ov::Dimension(-1),
                              ov::Dimension(4, -1),
                              ov::Dimension(-1, 6),
                              ov::Dimension(8, 16),
                              ov::Dimension(9, 30),
                              15};
    // const lhs
    const auto const_op = ov::op::v0::Constant::create(ov::element::f32, {6}, {8, 8, 12, 32, 90, 45});
    const auto reshape = createReshapeSubgraph<ov::op::v1::Divide>(op_shape, const_op, false);
    ov::PartialShape expected_shape{-1, -1, ov::Dimension(2, -1), ov::Dimension(2, 4), ov::Dimension(3, 10), 3};
    EXPECT_EQ(reshape->get_element_type(), ov::element::f32);
    EXPECT_EQ(reshape->get_output_partial_shape(0), expected_shape);
}

TEST(type_prop, interval_value_propagation_pow_rhs) {
    ov::PartialShape op_shape{ov::Dimension(-1),
                              ov::Dimension(4, -1),
                              ov::Dimension(-1, 4),
                              ov::Dimension(2, 3),
                              ov::Dimension(3, 4),
                              2};
    // const rhs
    const auto const_op = ov::op::v0::Constant::create(ov::element::f32, {6}, {2, 2, 2, 2, 2, 2});
    const auto reshape = createReshapeSubgraph<ov::op::v1::Power>(op_shape, const_op);
    EXPECT_EQ(reshape->get_element_type(), ov::element::f32);
    EXPECT_EQ(reshape->get_output_partial_shape(0),
              ov::PartialShape({-1, -1, ov::Dimension(-1, 16), ov::Dimension(4, 9), ov::Dimension(9, 16), 4}));
}

TEST(type_prop, interval_value_propagation_pow_lhs) {
    ov::PartialShape op_shape{ov::Dimension(-1),
                              ov::Dimension(4, -1),
                              ov::Dimension(-1, 4),
                              ov::Dimension(2, 3),
                              ov::Dimension(3, 4),
                              2};
    // const lhs
    const auto const_op = ov::op::v0::Constant::create(ov::element::f32, {6}, {2, 2, 2, 2, 2, 2});
    const auto reshape = createReshapeSubgraph<ov::op::v1::Power>(op_shape, const_op, false);
    EXPECT_EQ(reshape->get_element_type(), ov::element::f32);
    EXPECT_EQ(reshape->get_output_partial_shape(0),
              ov::PartialShape({-1, -1, ov::Dimension(1, 16), ov::Dimension(4, 8), ov::Dimension(8, 16), 4}));
}

TEST(type_prop, interval_value_propagation_max_rhs) {
    ov::PartialShape op_shape{ov::Dimension(-1),
                              ov::Dimension(4, -1),
                              ov::Dimension(-1, 4),
                              ov::Dimension(-1, 4),
                              ov::Dimension(3, 5),
                              ov::Dimension(3, 5),
                              ov::Dimension(3, 5),
                              5,
                              8};
    // const rhs
    const auto const_op = ov::op::v0::Constant::create(ov::element::f32, {9}, {2, 2, 2, 6, 2, 4, 7, 8, 5});
    const auto reshape = createReshapeSubgraph<ov::op::v1::Maximum>(op_shape, const_op);
    EXPECT_EQ(reshape->get_element_type(), ov::element::f32);
    EXPECT_EQ(reshape->get_output_partial_shape(0),
              ov::PartialShape({-1, -1, ov::Dimension(2, 4), 6, ov::Dimension(3, 5), ov::Dimension(4, 5), 7, 8, 8}));
}

TEST(type_prop, interval_value_propagation_max_lhs) {
    ov::PartialShape op_shape{ov::Dimension(-1),
                              ov::Dimension(4, -1),
                              ov::Dimension(-1, 4),
                              ov::Dimension(-1, 4),
                              ov::Dimension(3, 5),
                              ov::Dimension(3, 5),
                              ov::Dimension(3, 5),
                              5,
                              8};
    // const lhs
    const auto const_op = ov::op::v0::Constant::create(ov::element::f32, {9}, {2, 2, 2, 6, 2, 4, 7, 8, 5});
    const auto reshape = createReshapeSubgraph<ov::op::v1::Maximum>(op_shape, const_op, false);
    EXPECT_EQ(reshape->get_element_type(), ov::element::f32);
    EXPECT_EQ(reshape->get_output_partial_shape(0),
              ov::PartialShape({-1, -1, ov::Dimension(2, 4), 6, ov::Dimension(3, 5), ov::Dimension(4, 5), 7, 8, 8}));
}

TEST(type_prop, interval_value_propagation_min_rhs) {
    ov::PartialShape op_shape{ov::Dimension(-1),
                              ov::Dimension(4, -1),
                              ov::Dimension(-1, 4),
                              ov::Dimension(-1, 4),
                              ov::Dimension(3, 5),
                              ov::Dimension(3, 5),
                              ov::Dimension(3, 5),
                              5,
                              8};
    // const rhs
    const auto const_op = ov::op::v0::Constant::create(ov::element::f32, {9}, {2, 2, 2, 6, 2, 4, 7, 8, 5});
    const auto reshape = createReshapeSubgraph<ov::op::v1::Minimum>(op_shape, const_op);
    EXPECT_EQ(reshape->get_element_type(), ov::element::f32);
    EXPECT_EQ(
        reshape->get_output_partial_shape(0),
        ov::PartialShape(
            {-1, -1, ov::Dimension(-1, 2), ov::Dimension(-1, 4), 2, ov::Dimension(3, 4), ov::Dimension(3, 5), 5, 5}));
}

TEST(type_prop, interval_value_propagation_min_lhs) {
    ov::PartialShape op_shape{ov::Dimension(-1),
                              ov::Dimension(4, -1),
                              ov::Dimension(-1, 4),
                              ov::Dimension(-1, 4),
                              ov::Dimension(3, 5),
                              ov::Dimension(3, 5),
                              ov::Dimension(3, 5),
                              5,
                              8};
    // const lhs
    const auto const_op = ov::op::v0::Constant::create(ov::element::f32, {9}, {2, 2, 2, 6, 2, 4, 7, 8, 5});
    const auto reshape = createReshapeSubgraph<ov::op::v1::Minimum>(op_shape, const_op, false);
    EXPECT_EQ(reshape->get_element_type(), ov::element::f32);
    EXPECT_EQ(
        reshape->get_output_partial_shape(0),
        ov::PartialShape(
            {-1, -1, ov::Dimension(-1, 2), ov::Dimension(-1, 4), 2, ov::Dimension(3, 4), ov::Dimension(3, 5), 5, 5}));
}

TEST(type_prop, interval_value_propagation_add_sub) {
    // ov::Dimensions with bounds
    auto param = make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                    ov::PartialShape{ov::Dimension(2, 8), ov::Dimension(4, 16), 2});

    auto shape_of = make_shared<ov::op::v3::ShapeOf>(param);
    auto cast_fp = make_shared<ov::op::v0::Convert>(shape_of, ov::element::f32);
    auto add = make_shared<ov::op::v1::Add>(
        cast_fp,
        ov::op::v0::Constant::create(ov::element::f32, {3}, {2, 3, 4}));  // {(4, 10), (7, 19), 6}
    auto sub = make_shared<ov::op::v1::Subtract>(
        add,
        ov::op::v0::Constant::create(ov::element::f32, {3}, {3, 2, 1}));  // {(1, 7), (5, 17), 5}
    auto cast_int = make_shared<ov::op::v0::Convert>(sub, ov::element::i32);

    auto reshape = make_shared<ov::op::v1::Reshape>(param, cast_int, false);

    EXPECT_EQ(reshape->get_element_type(), ov::element::f32);
    EXPECT_EQ(reshape->get_output_partial_shape(0), ov::PartialShape({ov::Dimension(1, 7), ov::Dimension(5, 17), 5}));
}

TEST(type_prop, interval_value_propagation_add_sub_no_bounds) {
    // Fully dynamic dimension, no upper, no lower bound
    auto param = make_shared<ov::op::v0::Parameter>(
        ov::element::f32,
        ov::PartialShape{ov::Dimension(-1), ov::Dimension(4, -1), ov::Dimension(-1, 2)});

    auto shape_of = make_shared<ov::op::v3::ShapeOf>(param);
    auto cast_fp = make_shared<ov::op::v0::Convert>(shape_of, ov::element::f32);
    auto add = make_shared<ov::op::v1::Add>(cast_fp, ov::op::v0::Constant::create(ov::element::f32, {3}, {2, 3, 4}));
    auto sub = make_shared<ov::op::v1::Subtract>(add, ov::op::v0::Constant::create(ov::element::f32, {3}, {3, 2, 1}));
    auto cast_int = make_shared<ov::op::v0::Convert>(sub, ov::element::i32);

    auto reshape = make_shared<ov::op::v1::Reshape>(param, cast_int, false);

    EXPECT_EQ(reshape->get_element_type(), ov::element::f32);
    EXPECT_EQ(reshape->get_output_partial_shape(0),
              ov::PartialShape(
                  {ov::Dimension(-1), ov::Dimension(-1), ov::Dimension(3, 5)}));  // Fully dynamic if no upper bound
}

TEST(type_prop, interval_value_propagation_add_sub_div_mul) {
    auto param = make_shared<ov::op::v0::Parameter>(
        ov::element::f32,
        ov::PartialShape{ov::Dimension(-1), ov::Dimension(2, 8), ov::Dimension(4, 10), 6});

    auto shape_of = make_shared<ov::op::v3::ShapeOf>(param);
    auto cast_fp = make_shared<ov::op::v0::Convert>(shape_of, ov::element::f32);
    auto add = make_shared<ov::op::v1::Add>(
        cast_fp,
        ov::op::v0::Constant::create(ov::element::f32, {4}, {2, 2, -1, 3}));  // {(-1), (4, 10), (3, 9), (9)}
    auto div = make_shared<ov::op::v1::Divide>(
        add,
        ov::op::v0::Constant::create(ov::element::f32, {4}, {2, 2, -3, 3}));  // {(-1), (2, 5), (-3, -1), (3)}
    auto sub = make_shared<ov::op::v1::Subtract>(
        div,
        ov::op::v0::Constant::create(ov::element::f32, {4}, {2, 1, 2, -4}));  // {(-1), (1, 4), (-5, -3), (7)}
    auto mul = make_shared<ov::op::v1::Multiply>(
        sub,
        ov::op::v0::Constant::create(ov::element::f32, {4}, {2, 3, -4, 5}));  // {(-1), (3, 12), (12, 20), (35)}
    auto cast_int = make_shared<ov::op::v0::Convert>(mul, ov::element::i32);

    auto reshape = make_shared<ov::op::v1::Reshape>(param, cast_int, false);

    EXPECT_EQ(reshape->get_element_type(), ov::element::f32);
    EXPECT_EQ(reshape->get_output_partial_shape(0),
              ov::PartialShape({ov::Dimension(-1), ov::Dimension(3, 12), ov::Dimension(12, 20), 35}));
}
