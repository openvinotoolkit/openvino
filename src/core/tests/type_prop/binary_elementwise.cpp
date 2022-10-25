// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "util/type_prop.hpp"

using namespace std;
using namespace ngraph;
using namespace testing;

//
// Tests for binary elementwise ops.
//
void test_binary(std::string /* node_type */,
                 shared_ptr<Node>(f)(const shared_ptr<Node>& x, const shared_ptr<Node>& y)) {
    // Check for bad arguments
    auto tv0_2_4_param_0 = make_shared<op::Parameter>(element::f32, Shape{2, 4});
    auto tv0_2_4_param_1 = make_shared<op::Parameter>(element::f32, Shape{2, 4});
    auto tv0_2_4_param_2 = make_shared<op::Parameter>(element::i32, Shape{2, 4});
    auto tv0_4_2_param = make_shared<op::Parameter>(element::f32, Shape{4, 2});

    auto test_binary_bad_arguments_view_shapes = [&](const shared_ptr<Node>& x, const shared_ptr<Node>& y) {
        try {
            auto node = f(x, y);
            // Should have thrown, so fail if it didn't
            FAIL() << "Incompatible view arguments not detected.";
        } catch (const NodeValidationFailure& error) {
            EXPECT_HAS_SUBSTRING(error.what(), std::string("Argument shapes are inconsistent"));
        } catch (...) {
            FAIL() << "Deduced type check failed for unexpected reason";
        }
    };
    test_binary_bad_arguments_view_shapes(tv0_2_4_param_0, tv0_4_2_param);

    auto test_binary_bad_arguments_view_element_types = [&](const shared_ptr<Node>& x, const shared_ptr<Node>& y) {
        try {
            auto node = f(x, y);
            // Should have thrown, so fail if it didn't
            FAIL() << "Incompatible view arguments not detected.";
        } catch (const NodeValidationFailure& error) {
            EXPECT_HAS_SUBSTRING(error.what(), std::string("Arguments do not have the same element type"));
        } catch (...) {
            FAIL() << "Deduced type check failed for unexpected reason";
        }
    };

    test_binary_bad_arguments_view_element_types(tv0_2_4_param_0, tv0_2_4_param_2);

    auto test_binary_good_arguments = [&](const shared_ptr<Node>& x, const shared_ptr<Node>& y) {
        auto node = f(x, y);
        EXPECT_TRUE(node->has_same_type(node->input_values()[0].get_node_shared_ptr()));
    };
    test_binary_good_arguments(tv0_2_4_param_0, tv0_2_4_param_1);
}

TEST(type_prop, add_bad_arguments) {
    test_binary("Add", [](const shared_ptr<Node>& x, const shared_ptr<Node>& y) -> shared_ptr<Node> {
        return make_shared<op::v1::Add>(x, y);
    });
}

//
// Tests for binary elementwise logical ops.
//
void test_binary_logical(std::string /* node_type */,
                         shared_ptr<Node>(f)(const shared_ptr<Node>& x, const shared_ptr<Node>& y)) {
    // Check for bad arguments
    auto tv0_2_4_param_0 = make_shared<op::Parameter>(element::boolean, Shape{2, 4});
    auto tv0_2_4_param_1 = make_shared<op::Parameter>(element::boolean, Shape{2, 4});
    auto tv0_2_4_param_2 = make_shared<op::Parameter>(element::i32, Shape{2, 4});
    auto tv0_2_4_param_3 = make_shared<op::Parameter>(element::i32, Shape{2, 4});
    auto tv0_4_2_param = make_shared<op::Parameter>(element::boolean, Shape{4, 2});

    auto test_binary_bad_arguments_view_shapes = [&](const shared_ptr<Node>& x, const shared_ptr<Node>& y) {
        try {
            auto node = f(x, y);
            // Should have thrown, so fail if it didn't
            FAIL() << "Incompatible view arguments not detected.";
        } catch (const NodeValidationFailure& error) {
            EXPECT_HAS_SUBSTRING(error.what(), std::string("Argument shapes are inconsistent"));
        } catch (...) {
            FAIL() << "Deduced type check failed for unexpected reason";
        }
    };
    test_binary_bad_arguments_view_shapes(tv0_2_4_param_0, tv0_4_2_param);

    auto test_binary_differ_arguments_view_element_types = [&](const shared_ptr<Node>& x, const shared_ptr<Node>& y) {
        try {
            auto node = f(x, y);
            // Should have thrown, so fail if it didn't
            FAIL() << "Incompatible view arguments not detected.";
        } catch (const NodeValidationFailure& error) {
            EXPECT_HAS_SUBSTRING(error.what(), std::string("Arguments do not have the same element type"));
        } catch (...) {
            FAIL() << "Deduced type check failed for unexpected reason";
        }
    };

    auto test_binary_non_bool_arguments_view_element_types = [&](const shared_ptr<Node>& x, const shared_ptr<Node>& y) {
        try {
            auto node = f(x, y);
            // Should have thrown, so fail if it didn't
            FAIL() << "Incompatible view arguments not detected.";
        } catch (const ngraph_error& error) {
            EXPECT_HAS_SUBSTRING(error.what(), "must have boolean element type");
        } catch (...) {
            FAIL() << "Deduced type check failed for unexpected reason";
        }
    };

    test_binary_differ_arguments_view_element_types(tv0_2_4_param_0, tv0_2_4_param_2);
    test_binary_differ_arguments_view_element_types(tv0_2_4_param_2, tv0_2_4_param_0);
    test_binary_non_bool_arguments_view_element_types(tv0_2_4_param_2, tv0_2_4_param_3);

    auto test_binary_good_arguments = [&](const shared_ptr<Node>& x, const shared_ptr<Node>& y) {
        auto node = f(x, y);
        EXPECT_TRUE(node->has_same_type(node->input_values()[0].get_node_shared_ptr()));
    };
    test_binary_good_arguments(tv0_2_4_param_0, tv0_2_4_param_1);
}

TEST(type_prop, or_bad_arguments) {
    test_binary_logical("Or", [](const shared_ptr<Node>& x, const shared_ptr<Node>& y) -> shared_ptr<Node> {
        return make_shared<op::v1::LogicalOr>(x, y);
    });
}

TEST(type_prop, xor_bad_arguments) {
    test_binary_logical("Xor", [](const shared_ptr<Node>& x, const shared_ptr<Node>& y) -> shared_ptr<Node> {
        return make_shared<op::Xor>(x, y);
    });
}

template <typename T>
void test_binary_eltwise_numpy(const element::Type& et, const op::AutoBroadcastSpec& autob) {
    auto param1 = make_shared<op::Parameter>(et, Shape{1, 3, 6});
    auto param2 = make_shared<op::Parameter>(et, Shape{3, 1});
    auto param3 = make_shared<op::Parameter>(et, Shape{2, 3, 6});
    auto param4 = make_shared<op::Parameter>(et, Shape{6});
    auto param5 = make_shared<op::Parameter>(et, Shape{});

    EXPECT_EQ(make_shared<T>(param1, param2, autob)->get_shape(), (Shape{1, 3, 6}));
    EXPECT_EQ(make_shared<T>(param1, param3, autob)->get_shape(), (Shape{2, 3, 6}));
    EXPECT_EQ(make_shared<T>(param4, param3, autob)->get_shape(), (Shape{2, 3, 6}));
    EXPECT_EQ(make_shared<T>(param5, param3, autob)->get_shape(), (Shape{2, 3, 6}));
    EXPECT_EQ(make_shared<T>(param3, param5, autob)->get_shape(), (Shape{2, 3, 6}));

    auto pp1 = make_shared<op::Parameter>(et, PartialShape{1, Dimension::dynamic(), 6});
    auto pp2 = make_shared<op::Parameter>(et, PartialShape{3, 1});
    EXPECT_EQ(make_shared<T>(pp1, pp2, autob)->get_shape(), (Shape{1, 3, 6}));
}

TEST(type_prop, eltwise_auto_bcast) {
    test_binary_eltwise_numpy<op::v1::Add>(element::f32, op::AutoBroadcastType::NUMPY);
    test_binary_eltwise_numpy<op::v1::Equal>(element::f32, op::AutoBroadcastType::NUMPY);
    test_binary_eltwise_numpy<op::v1::Greater>(element::f32, op::AutoBroadcastType::NUMPY);
    test_binary_eltwise_numpy<op::v1::GreaterEqual>(element::f32, op::AutoBroadcastType::NUMPY);
    test_binary_eltwise_numpy<op::v1::Less>(element::f32, op::AutoBroadcastType::NUMPY);
    test_binary_eltwise_numpy<op::v1::LessEqual>(element::f32, op::AutoBroadcastType::NUMPY);
    test_binary_eltwise_numpy<op::v1::Maximum>(element::f32, op::AutoBroadcastType::NUMPY);
    test_binary_eltwise_numpy<op::v1::NotEqual>(element::f32, op::AutoBroadcastType::NUMPY);
    test_binary_eltwise_numpy<op::v1::LogicalOr>(element::boolean, op::AutoBroadcastType::NUMPY);
    test_binary_eltwise_numpy<op::Xor>(element::boolean, op::AutoBroadcastType::NUMPY);
}

TEST(type_prop, comparison_good) {
    auto tv0_2_4_param_0 = make_shared<op::Parameter>(element::f32, Shape{2, 4});
    auto tv0_2_4_param_1 = make_shared<op::Parameter>(element::f32, Shape{2, 4});
    auto eq = make_shared<op::v1::Equal>(tv0_2_4_param_0, tv0_2_4_param_1);
    EXPECT_EQ(eq->get_element_type(), element::boolean);
    EXPECT_EQ(eq->get_shape(), (Shape{2, 4}));
}

TEST(type_prop, binary_arithmetic_bad_argument_element_types) {
    auto tv0_2_4_param_0 = make_shared<op::Parameter>(element::boolean, Shape{2, 4});
    auto tv0_2_4_param_1 = make_shared<op::Parameter>(element::boolean, Shape{2, 4});
    try {
        auto bc = make_shared<op::v1::Add>(tv0_2_4_param_0, tv0_2_4_param_1);
        // Should have thrown, so fail if it didn't
        FAIL() << "Did not detect incorrect element types for arithmetic operator";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Arguments cannot have boolean element type"));
    } catch (...) {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

namespace {
template <typename T>
void test_binary_eltwise_bad_argument_shape(const element::Type& et) {
    auto input1 = make_shared<op::Parameter>(element::f32, Shape{2, 4});
    auto input2 = make_shared<op::Parameter>(element::f32, Shape{1, 2, 4});
    try {
        auto bc = make_shared<T>(input1, input2, op::AutoBroadcastType::NONE);
        // Should have thrown, so fail if it didn't
        FAIL() << "Did not detect incorrect element types for arithmetic operator";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Argument shapes are inconsistent"));
    } catch (...) {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}
}  // namespace

TEST(type_prop, binary_arithmetic_bad_argument_shape_with_none_autobroadcast_attribute) {
    test_binary_eltwise_bad_argument_shape<op::v1::Add>(element::f32);
    test_binary_eltwise_bad_argument_shape<op::v1::Equal>(element::f32);
    test_binary_eltwise_bad_argument_shape<op::v1::Greater>(element::f32);
    test_binary_eltwise_bad_argument_shape<op::v1::GreaterEqual>(element::f32);
    test_binary_eltwise_bad_argument_shape<op::v1::Less>(element::f32);
    test_binary_eltwise_bad_argument_shape<op::v1::LessEqual>(element::f32);
    test_binary_eltwise_bad_argument_shape<op::v1::Maximum>(element::f32);
    test_binary_eltwise_bad_argument_shape<op::v1::NotEqual>(element::f32);
    test_binary_eltwise_bad_argument_shape<op::v1::LogicalOr>(element::boolean);
    test_binary_eltwise_bad_argument_shape<op::Xor>(element::boolean);
}

TEST(type_prop, binary_elementwise_arithmetic_both_dynamic) {
    auto a = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto b = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto add = make_shared<op::v1::Add>(a, b);

    ASSERT_TRUE(add->get_output_partial_shape(0).rank().is_dynamic());
}

TEST(type_prop, binary_elementwise_arithmetic_left_rank_static_dynamic_right_rank_static_dynamic_result_static) {
    auto a = make_shared<op::Parameter>(element::f32, PartialShape{1, Dimension::dynamic(), 3});
    auto b = make_shared<op::Parameter>(element::f32, PartialShape{1, 2, Dimension::dynamic()});
    auto add = make_shared<op::v1::Add>(a, b);

    ASSERT_TRUE(add->get_output_partial_shape(0).is_static());
    ASSERT_EQ(add->get_shape(), (Shape{1, 2, 3}));
}

TEST(type_prop,
     binary_elementwise_arithmetic_left_rank_static_dynamic_right_rank_static_dynamic_result_rank_static_dynamic) {
    auto a = make_shared<op::Parameter>(element::f32, PartialShape{1, Dimension::dynamic(), Dimension::dynamic()});
    auto b = make_shared<op::Parameter>(element::f32, PartialShape{1, 2, Dimension::dynamic()});
    auto add = make_shared<op::v1::Add>(a, b);

    ASSERT_TRUE(add->get_output_partial_shape(0).rank().is_static());
    ASSERT_TRUE(add->get_output_partial_shape(0).is_dynamic());
    ASSERT_TRUE(add->get_output_partial_shape(0).same_scheme(PartialShape{1, 2, Dimension::dynamic()}));
}

TEST(type_prop, binary_elementwise_arithmetic_left_static_right_rank_static_dynamic) {
    auto a = make_shared<op::Parameter>(element::f32, PartialShape{1, 2, 3});
    auto b = make_shared<op::Parameter>(element::f32, PartialShape{1, 2, Dimension::dynamic()});
    auto add = make_shared<op::v1::Add>(a, b);

    ASSERT_TRUE(add->get_output_partial_shape(0).is_static());
    ASSERT_EQ(add->get_shape(), (Shape{1, 2, 3}));
}

TEST(type_prop, binary_elementwise_arithmetic_left_rank_static_dynamic_right_static) {
    auto a = make_shared<op::Parameter>(element::f32, PartialShape{1, 2, Dimension::dynamic()});
    auto b = make_shared<op::Parameter>(element::f32, PartialShape{1, 2, 3});
    auto add = make_shared<op::v1::Add>(a, b);

    ASSERT_TRUE(add->get_output_partial_shape(0).is_static());
    ASSERT_EQ(add->get_shape(), (Shape{1, 2, 3}));
}

TEST(type_prop, binary_elementwise_arithmetic_left_rank_static_dynamic_inconsistent) {
    auto a = make_shared<op::Parameter>(element::f32, PartialShape{1, 2, Dimension::dynamic()});
    auto b = make_shared<op::Parameter>(element::f32, PartialShape{1, 3, 3});

    try {
        auto add = make_shared<op::v1::Add>(a, b);
        FAIL() << "Inconsistent partial shapes not detected";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), "Argument shapes are inconsistent");
    } catch (...) {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, binary_elementwise_arithmetic_right_rank_static_dynamic_inconsistent) {
    auto a = make_shared<op::Parameter>(element::f32, PartialShape{1, 3, 3});
    auto b = make_shared<op::Parameter>(element::f32, PartialShape{1, 2, Dimension::dynamic()});

    try {
        auto add = make_shared<op::v1::Add>(a, b);
        FAIL() << "Inconsistent partial shapes not detected";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), "Argument shapes are inconsistent");
    } catch (...) {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, binary_elementwise_arithmetic_both_rank_static_dynamic_inconsistent) {
    auto a = make_shared<op::Parameter>(element::f32, PartialShape{Dimension::dynamic(), 3, 3});
    auto b = make_shared<op::Parameter>(element::f32, PartialShape{1, 2, Dimension::dynamic()});

    try {
        auto add = make_shared<op::v1::Add>(a, b);
        FAIL() << "Inconsistent partial shapes not detected";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), "Argument shapes are inconsistent");
    } catch (...) {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, binary_elementwise_arithmetic_left_rank_static_dynamic_different_rank) {
    auto a = make_shared<op::Parameter>(element::f32, PartialShape{1, 2, Dimension::dynamic()});
    auto b = make_shared<op::Parameter>(element::f32, PartialShape{1, 2, 3, 4});

    try {
        auto add = make_shared<op::v1::Add>(a, b);
        FAIL() << "Inconsistent partial shapes not detected";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), "Argument shapes are inconsistent");
    } catch (...) {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, binary_elementwise_arithmetic_right_rank_static_dynamic_different_rank) {
    auto a = make_shared<op::Parameter>(element::f32, PartialShape{1, 2, 3, 4});
    auto b = make_shared<op::Parameter>(element::f32, PartialShape{1, 2, Dimension::dynamic()});

    try {
        auto add = make_shared<op::v1::Add>(a, b);
        FAIL() << "Inconsistent partial shapes not detected";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), "Argument shapes are inconsistent");
    } catch (...) {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, binary_elementwise_arithmetic_both_rank_static_dynamic_different_rank) {
    auto a = make_shared<op::Parameter>(element::f32, PartialShape{1, Dimension::dynamic(), 3, 4});
    auto b = make_shared<op::Parameter>(element::f32, PartialShape{1, 2, Dimension::dynamic()});

    try {
        auto add = make_shared<op::v1::Add>(a, b);
        FAIL() << "Inconsistent partial shapes not detected";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), "Argument shapes are inconsistent");
    } catch (...) {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, binary_elementwise_arithmetic_both_et_dynamic) {
    auto a = make_shared<op::Parameter>(element::dynamic, Shape{1, 2, 3, 4});
    auto b = make_shared<op::Parameter>(element::dynamic, Shape{1, 2, 3, 4});
    auto add = make_shared<op::v1::Add>(a, b);

    ASSERT_TRUE(add->get_output_element_type(0).is_dynamic());
}

TEST(type_prop, binary_elementwise_arithmetic_left_et_dynamic) {
    auto a = make_shared<op::Parameter>(element::dynamic, Shape{1, 2, 3, 4});
    auto b = make_shared<op::Parameter>(element::u32, Shape{1, 2, 3, 4});
    auto add = make_shared<op::v1::Add>(a, b);

    ASSERT_EQ(add->get_output_element_type(0), element::u32);
}

TEST(type_prop, binary_elementwise_arithmetic_right_et_dynamic) {
    auto a = make_shared<op::Parameter>(element::i64, Shape{1, 2, 3, 4});
    auto b = make_shared<op::Parameter>(element::dynamic, Shape{1, 2, 3, 4});
    auto add = make_shared<op::v1::Add>(a, b);

    ASSERT_EQ(add->get_output_element_type(0), element::i64);
}

TEST(type_prop, logic_arith_compare_partial_et) {
    auto test_arith = [](element::Type et0, element::Type et1) -> std::shared_ptr<Node> {
        auto param0 = std::make_shared<op::Parameter>(et0, Shape{1, 2, 3});
        auto param1 = std::make_shared<op::Parameter>(et1, Shape{1, 2, 3});
        return std::make_shared<op::v1::Add>(param0, param1);
    };

    auto test_compare = [](element::Type et0, element::Type et1) -> std::shared_ptr<Node> {
        auto param0 = std::make_shared<op::Parameter>(et0, Shape{1, 2, 3});
        auto param1 = std::make_shared<op::Parameter>(et1, Shape{1, 2, 3});
        return std::make_shared<op::v1::Greater>(param0, param1);
    };

    auto test_logical_not = [](element::Type et) -> std::shared_ptr<Node> {
        auto param = std::make_shared<op::Parameter>(et, Shape{1, 2, 3});
        return std::make_shared<op::v1::LogicalNot>(param);
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
    ASSERT_EQ(test_arith(element::i32, element::i32)->get_element_type(), element::i32);
    ASSERT_ANY_THROW({ test_arith(element::i32, element::boolean); });
    ASSERT_EQ(test_arith(element::i32, element::dynamic)->get_element_type(), element::i32);
    ASSERT_ANY_THROW({ test_arith(element::boolean, element::i32); });
    ASSERT_ANY_THROW({ test_arith(element::boolean, element::boolean); });
    ASSERT_ANY_THROW({ test_arith(element::boolean, element::dynamic); });
    ASSERT_EQ(test_arith(element::dynamic, element::i32)->get_element_type(), element::i32);
    ASSERT_ANY_THROW({ test_arith(element::dynamic, element::boolean); });
    ASSERT_EQ(test_arith(element::dynamic, element::dynamic)->get_element_type(), element::dynamic);

    // Comparison ops:
    //
    // int int -> boo
    // int boo -> !
    // int dyn -> boo
    // boo int -> !
    // boo boo -> boo
    // boo dyn -> boo
    // dyn int -> boo
    // dyn boo -> boo
    // dyn dyn -> boo
    ASSERT_EQ(test_compare(element::i32, element::i32)->get_element_type(), element::boolean);
    ASSERT_ANY_THROW({ test_compare(element::i32, element::boolean); });
    ASSERT_EQ(test_compare(element::i32, element::dynamic)->get_element_type(), element::boolean);
    ASSERT_ANY_THROW({ test_compare(element::boolean, element::i32); });
    ASSERT_EQ(test_compare(element::boolean, element::boolean)->get_element_type(), element::boolean);
    ASSERT_EQ(test_compare(element::boolean, element::dynamic)->get_element_type(), element::boolean);
    ASSERT_EQ(test_compare(element::dynamic, element::i32)->get_element_type(), element::boolean);
    ASSERT_EQ(test_compare(element::dynamic, element::boolean)->get_element_type(), element::boolean);
    ASSERT_EQ(test_compare(element::dynamic, element::dynamic)->get_element_type(), element::boolean);

    // Logical negation op:
    //
    // Current behavior:
    // int -> int
    // boo -> boo
    // dyn -> dyn
    //
    // TODO(amprocte): I believe the behavior should actually be:
    // int -> !
    // boo -> boo
    // dyn -> boo
    ASSERT_EQ(test_logical_not(element::i32)->get_element_type(), element::i32);
    ASSERT_EQ(test_logical_not(element::boolean)->get_element_type(), element::boolean);
    ASSERT_EQ(test_logical_not(element::dynamic)->get_element_type(), element::dynamic);
}

namespace {
template <class T>
shared_ptr<op::v1::Reshape> createReshapeSubgraph(PartialShape param_shape,
                                                  shared_ptr<op::Constant> constant_op,
                                                  bool const_rhs = true) {
    auto param = make_shared<op::Parameter>(element::f32, param_shape);
    auto shape_of = make_shared<op::v3::ShapeOf>(param);
    auto cast_fp = make_shared<op::Convert>(shape_of, element::f32);

    Output<Node> op;
    if (const_rhs)
        op = make_shared<T>(cast_fp, constant_op);
    else
        op = make_shared<T>(constant_op, cast_fp);

    auto cast_int = make_shared<op::Convert>(op, element::i32);
    auto r = make_shared<op::v1::Reshape>(param, cast_int, false);

    return r;
}
}  // namespace

TEST(type_prop, interval_value_propagation_add_rhs) {
    PartialShape op_shape{Dimension(-1), Dimension(2, -1), Dimension(-1, 6), Dimension(7, 10), Dimension(7, 10), 5};
    const auto const_op = op::Constant::create(element::f32, {6}, {2, 3, 4, 5, -5, 6});
    // const rhs
    const auto reshape = createReshapeSubgraph<op::v1::Add>(op_shape, const_op);
    EXPECT_EQ(reshape->get_element_type(), element::f32);
    EXPECT_EQ(reshape->get_output_partial_shape(0),
              PartialShape({-1, -1, Dimension(4, 10), Dimension(12, 15), Dimension(2, 5), 11}));
}

TEST(type_prop, interval_value_propagation_add_lhs) {
    PartialShape op_shape{Dimension(-1), Dimension(2, -1), Dimension(-1, 6), Dimension(7, 10), Dimension(7, 10), 5};
    const auto const_op = op::Constant::create(element::f32, {6}, {2, 3, 4, 5, -5, 6});
    // const lhs
    const auto reshape = createReshapeSubgraph<op::v1::Add>(op_shape, const_op, false);
    EXPECT_EQ(reshape->get_element_type(), element::f32);
    EXPECT_EQ(reshape->get_output_partial_shape(0),
              PartialShape({-1, -1, Dimension(4, 10), Dimension(12, 15), Dimension(2, 5), 11}));
}

TEST(type_prop, interval_value_propagation_add_incorrect_dim) {
    // const rhs - result lower than 0
    PartialShape op_shape{Dimension(5, 7)};
    const auto const_op = op::Constant::create(element::f32, {1}, {-10});
    OV_EXPECT_THROW(createReshapeSubgraph<op::v1::Add>(op_shape, const_op),
                    NodeValidationFailure,
                    HasSubstr("Dim size cannot be less than -1"));
}

TEST(type_prop, interval_value_propagation_sub_rhs) {
    PartialShape op_shape{Dimension(-1), Dimension(24, -1), Dimension(4, 36), Dimension(13, 27), Dimension(13, 27), 15};
    // const rhs
    const auto const_op = op::Constant::create(element::f32, {6}, {2, 3, 4, 5, -5, 6});
    const auto reshape = createReshapeSubgraph<op::v1::Subtract>(op_shape, const_op);
    EXPECT_EQ(reshape->get_element_type(), element::f32);
    EXPECT_EQ(reshape->get_output_partial_shape(0),
              PartialShape({-1, -1, Dimension(-1, 32), Dimension(8, 22), Dimension(18, 32), 9}));
}

TEST(type_prop, interval_value_propagation_sub_lhs) {
    PartialShape op_shape{Dimension(-1), Dimension(24, -1), Dimension(4, 36), Dimension(13, 27), Dimension(13, 27), 15};
    // const lhs
    const auto const_op = op::Constant::create(element::f32, {6}, {12, 28, 36, 43, 27, 25});
    const auto reshape = createReshapeSubgraph<op::v1::Subtract>(op_shape, const_op, false);
    EXPECT_EQ(reshape->get_element_type(), element::f32);
    EXPECT_EQ(reshape->get_output_partial_shape(0),
              PartialShape({-1, -1, Dimension(0, 32), Dimension(16, 30), Dimension(0, 14), 10}));
}

TEST(type_prop, interval_value_propagation_sub_incorrect_dim) {
    // const lhs - result lower than 0
    PartialShape op_shape{Dimension(13, 27)};
    const auto const_op = op::Constant::create(element::f32, {1}, {5});
    OV_EXPECT_THROW(createReshapeSubgraph<op::v1::Subtract>(op_shape, const_op, false),
                    NodeValidationFailure,
                    HasSubstr("Dim size cannot be less than -1"));
}

TEST(type_prop, interval_value_propagation_mul_rhs) {
    PartialShape op_shape{Dimension(-1), Dimension(4, -1), Dimension(-1, 6), Dimension(5, 7), Dimension(9, 10), 15};
    // const rhs
    const auto const_op = op::Constant::create(element::f32, {6}, {7, 6, 5, 4, 3, 2});
    const auto reshape = createReshapeSubgraph<op::v1::Multiply>(op_shape, const_op);
    EXPECT_EQ(reshape->get_element_type(), element::f32);
    EXPECT_EQ(reshape->get_output_partial_shape(0),
              PartialShape({-1, -1, Dimension(-1, 30), Dimension(20, 28), Dimension(27, 30), 30}));
}

TEST(type_prop, interval_value_propagation_mul_lhs) {
    PartialShape op_shape{Dimension(-1), Dimension(4, -1), Dimension(-1, 6), Dimension(5, 7), Dimension(9, 10), 15};

    // const lhs
    const auto const_op = op::Constant::create(element::f32, {6}, {7, 6, 5, 4, 3, 2});
    const auto reshape = createReshapeSubgraph<op::v1::Multiply>(op_shape, const_op, false);
    EXPECT_EQ(reshape->get_element_type(), element::f32);
    EXPECT_EQ(reshape->get_output_partial_shape(0),
              PartialShape({-1, -1, Dimension(-1, 30), Dimension(20, 28), Dimension(27, 30), 30}));
}

TEST(type_prop, interval_value_propagation_mul_incorrect_dim_rhs) {
    // const rhs - result lower than 0
    PartialShape op_shape{Dimension(5, 7)};
    const auto const_op = op::Constant::create(element::f32, {1}, {-3});
    OV_EXPECT_THROW(createReshapeSubgraph<op::v1::Multiply>(op_shape, const_op),
                    NodeValidationFailure,
                    HasSubstr("Dim size cannot be less than -1"));
}

TEST(type_prop, interval_value_propagation_mul_incorrect_dim_lhs) {
    // const lhs - result lower than 0
    PartialShape op_shape{Dimension(5, 7)};
    const auto const_op = op::Constant::create(element::f32, {1}, {-3});
    OV_EXPECT_THROW(createReshapeSubgraph<op::v1::Multiply>(op_shape, const_op, false),
                    NodeValidationFailure,
                    HasSubstr("Dim size cannot be less than -1"));
}

TEST(type_prop, interval_value_propagation_div_rhs) {
    // const rhs
    PartialShape op_shape{Dimension(8, 16), Dimension(9, 30), 15};
    const auto const_op = op::Constant::create(element::f32, {3}, {4, 3, 5});
    const auto reshape = createReshapeSubgraph<op::v1::Divide>(op_shape, const_op);
    PartialShape expected_shape{Dimension(2, 4), Dimension(3, 10), 3};
    EXPECT_EQ(reshape->get_element_type(), element::f32);
    EXPECT_EQ(reshape->get_output_partial_shape(0), expected_shape);
}

TEST(type_prop, interval_value_propagation_div_rhs_full) {
    PartialShape op_shape{Dimension(-1), Dimension(4, -1), Dimension(-1, 6), Dimension(8, 16), Dimension(9, 30), 15};
    // const rhs
    const auto const_op = op::Constant::create(element::f32, {6}, {8, 2, 2, 4, 3, 5});
    const auto reshape = createReshapeSubgraph<op::v1::Divide>(op_shape, const_op);
    PartialShape expected_shape{-1, -1, Dimension(-1, 3), Dimension(2, 4), Dimension(3, 10), 3};
    EXPECT_EQ(reshape->get_element_type(), element::f32);
    EXPECT_EQ(reshape->get_output_partial_shape(0), expected_shape);
}

TEST(type_prop, interval_value_propagation_div_lhs) {
    PartialShape op_shape{Dimension(-1), Dimension(4, -1), Dimension(-1, 6), Dimension(8, 16), Dimension(9, 30), 15};
    // const lhs
    const auto const_op = op::Constant::create(element::f32, {6}, {8, 8, 12, 32, 90, 45});
    const auto reshape = createReshapeSubgraph<op::v1::Divide>(op_shape, const_op, false);
    PartialShape expected_shape{-1, -1, Dimension(2, -1), Dimension(2, 4), Dimension(3, 10), 3};
    EXPECT_EQ(reshape->get_element_type(), element::f32);
    EXPECT_EQ(reshape->get_output_partial_shape(0), expected_shape);
}

TEST(type_prop, interval_value_propagation_pow_rhs) {
    PartialShape op_shape{Dimension(-1), Dimension(4, -1), Dimension(-1, 4), Dimension(2, 3), Dimension(3, 4), 2};
    // const rhs
    const auto const_op = op::Constant::create(element::f32, {6}, {2, 2, 2, 2, 2, 2});
    const auto reshape = createReshapeSubgraph<op::v1::Power>(op_shape, const_op);
    EXPECT_EQ(reshape->get_element_type(), element::f32);
    EXPECT_EQ(reshape->get_output_partial_shape(0),
              PartialShape({-1, -1, Dimension(-1, 16), Dimension(4, 9), Dimension(9, 16), 4}));
}

TEST(type_prop, interval_value_propagation_pow_lhs) {
    PartialShape op_shape{Dimension(-1), Dimension(4, -1), Dimension(-1, 4), Dimension(2, 3), Dimension(3, 4), 2};
    // const lhs
    const auto const_op = op::Constant::create(element::f32, {6}, {2, 2, 2, 2, 2, 2});
    const auto reshape = createReshapeSubgraph<op::v1::Power>(op_shape, const_op, false);
    EXPECT_EQ(reshape->get_element_type(), element::f32);
    EXPECT_EQ(reshape->get_output_partial_shape(0),
              PartialShape({-1, -1, Dimension(1, 16), Dimension(4, 8), Dimension(8, 16), 4}));
}

TEST(type_prop, interval_value_propagation_max_rhs) {
    PartialShape op_shape{Dimension(-1),
                          Dimension(4, -1),
                          Dimension(-1, 4),
                          Dimension(-1, 4),
                          Dimension(3, 5),
                          Dimension(3, 5),
                          Dimension(3, 5),
                          5,
                          8};
    // const rhs
    const auto const_op = op::Constant::create(element::f32, {9}, {2, 2, 2, 6, 2, 4, 7, 8, 5});
    const auto reshape = createReshapeSubgraph<op::v1::Maximum>(op_shape, const_op);
    EXPECT_EQ(reshape->get_element_type(), element::f32);
    EXPECT_EQ(reshape->get_output_partial_shape(0),
              PartialShape({-1, -1, Dimension(2, 4), 6, Dimension(3, 5), Dimension(4, 5), 7, 8, 8}));
}

TEST(type_prop, interval_value_propagation_max_lhs) {
    PartialShape op_shape{Dimension(-1),
                          Dimension(4, -1),
                          Dimension(-1, 4),
                          Dimension(-1, 4),
                          Dimension(3, 5),
                          Dimension(3, 5),
                          Dimension(3, 5),
                          5,
                          8};
    // const lhs
    const auto const_op = op::Constant::create(element::f32, {9}, {2, 2, 2, 6, 2, 4, 7, 8, 5});
    const auto reshape = createReshapeSubgraph<op::v1::Maximum>(op_shape, const_op, false);
    EXPECT_EQ(reshape->get_element_type(), element::f32);
    EXPECT_EQ(reshape->get_output_partial_shape(0),
              PartialShape({-1, -1, Dimension(2, 4), 6, Dimension(3, 5), Dimension(4, 5), 7, 8, 8}));
}

TEST(type_prop, interval_value_propagation_min_rhs) {
    PartialShape op_shape{Dimension(-1),
                          Dimension(4, -1),
                          Dimension(-1, 4),
                          Dimension(-1, 4),
                          Dimension(3, 5),
                          Dimension(3, 5),
                          Dimension(3, 5),
                          5,
                          8};
    // const rhs
    const auto const_op = op::Constant::create(element::f32, {9}, {2, 2, 2, 6, 2, 4, 7, 8, 5});
    const auto reshape = createReshapeSubgraph<op::v1::Minimum>(op_shape, const_op);
    EXPECT_EQ(reshape->get_element_type(), element::f32);
    EXPECT_EQ(reshape->get_output_partial_shape(0),
              PartialShape({-1, -1, Dimension(-1, 2), Dimension(-1, 4), 2, Dimension(3, 4), Dimension(3, 5), 5, 5}));
}

TEST(type_prop, interval_value_propagation_min_lhs) {
    PartialShape op_shape{Dimension(-1),
                          Dimension(4, -1),
                          Dimension(-1, 4),
                          Dimension(-1, 4),
                          Dimension(3, 5),
                          Dimension(3, 5),
                          Dimension(3, 5),
                          5,
                          8};
    // const lhs
    const auto const_op = op::Constant::create(element::f32, {9}, {2, 2, 2, 6, 2, 4, 7, 8, 5});
    const auto reshape = createReshapeSubgraph<op::v1::Minimum>(op_shape, const_op, false);
    EXPECT_EQ(reshape->get_element_type(), element::f32);
    EXPECT_EQ(reshape->get_output_partial_shape(0),
              PartialShape({-1, -1, Dimension(-1, 2), Dimension(-1, 4), 2, Dimension(3, 4), Dimension(3, 5), 5, 5}));
}

TEST(type_prop, interval_value_propagation_add_sub) {
    // Dimensions with bounds
    auto param = make_shared<op::Parameter>(element::f32, PartialShape{Dimension(2, 8), Dimension(4, 16), 2});

    auto shape_of = make_shared<op::v3::ShapeOf>(param);
    auto cast_fp = make_shared<op::Convert>(shape_of, element::f32);
    auto add =
        make_shared<op::v1::Add>(cast_fp, op::Constant::create(element::f32, {3}, {2, 3, 4}));  // {(4, 10), (7, 19), 6}
    auto sub =
        make_shared<op::v1::Subtract>(add, op::Constant::create(element::f32, {3}, {3, 2, 1}));  // {(1, 7), (5, 17), 5}
    auto cast_int = make_shared<op::Convert>(sub, element::i32);

    auto reshape = make_shared<op::v1::Reshape>(param, cast_int, false);

    EXPECT_EQ(reshape->get_element_type(), element::f32);
    EXPECT_EQ(reshape->get_output_partial_shape(0), PartialShape({Dimension(1, 7), Dimension(5, 17), 5}));
}

TEST(type_prop, interval_value_propagation_add_sub_no_bounds) {
    // Fully dynamic dimension, no upper, no lower bound
    auto param =
        make_shared<op::Parameter>(element::f32, PartialShape{Dimension(-1), Dimension(4, -1), Dimension(-1, 2)});

    auto shape_of = make_shared<op::v3::ShapeOf>(param);
    auto cast_fp = make_shared<op::Convert>(shape_of, element::f32);
    auto add = make_shared<op::v1::Add>(cast_fp, op::Constant::create(element::f32, {3}, {2, 3, 4}));
    auto sub = make_shared<op::v1::Subtract>(add, op::Constant::create(element::f32, {3}, {3, 2, 1}));
    auto cast_int = make_shared<op::Convert>(sub, element::i32);

    auto reshape = make_shared<op::v1::Reshape>(param, cast_int, false);

    EXPECT_EQ(reshape->get_element_type(), element::f32);
    EXPECT_EQ(reshape->get_output_partial_shape(0),
              PartialShape({Dimension(-1), Dimension(-1), Dimension(3, 5)}));  // Fully dynamic if no upper bound
}

TEST(type_prop, interval_value_propagation_add_sub_div_mul) {
    auto param =
        make_shared<op::Parameter>(element::f32, PartialShape{Dimension(-1), Dimension(2, 8), Dimension(4, 10), 6});

    auto shape_of = make_shared<op::v3::ShapeOf>(param);
    auto cast_fp = make_shared<op::Convert>(shape_of, element::f32);
    auto add = make_shared<op::v1::Add>(
        cast_fp,
        op::Constant::create(element::f32, {4}, {2, 2, -1, 3}));  // {(-1), (4, 10), (3, 9), (9)}
    auto div = make_shared<op::v1::Divide>(
        add,
        op::Constant::create(element::f32, {4}, {2, 2, -3, 3}));  // {(-1), (2, 5), (-3, -1), (3)}
    auto sub = make_shared<op::v1::Subtract>(
        div,
        op::Constant::create(element::f32, {4}, {2, 1, 2, -4}));  // {(-1), (1, 4), (-5, -3), (7)}
    auto mul = make_shared<op::v1::Multiply>(
        sub,
        op::Constant::create(element::f32, {4}, {2, 3, -4, 5}));  // {(-1), (3, 12), (12, 20), (35)}
    auto cast_int = make_shared<op::Convert>(mul, element::i32);

    auto reshape = make_shared<op::v1::Reshape>(param, cast_int, false);

    EXPECT_EQ(reshape->get_element_type(), element::f32);
    EXPECT_EQ(reshape->get_output_partial_shape(0),
              PartialShape({Dimension(-1), Dimension(3, 12), Dimension(12, 20), 35}));
}
