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
#include "util/type_prop.hpp"

using namespace std;
using namespace ngraph;

//
// Tests for binary elementwise ops.
//
void test_binary(std::string /* node_type */,
                 shared_ptr<Node>(f)(const shared_ptr<Node>& x, const shared_ptr<Node>& y))
{
    // Check for bad arguments
    auto tv0_2_4_param_0 = make_shared<op::Parameter>(element::f32, Shape{2, 4});
    auto tv0_2_4_param_1 = make_shared<op::Parameter>(element::f32, Shape{2, 4});
    auto tv0_2_4_param_2 = make_shared<op::Parameter>(element::i32, Shape{2, 4});
    auto tv0_4_2_param = make_shared<op::Parameter>(element::f32, Shape{4, 2});

    auto test_binary_bad_arguments_view_shapes = [&](const shared_ptr<Node>& x,
                                                     const shared_ptr<Node>& y) {
        try
        {
            auto node = f(x, y);
            // Should have thrown, so fail if it didn't
            FAIL() << "Incompatible view arguments not detected.";
        }
        catch (const NodeValidationFailure& error)
        {
            EXPECT_HAS_SUBSTRING(error.what(), std::string("Argument shapes are inconsistent"));
        }
        catch (...)
        {
            FAIL() << "Deduced type check failed for unexpected reason";
        }
    };
    test_binary_bad_arguments_view_shapes(tv0_2_4_param_0, tv0_4_2_param);

    auto test_binary_bad_arguments_view_element_types = [&](const shared_ptr<Node>& x,
                                                            const shared_ptr<Node>& y) {
        try
        {
            auto node = f(x, y);
            // Should have thrown, so fail if it didn't
            FAIL() << "Incompatible view arguments not detected.";
        }
        catch (const NodeValidationFailure& error)
        {
            EXPECT_HAS_SUBSTRING(error.what(),
                                 std::string("Argument element types are inconsistent"));
        }
        catch (...)
        {
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

TEST(type_prop, add_bad_arguments)
{
    test_binary("Add",
                [](const shared_ptr<Node>& x, const shared_ptr<Node>& y) -> shared_ptr<Node> {
                    return make_shared<op::v1::Add>(x, y);
                });
}

TEST(type_prop, divide_bad_arguments)
{
    test_binary("Divide",
                [](const shared_ptr<Node>& x, const shared_ptr<Node>& y) -> shared_ptr<Node> {
                    return make_shared<op::v1::Divide>(x, y);
                });
}

TEST(type_prop, multiply_bad_arguments)
{
    test_binary("Multiply",
                [](const shared_ptr<Node>& x, const shared_ptr<Node>& y) -> shared_ptr<Node> {
                    return make_shared<op::v1::Multiply>(x, y);
                });
}

TEST(type_prop, subtract_bad_arguments)
{
    test_binary("Subtract",
                [](const shared_ptr<Node>& x, const shared_ptr<Node>& y) -> shared_ptr<Node> {
                    return make_shared<op::v1::Subtract>(x, y);
                });
}

//
// Tests for binary elementwise logical ops.
//
void test_binary_logical(std::string /* node_type */,
                         shared_ptr<Node>(f)(const shared_ptr<Node>& x, const shared_ptr<Node>& y))
{
    // Check for bad arguments
    auto tv0_2_4_param_0 = make_shared<op::Parameter>(element::boolean, Shape{2, 4});
    auto tv0_2_4_param_1 = make_shared<op::Parameter>(element::boolean, Shape{2, 4});
    auto tv0_2_4_param_2 = make_shared<op::Parameter>(element::i32, Shape{2, 4});
    auto tv0_2_4_param_3 = make_shared<op::Parameter>(element::i32, Shape{2, 4});
    auto tv0_4_2_param = make_shared<op::Parameter>(element::boolean, Shape{4, 2});

    auto test_binary_bad_arguments_view_shapes = [&](const shared_ptr<Node>& x,
                                                     const shared_ptr<Node>& y) {
        try
        {
            auto node = f(x, y);
            // Should have thrown, so fail if it didn't
            FAIL() << "Incompatible view arguments not detected.";
        }
        catch (const NodeValidationFailure& error)
        {
            EXPECT_HAS_SUBSTRING(error.what(), std::string("Argument shapes are inconsistent"));
        }
        catch (...)
        {
            FAIL() << "Deduced type check failed for unexpected reason";
        }
    };
    test_binary_bad_arguments_view_shapes(tv0_2_4_param_0, tv0_4_2_param);

    auto test_binary_differ_arguments_view_element_types = [&](const shared_ptr<Node>& x,
                                                               const shared_ptr<Node>& y) {
        try
        {
            auto node = f(x, y);
            // Should have thrown, so fail if it didn't
            FAIL() << "Incompatible view arguments not detected.";
        }
        catch (const NodeValidationFailure& error)
        {
            EXPECT_HAS_SUBSTRING(error.what(),
                                 std::string("Argument element types are inconsistent"));
        }
        catch (...)
        {
            FAIL() << "Deduced type check failed for unexpected reason";
        }
    };

    auto test_binary_non_bool_arguments_view_element_types = [&](const shared_ptr<Node>& x,
                                                                 const shared_ptr<Node>& y) {
        try
        {
            auto node = f(x, y);
            // Should have thrown, so fail if it didn't
            FAIL() << "Incompatible view arguments not detected.";
        }
        catch (const ngraph_error& error)
        {
            EXPECT_HAS_SUBSTRING(error.what(), "must have boolean element type");
        }
        catch (...)
        {
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

TEST(type_prop, or_bad_arguments)
{
    test_binary_logical(
        "Or", [](const shared_ptr<Node>& x, const shared_ptr<Node>& y) -> shared_ptr<Node> {
            return make_shared<op::v1::LogicalOr>(x, y);
        });
}

TEST(type_prop, xor_bad_arguments)
{
    test_binary_logical(
        "Xor", [](const shared_ptr<Node>& x, const shared_ptr<Node>& y) -> shared_ptr<Node> {
            return make_shared<op::Xor>(x, y);
        });
}

template <typename T>
void test_binary_eltwise_numpy(const element::Type& et, const op::AutoBroadcastSpec& autob)
{
    auto param1 = make_shared<op::Parameter>(et, Shape{1, 3, 6});
    auto param2 = make_shared<op::Parameter>(et, Shape{3, 1});
    auto param3 = make_shared<op::Parameter>(et, Shape{2, 3, 6});
    auto param4 = make_shared<op::Parameter>(et, Shape{6});
    EXPECT_EQ(make_shared<T>(param1, param2, autob)->get_shape(), (Shape{1, 3, 6}));
    EXPECT_EQ(make_shared<T>(param1, param3, autob)->get_shape(), (Shape{2, 3, 6}));
    EXPECT_EQ(make_shared<T>(param4, param3, autob)->get_shape(), (Shape{2, 3, 6}));

    auto pp1 = make_shared<op::Parameter>(et, PartialShape{1, Dimension::dynamic(), 6});
    auto pp2 = make_shared<op::Parameter>(et, PartialShape{3, 1});
    EXPECT_EQ(make_shared<T>(pp1, pp2, autob)->get_shape(), (Shape{1, 3, 6}));
}

TEST(type_prop, eltwise_auto_bcast)
{
    test_binary_eltwise_numpy<op::v1::Add>(element::f32, op::AutoBroadcastType::NUMPY);
    test_binary_eltwise_numpy<op::v1::Divide>(element::f32, op::AutoBroadcastType::NUMPY);
    test_binary_eltwise_numpy<op::v1::Equal>(element::f32, op::AutoBroadcastType::NUMPY);
    test_binary_eltwise_numpy<op::v1::Greater>(element::f32, op::AutoBroadcastType::NUMPY);
    test_binary_eltwise_numpy<op::v1::GreaterEqual>(element::f32, op::AutoBroadcastType::NUMPY);
    test_binary_eltwise_numpy<op::v1::Less>(element::f32, op::AutoBroadcastType::NUMPY);
    test_binary_eltwise_numpy<op::v1::LessEqual>(element::f32, op::AutoBroadcastType::NUMPY);
    test_binary_eltwise_numpy<op::v1::Maximum>(element::f32, op::AutoBroadcastType::NUMPY);
    test_binary_eltwise_numpy<op::v1::Minimum>(element::f32, op::AutoBroadcastType::NUMPY);
    test_binary_eltwise_numpy<op::v1::Multiply>(element::f32, op::AutoBroadcastType::NUMPY);
    test_binary_eltwise_numpy<op::v1::NotEqual>(element::f32, op::AutoBroadcastType::NUMPY);
    test_binary_eltwise_numpy<op::v1::LogicalOr>(element::boolean, op::AutoBroadcastType::NUMPY);
    test_binary_eltwise_numpy<op::v1::Power>(element::f32, op::AutoBroadcastType::NUMPY);
    test_binary_eltwise_numpy<op::v1::Subtract>(element::f32, op::AutoBroadcastType::NUMPY);
    test_binary_eltwise_numpy<op::Xor>(element::boolean, op::AutoBroadcastType::NUMPY);
}

TEST(type_prop, comparison_good)
{
    auto tv0_2_4_param_0 = make_shared<op::Parameter>(element::f32, Shape{2, 4});
    auto tv0_2_4_param_1 = make_shared<op::Parameter>(element::f32, Shape{2, 4});
    auto eq = make_shared<op::v1::Equal>(tv0_2_4_param_0, tv0_2_4_param_1);
    EXPECT_EQ(eq->get_element_type(), element::boolean);
    EXPECT_EQ(eq->get_shape(), (Shape{2, 4}));
}

TEST(type_prop, binary_arithmetic_bad_argument_element_types)
{
    auto tv0_2_4_param_0 = make_shared<op::Parameter>(element::boolean, Shape{2, 4});
    auto tv0_2_4_param_1 = make_shared<op::Parameter>(element::boolean, Shape{2, 4});
    try
    {
        auto bc = make_shared<op::v1::Add>(tv0_2_4_param_0, tv0_2_4_param_1);
        // Should have thrown, so fail if it didn't
        FAIL() << "Did not detect incorrect element types for arithmetic operator";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Arguments cannot have boolean element type"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

namespace
{
    template <typename T>
    void test_binary_eltwise_bad_argument_shape(const element::Type& et)
    {
        auto input1 = make_shared<op::Parameter>(element::f32, Shape{2, 4});
        auto input2 = make_shared<op::Parameter>(element::f32, Shape{1, 2, 4});
        try
        {
            auto bc = make_shared<T>(input1, input2, op::AutoBroadcastType::NONE);
            // Should have thrown, so fail if it didn't
            FAIL() << "Did not detect incorrect element types for arithmetic operator";
        }
        catch (const NodeValidationFailure& error)
        {
            EXPECT_HAS_SUBSTRING(error.what(), std::string("Argument shapes are inconsistent"));
        }
        catch (...)
        {
            FAIL() << "Deduced type check failed for unexpected reason";
        }
    }
} // namespace

TEST(type_prop, binary_arithmetic_bad_argument_shape_with_none_autobroadcast_attribute)
{
    test_binary_eltwise_bad_argument_shape<op::v1::Add>(element::f32);
    test_binary_eltwise_bad_argument_shape<op::v1::Divide>(element::f32);
    test_binary_eltwise_bad_argument_shape<op::v1::Equal>(element::f32);
    test_binary_eltwise_bad_argument_shape<op::v1::Greater>(element::f32);
    test_binary_eltwise_bad_argument_shape<op::v1::GreaterEqual>(element::f32);
    test_binary_eltwise_bad_argument_shape<op::v1::Less>(element::f32);
    test_binary_eltwise_bad_argument_shape<op::v1::LessEqual>(element::f32);
    test_binary_eltwise_bad_argument_shape<op::v1::Maximum>(element::f32);
    test_binary_eltwise_bad_argument_shape<op::v1::Minimum>(element::f32);
    test_binary_eltwise_bad_argument_shape<op::v1::Multiply>(element::f32);
    test_binary_eltwise_bad_argument_shape<op::v1::NotEqual>(element::f32);
    test_binary_eltwise_bad_argument_shape<op::v1::LogicalOr>(element::boolean);
    test_binary_eltwise_bad_argument_shape<op::v1::Power>(element::f32);
    test_binary_eltwise_bad_argument_shape<op::v1::Subtract>(element::f32);
    test_binary_eltwise_bad_argument_shape<op::Xor>(element::boolean);
}

TEST(type_prop, binary_elementwise_arithmetic_both_dynamic)
{
    auto a = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto b = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto add = make_shared<op::v1::Add>(a, b);

    ASSERT_TRUE(add->get_output_partial_shape(0).rank().is_dynamic());
}

TEST(type_prop,
     binary_elementwise_arithmetic_left_rank_static_dynamic_right_rank_static_dynamic_result_static)
{
    auto a = make_shared<op::Parameter>(element::f32, PartialShape{1, Dimension::dynamic(), 3});
    auto b = make_shared<op::Parameter>(element::f32, PartialShape{1, 2, Dimension::dynamic()});
    auto add = make_shared<op::v1::Add>(a, b);

    ASSERT_TRUE(add->get_output_partial_shape(0).is_static());
    ASSERT_EQ(add->get_shape(), (Shape{1, 2, 3}));
}

TEST(
    type_prop,
    binary_elementwise_arithmetic_left_rank_static_dynamic_right_rank_static_dynamic_result_rank_static_dynamic)
{
    auto a = make_shared<op::Parameter>(
        element::f32, PartialShape{1, Dimension::dynamic(), Dimension::dynamic()});
    auto b = make_shared<op::Parameter>(element::f32, PartialShape{1, 2, Dimension::dynamic()});
    auto add = make_shared<op::v1::Add>(a, b);

    ASSERT_TRUE(add->get_output_partial_shape(0).rank().is_static());
    ASSERT_TRUE(add->get_output_partial_shape(0).is_dynamic());
    ASSERT_TRUE(
        add->get_output_partial_shape(0).same_scheme(PartialShape{1, 2, Dimension::dynamic()}));
}

TEST(type_prop, binary_elementwise_arithmetic_left_static_right_rank_static_dynamic)
{
    auto a = make_shared<op::Parameter>(element::f32, PartialShape{1, 2, 3});
    auto b = make_shared<op::Parameter>(element::f32, PartialShape{1, 2, Dimension::dynamic()});
    auto add = make_shared<op::v1::Add>(a, b);

    ASSERT_TRUE(add->get_output_partial_shape(0).is_static());
    ASSERT_EQ(add->get_shape(), (Shape{1, 2, 3}));
}

TEST(type_prop, binary_elementwise_arithmetic_left_rank_static_dynamic_right_static)
{
    auto a = make_shared<op::Parameter>(element::f32, PartialShape{1, 2, Dimension::dynamic()});
    auto b = make_shared<op::Parameter>(element::f32, PartialShape{1, 2, 3});
    auto add = make_shared<op::v1::Add>(a, b);

    ASSERT_TRUE(add->get_output_partial_shape(0).is_static());
    ASSERT_EQ(add->get_shape(), (Shape{1, 2, 3}));
}

TEST(type_prop, binary_elementwise_arithmetic_left_rank_static_dynamic_inconsistent)
{
    auto a = make_shared<op::Parameter>(element::f32, PartialShape{1, 2, Dimension::dynamic()});
    auto b = make_shared<op::Parameter>(element::f32, PartialShape{1, 3, 3});

    try
    {
        auto add = make_shared<op::v1::Add>(a, b);
        FAIL() << "Inconsistent partial shapes not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "Argument shapes are inconsistent");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, binary_elementwise_arithmetic_right_rank_static_dynamic_inconsistent)
{
    auto a = make_shared<op::Parameter>(element::f32, PartialShape{1, 3, 3});
    auto b = make_shared<op::Parameter>(element::f32, PartialShape{1, 2, Dimension::dynamic()});

    try
    {
        auto add = make_shared<op::v1::Add>(a, b);
        FAIL() << "Inconsistent partial shapes not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "Argument shapes are inconsistent");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, binary_elementwise_arithmetic_both_rank_static_dynamic_inconsistent)
{
    auto a = make_shared<op::Parameter>(element::f32, PartialShape{Dimension::dynamic(), 3, 3});
    auto b = make_shared<op::Parameter>(element::f32, PartialShape{1, 2, Dimension::dynamic()});

    try
    {
        auto add = make_shared<op::v1::Add>(a, b);
        FAIL() << "Inconsistent partial shapes not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "Argument shapes are inconsistent");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, binary_elementwise_arithmetic_left_rank_static_dynamic_different_rank)
{
    auto a = make_shared<op::Parameter>(element::f32, PartialShape{1, 2, Dimension::dynamic()});
    auto b = make_shared<op::Parameter>(element::f32, PartialShape{1, 2, 3, 4});

    try
    {
        auto add = make_shared<op::v1::Add>(a, b);
        FAIL() << "Inconsistent partial shapes not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "Argument shapes are inconsistent");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, binary_elementwise_arithmetic_right_rank_static_dynamic_different_rank)
{
    auto a = make_shared<op::Parameter>(element::f32, PartialShape{1, 2, 3, 4});
    auto b = make_shared<op::Parameter>(element::f32, PartialShape{1, 2, Dimension::dynamic()});

    try
    {
        auto add = make_shared<op::v1::Add>(a, b);
        FAIL() << "Inconsistent partial shapes not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "Argument shapes are inconsistent");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, binary_elementwise_arithmetic_both_rank_static_dynamic_different_rank)
{
    auto a = make_shared<op::Parameter>(element::f32, PartialShape{1, Dimension::dynamic(), 3, 4});
    auto b = make_shared<op::Parameter>(element::f32, PartialShape{1, 2, Dimension::dynamic()});

    try
    {
        auto add = make_shared<op::v1::Add>(a, b);
        FAIL() << "Inconsistent partial shapes not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "Argument shapes are inconsistent");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, binary_elementwise_arithmetic_both_et_dynamic)
{
    auto a = make_shared<op::Parameter>(element::dynamic, Shape{1, 2, 3, 4});
    auto b = make_shared<op::Parameter>(element::dynamic, Shape{1, 2, 3, 4});
    auto add = make_shared<op::v1::Add>(a, b);

    ASSERT_TRUE(add->get_output_element_type(0).is_dynamic());
}

TEST(type_prop, binary_elementwise_arithmetic_left_et_dynamic)
{
    auto a = make_shared<op::Parameter>(element::dynamic, Shape{1, 2, 3, 4});
    auto b = make_shared<op::Parameter>(element::u32, Shape{1, 2, 3, 4});
    auto add = make_shared<op::v1::Add>(a, b);

    ASSERT_EQ(add->get_output_element_type(0), element::u32);
}

TEST(type_prop, binary_elementwise_arithmetic_right_et_dynamic)
{
    auto a = make_shared<op::Parameter>(element::i64, Shape{1, 2, 3, 4});
    auto b = make_shared<op::Parameter>(element::dynamic, Shape{1, 2, 3, 4});
    auto add = make_shared<op::v1::Add>(a, b);

    ASSERT_EQ(add->get_output_element_type(0), element::i64);
}

TEST(type_prop, logic_arith_compare_partial_et)
{
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
    ASSERT_EQ(test_compare(element::boolean, element::boolean)->get_element_type(),
              element::boolean);
    ASSERT_EQ(test_compare(element::boolean, element::dynamic)->get_element_type(),
              element::boolean);
    ASSERT_EQ(test_compare(element::dynamic, element::i32)->get_element_type(), element::boolean);
    ASSERT_EQ(test_compare(element::dynamic, element::boolean)->get_element_type(),
              element::boolean);
    ASSERT_EQ(test_compare(element::dynamic, element::dynamic)->get_element_type(),
              element::boolean);

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
