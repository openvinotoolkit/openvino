// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "util/type_prop.hpp"

NGRAPH_SUPPRESS_DEPRECATED_START

using namespace std;
using namespace ngraph;

TEST(type_prop, select_deduce)
{
    auto tv0_2_4_param_0 = make_shared<op::Parameter>(element::boolean, Shape{2, 4});
    auto tv0_2_4_param_1 = make_shared<op::Parameter>(element::f32, Shape{2, 4});
    auto tv0_2_4_param_2 = make_shared<op::Parameter>(element::f32, Shape{2, 4});
    auto bc = make_shared<op::v1::Select>(tv0_2_4_param_0, tv0_2_4_param_1, tv0_2_4_param_2);
    ASSERT_EQ(bc->get_element_type(), element::f32);
    ASSERT_EQ(bc->get_shape(), (Shape{2, 4}));
}

TEST(type_prop, select_shape_mismatch_a)
{
    auto tv0_2_4_param_0 = make_shared<op::Parameter>(element::boolean, Shape{3, 5});
    auto tv0_2_4_param_1 = make_shared<op::Parameter>(element::f32, Shape{2, 4});
    auto tv0_2_4_param_2 = make_shared<op::Parameter>(element::f32, Shape{2, 4});
    try
    {
        auto bc = make_shared<op::v1::Select>(tv0_2_4_param_0, tv0_2_4_param_1, tv0_2_4_param_2);
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

TEST(type_prop, select_shape_mismatch_b)
{
    auto tv0_2_4_param_0 = make_shared<op::Parameter>(element::boolean, Shape{2, 4});
    auto tv0_2_4_param_1 = make_shared<op::Parameter>(element::f32, Shape{3, 5});
    auto tv0_2_4_param_2 = make_shared<op::Parameter>(element::f32, Shape{2, 4});
    try
    {
        auto bc = make_shared<op::v1::Select>(tv0_2_4_param_0, tv0_2_4_param_1, tv0_2_4_param_2);
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

TEST(type_prop, select_shape_mismatch_c)
{
    auto tv0_2_4_param_0 = make_shared<op::Parameter>(element::boolean, Shape{2, 4});
    auto tv0_2_4_param_1 = make_shared<op::Parameter>(element::f32, Shape{2, 4});
    auto tv0_2_4_param_2 = make_shared<op::Parameter>(element::f32, Shape{3, 5});
    try
    {
        auto bc = make_shared<op::v1::Select>(tv0_2_4_param_0, tv0_2_4_param_1, tv0_2_4_param_2);
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

TEST(type_prop, select_elem_mismatch_a)
{
    auto tv0_2_4_param_0 = make_shared<op::Parameter>(element::f32, Shape{2, 4});
    auto tv0_2_4_param_1 = make_shared<op::Parameter>(element::f32, Shape{2, 4});
    auto tv0_2_4_param_2 = make_shared<op::Parameter>(element::f32, Shape{2, 4});
    try
    {
        auto bc = make_shared<op::v1::Select>(tv0_2_4_param_0, tv0_2_4_param_1, tv0_2_4_param_2);
        // Should have thrown, so fail if it didn't
        FAIL() << "Did not detect incorrect element types for arithmetic operator";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Argument 0 must have boolean element type"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, select_elem_mismatch_bc)
{
    auto tv0_2_4_param_0 = make_shared<op::Parameter>(element::boolean, Shape{2, 4});
    auto tv0_2_4_param_1 = make_shared<op::Parameter>(element::f32, Shape{2, 4});
    auto tv0_2_4_param_2 = make_shared<op::Parameter>(element::i32, Shape{2, 4});
    try
    {
        auto bc = make_shared<op::v1::Select>(tv0_2_4_param_0, tv0_2_4_param_1, tv0_2_4_param_2);
        // Should have thrown, so fail if it didn't
        FAIL() << "Did not detect incorrect element types for arithmetic operator";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Argument 1 and 2 element types must match"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, select_partial_all_rank_dynamic)
{
    auto param0 = make_shared<op::Parameter>(element::boolean, PartialShape::dynamic());
    auto param1 = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto param2 = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());

    auto sel = make_shared<op::v1::Select>(param0, param1, param2);

    ASSERT_EQ(sel->get_output_element_type(0), element::f32);
    ASSERT_TRUE(sel->get_output_partial_shape(0).rank().is_dynamic());
}

TEST(type_prop, select_partial_all_rank_dynamic_arg0_et_dynamic_arg1_arg2_et_mismatch)
{
    auto param0 = make_shared<op::Parameter>(element::dynamic, PartialShape::dynamic());
    auto param1 = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto param2 = make_shared<op::Parameter>(element::i32, PartialShape::dynamic());

    try
    {
        auto sel = make_shared<op::v1::Select>(param0, param1, param2);
        FAIL() << "Did not detect mismatched element types for args 1 and 2 (element type-dynamic "
                  "arg0)";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Argument 1 and 2 element types must match"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, select_partial_all_rank_dynamic_arg0_arg1_et_dynamic)
{
    auto param0 = make_shared<op::Parameter>(element::dynamic, PartialShape::dynamic());
    auto param1 = make_shared<op::Parameter>(element::dynamic, PartialShape::dynamic());
    auto param2 = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());

    auto sel = make_shared<op::v1::Select>(param0, param1, param2);

    ASSERT_EQ(sel->get_output_element_type(0), element::f32);
    ASSERT_TRUE(sel->get_output_partial_shape(0).rank().is_dynamic());
}

TEST(type_prop, select_partial_all_rank_dynamic_arg0_arg2_et_dynamic)
{
    auto param0 = make_shared<op::Parameter>(element::dynamic, PartialShape::dynamic());
    auto param1 = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto param2 = make_shared<op::Parameter>(element::dynamic, PartialShape::dynamic());

    auto sel = make_shared<op::v1::Select>(param0, param1, param2);

    ASSERT_EQ(sel->get_output_element_type(0), element::f32);
    ASSERT_TRUE(sel->get_output_partial_shape(0).rank().is_dynamic());
}

TEST(type_prop, select_partial_all_rank_dynamic_arg0_arg1_arg2_et_dynamic)
{
    auto param0 = make_shared<op::Parameter>(element::dynamic, PartialShape::dynamic());
    auto param1 = make_shared<op::Parameter>(element::dynamic, PartialShape::dynamic());
    auto param2 = make_shared<op::Parameter>(element::dynamic, PartialShape::dynamic());

    auto sel = make_shared<op::v1::Select>(param0, param1, param2);

    ASSERT_EQ(sel->get_output_element_type(0), element::dynamic);
    ASSERT_TRUE(sel->get_output_partial_shape(0).rank().is_dynamic());
}

TEST(type_prop, select_partial_all_rank_static_dynamic_ok)
{
    auto param0 = make_shared<op::Parameter>(
        element::boolean, PartialShape{2, Dimension::dynamic(), Dimension::dynamic()});
    auto param1 = make_shared<op::Parameter>(
        element::f32, PartialShape{Dimension::dynamic(), 8, Dimension::dynamic()});
    auto param2 = make_shared<op::Parameter>(
        element::f32, PartialShape{Dimension::dynamic(), Dimension::dynamic(), 3});

    auto sel = make_shared<op::v1::Select>(param0, param1, param2);

    ASSERT_EQ(sel->get_output_element_type(0), element::f32);
    ASSERT_TRUE(sel->get_output_partial_shape(0).is_static());
    ASSERT_EQ(sel->get_output_shape(0), (Shape{2, 8, 3}));
}

TEST(type_prop, select_partial_all_rank_static_intransitive_incompatibility)
{
    auto param0 = make_shared<op::Parameter>(
        element::boolean, PartialShape{2, Dimension::dynamic(), Dimension::dynamic()});
    auto param1 = make_shared<op::Parameter>(
        element::f32, PartialShape{Dimension::dynamic(), 8, Dimension::dynamic()});
    auto param2 =
        make_shared<op::Parameter>(element::f32, PartialShape{3, Dimension::dynamic(), 3});

    try
    {
        auto sel = make_shared<op::v1::Select>(param0, param1, param2);
        FAIL() << "Did not detect intransitive partial-shape incompatibility";
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

//------------------------------ v1::Select ---------------------------------//
//
//
struct SelectParams
{
    std::vector<Shape> shapes;
    std::vector<element::Type> ets;
    op::AutoBroadcastSpec auto_broadcast;

    SelectParams(const std::vector<Shape>& shape,
                 const std::vector<element::Type>& et,
                 const op::AutoBroadcastSpec& auto_broadcast)
        : shapes(shape)
        , ets(et)
        , auto_broadcast(auto_broadcast)
    {
    }
};

struct DeduceV1SelectTest : ::testing::TestWithParam<SelectParams>
{
};

TEST_P(DeduceV1SelectTest, output_shape)
{
    auto tp = GetParam();
    auto cond = make_shared<op::Parameter>(tp.ets[0], tp.shapes[0]);
    auto ptrue = make_shared<op::Parameter>(tp.ets[1], tp.shapes[1]);
    auto pfalse = make_shared<op::Parameter>(tp.ets[2], tp.shapes[2]);
    auto select = make_shared<op::v1::Select>(cond, ptrue, pfalse, tp.auto_broadcast);

    ASSERT_EQ(select->get_shape(), tp.shapes[3]);
    ASSERT_EQ(select->get_element_type(), tp.ets[3]);
}

INSTANTIATE_TEST_SUITE_P(
    type_prop,
    DeduceV1SelectTest,
    ::testing::Values(SelectParams({{2, 4}, {2, 4}, {2, 4}, {2, 4}},
                                   {element::boolean, element::f32, element::f32, element::f32},
                                   op::AutoBroadcastType::NONE),
                      SelectParams({{2, 4}, {2, 4}, {2, 4}, {2, 4}},
                                   {element::boolean, element::f32, element::f32, element::f32},
                                   op::AutoBroadcastType::NUMPY),
                      SelectParams({{}, {2, 4}, {2, 4}, {2, 4}},
                                   {element::boolean, element::f32, element::f32, element::f32},
                                   op::AutoBroadcastType::NUMPY),
                      SelectParams({{}, {4}, {2, 4}, {2, 4}},
                                   {element::boolean, element::f32, element::dynamic, element::f32},
                                   op::AutoBroadcastType::NUMPY),
                      SelectParams({{}, {2, 4}, {4}, {2, 4}},
                                   {element::boolean, element::f32, element::f32, element::f32},
                                   op::AutoBroadcastType::NUMPY),
                      SelectParams({{4}, {2, 4}, {4}, {2, 4}},
                                   {element::boolean, element::i8, element::dynamic, element::i8},
                                   op::AutoBroadcastType::NUMPY),
                      SelectParams({{4}, {4}, {2, 4}, {2, 4}},
                                   {element::dynamic, element::dynamic, element::i8, element::i8},
                                   op::AutoBroadcastType::NUMPY),
                      SelectParams({{2}, {2}, {2, 4}, {2, 4}},
                                   {element::boolean, element::f32, element::dynamic, element::f32},
                                   {op::AutoBroadcastType::PDPD, 0}),
                      // TODO: Whats the right behavior here?
                      // SelectParams({{2}, {2, 4}, {2}, {2, 4}}, {element::boolean, element::f32,
                      // element::dynamic, element::f32}, {op::AutoBroadcastType::PDPD, 0}),
                      SelectParams({{4}, {4}, {2, 4}, {2, 4}},
                                   {element::boolean, element::f32, element::dynamic, element::f32},
                                   {op::AutoBroadcastType::PDPD, 1})),
    PrintToDummyParamName());

TEST(type_prop, select_v1_partial_shape)
{
    auto a = make_shared<op::Parameter>(element::boolean, PartialShape::dynamic());
    auto b = make_shared<op::Parameter>(element::f32, Shape{2, 4});
    auto c = make_shared<op::Parameter>(element::f32, Shape{2, 4});

    auto select = make_shared<op::v1::Select>(a, b, c, op::AutoBroadcastType::NONE);
    ASSERT_EQ(select->get_shape(), (Shape{2, 4}));
}

TEST(type_prop, select_v1_partial_shape_autob)
{
    auto a = make_shared<op::Parameter>(element::boolean, PartialShape{Dimension::dynamic()});
    auto b = make_shared<op::Parameter>(element::f32, PartialShape{Dimension::dynamic()});
    auto c = make_shared<op::Parameter>(element::f32, PartialShape{2, Dimension::dynamic()});

    auto select = make_shared<op::v1::Select>(a, b, c);
    ASSERT_TRUE(
        select->get_output_partial_shape(0).same_scheme(PartialShape{2, Dimension::dynamic()}));
}

TEST(type_prop, select_v1_wrong_et)
{
    auto param0 = make_shared<op::Parameter>(element::i8, Shape{2, 4});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{2, 4});
    auto param2 = make_shared<op::Parameter>(element::f32, Shape{2, 4});

    try
    {
        auto sel = make_shared<op::v1::Select>(param0, param1, param2);
        FAIL() << "Did not detect wrong element type";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Argument 0 must have boolean element type"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, select_v1_et_mismatch)
{
    auto param0 = make_shared<op::Parameter>(element::boolean, Shape{2, 4});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{2, 4});
    auto param2 = make_shared<op::Parameter>(element::i8, Shape{2, 4});

    try
    {
        auto sel = make_shared<op::v1::Select>(param0, param1, param2);
        FAIL() << "Did not detect element type mismatch";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Argument 1 and 2 element types must match."));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, select_v1_shape_mismatch)
{
    auto param0 = make_shared<op::Parameter>(element::boolean, Shape{2, 4});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{2, 3});
    auto param2 = make_shared<op::Parameter>(element::f32, Shape{2, 4});

    try
    {
        auto sel = make_shared<op::v1::Select>(param0, param1, param2);
        FAIL() << "Did not detect shape mismatch";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Argument shapes are inconsistent."));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, select_v1_partial_shape_mismatch)
{
    auto param0 =
        make_shared<op::Parameter>(element::boolean, PartialShape{3, Dimension::dynamic()});
    auto param1 = make_shared<op::Parameter>(element::f32, PartialShape{2, Dimension::dynamic()});
    auto param2 = make_shared<op::Parameter>(element::f32, Shape{2, 4});

    try
    {
        auto sel = make_shared<op::v1::Select>(param0, param1, param2);
        FAIL() << "Did not detect shape mismatch";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Argument shapes are inconsistent."));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}
