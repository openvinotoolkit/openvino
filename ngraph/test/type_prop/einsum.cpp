// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "util/type_prop.hpp"

using namespace std;
using namespace ngraph;

TEST(type_prop, einsum_staticshape_dotproduct)
{
    std::string equation = "i,i->";
    Shape input1_shape{3};
    Shape input2_shape{3};
    Shape out_shape{};
    auto I1 = make_shared<op::Parameter>(element::f32, input1_shape);
    auto I2 = make_shared<op::Parameter>(element::f32, input2_shape);
    auto O = make_shared<op::v7::Einsum>(OutputVector{I1, I2}, equation);
    ASSERT_EQ(O->get_element_type(), element::f32);
    ASSERT_EQ(O->get_shape(), out_shape);
}

TEST(type_prop, einsum_staticshape_matmul)
{
    std::string equation = "ab,bc->ac";
    Shape input1_shape{2, 3};
    Shape input2_shape{3, 4};
    Shape out_shape{2, 4};
    auto I1 = make_shared<op::Parameter>(element::f32, input1_shape);
    auto I2 = make_shared<op::Parameter>(element::f32, input2_shape);
    auto O = make_shared<op::v7::Einsum>(OutputVector{I1, I2}, equation);
    ASSERT_EQ(O->get_element_type(), element::f32);
    ASSERT_EQ(O->get_shape(), out_shape);
}

TEST(type_prop, einsum_staticshape_trace)
{
    std::string equation = "kii->k";
    Shape input1_shape{2, 3, 3};
    Shape out_shape{2};
    auto I1 = make_shared<op::Parameter>(element::f32, input1_shape);
    auto O = make_shared<op::v7::Einsum>(OutputVector{I1}, equation);
    ASSERT_EQ(O->get_element_type(), element::f32);
    ASSERT_EQ(O->get_shape(), out_shape);
}

TEST(type_prop, einsum_staticshape_diagextraction)
{
    std::string equation = "kii->ki";
    Shape input1_shape{2, 3, 3};
    Shape out_shape{2, 3};
    auto I1 = make_shared<op::Parameter>(element::f32, input1_shape);
    auto O = make_shared<op::v7::Einsum>(OutputVector{I1}, equation);
    ASSERT_EQ(O->get_element_type(), element::f32);
    ASSERT_EQ(O->get_shape(), out_shape);
}

TEST(type_prop, einsum_staticshape_transpose)
{
    std::string equation = "ijk->kij";
    Shape input1_shape{1, 2, 3};
    Shape out_shape{3, 1, 2};
    auto I1 = make_shared<op::Parameter>(element::f32, input1_shape);
    auto O = make_shared<op::v7::Einsum>(OutputVector{I1}, equation);
    ASSERT_EQ(O->get_element_type(), element::f32);
    ASSERT_EQ(O->get_shape(), out_shape);
}

TEST(type_prop, einsum_staticshape_multimatmul)
{
    std::string equation = "ab,bcd,bc->ca";
    Shape input1_shape{2, 5};
    Shape input2_shape{5, 3, 6};
    Shape input3_shape{5, 3};
    Shape out_shape{3, 2};
    auto I1 = make_shared<op::Parameter>(element::i32, input1_shape);
    auto I2 = make_shared<op::Parameter>(element::i32, input2_shape);
    auto I3 = make_shared<op::Parameter>(element::i32, input3_shape);
    auto O = make_shared<op::v7::Einsum>(OutputVector{I1, I2, I3}, equation);
    ASSERT_EQ(O->get_element_type(), element::i32);
    ASSERT_EQ(O->get_shape(), out_shape);
}

TEST(type_prop, einsum_staticshape_ellipsis)
{
    std::string equation = "a...->...";
    Shape input1_shape{5, 3};
    Shape out_shape{3};
    auto I1 = make_shared<op::Parameter>(element::f32, input1_shape);
    auto O = make_shared<op::v7::Einsum>(OutputVector{I1}, equation);
    ASSERT_EQ(O->get_element_type(), element::f32);
    ASSERT_EQ(O->get_shape(), out_shape);
}

TEST(type_prop, einsum_staticshape_ellipsis2)
{
    std::string equation = "a...,...->a...";
    Shape input1_shape{3, 5};
    Shape input2_shape{1};
    Shape out_shape{3, 5};
    auto I1 = make_shared<op::Parameter>(element::f32, input1_shape);
    auto I2 = make_shared<op::Parameter>(element::f32, input2_shape);
    auto O = make_shared<op::v7::Einsum>(OutputVector{I1, I2}, equation);
    ASSERT_EQ(O->get_element_type(), element::f32);
    ASSERT_EQ(O->get_shape(), out_shape);
}

TEST(type_prop, einsum_staticshape_ellipsis3)
{
    std::string equation = "a...b,b...->a...";
    Shape input1_shape{11, 1, 4, 3};
    Shape input2_shape{3, 11, 7, 1};
    Shape out_shape{11, 11, 7, 4};
    auto I1 = make_shared<op::Parameter>(element::i32, input1_shape);
    auto I2 = make_shared<op::Parameter>(element::i32, input2_shape);
    auto O = make_shared<op::v7::Einsum>(OutputVector{I1, I2}, equation);
    ASSERT_EQ(O->get_element_type(), element::i32);
    ASSERT_EQ(O->get_shape(), out_shape);
}

TEST(type_prop, einsum_dynamicshape_dotproduct)
{
    std::string equation = "a,ab->ab";
    const auto input1_shape = PartialShape{Dimension(2, 7)};
    const auto input2_shape = PartialShape{Dimension(3, 10), 3};
    const auto out_shape = PartialShape{Dimension(3, 7), 3};
    auto I1 = make_shared<op::Parameter>(element::f32, input1_shape);
    auto I2 = make_shared<op::Parameter>(element::f32, input2_shape);
    auto O = make_shared<op::v7::Einsum>(OutputVector{I1, I2}, equation);
    ASSERT_EQ(O->get_element_type(), element::f32);
    ASSERT_TRUE(O->get_output_partial_shape(0).same_scheme(out_shape));
}

TEST(type_prop, einsum_dynamicshape_diagextraction)
{
    std::string equation = "xyzxy->xyz";
    const auto input1_shape = PartialShape{Dimension(2, 7), Dimension(1, 5), 4, Dimension(3, 5), 3};
    const auto out_shape = PartialShape{Dimension(3, 5), 3, 4};
    auto I1 = make_shared<op::Parameter>(element::i32, input1_shape);
    auto O = make_shared<op::v7::Einsum>(OutputVector{I1}, equation);
    ASSERT_EQ(O->get_element_type(), element::i32);
    ASSERT_TRUE(O->get_output_partial_shape(0).same_scheme(out_shape));
}

TEST(type_prop, DISABLED_einsum_dynamicshape_ellipsis1)
{
    // TODO: fix bug #53518 - PartialShape::broadcast_merge_into or Dimension::broadcast_merge
    //  to support broadcasting between Dimension(3, 5) and Dimension(1, 3)
    //  for which the result must be Dimension(3, 5)
    std::string equation = "a...b,b...->a...";
    const auto input1_shape = PartialShape{11, 1, Dimension(3, 5), 3};
    const auto input2_shape = PartialShape{3, 11, 7, Dimension(1, 3)};
    const auto out_shape = PartialShape{11, 11, 7, Dimension(3, 5)};
    auto I1 = make_shared<op::Parameter>(element::f32, input1_shape);
    auto I2 = make_shared<op::Parameter>(element::f32, input2_shape);
    auto O = make_shared<op::v7::Einsum>(OutputVector{I1, I2}, equation);
    ASSERT_EQ(O->get_element_type(), element::f32);
    ASSERT_TRUE(O->get_output_partial_shape(0).same_scheme(out_shape));
}

TEST(type_prop, einsum_implicitmode_mixedcaseletters)
{
    // the following equation is equivalent to "AbC->ACb"
    std::string equation = "AbC";
    const auto input1_shape = PartialShape{1, Dimension(2, 3), Dimension(4, 5)};
    auto I1 = make_shared<op::Parameter>(element::f32, input1_shape);
    const auto out_shape = PartialShape{1, Dimension(4, 5), Dimension(2, 3)};
    auto O = make_shared<op::v7::Einsum>(OutputVector{I1}, equation);
    ASSERT_EQ(O->get_element_type(), element::f32);
    ASSERT_TRUE(O->get_output_partial_shape(0).same_scheme(out_shape));
}

TEST(type_prop, einsum_implicitmode_mixedcaseletters2)
{
    // the following equation is equivalent to "a...b,B...->...Bab"
    std::string equation = "a...b,B...";
    const auto input1_shape = PartialShape{Dimension(3, 5), 11, 1, 3};
    const auto input2_shape = PartialShape{Dimension(1, 3), 3, 1, 7};
    const auto out_shape = PartialShape{3, 11, 7, Dimension(1, 3), Dimension(3, 5), 3};
    auto I1 = make_shared<op::Parameter>(element::f32, input1_shape);
    auto I2 = make_shared<op::Parameter>(element::f32, input2_shape);
    auto O = make_shared<op::v7::Einsum>(OutputVector{I1, I2}, equation);
    ASSERT_EQ(O->get_element_type(), element::f32);
    ASSERT_TRUE(O->get_output_partial_shape(0).same_scheme(out_shape));
}

TEST(type_prop, einsum_dynamicrank_multimatmul)
{
    std::string equation = "ab,bcd,bc->ca";
    Shape input1_shape{2, 5};
    PartialShape input2_shape = PartialShape::dynamic();
    Shape input3_shape{5, 3};
    Shape out_shape{3, 2};
    auto I1 = make_shared<op::Parameter>(element::i32, input1_shape);
    auto I2 = make_shared<op::Parameter>(element::i32, input2_shape);
    auto I3 = make_shared<op::Parameter>(element::i32, input3_shape);
    auto O = make_shared<op::v7::Einsum>(OutputVector{I1, I2, I3}, equation);
    ASSERT_EQ(O->get_element_type(), element::i32);
    ASSERT_EQ(O->get_shape(), out_shape);
}

TEST(type_prop, einsum_dynamicrank_multimatmul2)
{
    std::string equation = "ab,bcd,bc->ca";
    PartialShape input1_shape = PartialShape::dynamic();
    PartialShape input2_shape = PartialShape::dynamic();
    PartialShape input3_shape = PartialShape::dynamic();
    PartialShape out_shape{Dimension(), Dimension()};
    auto I1 = make_shared<op::Parameter>(element::i32, input1_shape);
    auto I2 = make_shared<op::Parameter>(element::i32, input2_shape);
    auto I3 = make_shared<op::Parameter>(element::i32, input3_shape);
    auto O = make_shared<op::v7::Einsum>(OutputVector{I1, I2, I3}, equation);
    ASSERT_EQ(O->get_element_type(), element::i32);
    ASSERT_TRUE(O->get_output_partial_shape(0).same_scheme(out_shape));
}

TEST(type_prop, einsum_incorrectequation_subscriptnumber)
{
    std::string equation = "ab,bc,cd->ac";
    Shape input1_shape{2, 3};
    Shape input2_shape{3, 4};
    auto I1 = make_shared<op::Parameter>(element::f32, input1_shape);
    auto I2 = make_shared<op::Parameter>(element::f32, input2_shape);

    try
    {
        auto O = make_shared<op::v7::Einsum>(OutputVector{I1, I2}, equation);
        // Should have thrown, so fail if it didn't
        FAIL() << "Incorrect number of input subscripts";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Equation must contain a number of subscripts equal to a "
                                         "number of Einsum inputs."));
    }
    catch (...)
    {
        FAIL() << "Equation format check failed";
    }
}

TEST(type_prop, einsum_incorrectequation_invalidlabels)
{
    std::string equation = "a$,Bc->ac";
    Shape input1_shape{2, 3};
    Shape input2_shape{3, 4};
    auto I1 = make_shared<op::Parameter>(element::f32, input1_shape);
    auto I2 = make_shared<op::Parameter>(element::f32, input2_shape);

    try
    {
        auto O = make_shared<op::v7::Einsum>(OutputVector{I1, I2}, equation);
        // Should have thrown, so fail if it didn't
        FAIL() << "Incorrect number of input subscripts";
    }
    catch (const CheckFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Input subscript of Einsum equation must consist of either only alphabetic "
                        "letters or alphabetic letters with one ellipsis."));
    }
    catch (...)
    {
        FAIL() << "Equation format check failed";
    }
}

TEST(type_prop, einsum_incorrectequation_incompatibleshapes)
{
    std::string equation = "ab,bc->ac";
    Shape input1_shape{2, 10};
    Shape input2_shape{3, 4};
    auto I1 = make_shared<op::Parameter>(element::f32, input1_shape);
    auto I2 = make_shared<op::Parameter>(element::f32, input2_shape);

    try
    {
        auto O = make_shared<op::v7::Einsum>(OutputVector{I1, I2}, equation);
        // Should have thrown, so fail if it didn't
        FAIL() << "Incompatible dimension indicated by the same labels";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Different input dimensions indicated by the same labels "
                                         "for Einsum must be compatible."));
    }
    catch (...)
    {
        FAIL() << "Equation format check failed";
    }
}

TEST(type_prop, einsum_incorrectequation_notbroadcastableshapes)
{
    std::string equation = "a...b,b...->a...";
    Shape input1_shape{11, 1, 4, 3};
    Shape input2_shape{3, 11, 7, 5};
    auto I1 = make_shared<op::Parameter>(element::f32, input1_shape);
    auto I2 = make_shared<op::Parameter>(element::f32, input2_shape);

    try
    {
        auto O = make_shared<op::v7::Einsum>(OutputVector{I1, I2}, equation);
        // Should have thrown, so fail if it didn't
        FAIL() << "Non-broadcastable shapes covered by ellipsis";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string(
                "Input dimensions labeled with ellipsis for Einsum must be broadcastable."));
    }
    catch (...)
    {
        FAIL() << "Equation format check failed";
    }
}

TEST(type_prop, einsum_incorrectequation_missedellipsis)
{
    std::string equation = "a...b,b...->a";
    Shape input1_shape{11, 1, 4, 3};
    Shape input2_shape{3, 11, 7, 5};
    auto I1 = make_shared<op::Parameter>(element::f32, input1_shape);
    auto I2 = make_shared<op::Parameter>(element::f32, input2_shape);

    try
    {
        auto O = make_shared<op::v7::Einsum>(OutputVector{I1, I2}, equation);
        // Should have thrown, so fail if it didn't
        FAIL() << "Non-broadcastable shapes covered by ellipsis";
    }
    catch (const CheckFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Output subscript of Einsum equation must contain one "
                                         "ellipsis if ellipsis is met in any input subscript."));
    }
    catch (...)
    {
        FAIL() << "Equation format check failed";
    }
}
