//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
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

NGRAPH_SUPPRESS_DEPRECATED_START

using namespace std;
using namespace ngraph;

TEST(type_prop, pad_deduce_1d)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{50});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{});
    CoordinateDiff padding_below{2};
    CoordinateDiff padding_above{3};
    auto pad = make_shared<op::Pad>(param0, param1, padding_below, padding_above);
    EXPECT_EQ(pad->get_element_type(), element::f32);
    EXPECT_EQ(pad->get_shape(), (Shape{55}));

    EXPECT_EQ(pad->get_padding_below(), (CoordinateDiff{2}));
    EXPECT_EQ(pad->get_padding_above(), (CoordinateDiff{3}));
}

TEST(type_prop, pad_deduce_2d)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{50, 40});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{});
    CoordinateDiff padding_below{5, 3};
    CoordinateDiff padding_above{6, 9};
    auto pad = make_shared<op::Pad>(param0, param1, padding_below, padding_above);
    EXPECT_EQ(pad->get_element_type(), element::f32);
    EXPECT_EQ(pad->get_shape(), (Shape{61, 52}));

    EXPECT_EQ(pad->get_padding_below(), (CoordinateDiff{5, 3}));
    EXPECT_EQ(pad->get_padding_above(), (CoordinateDiff{6, 9}));
}

TEST(type_prop, pad_deduce_3d)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{50, 40, 20});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{});
    CoordinateDiff padding_below{5, 3, 0};
    CoordinateDiff padding_above{6, 9, 4};
    auto pad = make_shared<op::Pad>(param0, param1, padding_below, padding_above);
    EXPECT_EQ(pad->get_element_type(), element::f32);
    EXPECT_EQ(pad->get_shape(), (Shape{61, 52, 24}));

    EXPECT_EQ(pad->get_padding_below(), (CoordinateDiff{5, 3, 0}));
    EXPECT_EQ(pad->get_padding_above(), (CoordinateDiff{6, 9, 4}));
}

TEST(type_prop, pad_deduce_3d_neg)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{50, 40, 20});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{});
    CoordinateDiff padding_below{-5, 3, -2};
    CoordinateDiff padding_above{-6, -9, 4};
    auto pad = make_shared<op::Pad>(param0, param1, padding_below, padding_above);
    EXPECT_EQ(pad->get_element_type(), element::f32);
    EXPECT_EQ(pad->get_shape(), (Shape{39, 34, 22}));

    EXPECT_EQ(pad->get_padding_below(), (CoordinateDiff{-5, 3, -2}));
    EXPECT_EQ(pad->get_padding_above(), (CoordinateDiff{-6, -9, 4}));
}

TEST(type_prop, pad_deduce_element_type_mismatch)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{50, 40, 20});
    auto param1 = make_shared<op::Parameter>(element::i32, Shape{});
    CoordinateDiff padding_below{5, 3, 0};
    CoordinateDiff padding_above{6, 9, 4};
    try
    {
        auto pad = make_shared<op::Pad>(param0, param1, padding_below, padding_above);

        // Should have thrown, so fail if it didn't
        FAIL() << "Element tpye mismatch not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Argument element types do not match"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, pad_deduce_nonscalar_pad_value)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{50, 40, 20});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{6});
    CoordinateDiff padding_below{5, 3, 0};
    CoordinateDiff padding_above{6, 9, 4};
    try
    {
        auto pad = make_shared<op::Pad>(param0, param1, padding_below, padding_above);

        // Should have thrown, so fail if it didn't
        FAIL() << "Non-scalar pad value not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Argument for padding value is not a scalar"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, pad_deduce_below_padding_wrong_rank)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{50, 40, 20});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{});
    CoordinateDiff padding_below{5, 3, 0, 6};
    CoordinateDiff padding_above{6, 9, 4};
    try
    {
        auto pad = make_shared<op::Pad>(param0, param1, padding_below, padding_above);

        // Should have thrown, so fail if it didn't
        FAIL() << "Wrong below-padding rank not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Ranks for padding below (CoordinateDiff{5, 3, 0, 6}) and "
                                         "padding above (CoordinateDiff{6, 9, "
                                         "4}) do not match"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, pad_deduce_above_padding_wrong_rank)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{50, 40, 20});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{});
    CoordinateDiff padding_below{5, 3, 0};
    CoordinateDiff padding_above{6, 9};
    try
    {
        auto pad = make_shared<op::Pad>(param0, param1, padding_below, padding_above);

        // Should have thrown, so fail if it didn't
        FAIL() << "Wrong above-padding rank not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Ranks for padding below (CoordinateDiff{5, 3, 0}) and "
                                         "padding above (CoordinateDiff{6, 9}) do not match"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, pad_deduce_too_small_for_edge)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{1, 5, 0, 2});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{});
    CoordinateDiff padding_below{0, 1, 2, 3};
    CoordinateDiff padding_above{0, 1, 2, 3};
    try
    {
        auto pad =
            make_shared<op::Pad>(param0, param1, padding_below, padding_above, op::PadMode::EDGE);

        // Should have thrown, so fail if it didn't
        FAIL() << "Input too small for edge padding not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("EDGE padding mode requires an input of dimension of at "
                                         "least 1 at each spatial axis"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, pad_deduce_too_small_for_reflect)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{1, 5, 1, 2});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{});
    CoordinateDiff padding_below{0, 1, 2, 3};
    CoordinateDiff padding_above{0, 1, 2, 3};
    try
    {
        auto pad = make_shared<op::Pad>(
            param0, param1, padding_below, padding_above, op::PadMode::REFLECT);

        // Should have thrown, so fail if it didn't
        FAIL() << "Input too small for reflect padding not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("REFLECT padding mode requires an input of dimension of "
                                         "at least 2 at each spatial axis"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, pad_deduce_too_much_negative_padding)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{5, 4, 2});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{});
    CoordinateDiff padding_below{5, 3, 0};
    CoordinateDiff padding_above{6, 9, -3};
    try
    {
        auto pad = make_shared<op::Pad>(param0, param1, padding_below, padding_above);

        // Should have thrown, so fail if it didn't
        FAIL() << "Too much negative padding not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Inferred result dimension at axis 2 is negative after padding"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, pad_partial_data_rank_dynamic_padding_rank_dynamic_ok)
{
    auto param0 = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto param1 = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());

    CoordinateDiff padding_below{2, 4, 6};
    CoordinateDiff padding_above{8, 2, 3};

    auto pad = make_shared<op::Pad>(param0, param1, padding_below, padding_above);

    ASSERT_EQ(pad->get_output_element_type(0), element::f32);
    ASSERT_TRUE(pad->get_output_partial_shape(0).same_scheme(
        PartialShape{Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic()}));
}

TEST(type_prop, pad_partial_data_rank_dynamic_padding_rank_dynamic_attribs_rank_inconsistent)
{
    auto param0 = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto param1 = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());

    CoordinateDiff padding_below{2, 4, 6};
    CoordinateDiff padding_above{8, 2, 3, 0};

    try
    {
        auto pad = make_shared<op::Pad>(param0, param1, padding_below, padding_above);
        FAIL() << "Inconsistent attribute ranks not detected (rank-dynamic/rank-dynamic arguments)";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Ranks for padding below (CoordinateDiff{2, 4, 6}) and "
                                         "padding above (CoordinateDiff{8, 2, 3, "
                                         "0}) do not match"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, pad_partial_data_rank_static_dynamic_padding_rank_dynamic_ok)
{
    auto param0 = make_shared<op::Parameter>(
        element::f32,
        PartialShape{Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic()});
    auto param1 = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());

    CoordinateDiff padding_below{2, 4, 6};
    CoordinateDiff padding_above{8, 2, 3};

    auto pad = make_shared<op::Pad>(param0, param1, padding_below, padding_above);

    ASSERT_EQ(pad->get_output_element_type(0), element::f32);
    ASSERT_TRUE(pad->get_output_partial_shape(0).same_scheme(
        PartialShape{Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic()}));
}

TEST(type_prop, pad_partial_data_rank_static_dynamic_some_dims_known_padding_rank_dynamic_ok)
{
    auto param0 =
        make_shared<op::Parameter>(element::f32, PartialShape{3, 5, Dimension::dynamic()});
    auto param1 = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());

    CoordinateDiff padding_below{2, 4, 6};
    CoordinateDiff padding_above{8, 2, 3};

    auto pad = make_shared<op::Pad>(param0, param1, padding_below, padding_above);

    ASSERT_EQ(pad->get_output_element_type(0), element::f32);
    ASSERT_TRUE(
        pad->get_output_partial_shape(0).same_scheme(PartialShape{13, 11, Dimension::dynamic()}));
}

TEST(type_prop, pad_partial_data_rank_dynamic_padding_static_ok)
{
    auto param0 = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{});

    CoordinateDiff padding_below{2, 4, 6};
    CoordinateDiff padding_above{8, 2, 3};

    auto pad = make_shared<op::Pad>(param0, param1, padding_below, padding_above);

    ASSERT_EQ(pad->get_output_element_type(0), element::f32);
    ASSERT_TRUE(pad->get_output_partial_shape(0).same_scheme(
        PartialShape{Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic()}));
}

TEST(type_prop, pad_partial_data_rank_dynamic_padding_static_wrong_padding_rank)
{
    auto param0 = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{2, 3, 8});

    CoordinateDiff padding_below{2, 4, 6};
    CoordinateDiff padding_above{8, 2, 3};

    try
    {
        auto pad = make_shared<op::Pad>(param0, param1, padding_below, padding_above);
        FAIL() << "Wrong padding rank not detected (rank-dynamic/static arguments)";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Argument for padding value is not a scalar (shape: {2,3,8})"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, pad_partial_data_rank_dynamic_padding_static_attribs_rank_inconsistent)
{
    auto param0 = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{});

    CoordinateDiff padding_below{2, 4, 6};
    CoordinateDiff padding_above{8, 2, 3, 4};

    try
    {
        auto pad = make_shared<op::Pad>(param0, param1, padding_below, padding_above);
        FAIL() << "Wrong padding rank not detected (rank-dynamic/static arguments)";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Ranks for padding below (CoordinateDiff{2, 4, 6}) and "
                                         "padding above (CoordinateDiff{8, 2, 3, "
                                         "4}) do not match"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, pad_v1_arg_pad_value_type_mismatch)
{
    auto arg = make_shared<op::Parameter>(element::f32, Shape{1, 2, 3});
    auto pads_begin = make_shared<op::Parameter>(element::i64, Shape{1});
    auto pads_end = make_shared<op::Parameter>(element::i64, Shape{1});
    auto arg_pad_value = make_shared<op::Parameter>(element::f16, Shape{1});

    try
    {
        auto pad_v1 = make_shared<op::v1::Pad>(
            arg, pads_begin, pads_end, arg_pad_value, op::PadMode::CONSTANT);

        // Should have thrown, so fail if it didn't
        FAIL() << "Incorrect arg_pad_value type exception not handled";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Argument element types do not match (input arg element type:"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, pad_v1_arg_pad_value_shape_not_compatible)
{
    auto arg = make_shared<op::Parameter>(element::f32, Shape{1, 2, 3});
    auto pads_begin = make_shared<op::Parameter>(element::i64, Shape{1});
    auto pads_end = make_shared<op::Parameter>(element::i64, Shape{1});
    auto arg_pad_value = make_shared<op::Parameter>(element::f32, Shape{1});

    try
    {
        auto pad_v1 = make_shared<op::v1::Pad>(
            arg, pads_begin, pads_end, arg_pad_value, op::PadMode::CONSTANT);

        // Should have thrown, so fail if it didn't
        FAIL() << "Incorrect arg_pad_value shape exception not handled";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Argument for padding value is not a scalar (shape:"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, pad_v1_pads_begin_shape_not_1D)
{
    auto arg = make_shared<op::Parameter>(element::f32, Shape{1, 2, 3});
    auto pads_begin = make_shared<op::Parameter>(element::i64, Shape{1, 2});
    auto pads_end = make_shared<op::Parameter>(element::i64, Shape{1});

    try
    {
        auto pad_v1 = make_shared<op::v1::Pad>(arg, pads_begin, pads_end, op::PadMode::SYMMETRIC);

        // Should have thrown, so fail if it didn't
        FAIL() << "Incorrect pads_begin shape exception not handled";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Argument for pads_begin is not 1D (shape:"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, pad_v1_pads_end_shape_not_1D)
{
    auto arg = make_shared<op::Parameter>(element::f32, Shape{1, 2, 3});
    auto pads_begin = make_shared<op::Parameter>(element::i64, Shape{1});
    auto pads_end = make_shared<op::Parameter>(element::i64, Shape{1, 2});

    try
    {
        auto pad_v1 = make_shared<op::v1::Pad>(arg, pads_begin, pads_end, op::PadMode::SYMMETRIC);

        // Should have thrown, so fail if it didn't
        FAIL() << "Incorrect pads_end shape exception not handled";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Argument for pads_end is not 1D (shape:"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, pad_v1_pads_begin_size_not_correct)
{
    auto arg = make_shared<op::Parameter>(element::f32, Shape{1, 2, 3});
    auto pads_begin = make_shared<op::Parameter>(element::i64, Shape{4});
    auto pads_end = make_shared<op::Parameter>(element::i64, Shape{1});

    try
    {
        auto pad_v1 = make_shared<op::v1::Pad>(arg, pads_begin, pads_end, op::PadMode::SYMMETRIC);

        // Should have thrown, so fail if it didn't
        FAIL() << "Incorrect pads_begin size exception not handled";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Number of elements of pads_begin must be >= 0 and <= arg "
                                         "rank (pads_begin_shape[0]:"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, pad_v1_pads_end_size_not_correct)
{
    auto arg = make_shared<op::Parameter>(element::f32, Shape{1, 2, 3});
    auto pads_begin = make_shared<op::Parameter>(element::i64, Shape{1});
    auto pads_end = make_shared<op::Parameter>(element::i64, Shape{4});
    auto arg_pad_value = make_shared<op::Parameter>(element::f32, Shape{});

    try
    {
        auto pad_v1 = make_shared<op::v1::Pad>(
            arg, pads_begin, pads_end, arg_pad_value, op::PadMode::CONSTANT);

        // Should have thrown, so fail if it didn't
        FAIL() << "Incorrect pads_end size exception not handled";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string(
                "Number of elements of pads_end must be >= 0 and <= arg rank (pads_end_shape[0]:"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, pad_v1_arg_pads_begin_incompatible_type)
{
    auto arg = make_shared<op::Parameter>(element::f32, Shape{1, 2, 3});
    auto pads_begin = make_shared<op::Parameter>(element::f32, Shape{1});
    auto pads_end = make_shared<op::Parameter>(element::i64, Shape{1});

    try
    {
        auto pad_v1 = make_shared<op::v1::Pad>(arg, pads_begin, pads_end, op::PadMode::REFLECT);

        // Should have thrown, so fail if it didn't
        FAIL() << "Incorrect pad_begin type exception not handled";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("pads_begin must be an integral number, but is:"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, pad_v1_arg_pads_end_incompatible_type)
{
    auto arg = make_shared<op::Parameter>(element::f32, Shape{1, 2, 3});
    auto pads_begin = make_shared<op::Parameter>(element::i64, Shape{1});
    auto pads_end = make_shared<op::Parameter>(element::f32, Shape{1});

    try
    {
        auto pad = make_shared<op::v1::Pad>(arg, pads_begin, pads_end, op::PadMode::REFLECT);

        // Should have thrown, so fail if it didn't
        FAIL() << "Incorrect pads_end type exception not thrown";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("pads_end must be an integral number, but is:"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, pad_v1_deduce_too_small_for_edge)
{
    auto arg = make_shared<op::Parameter>(element::f32, Shape{1, 5, 0, 2});
    auto pads_begin =
        make_shared<op::Constant>(element::i64, Shape{4}, std::vector<int64_t>{0, 1, 2, 3});
    auto pads_end =
        make_shared<op::Constant>(element::i64, Shape{4}, std::vector<int64_t>{0, 1, 2, 3});
    auto arg_pad_value = make_shared<op::Parameter>(element::f32, Shape{});

    try
    {
        auto pad_v1 =
            make_shared<op::v1::Pad>(arg, pads_begin, pads_end, arg_pad_value, op::PadMode::EDGE);

        // Should have thrown, so fail if it didn't
        FAIL() << "Incorrect input shape exception for EDGE mode not thrown";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("EDGE padding mode requires an input of dimension of at "
                                         "least 1 at each spatial axis"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, pad_v1_deduce_too_small_for_reflect)
{
    auto arg = make_shared<op::Parameter>(element::f32, Shape{1, 5, 1, 2});
    auto pads_begin =
        make_shared<op::Constant>(element::i64, Shape{4}, std::vector<int64_t>{0, 1, 2, 3});
    auto pads_end =
        make_shared<op::Constant>(element::i64, Shape{4}, std::vector<int64_t>{0, 1, 2, 3});
    auto arg_pad_value = make_shared<op::Parameter>(element::f32, Shape{});

    try
    {
        auto pad_v1 = make_shared<op::v1::Pad>(
            arg, pads_begin, pads_end, arg_pad_value, op::PadMode::REFLECT);

        // Should have thrown, so fail if it didn't
        FAIL() << "Incorrect input shape exception for REFLECT mode not thrown";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("REFLECT padding mode requires an input of dimension of "
                                         "at least 2 at each spatial axis"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}
