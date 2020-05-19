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

using namespace std;
using namespace ngraph;

TEST(type_prop, range_nonconst_ok)
{
    auto start = make_shared<op::Parameter>(element::i32, Shape{});
    auto stop = make_shared<op::Parameter>(element::i32, Shape{});
    auto step = make_shared<op::Parameter>(element::i32, Shape{});

    auto range = make_shared<op::Range>(start, stop, step);

    EXPECT_EQ(range->get_element_type(), element::i32);
    EXPECT_TRUE(range->get_output_partial_shape(0).same_scheme(PartialShape::dynamic(1)));
}

TEST(type_prop, range_nonconst_some_dyn_et_ok)
{
    auto start = make_shared<op::Parameter>(element::i32, Shape{});
    auto stop = make_shared<op::Parameter>(element::dynamic, Shape{});
    auto step = make_shared<op::Parameter>(element::i32, Shape{});

    auto range = make_shared<op::Range>(start, stop, step);

    EXPECT_EQ(range->get_element_type(), element::i32);
    EXPECT_TRUE(range->get_output_partial_shape(0).same_scheme(PartialShape::dynamic(1)));
}

TEST(type_prop, range_nonconst_all_dyn_et_ok)
{
    auto start = make_shared<op::Parameter>(element::dynamic, Shape{});
    auto stop = make_shared<op::Parameter>(element::dynamic, Shape{});
    auto step = make_shared<op::Parameter>(element::dynamic, Shape{});

    auto range = make_shared<op::Range>(start, stop, step);

    EXPECT_EQ(range->get_element_type(), element::dynamic);
    EXPECT_TRUE(range->get_output_partial_shape(0).same_scheme(PartialShape::dynamic(1)));
}

TEST(type_prop, range_nonconst_f32_ok)
{
    auto start = make_shared<op::Parameter>(element::dynamic, Shape{});
    auto stop = make_shared<op::Parameter>(element::f32, Shape{});
    auto step = make_shared<op::Parameter>(element::dynamic, Shape{});

    auto range = make_shared<op::Range>(start, stop, step);

    EXPECT_EQ(range->get_element_type(), element::f32);
    EXPECT_TRUE(range->get_output_partial_shape(0).same_scheme(PartialShape::dynamic(1)));
}

TEST(type_prop, range_nonconst_boolean_fails)
{
    auto start = make_shared<op::Parameter>(element::dynamic, Shape{});
    auto stop = make_shared<op::Parameter>(element::boolean, Shape{});
    auto step = make_shared<op::Parameter>(element::dynamic, Shape{});

    try
    {
        auto range = make_shared<op::Range>(start, stop, step);
        FAIL() << "Boolean element type not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             "Element type for start, stop, and step, must not be boolean.");
    }
    catch (...)
    {
        FAIL() << "Test failed for unexpected reason";
    }
}

TEST(type_prop, range_some_const_ok)
{
    auto start = make_shared<op::Constant>(element::i32, Shape{}, std::vector<int32_t>{3});
    auto stop = make_shared<op::Parameter>(element::i32, Shape{});
    auto step = make_shared<op::Constant>(element::i32, Shape{}, std::vector<int32_t>{2});

    auto range = make_shared<op::Range>(start, stop, step);

    EXPECT_EQ(range->get_element_type(), element::i32);
    EXPECT_TRUE(range->get_output_partial_shape(0).same_scheme(PartialShape::dynamic(1)));
}

TEST(type_prop, range_some_const_zero_stride_fails)
{
    auto start = make_shared<op::Constant>(element::i32, Shape{}, std::vector<int32_t>{3});
    auto stop = make_shared<op::Parameter>(element::i32, Shape{});
    auto step = make_shared<op::Constant>(element::i32, Shape{}, std::vector<int32_t>{0});

    try
    {
        auto range = make_shared<op::Range>(start, stop, step);
        FAIL() << "Zero stride not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "'step' cannot be zero");
    }
    catch (...)
    {
        FAIL() << "Test failed for unexpected reason";
    }
}

TEST(type_prop, range_some_const_plus_inf_start_fails)
{
    auto start = make_shared<op::Constant>(
        element::f32, Shape{}, std::vector<float>{std::numeric_limits<float>::infinity()});
    auto stop = make_shared<op::Parameter>(element::f32, Shape{});
    auto step = make_shared<op::Constant>(element::f32, Shape{}, std::vector<float>{1});

    try
    {
        auto range = make_shared<op::Range>(start, stop, step);
        FAIL() << "+Infinity start not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "'start' cannot be nan or infinite.");
    }
    catch (...)
    {
        FAIL() << "Test failed for unexpected reason";
    }
}

TEST(type_prop, range_some_const_minus_inf_start_fails)
{
    auto start = make_shared<op::Constant>(
        element::f32, Shape{}, std::vector<float>{-std::numeric_limits<float>::infinity()});
    auto stop = make_shared<op::Parameter>(element::f32, Shape{});
    auto step = make_shared<op::Constant>(element::f32, Shape{}, std::vector<float>{1});

    try
    {
        auto range = make_shared<op::Range>(start, stop, step);
        FAIL() << "-Infinity start not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "'start' cannot be nan or infinite.");
    }
    catch (...)
    {
        FAIL() << "Test failed for unexpected reason";
    }
}

TEST(type_prop, range_some_const_nan_start_fails)
{
    auto start =
        make_shared<op::Constant>(element::f32, Shape{}, std::vector<float>{std::nanf("")});
    auto stop = make_shared<op::Parameter>(element::f32, Shape{});
    auto step = make_shared<op::Constant>(element::f32, Shape{}, std::vector<float>{1});

    try
    {
        auto range = make_shared<op::Range>(start, stop, step);
        FAIL() << "NaN start not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "'start' cannot be nan or infinite.");
    }
    catch (...)
    {
        FAIL() << "Test failed for unexpected reason";
    }
}

TEST(type_prop, range_some_const_plus_inf_stop_fails)
{
    auto start = make_shared<op::Parameter>(element::f32, Shape{});
    auto stop = make_shared<op::Constant>(
        element::f32, Shape{}, std::vector<float>{std::numeric_limits<float>::infinity()});
    auto step = make_shared<op::Constant>(element::f32, Shape{}, std::vector<float>{1});

    try
    {
        auto range = make_shared<op::Range>(start, stop, step);
        FAIL() << "+Infinity stop not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "'stop' cannot be nan or infinite.");
    }
    catch (...)
    {
        FAIL() << "Test failed for unexpected reason";
    }
}

TEST(type_prop, range_some_const_minus_inf_stop_fails)
{
    auto start = make_shared<op::Parameter>(element::f32, Shape{});
    auto stop = make_shared<op::Constant>(
        element::f32, Shape{}, std::vector<float>{-std::numeric_limits<float>::infinity()});
    auto step = make_shared<op::Constant>(element::f32, Shape{}, std::vector<float>{1});

    try
    {
        auto range = make_shared<op::Range>(start, stop, step);
        FAIL() << "-Infinity stop not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "'stop' cannot be nan or infinite.");
    }
    catch (...)
    {
        FAIL() << "Test failed for unexpected reason";
    }
}

TEST(type_prop, range_some_const_nan_stio_fails)
{
    auto start = make_shared<op::Parameter>(element::f32, Shape{});
    auto stop = make_shared<op::Constant>(element::f32, Shape{}, std::vector<float>{std::nanf("")});
    auto step = make_shared<op::Constant>(element::f32, Shape{}, std::vector<float>{1});

    try
    {
        auto range = make_shared<op::Range>(start, stop, step);
        FAIL() << "NaN stop not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "'stop' cannot be nan or infinite.");
    }
    catch (...)
    {
        FAIL() << "Test failed for unexpected reason";
    }
}

TEST(type_prop, range_some_const_plus_inf_stride_fails)
{
    auto start = make_shared<op::Constant>(element::f32, Shape{}, std::vector<float>{3});
    auto stop = make_shared<op::Parameter>(element::f32, Shape{});
    auto step = make_shared<op::Constant>(
        element::f32, Shape{}, std::vector<float>{std::numeric_limits<float>::infinity()});

    try
    {
        auto range = make_shared<op::Range>(start, stop, step);
        FAIL() << "+Infinity stride not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "'step' cannot be zero, nan, or infinite.");
    }
    catch (...)
    {
        FAIL() << "Test failed for unexpected reason";
    }
}

TEST(type_prop, range_some_const_minus_inf_stride_fails)
{
    auto start = make_shared<op::Constant>(element::f32, Shape{}, std::vector<float>{3});
    auto stop = make_shared<op::Parameter>(element::f32, Shape{});
    auto step = make_shared<op::Constant>(
        element::f32, Shape{}, std::vector<float>{-std::numeric_limits<float>::infinity()});

    try
    {
        auto range = make_shared<op::Range>(start, stop, step);
        FAIL() << "-Infinity stride not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "'step' cannot be zero, nan, or infinite.");
    }
    catch (...)
    {
        FAIL() << "Test failed for unexpected reason";
    }
}

TEST(type_prop, range_some_const_nan_stride_fails)
{
    auto start = make_shared<op::Constant>(element::f32, Shape{}, std::vector<float>{3});
    auto stop = make_shared<op::Parameter>(element::f32, Shape{});
    auto step = make_shared<op::Constant>(element::f32, Shape{}, std::vector<float>{std::nanf("")});

    try
    {
        auto range = make_shared<op::Range>(start, stop, step);
        FAIL() << "NaN stride not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "'step' cannot be zero, nan, or infinite.");
    }
    catch (...)
    {
        FAIL() << "Test failed for unexpected reason";
    }
}

TEST(type_prop, range_all_const_zero_stride_fails)
{
    auto start = make_shared<op::Constant>(element::i32, Shape{}, std::vector<int32_t>{3});
    auto stop = make_shared<op::Constant>(element::i32, Shape{}, std::vector<int32_t>{5});
    auto step = make_shared<op::Constant>(element::i32, Shape{}, std::vector<int32_t>{0});

    try
    {
        auto range = make_shared<op::Range>(start, stop, step);
        FAIL() << "Zero stride not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "'step' cannot be zero");
    }
    catch (...)
    {
        FAIL() << "Test failed for unexpected reason";
    }
}

struct RangeParams
{
    double start;
    double stop;
    double step;
    PartialShape expected_shape;
};

template <typename T>
void run_range_test(const element::Type& et, const RangeParams& params)
{
    auto start =
        make_shared<op::Constant>(et, Shape{}, std::vector<T>{static_cast<T>(params.start)});
    auto stop = make_shared<op::Constant>(et, Shape{}, std::vector<T>{static_cast<T>(params.stop)});
    auto step = make_shared<op::Constant>(et, Shape{}, std::vector<T>{static_cast<T>(params.step)});

    auto range = make_shared<op::Range>(start, stop, step);

    EXPECT_EQ(range->get_element_type(), et);
    EXPECT_TRUE(range->get_output_partial_shape(0).same_scheme(params.expected_shape))
        << "Expected shape " << params.expected_shape << " but got "
        << range->get_output_partial_shape(0);
}

struct RangeTest : ::testing::TestWithParam<RangeParams>
{
};

TEST_P(RangeTest, deduce_shape_i8)
{
    run_range_test<int8_t>(element::i8, GetParam());
}

TEST_P(RangeTest, deduce_shape_i16)
{
    run_range_test<int16_t>(element::i16, GetParam());
}

TEST_P(RangeTest, deduce_shape_i32)
{
    run_range_test<int32_t>(element::i32, GetParam());
}

TEST_P(RangeTest, deduce_shape_i64)
{
    run_range_test<int64_t>(element::i64, GetParam());
}

TEST_P(RangeTest, deduce_shape_u8)
{
    run_range_test<uint8_t>(element::u8, GetParam());
}

TEST_P(RangeTest, deduce_shape_u16)
{
    run_range_test<uint16_t>(element::u16, GetParam());
}

TEST_P(RangeTest, deduce_shape_u32)
{
    run_range_test<uint32_t>(element::u32, GetParam());
}

TEST_P(RangeTest, deduce_shape_u64)
{
    run_range_test<uint64_t>(element::u64, GetParam());
}

TEST_P(RangeTest, deduce_shape_bf16)
{
    run_range_test<bfloat16>(element::bf16, GetParam());
}

TEST_P(RangeTest, deduce_shape_f16)
{
    run_range_test<float16>(element::f16, GetParam());
}

TEST_P(RangeTest, deduce_shape_f32)
{
    run_range_test<float>(element::f32, GetParam());
}

TEST_P(RangeTest, deduce_shape_f64)
{
    run_range_test<double>(element::f64, GetParam());
}

INSTANTIATE_TEST_CASE_P(type_prop,
                        RangeTest,
                        ::testing::Values(RangeParams{0, 5, 1, PartialShape{5}},
                                          RangeParams{0, 22, 2, PartialShape{11}},
                                          RangeParams{1, 23, 2, PartialShape{11}},
                                          RangeParams{1, 22, 2, PartialShape{11}},
                                          RangeParams{0, 0, 1, PartialShape{0}},
                                          RangeParams{1, 0, 2, PartialShape{0}}),
                        PrintToDummyParamName());

struct RangeTestWithNegatives : ::testing::TestWithParam<RangeParams>
{
};

TEST_P(RangeTestWithNegatives, deduce_shape_i8)
{
    run_range_test<int8_t>(element::i8, GetParam());
}

TEST_P(RangeTestWithNegatives, deduce_shape_i16)
{
    run_range_test<int16_t>(element::i16, GetParam());
}

TEST_P(RangeTestWithNegatives, deduce_shape_i32)
{
    run_range_test<int32_t>(element::i32, GetParam());
}

TEST_P(RangeTestWithNegatives, deduce_shape_i64)
{
    run_range_test<int64_t>(element::i64, GetParam());
}

TEST_P(RangeTestWithNegatives, deduce_shape_bf16)
{
    run_range_test<bfloat16>(element::bf16, GetParam());
}

TEST_P(RangeTestWithNegatives, deduce_shape_f16)
{
    run_range_test<float16>(element::f16, GetParam());
}

TEST_P(RangeTestWithNegatives, deduce_shape_f32)
{
    run_range_test<float>(element::f32, GetParam());
}

TEST_P(RangeTestWithNegatives, deduce_shape_f64)
{
    run_range_test<double>(element::f64, GetParam());
}

INSTANTIATE_TEST_CASE_P(type_prop,
                        RangeTestWithNegatives,
                        ::testing::Values(RangeParams{2, 0, -2, PartialShape{1}},
                                          RangeParams{2, 0, -1, PartialShape{2}},
                                          RangeParams{-19, 19, 1, PartialShape{38}},
                                          RangeParams{-19, 19, 3, PartialShape{13}},
                                          RangeParams{20, -19, 1, PartialShape{0}}),
                        PrintToDummyParamName());

struct RangeTestFloating : ::testing::TestWithParam<RangeParams>
{
};

TEST_P(RangeTestFloating, deduce_shape_bf16)
{
    run_range_test<bfloat16>(element::bf16, GetParam());
}

TEST_P(RangeTestFloating, deduce_shape_f16)
{
    run_range_test<float16>(element::f16, GetParam());
}

TEST_P(RangeTestFloating, deduce_shape_f32)
{
    run_range_test<float>(element::f32, GetParam());
}

TEST_P(RangeTestFloating, deduce_shape_f64)
{
    run_range_test<double>(element::f64, GetParam());
}

INSTANTIATE_TEST_CASE_P(type_prop,
                        RangeTestFloating,
                        ::testing::Values(RangeParams{0, 1, 0.25, PartialShape{4}},
                                          RangeParams{-1, 1, 0.25, PartialShape{8}},
                                          RangeParams{-1, 0.875, 0.25, PartialShape{8}}),
                        PrintToDummyParamName());
