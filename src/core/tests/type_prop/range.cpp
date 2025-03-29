// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/range.hpp"

#include "common_test_utils/type_prop.hpp"
#include "openvino/op/constant.hpp"

using namespace std;
using namespace ov;

struct RangeParams {
    double start;
    double stop;
    double step;
    PartialShape expected_shape;
};

// ------------------------------ V0 ------------------------------

TEST(type_prop, range_nonconst_ok) {
    auto start = make_shared<ov::op::v0::Parameter>(element::i32, Shape{});
    auto stop = make_shared<ov::op::v0::Parameter>(element::i32, Shape{});
    auto step = make_shared<ov::op::v0::Parameter>(element::i32, Shape{});

    auto range = make_shared<op::v0::Range>(start, stop, step);

    EXPECT_EQ(range->get_element_type(), element::i32);
    EXPECT_TRUE(range->get_output_partial_shape(0).same_scheme(PartialShape::dynamic(1)));
}

TEST(type_prop, range_nonconst_some_dyn_et_ok) {
    auto start = make_shared<ov::op::v0::Parameter>(element::i32, Shape{});
    auto stop = make_shared<ov::op::v0::Parameter>(element::dynamic, Shape{});
    auto step = make_shared<ov::op::v0::Parameter>(element::i32, Shape{});

    auto range = make_shared<op::v0::Range>(start, stop, step);

    EXPECT_EQ(range->get_element_type(), element::i32);
    EXPECT_TRUE(range->get_output_partial_shape(0).same_scheme(PartialShape::dynamic(1)));
}

TEST(type_prop, range_nonconst_all_dyn_et_ok) {
    auto start = make_shared<ov::op::v0::Parameter>(element::dynamic, Shape{});
    auto stop = make_shared<ov::op::v0::Parameter>(element::dynamic, Shape{});
    auto step = make_shared<ov::op::v0::Parameter>(element::dynamic, Shape{});

    auto range = make_shared<op::v0::Range>(start, stop, step);

    EXPECT_EQ(range->get_element_type(), element::dynamic);
    EXPECT_TRUE(range->get_output_partial_shape(0).same_scheme(PartialShape::dynamic(1)));
}

TEST(type_prop, range_nonconst_f32_ok) {
    auto start = make_shared<ov::op::v0::Parameter>(element::dynamic, Shape{});
    auto stop = make_shared<ov::op::v0::Parameter>(element::f32, Shape{});
    auto step = make_shared<ov::op::v0::Parameter>(element::dynamic, Shape{});

    auto range = make_shared<op::v0::Range>(start, stop, step);

    EXPECT_EQ(range->get_element_type(), element::f32);
    EXPECT_TRUE(range->get_output_partial_shape(0).same_scheme(PartialShape::dynamic(1)));
}

TEST(type_prop, range_nonconst_boolean_fails) {
    auto start = make_shared<ov::op::v0::Parameter>(element::dynamic, Shape{});
    auto stop = make_shared<ov::op::v0::Parameter>(element::boolean, Shape{});
    auto step = make_shared<ov::op::v0::Parameter>(element::dynamic, Shape{});

    try {
        auto range = make_shared<op::v0::Range>(start, stop, step);
        FAIL() << "Boolean element type not detected";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), "Element type for start, stop, and step, must not be boolean.");
    } catch (...) {
        FAIL() << "Test failed for unexpected reason";
    }
}

TEST(type_prop, range_some_const_ok) {
    auto start = make_shared<ov::op::v0::Constant>(element::i32, Shape{}, std::vector<int32_t>{3});
    auto stop = make_shared<ov::op::v0::Parameter>(element::i32, Shape{});
    auto step = make_shared<ov::op::v0::Constant>(element::i32, Shape{}, std::vector<int32_t>{2});

    auto range = make_shared<op::v0::Range>(start, stop, step);

    EXPECT_EQ(range->get_element_type(), element::i32);
    EXPECT_TRUE(range->get_output_partial_shape(0).same_scheme(PartialShape::dynamic(1)));
}

TEST(type_prop, range_some_const_zero_stride_fails) {
    auto start = make_shared<ov::op::v0::Constant>(element::i32, Shape{}, std::vector<int32_t>{3});
    auto stop = make_shared<ov::op::v0::Parameter>(element::i32, Shape{});
    auto step = make_shared<ov::op::v0::Constant>(element::i32, Shape{}, std::vector<int32_t>{0});

    try {
        auto range = make_shared<op::v0::Range>(start, stop, step);
        FAIL() << "Zero stride not detected";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), "'step' cannot be zero");
    } catch (...) {
        FAIL() << "Test failed for unexpected reason";
    }
}

TEST(type_prop, range_some_const_plus_inf_start_fails) {
    auto start = make_shared<ov::op::v0::Constant>(element::f32,
                                                   Shape{},
                                                   std::vector<float>{std::numeric_limits<float>::infinity()});
    auto stop = make_shared<ov::op::v0::Parameter>(element::f32, Shape{});
    auto step = make_shared<ov::op::v0::Constant>(element::f32, Shape{}, std::vector<float>{1});

    try {
        auto range = make_shared<op::v0::Range>(start, stop, step);
        FAIL() << "+Infinity start not detected";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), "'start' cannot be nan or infinite.");
    } catch (const ov::Exception& error) {
        EXPECT_HAS_SUBSTRING(error.what(), "Check 'std::numeric_limits<OUT_T>::max() >= c");
    } catch (...) {
        FAIL() << "Test failed for unexpected reason";
    }
}

TEST(type_prop, range_some_const_minus_inf_start_fails) {
    auto start = make_shared<ov::op::v0::Constant>(element::f32,
                                                   Shape{},
                                                   std::vector<float>{-std::numeric_limits<float>::infinity()});
    auto stop = make_shared<ov::op::v0::Parameter>(element::f32, Shape{});
    auto step = make_shared<ov::op::v0::Constant>(element::f32, Shape{}, std::vector<float>{1});

    try {
        auto range = make_shared<op::v0::Range>(start, stop, step);
        FAIL() << "-Infinity start not detected";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), "'start' cannot be nan or infinite.");
    } catch (const ov::Exception& error) {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            "Check '!std::numeric_limits<IN_T>::is_signed || std::numeric_limits<OUT_T>::lowest() <= c");
    } catch (...) {
        FAIL() << "Test failed for unexpected reason";
    }
}

TEST(type_prop, range_some_const_nan_start_fails) {
    auto start = make_shared<ov::op::v0::Constant>(element::f32, Shape{}, std::vector<float>{std::nanf("")});
    auto stop = make_shared<ov::op::v0::Parameter>(element::f32, Shape{});
    auto step = make_shared<ov::op::v0::Constant>(element::f32, Shape{}, std::vector<float>{1});

    try {
        auto range = make_shared<op::v0::Range>(start, stop, step);
        FAIL() << "NaN start not detected";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), "'start' cannot be nan or infinite.");
    } catch (const ov::Exception& error) {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            "Check '!std::numeric_limits<IN_T>::is_signed || std::numeric_limits<OUT_T>::lowest() <= c");
    } catch (...) {
        FAIL() << "Test failed for unexpected reason";
    }
}

TEST(type_prop, range_some_const_plus_inf_stop_fails) {
    auto start = make_shared<ov::op::v0::Parameter>(element::f32, Shape{});
    auto stop = make_shared<ov::op::v0::Constant>(element::f32,
                                                  Shape{},
                                                  std::vector<float>{std::numeric_limits<float>::infinity()});
    auto step = make_shared<ov::op::v0::Constant>(element::f32, Shape{}, std::vector<float>{1});

    try {
        auto range = make_shared<op::v0::Range>(start, stop, step);
        FAIL() << "+Infinity stop not detected";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), "'stop' cannot be nan or infinite.");
    } catch (const ov::Exception& error) {
        EXPECT_HAS_SUBSTRING(error.what(), "Check 'std::numeric_limits<OUT_T>::max() >= c");
    } catch (...) {
        FAIL() << "Test failed for unexpected reason";
    }
}

TEST(type_prop, range_some_const_minus_inf_stop_fails) {
    auto start = make_shared<ov::op::v0::Parameter>(element::f32, Shape{});
    auto stop = make_shared<ov::op::v0::Constant>(element::f32,
                                                  Shape{},
                                                  std::vector<float>{-std::numeric_limits<float>::infinity()});
    auto step = make_shared<ov::op::v0::Constant>(element::f32, Shape{}, std::vector<float>{1});

    try {
        auto range = make_shared<op::v0::Range>(start, stop, step);
        FAIL() << "-Infinity stop not detected";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), "'stop' cannot be nan or infinite.");
    } catch (const ov::Exception& error) {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            "Check '!std::numeric_limits<IN_T>::is_signed || std::numeric_limits<OUT_T>::lowest() <= c");
    } catch (...) {
        FAIL() << "Test failed for unexpected reason";
    }
}

TEST(type_prop, range_some_const_nan_stio_fails) {
    auto start = make_shared<ov::op::v0::Parameter>(element::f32, Shape{});
    auto stop = make_shared<ov::op::v0::Constant>(element::f32, Shape{}, std::vector<float>{std::nanf("")});
    auto step = make_shared<ov::op::v0::Constant>(element::f32, Shape{}, std::vector<float>{1});

    try {
        auto range = make_shared<op::v0::Range>(start, stop, step);
        FAIL() << "NaN stop not detected";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), "'stop' cannot be nan or infinite.");
    } catch (const ov::Exception& error) {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            "Check '!std::numeric_limits<IN_T>::is_signed || std::numeric_limits<OUT_T>::lowest() <= c");
    } catch (...) {
        FAIL() << "Test failed for unexpected reason";
    }
}

TEST(type_prop, range_some_const_plus_inf_stride_fails) {
    auto start = make_shared<ov::op::v0::Constant>(element::f32, Shape{}, std::vector<float>{3});
    auto stop = make_shared<ov::op::v0::Parameter>(element::f32, Shape{});
    auto step = make_shared<ov::op::v0::Constant>(element::f32,
                                                  Shape{},
                                                  std::vector<float>{std::numeric_limits<float>::infinity()});

    try {
        auto range = make_shared<op::v0::Range>(start, stop, step);
        FAIL() << "+Infinity stride not detected";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), "'step' cannot be zero, nan, or infinite.");
    } catch (const ov::Exception& error) {
        EXPECT_HAS_SUBSTRING(error.what(), "Check 'std::numeric_limits<OUT_T>::max() >= c");
    } catch (...) {
        FAIL() << "Test failed for unexpected reason";
    }
}

TEST(type_prop, range_some_const_minus_inf_stride_fails) {
    auto start = make_shared<ov::op::v0::Constant>(element::f32, Shape{}, std::vector<float>{3});
    auto stop = make_shared<ov::op::v0::Parameter>(element::f32, Shape{});
    auto step = make_shared<ov::op::v0::Constant>(element::f32,
                                                  Shape{},
                                                  std::vector<float>{-std::numeric_limits<float>::infinity()});

    try {
        auto range = make_shared<op::v0::Range>(start, stop, step);
        FAIL() << "-Infinity stride not detected";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), "'step' cannot be zero, nan, or infinite.");
    } catch (const ov::Exception& error) {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            "Check '!std::numeric_limits<IN_T>::is_signed || std::numeric_limits<OUT_T>::lowest() <= c");
    } catch (...) {
        FAIL() << "Test failed for unexpected reason";
    }
}

TEST(type_prop, range_some_const_nan_stride_fails) {
    auto start = make_shared<ov::op::v0::Constant>(element::f32, Shape{}, std::vector<float>{3});
    auto stop = make_shared<ov::op::v0::Parameter>(element::f32, Shape{});
    auto step = make_shared<ov::op::v0::Constant>(element::f32, Shape{}, std::vector<float>{std::nanf("")});

    try {
        auto range = make_shared<op::v0::Range>(start, stop, step);
        FAIL() << "NaN stride not detected";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), "'step' cannot be zero, nan, or infinite.");
    } catch (const ov::Exception& error) {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            "Check '!std::numeric_limits<IN_T>::is_signed || std::numeric_limits<OUT_T>::lowest() <= c");
    } catch (...) {
        FAIL() << "Test failed for unexpected reason";
    }
}

TEST(type_prop, range_all_const_zero_stride_fails) {
    auto start = make_shared<ov::op::v0::Constant>(element::i32, Shape{}, std::vector<int32_t>{3});
    auto stop = make_shared<ov::op::v0::Constant>(element::i32, Shape{}, std::vector<int32_t>{5});
    auto step = make_shared<ov::op::v0::Constant>(element::i32, Shape{}, std::vector<int32_t>{0});

    try {
        auto range = make_shared<op::v0::Range>(start, stop, step);
        FAIL() << "Zero stride not detected";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), "'step' cannot be zero");
    } catch (...) {
        FAIL() << "Test failed for unexpected reason";
    }
}

template <typename T>
void run_range_test(const element::Type& et, const RangeParams& params) {
    auto start = make_shared<ov::op::v0::Constant>(et, Shape{}, std::vector<T>{static_cast<T>(params.start)});
    auto stop = make_shared<ov::op::v0::Constant>(et, Shape{}, std::vector<T>{static_cast<T>(params.stop)});
    auto step = make_shared<ov::op::v0::Constant>(et, Shape{}, std::vector<T>{static_cast<T>(params.step)});

    auto range = make_shared<op::v0::Range>(start, stop, step);

    EXPECT_EQ(range->get_element_type(), et);
    EXPECT_TRUE(range->get_output_partial_shape(0).same_scheme(params.expected_shape))
        << "Expected shape " << params.expected_shape << " but got " << range->get_output_partial_shape(0);
}

struct RangeTest : ::testing::TestWithParam<RangeParams> {};

TEST_P(RangeTest, deduce_shape_i8) {
    run_range_test<int8_t>(element::i8, GetParam());
}

TEST_P(RangeTest, deduce_shape_i16) {
    run_range_test<int16_t>(element::i16, GetParam());
}

TEST_P(RangeTest, deduce_shape_i32) {
    run_range_test<int32_t>(element::i32, GetParam());
}

TEST_P(RangeTest, deduce_shape_i64) {
    run_range_test<int64_t>(element::i64, GetParam());
}

TEST_P(RangeTest, deduce_shape_u8) {
    run_range_test<uint8_t>(element::u8, GetParam());
}

TEST_P(RangeTest, deduce_shape_u16) {
    run_range_test<uint16_t>(element::u16, GetParam());
}

TEST_P(RangeTest, deduce_shape_u32) {
    run_range_test<uint32_t>(element::u32, GetParam());
}

TEST_P(RangeTest, deduce_shape_u64) {
    run_range_test<uint64_t>(element::u64, GetParam());
}

TEST_P(RangeTest, deduce_shape_bf16) {
    run_range_test<bfloat16>(element::bf16, GetParam());
}

TEST_P(RangeTest, deduce_shape_f16) {
    run_range_test<float16>(element::f16, GetParam());
}

TEST_P(RangeTest, deduce_shape_f32) {
    run_range_test<float>(element::f32, GetParam());
}

TEST_P(RangeTest, deduce_shape_f64) {
    run_range_test<double>(element::f64, GetParam());
}

INSTANTIATE_TEST_SUITE_P(type_prop,
                         RangeTest,
                         ::testing::Values(RangeParams{0, 5, 1, PartialShape{5}},
                                           RangeParams{0, 22, 2, PartialShape{11}},
                                           RangeParams{1, 23, 2, PartialShape{11}},
                                           RangeParams{1, 22, 2, PartialShape{11}},
                                           RangeParams{0, 0, 1, PartialShape{0}},
                                           RangeParams{1, 0, 2, PartialShape{0}}),
                         PrintToDummyParamName());

struct RangeTestWithNegatives : ::testing::TestWithParam<RangeParams> {};

TEST_P(RangeTestWithNegatives, deduce_shape_i8) {
    run_range_test<int8_t>(element::i8, GetParam());
}

TEST_P(RangeTestWithNegatives, deduce_shape_i16) {
    run_range_test<int16_t>(element::i16, GetParam());
}

TEST_P(RangeTestWithNegatives, deduce_shape_i32) {
    run_range_test<int32_t>(element::i32, GetParam());
}

TEST_P(RangeTestWithNegatives, deduce_shape_i64) {
    run_range_test<int64_t>(element::i64, GetParam());
}

TEST_P(RangeTestWithNegatives, deduce_shape_bf16) {
    run_range_test<bfloat16>(element::bf16, GetParam());
}

TEST_P(RangeTestWithNegatives, deduce_shape_f16) {
    run_range_test<float16>(element::f16, GetParam());
}

TEST_P(RangeTestWithNegatives, deduce_shape_f32) {
    run_range_test<float>(element::f32, GetParam());
}

TEST_P(RangeTestWithNegatives, deduce_shape_f64) {
    run_range_test<double>(element::f64, GetParam());
}

INSTANTIATE_TEST_SUITE_P(type_prop,
                         RangeTestWithNegatives,
                         ::testing::Values(RangeParams{2, 0, -2, PartialShape{1}},
                                           RangeParams{2, 0, -1, PartialShape{2}},
                                           RangeParams{-19, 19, 1, PartialShape{38}},
                                           RangeParams{-19, 19, 3, PartialShape{13}},
                                           RangeParams{20, -19, 1, PartialShape{0}}),
                         PrintToDummyParamName());

struct RangeTestFloating : ::testing::TestWithParam<RangeParams> {};

TEST_P(RangeTestFloating, deduce_shape_bf16) {
    run_range_test<bfloat16>(element::bf16, GetParam());
}

TEST_P(RangeTestFloating, deduce_shape_f16) {
    run_range_test<float16>(element::f16, GetParam());
}

TEST_P(RangeTestFloating, deduce_shape_f32) {
    run_range_test<float>(element::f32, GetParam());
}

TEST_P(RangeTestFloating, deduce_shape_f64) {
    run_range_test<double>(element::f64, GetParam());
}

INSTANTIATE_TEST_SUITE_P(type_prop,
                         RangeTestFloating,
                         ::testing::Values(RangeParams{0, 1, 0.25, PartialShape{4}},
                                           RangeParams{-1, 1, 0.25, PartialShape{8}},
                                           RangeParams{-1, 0.875, 0.25, PartialShape{8}}),
                         PrintToDummyParamName());

// ------------------------------ V4 ------------------------------

TEST(type_prop, range_v4_all_const_shape_inference) {
    int num_elems = 100;
    int step_val = 5;
    int start_val = 0;
    int stop_val = num_elems * step_val + start_val;
    element::Type_t et = element::i32;
    auto start = make_shared<ov::op::v0::Constant>(et, Shape{}, std::vector<int>{start_val});
    auto stop = make_shared<ov::op::v0::Constant>(et, Shape{}, std::vector<int>{stop_val});
    auto step = make_shared<ov::op::v0::Constant>(et, Shape{}, std::vector<int>{step_val});
    auto range = make_shared<op::v4::Range>(start, stop, step, et);
    auto pshape_out = range->get_output_partial_shape(0);
    ASSERT_TRUE(pshape_out.rank().is_static() && pshape_out.rank() == Dimension{1});
    ASSERT_TRUE(pshape_out.same_scheme(PartialShape{Dimension{num_elems}}));
}

TEST(type_prop, range_v4_some_const_shape_inference) {
    int step_val = 5;
    int start_val = 0;
    element::Type_t et = element::i32;
    auto start = make_shared<ov::op::v0::Constant>(et, Shape{}, std::vector<int>{start_val});
    auto stop = make_shared<ov::op::v0::Parameter>(et, Shape{});
    auto step = make_shared<ov::op::v0::Constant>(et, Shape{}, std::vector<int>{step_val});
    auto range = make_shared<op::v4::Range>(start, stop, step, et);
    auto pshape_out = range->get_output_partial_shape(0);
    ASSERT_TRUE(pshape_out.rank().is_static() && pshape_out.rank() == Dimension{1});
    ASSERT_TRUE(pshape_out.same_scheme(PartialShape{Dimension::dynamic()}));
}

TEST(type_prop, range_v4_trunc_inputs_shape_inference) {
    element::Type_t et = element::f32;
    auto start = make_shared<ov::op::v0::Constant>(et, Shape{}, std::vector<float>{0.9f});
    auto stop = make_shared<ov::op::v0::Constant>(et, Shape{}, std::vector<float>{10.3f});
    auto step = make_shared<ov::op::v0::Constant>(et, Shape{}, std::vector<float>{1.7f});
    auto range = make_shared<op::v4::Range>(start, stop, step, element::i32);
    auto pshape_out = range->get_output_partial_shape(0);
    ASSERT_TRUE(pshape_out.rank().is_static() && pshape_out.rank() == Dimension{1});
    ASSERT_TRUE(pshape_out.same_scheme(PartialShape{Dimension{10}}));
}

TEST(type_prop, range_v4_invalid_inputs_elem_type) {
    // invalid element type for start scalar
    try {
        auto start = make_shared<ov::op::v0::Parameter>(element::boolean, Shape{});
        auto stop = make_shared<ov::op::v0::Parameter>(element::i32, Shape{});
        auto step = make_shared<ov::op::v0::Parameter>(element::i32, Shape{});
        auto range = make_shared<op::v4::Range>(start, stop, step, element::i32);
        FAIL() << "Exception expected";
    } catch (ov::NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("'start' input scalar should be a numeric type"));
    } catch (...) {
        FAIL() << "Unknown exception was thrown";
    }

    // invalid element type for stop scalar
    try {
        auto start = make_shared<ov::op::v0::Parameter>(element::dynamic, Shape{});
        auto stop = make_shared<ov::op::v0::Parameter>(element::boolean, Shape{});
        auto step = make_shared<ov::op::v0::Parameter>(element::i32, Shape{});
        auto range = make_shared<op::v4::Range>(start, stop, step, element::i32);
        FAIL() << "Exception expected";
    } catch (ov::NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("'stop' input scalar should be a numeric type"));
    } catch (...) {
        FAIL() << "Unknown exception was thrown";
    }

    // invalid element type for step scalar
    try {
        auto start = make_shared<ov::op::v0::Parameter>(element::i32, Shape{});
        auto stop = make_shared<ov::op::v0::Parameter>(element::dynamic, Shape{});
        auto step = make_shared<ov::op::v0::Parameter>(element::boolean, Shape{});
        auto range = make_shared<op::v4::Range>(start, stop, step, element::i32);
        FAIL() << "Exception expected";
    } catch (const ov::NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("'step' input scalar should be a numeric type"));
    } catch (...) {
        FAIL() << "Unknown exception was thrown";
    }
}

TEST(type_prop, range_v4_invalid_output_elem_type) {
    try {
        auto start = make_shared<ov::op::v0::Parameter>(element::f16, Shape{1});
        auto stop = make_shared<ov::op::v0::Parameter>(element::f16, Shape{});
        auto step = make_shared<ov::op::v0::Parameter>(element::f16, Shape{});
        auto range = make_shared<op::v4::Range>(start, stop, step, element::boolean);
    } catch (const ov::NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("output tensor type should be a numeric type"));
    } catch (...) {
        FAIL() << "Unknown exception was thrown";
    }
}

TEST(type_prop, range_v4_invalid_inputs_non_scalar) {
    // start input not a scalar
    try {
        auto start = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1});
        auto stop = make_shared<ov::op::v0::Parameter>(element::f32, Shape{});
        auto step = make_shared<ov::op::v0::Parameter>(element::f32, Shape{});
        auto range = make_shared<op::v4::Range>(start, stop, step, element::f32);
        FAIL() << "Exception expected";
    } catch (const ov::NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("'start' input is not a scalar"));
    } catch (...) {
        FAIL() << "Unknown exception was thrown";
    }

    // stop input not a scalar
    try {
        auto start = make_shared<ov::op::v0::Parameter>(element::f32, Shape{});
        auto stop = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{Dimension::dynamic()});
        auto step = make_shared<ov::op::v0::Parameter>(element::f32, Shape{});
        auto range = make_shared<op::v4::Range>(start, stop, step, element::f32);
        FAIL() << "Exception expected";
    } catch (const ov::NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("'stop' input is not a scalar"));
    } catch (...) {
        FAIL() << "Unknown exception was thrown";
    }

    // step input not a scalar
    try {
        auto start = make_shared<ov::op::v0::Parameter>(element::f32, Shape{});
        auto stop = make_shared<ov::op::v0::Parameter>(element::f32, Shape{});
        auto step = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic(2));
        auto range = make_shared<op::v4::Range>(start, stop, step, element::f32);
        FAIL() << "Exception expected";
    } catch (const ov::NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("'step' input is not a scalar"));
    } catch (...) {
        FAIL() << "Unknown exception was thrown";
    }
}

TEST(type_prop, range_v4_invalid_inputs_plus_inf) {
    // invalid start input scalar, +inf
    try {
        auto start = make_shared<ov::op::v0::Constant>(element::f32,
                                                       Shape{},
                                                       std::vector<float>{std::numeric_limits<float>::infinity()});
        auto stop = make_shared<ov::op::v0::Parameter>(element::f32, Shape{});
        auto step = make_shared<ov::op::v0::Constant>(element::f32, Shape{}, std::vector<float>{1});
        auto range = make_shared<op::v4::Range>(start, stop, step, element::f32);
        FAIL() << "+Infinity start not detected";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), "'start' cannot be nan or infinite.");
    } catch (const ov::Exception& error) {
        EXPECT_HAS_SUBSTRING(error.what(), "Check 'std::numeric_limits<OUT_T>::max() >= c");
    } catch (...) {
        FAIL() << "Test failed for unexpected reason";
    }

    // invalid stop input scalar, +inf
    try {
        auto start = make_shared<ov::op::v0::Parameter>(element::f32, Shape{});
        auto stop = make_shared<ov::op::v0::Constant>(element::f32,
                                                      Shape{},
                                                      std::vector<float>{std::numeric_limits<float>::infinity()});
        auto step = make_shared<ov::op::v0::Constant>(element::f32, Shape{}, std::vector<float>{1});
        auto range = make_shared<op::v4::Range>(start, stop, step, element::f32);
        FAIL() << "+Infinity stop not detected";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), "'stop' cannot be nan or infinite.");
    } catch (const ov::Exception& error) {
        EXPECT_HAS_SUBSTRING(error.what(), "Check 'std::numeric_limits<OUT_T>::max() >= c");
    } catch (...) {
        FAIL() << "Test failed for unexpected reason";
    }

    // invalid step input scalar, +inf
    try {
        auto start = make_shared<ov::op::v0::Constant>(element::f32, Shape{}, std::vector<float>{3});
        auto stop = make_shared<ov::op::v0::Parameter>(element::f32, Shape{});
        auto step = make_shared<ov::op::v0::Constant>(element::f32,
                                                      Shape{},
                                                      std::vector<float>{std::numeric_limits<float>::infinity()});
        auto range = make_shared<op::v4::Range>(start, stop, step, element::f32);
        FAIL() << "+Infinity step not detected";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), "'step' cannot be nan or infinite.");
    } catch (const ov::Exception& error) {
        EXPECT_HAS_SUBSTRING(error.what(), "Check 'std::numeric_limits<OUT_T>::max() >= c");
    } catch (...) {
        FAIL() << "Test failed for unexpected reason";
    }
}

TEST(type_prop, range_v4_invalid_inputs_minus_inf) {
    // invalid start input scalar, -inf
    try {
        auto start = make_shared<ov::op::v0::Constant>(element::f32,
                                                       Shape{},
                                                       std::vector<float>{-std::numeric_limits<float>::infinity()});
        auto stop = make_shared<ov::op::v0::Parameter>(element::f32, Shape{});
        auto step = make_shared<ov::op::v0::Constant>(element::f32, Shape{}, std::vector<float>{1});
        auto range = make_shared<op::v4::Range>(start, stop, step, element::f32);
        FAIL() << "-Infinity start not detected";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), "'start' cannot be nan or infinite.");
    } catch (const ov::Exception& error) {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            "Check '!std::numeric_limits<IN_T>::is_signed || std::numeric_limits<OUT_T>::lowest() <= c");
    } catch (...) {
        FAIL() << "Test failed for unexpected reason";
    }

    // invalid stop input scalar, -inf
    try {
        auto start = make_shared<ov::op::v0::Parameter>(element::f32, Shape{});
        auto stop = make_shared<ov::op::v0::Constant>(element::f32,
                                                      Shape{},
                                                      std::vector<float>{-std::numeric_limits<float>::infinity()});
        auto step = make_shared<ov::op::v0::Constant>(element::f32, Shape{}, std::vector<float>{1});
        auto range = make_shared<op::v4::Range>(start, stop, step, element::f32);
        FAIL() << "-Infinity stop not detected";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), "'stop' cannot be nan or infinite.");
    } catch (const ov::Exception& error) {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            "Check '!std::numeric_limits<IN_T>::is_signed || std::numeric_limits<OUT_T>::lowest() <= c");
    } catch (...) {
        FAIL() << "Test failed for unexpected reason";
    }

    // invalid step input scalar, -inf
    try {
        auto start = make_shared<ov::op::v0::Constant>(element::f32, Shape{}, std::vector<float>{3});
        auto stop = make_shared<ov::op::v0::Parameter>(element::f32, Shape{});
        auto step = make_shared<ov::op::v0::Constant>(element::f32,
                                                      Shape{},
                                                      std::vector<float>{-std::numeric_limits<float>::infinity()});
        auto range = make_shared<op::v4::Range>(start, stop, step, element::f32);
        FAIL() << "-Infinity step not detected";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), "'step' cannot be nan or infinite.");
    } catch (const ov::Exception& error) {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            "Check '!std::numeric_limits<IN_T>::is_signed || std::numeric_limits<OUT_T>::lowest() <= c");
    } catch (...) {
        FAIL() << "Test failed for unexpected reason";
    }
}

TEST(type_prop, range_v4_invalid_inputs_nan) {
    // invalid start input scalar, nan
    try {
        auto start = make_shared<ov::op::v0::Constant>(element::f32, Shape{}, std::vector<float>{std::nanf("")});
        auto stop = make_shared<ov::op::v0::Parameter>(element::f32, Shape{});
        auto step = make_shared<ov::op::v0::Constant>(element::f32, Shape{}, std::vector<float>{1});
        auto range = make_shared<op::v4::Range>(start, stop, step, element::f32);
        FAIL() << "NaN start not detected";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), "'start' cannot be nan or infinite.");
    } catch (const ov::Exception& error) {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            "Check '!std::numeric_limits<IN_T>::is_signed || std::numeric_limits<OUT_T>::lowest() <= c");
    } catch (...) {
        FAIL() << "Test failed for unexpected reason";
    }

    // invalid stop input scalar, nan
    try {
        auto start = make_shared<ov::op::v0::Parameter>(element::f32, Shape{});
        auto stop = make_shared<ov::op::v0::Constant>(element::f32, Shape{}, std::vector<float>{std::nanf("")});
        auto step = make_shared<ov::op::v0::Constant>(element::f32, Shape{}, std::vector<float>{1});
        auto range = make_shared<op::v4::Range>(start, stop, step, element::f32);
        FAIL() << "NaN stop not detected";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), "'stop' cannot be nan or infinite.");
    } catch (const ov::Exception& error) {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            "Check '!std::numeric_limits<IN_T>::is_signed || std::numeric_limits<OUT_T>::lowest() <= c");
    } catch (...) {
        FAIL() << "Test failed for unexpected reason";
    }

    // invalid step input scalar, nan
    try {
        auto start = make_shared<ov::op::v0::Constant>(element::f32, Shape{}, std::vector<float>{1});
        auto stop = make_shared<ov::op::v0::Parameter>(element::f32, Shape{});
        auto step = make_shared<ov::op::v0::Constant>(element::f32, Shape{}, std::vector<float>{std::nanf("")});
        auto range = make_shared<op::v4::Range>(start, stop, step, element::f32);
        FAIL() << "NaN step not detected";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), "'step' cannot be nan or infinite.");
    } catch (const ov::Exception& error) {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            "Check '!std::numeric_limits<IN_T>::is_signed || std::numeric_limits<OUT_T>::lowest() <= c");
    } catch (...) {
        FAIL() << "Test failed for unexpected reason";
    }
}

TEST(type_prop, range_v4_zero_output_elem_pos_step) {
    auto start = make_shared<ov::op::v0::Constant>(element::f32, Shape{}, std::vector<float>{5});
    auto stop = make_shared<ov::op::v0::Constant>(element::f32, Shape{}, std::vector<float>{1});
    auto step = make_shared<ov::op::v0::Constant>(element::f32, Shape{}, std::vector<float>{1});
    auto range = make_shared<op::v4::Range>(start, stop, step, element::f32);
    // if step is positive and start >= stop, number of output elements is zero
    ASSERT_TRUE(range->get_output_partial_shape(0).same_scheme(PartialShape{Dimension(0)}));
}

TEST(type_prop, range_v4_zero_output_elem_neg_step) {
    auto start = make_shared<ov::op::v0::Constant>(element::f32, Shape{}, std::vector<float>{1});
    auto stop = make_shared<ov::op::v0::Constant>(element::f32, Shape{}, std::vector<float>{5});
    auto step = make_shared<ov::op::v0::Constant>(element::f32, Shape{}, std::vector<float>{-1});
    auto range = make_shared<op::v4::Range>(start, stop, step, element::f32);
    // if step is negative and start <= stop, number of output elements is zero
    ASSERT_TRUE(range->get_output_partial_shape(0).same_scheme(PartialShape{Dimension(0)}));
}

template <typename T>
void run_range_v4_test(const element::Type& et, const RangeParams& params) {
    auto start = make_shared<ov::op::v0::Constant>(et, Shape{}, std::vector<T>{static_cast<T>(params.start)});
    auto stop = make_shared<ov::op::v0::Constant>(et, Shape{}, std::vector<T>{static_cast<T>(params.stop)});
    auto step = make_shared<ov::op::v0::Constant>(et, Shape{}, std::vector<T>{static_cast<T>(params.step)});

    auto range = make_shared<op::v4::Range>(start, stop, step, et);

    EXPECT_TRUE(range->get_output_partial_shape(0).same_scheme(params.expected_shape))
        << "Expected shape " << params.expected_shape << " but got " << range->get_output_partial_shape(0);
}

struct RangeNumpyTest : ::testing::TestWithParam<RangeParams> {};

TEST_P(RangeNumpyTest, deduce_shape_i8) {
    run_range_v4_test<int8_t>(element::i8, GetParam());
}

TEST_P(RangeNumpyTest, deduce_shape_i16) {
    run_range_v4_test<int16_t>(element::i16, GetParam());
}

TEST_P(RangeNumpyTest, deduce_shape_i32) {
    run_range_v4_test<int32_t>(element::i32, GetParam());
}

TEST_P(RangeNumpyTest, deduce_shape_i64) {
    run_range_v4_test<int64_t>(element::i64, GetParam());
}

TEST_P(RangeNumpyTest, deduce_shape_u8) {
    run_range_v4_test<uint8_t>(element::u8, GetParam());
}

TEST_P(RangeNumpyTest, deduce_shape_u16) {
    run_range_v4_test<uint16_t>(element::u16, GetParam());
}

TEST_P(RangeNumpyTest, deduce_shape_u32) {
    run_range_v4_test<uint32_t>(element::u32, GetParam());
}

TEST_P(RangeNumpyTest, deduce_shape_u64) {
    run_range_v4_test<uint64_t>(element::u64, GetParam());
}

TEST_P(RangeNumpyTest, deduce_shape_bf16) {
    run_range_v4_test<bfloat16>(element::bf16, GetParam());
}

TEST_P(RangeNumpyTest, deduce_shape_f16) {
    run_range_v4_test<float16>(element::f16, GetParam());
}

TEST_P(RangeNumpyTest, deduce_shape_f32) {
    run_range_v4_test<float>(element::f32, GetParam());
}

TEST_P(RangeNumpyTest, deduce_shape_f64) {
    run_range_v4_test<double>(element::f64, GetParam());
}

INSTANTIATE_TEST_SUITE_P(type_prop,
                         RangeNumpyTest,
                         ::testing::Values(RangeParams{0, 5, 1, PartialShape{5}},
                                           RangeParams{0, 22, 2, PartialShape{11}},
                                           RangeParams{1, 23, 2, PartialShape{11}},
                                           RangeParams{1, 22, 2, PartialShape{11}},
                                           RangeParams{0, 0, 1, PartialShape{0}},
                                           RangeParams{1, 0, 2, PartialShape{0}}),
                         PrintToDummyParamName());

struct RangeNumpyTestWithNegatives : ::testing::TestWithParam<RangeParams> {};

TEST_P(RangeNumpyTestWithNegatives, deduce_shape_i8) {
    run_range_v4_test<int8_t>(element::i8, GetParam());
}

TEST_P(RangeNumpyTestWithNegatives, deduce_shape_i16) {
    run_range_v4_test<int16_t>(element::i16, GetParam());
}

TEST_P(RangeNumpyTestWithNegatives, deduce_shape_i32) {
    run_range_v4_test<int32_t>(element::i32, GetParam());
}

TEST_P(RangeNumpyTestWithNegatives, deduce_shape_i64) {
    run_range_v4_test<int64_t>(element::i64, GetParam());
}

TEST_P(RangeNumpyTestWithNegatives, deduce_shape_bf16) {
    run_range_v4_test<bfloat16>(element::bf16, GetParam());
}

TEST_P(RangeNumpyTestWithNegatives, deduce_shape_f16) {
    run_range_v4_test<float16>(element::f16, GetParam());
}

TEST_P(RangeNumpyTestWithNegatives, deduce_shape_f32) {
    run_range_v4_test<float>(element::f32, GetParam());
}

TEST_P(RangeNumpyTestWithNegatives, deduce_shape_f64) {
    run_range_v4_test<double>(element::f64, GetParam());
}

INSTANTIATE_TEST_SUITE_P(type_prop,
                         RangeNumpyTestWithNegatives,
                         ::testing::Values(RangeParams{2, 0, -2, PartialShape{1}},
                                           RangeParams{2, 0, -1, PartialShape{2}},
                                           RangeParams{-19, 19, 1, PartialShape{38}},
                                           RangeParams{-19, 19, 3, PartialShape{13}},
                                           RangeParams{20, -19, 1, PartialShape{0}}),
                         PrintToDummyParamName());

struct RangeNumpyTestFloating : ::testing::TestWithParam<RangeParams> {};

TEST_P(RangeNumpyTestFloating, deduce_shape_bf16) {
    run_range_v4_test<bfloat16>(element::bf16, GetParam());
}

TEST_P(RangeNumpyTestFloating, deduce_shape_f16) {
    run_range_v4_test<float16>(element::f16, GetParam());
}

TEST_P(RangeNumpyTestFloating, deduce_shape_f32) {
    run_range_v4_test<float>(element::f32, GetParam());
}

TEST_P(RangeNumpyTestFloating, deduce_shape_f64) {
    run_range_v4_test<double>(element::f64, GetParam());
}

INSTANTIATE_TEST_SUITE_P(type_prop,
                         RangeNumpyTestFloating,
                         ::testing::Values(RangeParams{0, 1, 0.25, PartialShape{4}},
                                           RangeParams{-1, 1, 0.25, PartialShape{8}},
                                           RangeParams{-1, 0.875, 0.25, PartialShape{8}}),
                         PrintToDummyParamName());

TEST(type_prop, range_symbol_start_0_stop_A_step_1) {
    auto stop_symbol = std::make_shared<ov::Symbol>();
    auto source_shape = PartialShape::dynamic(1);
    source_shape[0].set_symbol(stop_symbol);
    auto symbol_source =
        make_shared<ov::op::v0::ShapeOf>(make_shared<ov::op::v0::Parameter>(element::i64, source_shape));

    auto start = make_shared<ov::op::v0::Constant>(element::i64, Shape{}, 0);
    auto stop = make_shared<ov::op::v8::Gather>(symbol_source,
                                                make_shared<ov::op::v0::Constant>(element::i64, Shape{}, 0),
                                                make_shared<ov::op::v0::Constant>(element::i64, Shape{}, 0));
    auto step = make_shared<ov::op::v0::Constant>(element::i64, Shape{}, 1);

    auto range = make_shared<op::v0::Range>(start, stop, step);

    ASSERT_TRUE(ov::symbol::are_equal(range->get_output_partial_shape(0)[0].get_symbol(), stop_symbol));
}
