// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "util/type_prop.hpp"

NGRAPH_SUPPRESS_DEPRECATED_START

using namespace std;
using namespace ngraph;

TEST(type_prop, pad_v1_arg_pad_value_type_mismatch) {
    auto arg = make_shared<op::Parameter>(element::f32, Shape{1, 2, 3});
    auto pads_begin = make_shared<op::Parameter>(element::i64, Shape{1});
    auto pads_end = make_shared<op::Parameter>(element::i64, Shape{1});
    auto arg_pad_value = make_shared<op::Parameter>(element::f16, Shape{1});

    try {
        auto pad_v1 = make_shared<op::v1::Pad>(arg, pads_begin, pads_end, arg_pad_value, op::PadMode::CONSTANT);

        // Should have thrown, so fail if it didn't
        FAIL() << "Incorrect arg_pad_value type exception not handled";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Argument element types do not match (input arg element type:"));
    } catch (...) {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, pad_v1_arg_pad_value_shape_not_compatible) {
    auto arg = make_shared<op::Parameter>(element::f32, Shape{1, 2, 3});
    auto pads_begin = make_shared<op::Parameter>(element::i64, Shape{1});
    auto pads_end = make_shared<op::Parameter>(element::i64, Shape{1});
    auto arg_pad_value = make_shared<op::Parameter>(element::f32, Shape{1});

    try {
        auto pad_v1 = make_shared<op::v1::Pad>(arg, pads_begin, pads_end, arg_pad_value, op::PadMode::CONSTANT);

        // Should have thrown, so fail if it didn't
        FAIL() << "Incorrect arg_pad_value shape exception not handled";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Argument for padding value is not a scalar (shape:"));
    } catch (...) {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, pad_v1_pads_begin_shape_not_1D) {
    auto arg = make_shared<op::Parameter>(element::f32, Shape{1, 2, 3});
    auto pads_begin = make_shared<op::Parameter>(element::i64, Shape{1, 2});
    auto pads_end = make_shared<op::Parameter>(element::i64, Shape{1});

    try {
        auto pad_v1 = make_shared<op::v1::Pad>(arg, pads_begin, pads_end, op::PadMode::SYMMETRIC);

        // Should have thrown, so fail if it didn't
        FAIL() << "Incorrect pads_begin shape exception not handled";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Argument for pads_begin is not 1D (shape:"));
    } catch (...) {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, pad_v1_pads_end_shape_not_1D) {
    auto arg = make_shared<op::Parameter>(element::f32, Shape{1, 2, 3});
    auto pads_begin = make_shared<op::Parameter>(element::i64, Shape{1});
    auto pads_end = make_shared<op::Parameter>(element::i64, Shape{1, 2});

    try {
        auto pad_v1 = make_shared<op::v1::Pad>(arg, pads_begin, pads_end, op::PadMode::SYMMETRIC);

        // Should have thrown, so fail if it didn't
        FAIL() << "Incorrect pads_end shape exception not handled";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Argument for pads_end is not 1D (shape:"));
    } catch (...) {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, pad_v1_pads_begin_size_not_correct) {
    auto arg = make_shared<op::Parameter>(element::f32, Shape{1, 2, 3});
    auto pads_begin = make_shared<op::Parameter>(element::i64, Shape{4});
    auto pads_end = make_shared<op::Parameter>(element::i64, Shape{1});

    try {
        auto pad_v1 = make_shared<op::v1::Pad>(arg, pads_begin, pads_end, op::PadMode::SYMMETRIC);

        // Should have thrown, so fail if it didn't
        FAIL() << "Incorrect pads_begin size exception not handled";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Number of elements of pads_begin must be >= 0 and <= arg "
                                         "rank (pads_begin_shape[0]:"));
    } catch (...) {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, pad_v1_pads_end_size_not_correct) {
    auto arg = make_shared<op::Parameter>(element::f32, Shape{1, 2, 3});
    auto pads_begin = make_shared<op::Parameter>(element::i64, Shape{1});
    auto pads_end = make_shared<op::Parameter>(element::i64, Shape{4});
    auto arg_pad_value = make_shared<op::Parameter>(element::f32, Shape{});

    try {
        auto pad_v1 = make_shared<op::v1::Pad>(arg, pads_begin, pads_end, arg_pad_value, op::PadMode::CONSTANT);

        // Should have thrown, so fail if it didn't
        FAIL() << "Incorrect pads_end size exception not handled";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Number of elements of pads_end must be >= 0 and <= arg rank (pads_end_shape[0]:"));
    } catch (...) {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, pad_v1_arg_pads_begin_incompatible_type) {
    auto arg = make_shared<op::Parameter>(element::f32, Shape{1, 2, 3});
    auto pads_begin = make_shared<op::Parameter>(element::f32, Shape{1});
    auto pads_end = make_shared<op::Parameter>(element::i64, Shape{1});

    try {
        auto pad_v1 = make_shared<op::v1::Pad>(arg, pads_begin, pads_end, op::PadMode::REFLECT);

        // Should have thrown, so fail if it didn't
        FAIL() << "Incorrect pad_begin type exception not handled";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("pads_begin must be an integral number, but is:"));
    } catch (...) {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, pad_v1_arg_pads_end_incompatible_type) {
    auto arg = make_shared<op::Parameter>(element::f32, Shape{1, 2, 3});
    auto pads_begin = make_shared<op::Parameter>(element::i64, Shape{1});
    auto pads_end = make_shared<op::Parameter>(element::f32, Shape{1});

    try {
        auto pad = make_shared<op::v1::Pad>(arg, pads_begin, pads_end, op::PadMode::REFLECT);

        // Should have thrown, so fail if it didn't
        FAIL() << "Incorrect pads_end type exception not thrown";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("pads_end must be an integral number, but is:"));
    } catch (...) {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, pad_v1_deduce_too_small_for_edge) {
    auto arg = make_shared<op::Parameter>(element::f32, Shape{1, 5, 0, 2});
    auto pads_begin = make_shared<op::Constant>(element::i64, Shape{4}, std::vector<int64_t>{0, 1, 2, 3});
    auto pads_end = make_shared<op::Constant>(element::i64, Shape{4}, std::vector<int64_t>{0, 1, 2, 3});
    auto arg_pad_value = make_shared<op::Parameter>(element::f32, Shape{});

    try {
        auto pad_v1 = make_shared<op::v1::Pad>(arg, pads_begin, pads_end, arg_pad_value, op::PadMode::EDGE);

        // Should have thrown, so fail if it didn't
        FAIL() << "Incorrect input shape exception for EDGE mode not thrown";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("EDGE padding mode requires an input of dimension of at "
                                         "least 1 at each spatial axis"));
    } catch (...) {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, pad_v1_deduce_too_small_for_reflect) {
    auto arg = make_shared<op::Parameter>(element::f32, Shape{1, 5, 1, 2});
    auto pads_begin = make_shared<op::Constant>(element::i64, Shape{4}, std::vector<int64_t>{0, 1, 2, 3});
    auto pads_end = make_shared<op::Constant>(element::i64, Shape{4}, std::vector<int64_t>{0, 1, 2, 3});
    auto arg_pad_value = make_shared<op::Parameter>(element::f32, Shape{});

    try {
        auto pad_v1 = make_shared<op::v1::Pad>(arg, pads_begin, pads_end, arg_pad_value, op::PadMode::REFLECT);

        // Should have thrown, so fail if it didn't
        FAIL() << "Incorrect input shape exception for REFLECT mode not thrown";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("REFLECT padding mode requires an input of dimension of "
                                         "at least 2 at each spatial axis"));
    } catch (...) {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, pad_v1_dynamic_output_with_dynamic_rank) {
    auto arg = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto pads_begin = make_shared<op::Parameter>(element::i32, Shape{1});
    auto pads_end = make_shared<op::Parameter>(element::i32, Shape{1});
    auto arg_pad_value = op::Constant::create(element::f32, Shape{}, {0});

    auto pad = make_shared<op::v1::Pad>(arg, pads_begin, pads_end, arg_pad_value, op::PadMode::CONSTANT);
    ASSERT_EQ(pad->get_output_partial_shape(0), PartialShape::dynamic());
}

TEST(type_prop, pad_v1_dynamic_output_with_static_rank) {
    auto arg = make_shared<op::Parameter>(element::f32, Shape{1, 2, 3});
    auto pads_begin = make_shared<op::Parameter>(element::i32, Shape{1});
    auto pads_end = make_shared<op::Parameter>(element::i32, Shape{1});
    auto arg_pad_value = op::Constant::create(element::f32, Shape{}, {0});

    auto pad = make_shared<op::v1::Pad>(arg, pads_begin, pads_end, arg_pad_value, op::PadMode::CONSTANT);
    ASSERT_EQ(pad->get_output_partial_shape(0), PartialShape::dynamic(3));
}
