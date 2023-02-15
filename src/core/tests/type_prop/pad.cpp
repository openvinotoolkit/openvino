// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/test_assertions.hpp"
#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "util/type_prop.hpp"

NGRAPH_SUPPRESS_DEPRECATED_START

using namespace std;
using namespace ngraph;
using namespace testing;

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

TEST(type_prop, pad_v1_pads_end_got_negative_value) {
    auto arg_shape = PartialShape{-1, {0, 10}, {2, -1}, {2, 8}, {3, 10}, 5};
    set_shape_labels(arg_shape, 10);
    const auto arg = std::make_shared<op::Parameter>(element::f32, arg_shape);
    const auto pads_begin = op::Constant::create(element::i64, Shape{6}, {2, 0, 1, 3, 2, 1});
    const auto pads_end = op::Constant::create(element::i64, Shape{6}, {-3, -2, -2, -3, -1, -3});

    const auto pad = make_shared<op::v1::Pad>(arg, pads_begin, pads_end, op::PadMode::REFLECT);

    EXPECT_EQ(pad->get_output_partial_shape(0), PartialShape({-1, {0, 8}, {1, -1}, {2, 8}, {4, 11}, 3}));
    EXPECT_THAT(get_shape_labels(pad->get_output_partial_shape(0)),
                ElementsAre(ov::no_label, ov::no_label, ov::no_label, 13, ov::no_label, ov::no_label));
}

TEST(type_prop, pad_v1_pads_begin_got_negative_value) {
    auto arg_shape = PartialShape{-1, {0, 10}, {2, -1}, {2, 8}, {3, 10}, 5};
    set_shape_labels(arg_shape, 10);
    const auto arg = std::make_shared<op::Parameter>(element::f32, arg_shape);
    const auto pads_begin = op::Constant::create(element::i64, Shape{6}, {-1, -1, -2, -3, -8, -4});
    const auto pads_end = op::Constant::create(element::i64, Shape{6}, {0, 2, 0, 3, 5, 4});

    const auto pad = make_shared<op::v1::Pad>(arg, pads_begin, pads_end, op::PadMode::REFLECT);
    EXPECT_EQ(pad->get_output_partial_shape(0), PartialShape({-1, {1, 11}, {0, -1}, {2, 8}, {0, 7}, 5}));
    EXPECT_THAT(get_shape_labels(pad->get_output_partial_shape(0)),
                ElementsAre(ov::no_label, ov::no_label, ov::no_label, 13, ov::no_label, 15));
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

TEST(type_prop, pad_v1_any_dim_for_padding_reflect) {
    auto arg_shape = PartialShape{1, {23, 48}, {23, 48}, 1};
    set_shape_labels(arg_shape, 10);
    auto arg = make_shared<op::Parameter>(element::f32, arg_shape);
    auto pads_begin = make_shared<op::Constant>(element::i64, Shape{4}, std::vector<int64_t>{0, 1, 1, 0});
    auto pads_end = make_shared<op::Constant>(element::i64, Shape{4}, std::vector<int64_t>{0, 1, 1, 0});

    auto pad = make_shared<op::v1::Pad>(arg, pads_begin, pads_end, op::PadMode::REFLECT);
    EXPECT_EQ(pad->get_output_partial_shape(0), PartialShape({1, {25, 50}, {25, 50}, 1}));
    EXPECT_THAT(get_shape_labels(pad->get_output_partial_shape(0)), ElementsAre(10, ov::no_label, ov::no_label, 13));
}

TEST(type_prop, pad_v1_any_dim_for_padding_edge) {
    auto arg_shape = PartialShape{1, {0, 48}, -1, {20, -1}, {5, -1}, 10, 12};
    set_shape_labels(arg_shape, 10);
    auto arg = make_shared<op::Parameter>(element::f32, arg_shape);
    auto pads_begin = make_shared<op::Constant>(element::i64, Shape{7}, std::vector<int64_t>{1, 2, 1, 2, 0, 0, 0});
    auto pads_end = make_shared<op::Constant>(element::i64, Shape{7}, std::vector<int64_t>{0, 3, 0, 1, 0, 5, 0});

    auto pad = make_shared<op::v1::Pad>(arg, pads_begin, pads_end, op::PadMode::EDGE);
    EXPECT_EQ(pad->get_output_partial_shape(0), PartialShape({2, {5, 53}, {1, -1}, {23, -1}, {5, -1}, 15, 12}));
    EXPECT_THAT(get_shape_labels(pad->get_output_partial_shape(0)),
                ElementsAre(ov::no_label, ov::no_label, ov::no_label, ov::no_label, 14, ov::no_label, 16));
}

TEST(type_prop, pad_v1_dynamic_input_type_with_static_value) {
    auto arg = make_shared<op::Parameter>(element::dynamic, Shape{1, 2, 3});
    auto pads_begin = make_shared<op::Parameter>(element::i32, Shape{1});
    auto pads_end = make_shared<op::Parameter>(element::i32, Shape{1});
    auto arg_pad_value = op::Constant::create(element::f32, Shape{}, {0});

    auto pad = make_shared<op::v1::Pad>(arg, pads_begin, pads_end, arg_pad_value, op::PadMode::CONSTANT);
    EXPECT_EQ(pad->get_output_element_type(0), element::f32);
    EXPECT_EQ(pad->get_output_partial_shape(0), PartialShape::dynamic(3));
}

TEST(type_prop, pad_v1_preserve_partial_values_and_labels_via_evaluates_bounds) {
    auto arg_shape = PartialShape{1, {2, 5}, {1, 3}};
    auto begin_shape = PartialShape{{2, 4}, 0, {0, 2}};
    auto end_shape = PartialShape{{1, 2}, 0, 1};
    set_shape_labels(arg_shape, 10);
    set_shape_labels(begin_shape, 20);
    set_shape_labels(end_shape, 30);

    auto arg = make_shared<op::Parameter>(element::f32, arg_shape);
    auto s_begin = make_shared<op::ShapeOf>(make_shared<op::Parameter>(element::i64, begin_shape));
    auto s_end = make_shared<op::ShapeOf>(make_shared<op::Parameter>(element::i64, end_shape));

    auto pad = make_shared<op::v1::Pad>(arg, s_begin, s_end, op::PadMode::EDGE);

    EXPECT_EQ(pad->get_output_partial_shape(0), PartialShape({{4, 7}, {2, 5}, {2, 6}}));
    EXPECT_THAT(get_shape_labels(pad->get_output_partial_shape(0)), ElementsAre(ov::no_label, 11, ov::no_label));
}

TEST(type_prop, pad_v1_preserve_partial_values_and_labels_on_inputs) {
    auto arg_shape = PartialShape{1, {2, 5}, {1, 3}};
    set_shape_labels(arg_shape, 10);
    auto arg = make_shared<op::Parameter>(element::i32, arg_shape);
    auto s = make_shared<op::ShapeOf>(arg);

    auto pads_begin = make_shared<op::Constant>(element::i64, Shape{1}, std::vector<int64_t>{1});
    auto pads_end = make_shared<op::Constant>(element::i64, Shape{1}, std::vector<int64_t>{2});

    auto pad = make_shared<op::v1::Pad>(s, pads_begin, pads_end, op::PadMode::EDGE);
    auto param = make_shared<op::Parameter>(element::f32, PartialShape{1});
    auto bc = std::make_shared<op::v3::Broadcast>(param, pad, op::BroadcastType::BIDIRECTIONAL);

    EXPECT_EQ(bc->get_output_partial_shape(0), PartialShape({1, 1, {2, 5}, {1, 3}, {1, 3}, {1, 3}}));
    EXPECT_THAT(get_shape_labels(bc->get_output_partial_shape(0)), ElementsAre(10, 10, 11, 12, 12, 12));
}

TEST(type_prop, pad_v1_default_ctor) {
    const auto arg_shape = PartialShape{{1, 2}, {4, 10}, {3, 8}, {1, 2}};
    const auto arg = make_shared<op::Parameter>(element::f32, arg_shape);
    const auto pads_begin = make_shared<op::Constant>(element::i64, Shape{4}, std::vector<int64_t>{0, 2, 1, 0});
    const auto pads_end = make_shared<op::Constant>(element::i64, Shape{4}, std::vector<int64_t>{0, 1, 1, 0});

    const auto pad = make_shared<op::v1::Pad>();
    pad->set_arguments(OutputVector{arg, pads_begin, pads_end});
    pad->set_pad_mode(op::PadMode::REFLECT);
    pad->validate_and_infer_types();

    EXPECT_EQ(pad->get_output_element_type(0), element::f32);
    EXPECT_EQ(pad->get_output_partial_shape(0), PartialShape({{1, 2}, {7, 13}, {5, 10}, {1, 2}}));
}
