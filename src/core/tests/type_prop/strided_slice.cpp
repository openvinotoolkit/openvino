// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <dimension_tracker.hpp>
#include <memory>
#include <strided_slice_shape_inference.hpp>

#include "common_test_utils/test_assertions.hpp"
#include "gmock/gmock.h"
#include "ngraph/ngraph.hpp"
#include "openvino/opsets/opset9.hpp"
#include "util/type_prop.hpp"

using namespace std;
using namespace ngraph;
using namespace testing;

TEST(type_prop, strided_slice_begin_incorrect_type) {
    auto data = make_shared<op::Parameter>(element::f32, Shape{2, 4, 6, 8});
    auto begin = make_shared<op::Parameter>(element::f16, Shape{4});
    auto end = make_shared<op::Parameter>(element::i64, Shape{4});
    try {
        auto strided_slice = make_shared<op::v1::StridedSlice>(data,
                                                               begin,
                                                               end,
                                                               vector<int64_t>{1, 0, 1, 0},
                                                               vector<int64_t>{1, 0, 1, 0});
        // Should have thrown, so fail if it didn't
        FAIL() << "Incorrect begin type exception not thrown.";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Begin mask must be an integral number"));
    } catch (...) {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, strided_slice_end_incorrect_type) {
    auto data = make_shared<op::Parameter>(element::f32, Shape{2, 4, 6, 8});
    auto begin = make_shared<op::Parameter>(element::i64, Shape{4});
    auto end = make_shared<op::Parameter>(element::boolean, Shape{4});
    try {
        auto strided_slice = make_shared<op::v1::StridedSlice>(data,
                                                               begin,
                                                               end,
                                                               vector<int64_t>{1, 0, 1, 0},
                                                               vector<int64_t>{1, 0, 1, 0});
        // Should have thrown, so fail if it didn't
        FAIL() << "Incorrect end type exception not thrown.";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("End mask must be an integral number"));
    } catch (...) {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, strided_slice_incompatible_size_of_masks_attr) {
    auto data = make_shared<op::Parameter>(element::f32, Shape{2, 4, 6, 8});
    auto begin = make_shared<op::Parameter>(element::i64, Shape{4});
    auto end = make_shared<op::Parameter>(element::i64, Shape{4});
    try {
        auto strided_slice = make_shared<op::v1::StridedSlice>(data,
                                                               begin,
                                                               end,
                                                               vector<int64_t>{1, 0, 1, 0},
                                                               vector<int64_t>{1, 0, 1, 0},
                                                               vector<int64_t>{1, 0, 1, 0, 1});
        // Should have thrown, so fail if it didn't
        FAIL() << "Incompatible size od masks exception not thrown.";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("All masks of StridedSlice must have the same size"));
    } catch (...) {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, strided_slice_mask_incorrect_value) {
    auto data = make_shared<op::Parameter>(element::f32, Shape{2, 4, 6, 8});
    auto begin = make_shared<op::Parameter>(element::i64, Shape{4, 5});
    auto end = make_shared<op::Parameter>(element::i64, Shape{4});
    try {
        auto strided_slice = make_shared<op::v1::StridedSlice>(data,
                                                               begin,
                                                               end,
                                                               vector<int64_t>{1, 0, 1, 0},
                                                               vector<int64_t>{1, 0, 1, 2});
        // Should have thrown, so fail if it didn't
        FAIL() << "Incorrect values of StridedSlice mask exception not thrown.";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("All masks of StridedSlice must have be 0 or 1"));
    } catch (...) {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, strided_slice_begin_incorrect_shape) {
    auto data = make_shared<op::Parameter>(element::f32, Shape{2, 4, 6, 8});
    auto begin = make_shared<op::Parameter>(element::i64, Shape{4, 5});
    auto end = make_shared<op::Parameter>(element::i64, Shape{4});
    try {
        auto strided_slice = make_shared<op::v1::StridedSlice>(data,
                                                               begin,
                                                               end,
                                                               vector<int64_t>{1, 0, 1, 0},
                                                               vector<int64_t>{1, 0, 1, 0});
        // Should have thrown, so fail if it didn't
        FAIL() << "Incorrect shape of begin exception not thrown.";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Begin input must be 1D (begin rank:"));
    } catch (...) {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, strided_slice_end_incorrect_shape) {
    auto data = make_shared<op::Parameter>(element::f32, Shape{2, 4, 6, 8});
    auto begin = make_shared<op::Parameter>(element::i64, Shape{4});
    auto end = make_shared<op::Parameter>(element::i64, Shape{4, 5});
    try {
        auto strided_slice = make_shared<op::v1::StridedSlice>(data,
                                                               begin,
                                                               end,
                                                               vector<int64_t>{1, 0, 1, 0},
                                                               vector<int64_t>{1, 0, 1, 0});
        // Should have thrown, so fail if it didn't
        FAIL() << "Incorrect shape of end exception not thrown.";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("End input must be 1D (end rank:"));
    } catch (...) {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, strided_slice_default_stride_dynamic_shape_input_begin_not_1d) {
    auto data = make_shared<op::Parameter>(element::f32, Shape{2, 4, 6, 8});
    auto begin = make_shared<op::Parameter>(element::i64, PartialShape::dynamic());
    const auto end = make_shared<op::Parameter>(element::i64, PartialShape::dynamic());

    OV_EXPECT_THROW(
        const auto strided_slice =
            make_shared<op::v1::StridedSlice>(data, begin, end, vector<int64_t>{0, 0}, vector<int64_t>{0, 0}),
        CheckFailure,
        HasSubstr("Begin input must be 1D"));
}

TEST(type_prop, strided_slice_default_stride_dynamic_shape_input) {
    auto shape = PartialShape{2, 4, 6, 8};
    set_shape_labels(shape, 11);
    auto data = make_shared<op::Parameter>(element::f32, shape);
    auto begin = make_shared<op::Parameter>(element::i64, PartialShape::dynamic());
    auto end = make_shared<op::Parameter>(element::i64, Shape{2});

    auto strided_slice =
        make_shared<op::v1::StridedSlice>(data, begin, end, vector<int64_t>{0, 0}, vector<int64_t>{0, 0});

    EXPECT_EQ(strided_slice->get_output_partial_shape(0), PartialShape({{0, 2}, {0, 4}, 6, 8}));
    EXPECT_THAT(get_shape_labels(strided_slice->get_output_partial_shape(0)),
                ElementsAre(ov::no_label, ov::no_label, 13, 14));
}

TEST(type_prop, strided_slice_reverse_out_of_bounds_on_dims_0_1) {
    auto shape = PartialShape{3, 4, 5};
    set_shape_labels(shape, 10);
    auto data = std::make_shared<op::Parameter>(element::f32, shape);
    auto begin = op::Constant::create(element::i64, Shape{2}, {10, 2});
    auto end = op::Constant::create(element::i64, Shape{2}, {-10, -10});
    auto stride = op::Constant::create(element::i64, Shape{2}, {-1});

    auto mask = std::vector<int64_t>(shape.size(), 0);

    auto strided_slice = std::make_shared<op::v1::StridedSlice>(data, begin, end, stride, mask, mask);

    EXPECT_EQ(strided_slice->get_output_partial_shape(0), PartialShape({3, 3, 5}));
    EXPECT_THAT(get_shape_labels(strided_slice->get_output_partial_shape(0)), ElementsAre(10, ov::no_label, 12));
}

TEST(type_prop, strided_slice_ignore_begin_mask_stride_pos_1) {
    auto shape = PartialShape{4, 4, 4, 4, 4, 4, 4, 4};
    set_shape_labels(shape, 10);
    auto data = std::make_shared<op::Parameter>(element::f32, shape);
    auto begin = op::Constant::create(element::i64, Shape{shape.size()}, {0, 2, 4, 10, -1, -2, -4, 10});
    auto end = op::Constant::create(element::i64, Shape{shape.size()}, {1, 3, 4, 5, -1, -3, -4, -5});
    auto stride = op::Constant::create(element::i64, Shape{shape.size()}, {1});

    auto begin_mask = std::vector<int64_t>(shape.size(), 1);
    auto end_mask = std::vector<int64_t>(shape.size(), 0);

    auto strided_slice = std::make_shared<op::v1::StridedSlice>(data, begin, end, stride, begin_mask, end_mask);

    EXPECT_EQ(strided_slice->get_output_partial_shape(0), PartialShape({1, 3, 4, 4, 3, 1, 0, 0}));
    EXPECT_THAT(
        get_shape_labels(strided_slice->get_output_partial_shape(0)),
        ElementsAre(ov::no_label, ov::no_label, 12, 13, ov::no_label, ov::no_label, ov::no_label, ov::no_label));
}

TEST(type_prop, strided_slice_ignore_begin_mask_stride_neg_1) {
    auto shape = PartialShape{4, 4, 4, 4, 4, 4, 4, 4};
    set_shape_labels(shape, 10);
    auto data = std::make_shared<op::Parameter>(element::f32, shape);
    auto begin = op::Constant::create(element::i64, Shape{shape.size()}, {0, 2, 4, 10, -1, -2, -4, -10});
    auto end = op::Constant::create(element::i64, Shape{shape.size()}, {1, 3, 4, 5, -1, -3, -4, -5});
    auto stride = op::Constant::create(element::i64, Shape{shape.size()}, {-1, -1, -1, -1, -1, -1, -1, -1});

    auto begin_mask = std::vector<int64_t>(shape.size(), 1);
    auto end_mask = std::vector<int64_t>(shape.size(), 0);

    auto strided_slice = std::make_shared<op::v1::StridedSlice>(data, begin, end, stride, begin_mask, end_mask);

    EXPECT_EQ(strided_slice->get_output_partial_shape(0), PartialShape({2, 0, 0, 0, 0, 2, 3, 4}));
    EXPECT_THAT(get_shape_labels(strided_slice->get_output_partial_shape(0)),
                ElementsAre(ov::no_label,
                            ov::no_label,
                            ov::no_label,
                            ov::no_label,
                            ov::no_label,
                            ov::no_label,
                            ov::no_label,
                            17));
}

TEST(type_prop, strided_slice_ignore_end_mask_stride_pos_1) {
    auto shape = PartialShape{4, 4, 4, 4, 4, 4, 4, 4};
    set_shape_labels(shape, 10);
    auto data = std::make_shared<op::Parameter>(element::f32, shape);
    auto begin = op::Constant::create(element::i64, Shape{shape.size()}, {0, 2, 4, 10, -1, -2, -4, -10});
    auto end = op::Constant::create(element::i64, Shape{shape.size()}, {1, 3, 4, 5, -1, -3, -4, -5});
    auto stride = op::Constant::create(element::i64, Shape{shape.size()}, {1});

    auto begin_mask = std::vector<int64_t>(shape.size(), 0);
    auto end_mask = std::vector<int64_t>(shape.size(), 1);

    auto strided_slice = std::make_shared<op::v1::StridedSlice>(data, begin, end, stride, begin_mask, end_mask);

    EXPECT_EQ(strided_slice->get_output_partial_shape(0), PartialShape({4, 2, 0, 0, 1, 2, 4, 4}));
    EXPECT_THAT(get_shape_labels(strided_slice->get_output_partial_shape(0)),
                ElementsAre(10, ov::no_label, ov::no_label, ov::no_label, ov::no_label, ov::no_label, 16, 17));
}

TEST(type_prop, strided_slice_ignore_end_mask_stride_neg_1) {
    auto shape = PartialShape{4, 4, 4, 4, 4, 4, 4, 4};
    set_shape_labels(shape, 10);
    auto data = std::make_shared<op::Parameter>(element::f32, shape);
    auto begin = op::Constant::create(element::i64, Shape{shape.size()}, {0, 2, 4, 10, -1, -2, -4, -10});
    auto end = op::Constant::create(element::i64, Shape{shape.size()}, {1, 3, 4, 5, -1, -3, -4, -5});
    auto stride = op::Constant::create(element::i64, Shape{shape.size()}, {-1, -1, -1, -1, -1, -1, -1, -1});

    auto begin_mask = std::vector<int64_t>(shape.size(), 0);
    auto end_mask = std::vector<int64_t>(shape.size(), 1);

    auto strided_slice = std::make_shared<op::v1::StridedSlice>(data, begin, end, stride, begin_mask, end_mask);

    EXPECT_EQ(strided_slice->get_output_partial_shape(0), PartialShape({1, 3, 4, 4, 4, 3, 1, 1}));
    EXPECT_THAT(get_shape_labels(strided_slice->get_output_partial_shape(0)),
                ElementsAre(ov::no_label, ov::no_label, 12, 13, 14, ov::no_label, ov::no_label, ov::no_label));
}

TEST(type_prop, strided_slice_ignore_begin_end_masks_variadic_stride) {
    auto shape = PartialShape{4, 4, 4, 4, 4, 4, 4, 4};
    set_shape_labels(shape, 10);
    auto data = std::make_shared<op::Parameter>(element::f32, shape);
    auto begin = op::Constant::create(element::i64, Shape{shape.size()}, {0, 2, 4, 10, -1, -2, -4, -10});
    auto end = op::Constant::create(element::i64, Shape{shape.size()}, {1, 3, 4, 5, -1, -3, -4, -5});
    auto stride = op::Constant::create(element::i64, Shape{shape.size()}, {1, 2, 3, 10, -1, -2, -3, -10});

    auto mask = std::vector<int64_t>(shape.size(), 1);

    auto strided_slice = std::make_shared<op::v1::StridedSlice>(data, begin, end, stride, mask, mask);

    EXPECT_EQ(strided_slice->get_output_partial_shape(0), PartialShape({4, 2, 2, 1, 4, 2, 2, 1}));
    EXPECT_THAT(
        get_shape_labels(strided_slice->get_output_partial_shape(0)),
        ElementsAre(10, ov::no_label, ov::no_label, ov::no_label, 14, ov::no_label, ov::no_label, ov::no_label));
}

TEST(type_prop, strided_slice_end_over_dimension_size) {
    auto shape = PartialShape{3, 3, 3, 3, 3, 3, 3, 3};
    set_shape_labels(shape, 10);
    auto data = std::make_shared<op::Parameter>(element::f32, shape);
    auto begin = op::Constant::create(element::i64, Shape{shape.size()}, {0, 0, 0, 0, 1, 1, 1, 1});
    auto end = op::Constant::create(element::i64, Shape{shape.size()}, {1, 2, 3, 4, 1, 2, 3, 4});
    auto stride = op::Constant::create(element::i64, Shape{shape.size()}, {1});

    auto mask = std::vector<int64_t>(shape.size(), 0);

    auto strided_slice = std::make_shared<op::v1::StridedSlice>(data, begin, end, stride, mask, mask);

    EXPECT_EQ(strided_slice->get_output_partial_shape(0), PartialShape({1, 2, 3, 3, 0, 1, 2, 2}));
    EXPECT_THAT(
        get_shape_labels(strided_slice->get_output_partial_shape(0)),
        ElementsAre(ov::no_label, ov::no_label, 12, 13, ov::no_label, ov::no_label, ov::no_label, ov::no_label));
}

TEST(type_prop, strided_slice_begin_over_dimension_size) {
    auto shape = PartialShape{3, 3, 3, 3, 3, 3, 3, 3};
    set_shape_labels(shape, 10);
    auto data = std::make_shared<op::Parameter>(element::f32, shape);
    auto begin = op::Constant::create(element::i64, Shape{shape.size()}, {-1, -1, -1, -1, -2, -2, -2, -2});
    auto end = op::Constant::create(element::i64, Shape{shape.size()}, {1, 2, 3, 4, 1, 2, 3, 4});
    auto stride = op::Constant::create(element::i64, Shape{shape.size()}, {1});

    auto mask = std::vector<int64_t>(shape.size(), 0);

    auto strided_slice = std::make_shared<op::v1::StridedSlice>(data, begin, end, stride, mask, mask);

    EXPECT_EQ(strided_slice->get_output_partial_shape(0), PartialShape({0, 0, 1, 1, 0, 1, 2, 2}));
    EXPECT_THAT(get_shape_labels(strided_slice->get_output_partial_shape(0)), Each(ov::no_label));
}

TEST(type_prop, strided_slice_end_is_shape_of_with_bounds) {
    auto shape = PartialShape{1, {5, 7}};
    set_shape_labels(shape, 20);
    const auto p_end = std::make_shared<op::Parameter>(element::i64, shape);
    const auto shape_of_end = std::make_shared<op::ShapeOf>(p_end);

    auto data = op::Constant::create(element::i64, Shape{1, 10}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 0});
    auto begin = op::Constant::create(element::i64, Shape{2}, {0, 0});
    auto stride = op::Constant::create(element::i64, Shape{2}, {1, 1});

    auto mask = std::vector<int64_t>(2, 0);

    auto strided_slice = std::make_shared<op::v1::StridedSlice>(data, begin, shape_of_end, stride, mask, mask);

    EXPECT_EQ(strided_slice->get_output_partial_shape(0), PartialShape({1, {5, 7}}));
    EXPECT_THAT(get_shape_labels(strided_slice->get_output_partial_shape(0)), Each(ov::no_label));
}

TEST(type_prop, strided_slice_begin_is_shape_of_with_bounds) {
    auto shape = PartialShape{0, {3, 5}};
    set_shape_labels(shape, 20);
    const auto p_begin = std::make_shared<op::Parameter>(element::i64, shape);
    const auto shape_of_begin = std::make_shared<op::ShapeOf>(p_begin);

    auto data = op::Constant::create(element::i64, Shape{1, 10}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 0});
    auto end = op::Constant::create(element::i64, Shape{2}, {1, 7});
    auto stride = op::Constant::create(element::i64, Shape{2}, {1, 1});

    auto mask = std::vector<int64_t>(2, 0);

    auto strided_slice = std::make_shared<op::v1::StridedSlice>(data, shape_of_begin, end, stride, mask, mask);

    EXPECT_EQ(strided_slice->get_output_partial_shape(0), PartialShape({1, {2, 4}}));
    EXPECT_THAT(get_shape_labels(strided_slice->get_output_partial_shape(0)), Each(ov::no_label));
}

TEST(type_prop, strided_slice_begin_end_is_shape_of_with_bounds) {
    auto begin_shape = PartialShape{0, {3, 5}};
    auto end_shape = PartialShape{2, {6, 7}};
    set_shape_labels(begin_shape, 10);
    set_shape_labels(end_shape, 20);
    const auto p_begin = std::make_shared<op::Parameter>(element::i64, begin_shape);
    const auto p_end = std::make_shared<op::Parameter>(element::i64, end_shape);
    const auto shape_of_begin = std::make_shared<op::ShapeOf>(p_begin);
    const auto shape_of_end = std::make_shared<op::ShapeOf>(p_end);

    auto data = op::Constant::create(element::i64, Shape{1, 10}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 0});
    auto stride = op::Constant::create(element::i64, Shape{2}, {1, 1});

    auto mask = std::vector<int64_t>(2, 0);

    auto strided_slice = std::make_shared<op::v1::StridedSlice>(data, shape_of_begin, shape_of_end, stride, mask, mask);

    EXPECT_EQ(strided_slice->get_output_partial_shape(0), PartialShape({1, {1, 4}}));
    EXPECT_THAT(get_shape_labels(strided_slice->get_output_partial_shape(0)), Each(ov::no_label));
}

TEST(type_prop, strided_slice_out_of_bounds_different_stride) {
    auto data = std::make_shared<op::Parameter>(element::f32, PartialShape{5, 5, 5, 5, 5});
    const auto data_rank_size = data->get_partial_shape().size();
    auto begin = op::Constant::create(element::i64, Shape{data_rank_size}, {5, 5, 5, 5, 5});
    auto end = op::Constant::create(element::i64, Shape{data_rank_size}, {5, 5, 5, 5, 5});
    auto stride = op::Constant::create(element::i64, Shape{data_rank_size}, {1, 2, 5, -2, -5});

    const auto mask = std::vector<int64_t>(data_rank_size, 0);

    auto strided_slice = std::make_shared<op::v1::StridedSlice>(data, begin, end, stride, mask, mask);

    EXPECT_EQ(strided_slice->get_output_partial_shape(0), PartialShape({0, 0, 0, 0, 0}));
}

TEST(type_prop, strided_slice_reverse_end_is_int64_min) {
    auto data = std::make_shared<op::Parameter>(element::f32, PartialShape{{0, 20}, -1});
    const auto data_rank_size = data->get_partial_shape().size();
    auto begin = op::Constant::create(element::i64, Shape{data_rank_size}, {20, 20});
    auto end = op::Constant::create(element::i64, Shape{data_rank_size}, std::vector<int64_t>{INT64_MIN, INT64_MIN});
    auto stride = op::Constant::create(element::i64, Shape{data_rank_size}, std::vector<int64_t>{-1, -1});

    const auto mask = std::vector<int64_t>(data_rank_size, 0);

    auto ss = std::make_shared<op::v1::StridedSlice>(data, begin, end, stride, mask, mask);

    EXPECT_EQ(ss->get_output_partial_shape(0), PartialShape({{0, 20}, -1}));
}

TEST(type_prop, strided_slice_dynamic_value_and_label_propagation) {
    // Use evaluate upper,lower and labels
    auto marked_0 = Dimension(3, 5);
    ov::DimensionTracker::set_label(marked_0, 10);
    PartialShape target_0 = PartialShape{marked_0, 4};

    auto param = std::make_shared<op::Parameter>(element::f32, Shape{1});
    auto param_0 = std::make_shared<op::Parameter>(element::f32, target_0);
    auto shape_0 = std::make_shared<op::ShapeOf>(param_0);

    constexpr auto et = element::i64;
    std::vector<int64_t> start_val{0}, stop_val{1}, step_val{1};
    const auto start = std::make_shared<op::v0::Constant>(et, Shape{start_val.size()}, start_val);
    const auto stop = std::make_shared<op::v0::Constant>(et, Shape{stop_val.size()}, stop_val);
    const auto step = std::make_shared<op::v0::Constant>(et, Shape{step_val.size()}, step_val);
    const auto slice = std::make_shared<op::v1::StridedSlice>(shape_0,
                                                              start,
                                                              stop,
                                                              step,
                                                              std::vector<int64_t>{0},
                                                              std::vector<int64_t>{0});

    const auto bc = std::make_shared<op::v1::Broadcast>(param, slice);

    const auto& output_shape = bc->get_output_partial_shape(0);
    EXPECT_EQ(output_shape, PartialShape({marked_0}));
    EXPECT_THAT(get_shape_labels(output_shape), ElementsAre(10));
}

TEST(type_prop, strided_slice_use_default_ctor) {
    const auto zero_mask = std::vector<int64_t>(3, 0);

    auto data = std::make_shared<op::Parameter>(element::f32, PartialShape{10, 11, 12});
    auto begin = op::Constant::create(element::i64, Shape{4}, {0, 0, 0, 0});
    auto end = op::Constant::create(element::i64, Shape{4}, {1, 5, 20, 20});
    auto stride = op::Constant::create(element::i64, Shape{4}, {1, 1, 1, 1});

    auto slice = std::make_shared<op::v1::StridedSlice>();
    slice->set_begin_mask(zero_mask);
    slice->set_end_mask(zero_mask);
    slice->set_new_axis_mask({1, 0, 0});
    slice->set_shrink_axis_mask({0, 0, 1});
    slice->set_ellipsis_mask_mask(zero_mask);
    slice->set_arguments(ov::OutputVector{data, begin, end, stride});
    slice->validate_and_infer_types();

    ASSERT_EQ(slice->get_output_partial_shape(0), PartialShape({1, 5, 12}));
}

struct StridedSliceTestParams {
    std::string case_name;
    PartialShape input_shape;
    PartialShape begin_shape;
    PartialShape end_shape;
    PartialShape strides_shape;
    std::vector<int64_t> begin_mask;
    std::vector<int64_t> end_mask;
    std::vector<int64_t> new_axis_mask;
    std::vector<int64_t> shrink_axis_mask;
    std::vector<int64_t> ellipsis_mask;

    PartialShape ref_shape;
    ov::element::Type ref_type;
};

struct StridedSliceShapeInferTest : public TypePropOpTest<op::v1::StridedSlice>,
                                    public WithParamInterface<StridedSliceTestParams> {
public:
    static std::string get_test_case_name(const TestParamInfo<StridedSliceTestParams>& obj) {
        return obj.param.case_name;
    }
};

TEST_P(StridedSliceShapeInferTest, begin_end_strides_are_not_constants) {
    using namespace ov::opset9;

    const auto& params = GetParam();

    const auto input_data = std::make_shared<Parameter>(params.ref_type, params.input_shape);
    const auto begin = std::make_shared<Parameter>(ov::element::i32, params.begin_shape);
    const auto end = std::make_shared<Parameter>(ov::element::i32, params.end_shape);
    const auto strides = std::make_shared<Parameter>(ov::element::i32, params.strides_shape);
    const auto& begin_mask = params.begin_mask;
    const auto& end_mask = params.end_mask;
    const auto& new_axis_mask = params.new_axis_mask;
    const auto& shrink_axis_mask = params.shrink_axis_mask;
    const auto& ellipsis_mask = params.ellipsis_mask;

    const auto strided_slice =
        make_op(input_data, begin, end, strides, begin_mask, end_mask, new_axis_mask, shrink_axis_mask, ellipsis_mask);

    EXPECT_EQ(strided_slice->get_element_type(), params.ref_type);
    EXPECT_EQ(strided_slice->get_output_size(), 1);
    EXPECT_EQ(strided_slice->get_output_partial_shape(0), params.ref_shape);
}

INSTANTIATE_TEST_SUITE_P(type_prop,
                         StridedSliceShapeInferTest,
                         Values(
                             StridedSliceTestParams{
                                 "shrink_2nd_dim",
                                 {1, 200, 300, 3},            // input_shape
                                 {4},                         // begin shape
                                 {4},                         // end shape
                                 {4},                         // strides shape
                                 {0, 0, 0, 0},                // begin mask
                                 {0, 0, 0, 0},                // end mask
                                 {0, 0, 0, 0},                // new axis mask
                                 {0, 1, 0, 0},                // shrink axis mask
                                 {0, 0, 0, 0},                // ellipsis mask
                                 {{0, 1}, {0, 300}, {0, 3}},  // reference shape
                                 element::f32                 // reference type
                             },
                             StridedSliceTestParams{
                                 "shrink_output_shape_by_axes",
                                 {1, 200, 300, 3},    // input_shape
                                 {4},                 // begin shape
                                 {4},                 // end shape
                                 {4},                 // strides shape
                                 {1, 0, 0, 0},        // begin mask
                                 {1, 0, 0, 0},        // end mask
                                 {0, 0, 0, 0},        // new axis mask
                                 {0, 1, 0, 1},        // shrink axis mask
                                 {0, 0, 0, 0},        // ellipsis mask
                                 {{0, 1}, {0, 300}},  // reference shape
                                 element::f32         // reference type
                             },
                             StridedSliceTestParams{
                                 "extend_output_shape_by_new_axes",
                                 {1, 200, 200, 3},                    // input_shape
                                 {4},                                 // begin shape
                                 {4},                                 // end shape
                                 {4},                                 // strides shape
                                 {1, 0, 0, 0},                        // begin mask
                                 {1, 0, 0, 0},                        // end mask
                                 {0, 0, 1, 0},                        // new axis mask
                                 {0, 0, 0, 0},                        // shrink axis mask
                                 {0, 0, 0, 0},                        // ellipsis mask
                                 {{0, 1}, {0, 200}, 1, {0, 200}, 3},  // reference shape
                                 element::f32                         // reference type
                             },
                             StridedSliceTestParams{
                                 "there_are_empty_masks",
                                 {1, 200, 200, 3},                       // input_shape
                                 {4},                                    // begin shape
                                 {4},                                    // end shape
                                 {4},                                    // strides shape
                                 {1, 0, 0, 0},                           // begin mask
                                 {1, 0, 0, 0},                           // end mask
                                 {},                                     // new axis mask (empty)
                                 {},                                     // shrink axis mask
                                 {},                                     // ellipsis mask
                                 {{0, 1}, {-1, 200}, {0, 200}, {0, 3}},  // reference shape
                                 element::f32                            // reference type
                             },
                             StridedSliceTestParams{
                                 "when_input_has_dynamic_dimension",
                                 {{8, -1}, 200, 200, 3},                              // input_shape
                                 {4},                                                 // begin shape
                                 {4},                                                 // end shape
                                 {4},                                                 // strides shape
                                 {0, 0, 0, 0},                                        // begin mask
                                 {0, 0, 0, 0},                                        // end mask
                                 {0, 0, 0, 0},                                        // new axis mask
                                 {0, 0, 0, 0},                                        // shrink axis mask
                                 {0, 0, 0, 0},                                        // ellipsis mask
                                 {Dimension::dynamic(), {0, 200}, {0, 200}, {0, 3}},  // reference shape
                                 element::f32                                         // reference type
                             },
                             StridedSliceTestParams{
                                 "input_has_dynamic_dimensions_and_shrink_one",
                                 {{8, 40}, 200, {100, 200}, 3},             // input_shape
                                 {4},                                       // begin shape
                                 {4},                                       // end shape
                                 {4},                                       // strides shape
                                 {0, 0, 0, 0},                              // begin mask
                                 {0, 0, 0, 0},                              // end mask
                                 {0, 0, 0, 0},                              // new axis mask
                                 {1, 0, 0, 0},                              // shrink axis mask
                                 {0, 0, 0, 0},                              // ellipsis mask
                                 {{0, 200}, Dimension::dynamic(), {0, 3}},  // reference shape
                                 element::f32                               // reference type
                             },
                             StridedSliceTestParams{
                                 "input_is_dynamic_rank",
                                 PartialShape::dynamic(),  // input_shape
                                 {4},                      // begin shape
                                 {4},                      // end shape
                                 {4},                      // strides shape
                                 {1, 0, 1, 0},             // begin mask
                                 {0, 0, 0, 0},             // end mask
                                 {1, 0, 0, 0},             // new axis mask
                                 {0, 0, 0, 0},             // shrink axis mask
                                 {0, 0, 0, 0},             // ellipsis mask
                                 PartialShape::dynamic(),  // reference shape
                                 element::f32              // reference type
                             },
                             StridedSliceTestParams{
                                 "extend_input_with_static_rank_and_all_dynamic_dims",
                                 PartialShape::dynamic(4),  // input_shape
                                 {4},                       // begin shape
                                 {4},                       // end shape
                                 {4},                       // strides shape
                                 {1, 0, 1, 0},              // begin mask
                                 {0, 0, 0, 0},              // end mask
                                 {1, 0, 0, 0},              // new axis mask
                                 {0, 0, 0, 0},              // shrink axis mask
                                 {0, 0, 0, 0},              // ellipsis mask
                                 {1, -1, -1, -1, -1},       // reference shape
                                 element::f32               // reference type
                             },
                             StridedSliceTestParams{
                                 "begin_strides_are_dynamic_rank",
                                 {3, 5, 4},                 // input_shape
                                 PartialShape::dynamic(),   // begin shape
                                 {3},                       // end shape
                                 PartialShape::dynamic(),   // strides shape
                                 {1, 0, 1, 0},              // begin mask
                                 {0, 0, 0, 0},              // end mask
                                 {0, 0, 0, 0},              // new axis mask
                                 {0, 0, 0, 0},              // shrink axis mask
                                 {0, 0, 0, 0},              // ellipsis mask
                                 {{0, 3}, {0, 5}, {0, 4}},  // reference shape
                                 element::f32               // reference type
                             },
                             StridedSliceTestParams{
                                 "begin_end_strides_are_dynamic_rank",
                                 {3, 5, 4},                // input_shape
                                 PartialShape::dynamic(),  // begin shape
                                 PartialShape::dynamic(),  // end shape
                                 PartialShape::dynamic(),  // strides shape
                                 {1, 0, 1, 0},             // begin mask
                                 {0, 0, 0, 0},             // end mask
                                 {0, 0, 0, 0},             // new axis mask
                                 {0, 0, 0, 0},             // shrink axis mask
                                 {0, 0, 0, 0},             // ellipsis mask
                                 PartialShape::dynamic(),  // reference shape
                                 element::f32              // reference type
                             }),
                         StridedSliceShapeInferTest::get_test_case_name);
