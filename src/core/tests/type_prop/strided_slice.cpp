// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>

#include "common_test_utils/test_assertions.hpp"
#include "common_test_utils/type_prop.hpp"
#include "openvino/core/except.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/opsets/opset9.hpp"
#include "strided_slice_shape_inference.hpp"

using namespace std;
using namespace ov;
using namespace testing;

TEST(type_prop, strided_slice_begin_incorrect_type) {
    auto data = make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 4, 6, 8});
    auto begin = make_shared<ov::op::v0::Parameter>(element::f16, Shape{4});
    auto end = make_shared<ov::op::v0::Parameter>(element::i64, Shape{4});
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
    auto data = make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 4, 6, 8});
    auto begin = make_shared<ov::op::v0::Parameter>(element::i64, Shape{4});
    auto end = make_shared<ov::op::v0::Parameter>(element::boolean, Shape{4});
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
    auto data = make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 4, 6, 8});
    auto begin = make_shared<ov::op::v0::Parameter>(element::i64, Shape{4});
    auto end = make_shared<ov::op::v0::Parameter>(element::i64, Shape{4});
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
    auto data = make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 4, 6, 8});
    auto begin = make_shared<ov::op::v0::Parameter>(element::i64, Shape{4, 5});
    auto end = make_shared<ov::op::v0::Parameter>(element::i64, Shape{4});
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
    auto data = make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 4, 6, 8});
    auto begin = make_shared<ov::op::v0::Parameter>(element::i64, Shape{4, 5});
    auto end = make_shared<ov::op::v0::Parameter>(element::i64, Shape{4});

    OV_EXPECT_THROW(auto strided_slice = make_shared<op::v1::StridedSlice>(data,
                                                                           begin,
                                                                           end,
                                                                           vector<int64_t>{1, 0, 1, 0},
                                                                           vector<int64_t>{1, 0, 1, 0}),
                    NodeValidationFailure,
                    HasSubstr("Begin input must be 1D (has rank:"));
}

TEST(type_prop, strided_slice_end_incorrect_shape) {
    auto data = make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 4, 6, 8});
    auto begin = make_shared<ov::op::v0::Parameter>(element::i64, Shape{4});
    auto end = make_shared<ov::op::v0::Parameter>(element::i64, Shape{4, 5});

    OV_EXPECT_THROW(auto strided_slice = make_shared<op::v1::StridedSlice>(data,
                                                                           begin,
                                                                           end,
                                                                           vector<int64_t>{1, 0, 1, 0},
                                                                           vector<int64_t>{1, 0, 1, 0}),
                    NodeValidationFailure,
                    HasSubstr("End input must be 1D (has rank:"));
}

TEST(type_prop, strided_slice_default_stride_dynamic_shape_input_begin_not_1d) {
    auto data = make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 4, 6, 8});
    auto begin = make_shared<ov::op::v0::Parameter>(element::i64, PartialShape::dynamic());
    const auto end = make_shared<ov::op::v0::Parameter>(element::i64, PartialShape::dynamic());

    OV_EXPECT_THROW(
        const auto strided_slice =
            make_shared<op::v1::StridedSlice>(data, begin, end, vector<int64_t>{0, 0}, vector<int64_t>{0, 0}),
        AssertFailure,
        HasSubstr("Begin input must be 1D"));
}

TEST(type_prop, strided_slice_default_stride_dynamic_shape_input) {
    auto shape = PartialShape{2, 4, 6, 8};
    auto symbols = set_shape_symbols(shape);
    auto data = make_shared<ov::op::v0::Parameter>(element::f32, shape);
    auto begin = make_shared<ov::op::v0::Parameter>(element::i64, PartialShape::dynamic());
    auto end = make_shared<ov::op::v0::Parameter>(element::i64, Shape{2});
    auto strided_slice =
        make_shared<op::v1::StridedSlice>(data, begin, end, vector<int64_t>{0, 0}, vector<int64_t>{0, 0});

    EXPECT_EQ(strided_slice->get_output_partial_shape(0), PartialShape({{0, 2}, {0, 4}, 6, 8}));
    EXPECT_THAT(get_shape_symbols(strided_slice->get_output_partial_shape(0)),
                ElementsAre(nullptr, nullptr, symbols[2], symbols[3]));
}

TEST(type_prop, strided_slice_reverse_out_of_bounds_on_dims_0_1) {
    auto shape = PartialShape{3, 4, 5};
    auto symbols = set_shape_symbols(shape);
    auto data = std::make_shared<ov::op::v0::Parameter>(element::f32, shape);
    auto begin = ov::op::v0::Constant::create(element::i64, Shape{2}, {10, 2});
    auto end = ov::op::v0::Constant::create(element::i64, Shape{2}, {-10, -10});
    auto stride = ov::op::v0::Constant::create(element::i64, Shape{2}, {-1});

    auto mask = std::vector<int64_t>(shape.size(), 0);

    auto strided_slice = std::make_shared<op::v1::StridedSlice>(data, begin, end, stride, mask, mask);

    EXPECT_EQ(strided_slice->get_output_partial_shape(0), PartialShape({3, 3, 5}));
    EXPECT_THAT(get_shape_symbols(strided_slice->get_output_partial_shape(0)),
                ElementsAre(symbols[0], nullptr, symbols[2]));
}

TEST(type_prop, strided_slice_ignore_begin_mask_stride_pos_1) {
    auto shape = PartialShape{4, 4, 4, 4, 4, 4, 4, 4};
    auto symbols = set_shape_symbols(shape);
    auto data = std::make_shared<ov::op::v0::Parameter>(element::f32, shape);
    auto begin = ov::op::v0::Constant::create(element::i64, Shape{shape.size()}, {0, 2, 4, 10, -1, -2, -4, 10});
    auto end = ov::op::v0::Constant::create(element::i64, Shape{shape.size()}, {1, 3, 4, 5, -1, -3, -4, -5});
    auto stride = ov::op::v0::Constant::create(element::i64, Shape{shape.size()}, {1});

    auto begin_mask = std::vector<int64_t>(shape.size(), 1);
    auto end_mask = std::vector<int64_t>(shape.size(), 0);

    auto strided_slice = std::make_shared<op::v1::StridedSlice>(data, begin, end, stride, begin_mask, end_mask);

    EXPECT_EQ(strided_slice->get_output_partial_shape(0), PartialShape({1, 3, 4, 4, 3, 1, 0, 0}));
    EXPECT_THAT(get_shape_symbols(strided_slice->get_output_partial_shape(0)),
                ElementsAre(nullptr, nullptr, symbols[2], symbols[3], nullptr, nullptr, nullptr, nullptr));
}

TEST(type_prop, strided_slice_ignore_begin_mask_stride_neg_1) {
    auto shape = PartialShape{4, 4, 4, 4, 4, 4, 4, 4};
    auto symbols = set_shape_symbols(shape);
    auto data = std::make_shared<ov::op::v0::Parameter>(element::f32, shape);
    auto begin = ov::op::v0::Constant::create(element::i64, Shape{shape.size()}, {0, 2, 4, 10, -1, -2, -4, -10});
    auto end = ov::op::v0::Constant::create(element::i64, Shape{shape.size()}, {1, 3, 4, 5, -1, -3, -4, -5});
    auto stride = ov::op::v0::Constant::create(element::i64, Shape{shape.size()}, {-1, -1, -1, -1, -1, -1, -1, -1});

    auto begin_mask = std::vector<int64_t>(shape.size(), 1);
    auto end_mask = std::vector<int64_t>(shape.size(), 0);

    auto strided_slice = std::make_shared<op::v1::StridedSlice>(data, begin, end, stride, begin_mask, end_mask);

    EXPECT_EQ(strided_slice->get_output_partial_shape(0), PartialShape({2, 0, 0, 0, 0, 2, 3, 4}));
    EXPECT_THAT(get_shape_symbols(strided_slice->get_output_partial_shape(0)),
                ElementsAre(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, symbols[7]));
}

TEST(type_prop, strided_slice_ignore_end_mask_stride_pos_1) {
    auto shape = PartialShape{4, 4, 4, 4, 4, 4, 4, 4};
    auto symbols = set_shape_symbols(shape);
    auto data = std::make_shared<ov::op::v0::Parameter>(element::f32, shape);
    auto begin = ov::op::v0::Constant::create(element::i64, Shape{shape.size()}, {0, 2, 4, 10, -1, -2, -4, -10});
    auto end = ov::op::v0::Constant::create(element::i64, Shape{shape.size()}, {1, 3, 4, 5, -1, -3, -4, -5});
    auto stride = ov::op::v0::Constant::create(element::i64, Shape{shape.size()}, {1});

    auto begin_mask = std::vector<int64_t>(shape.size(), 0);
    auto end_mask = std::vector<int64_t>(shape.size(), 1);

    auto strided_slice = std::make_shared<op::v1::StridedSlice>(data, begin, end, stride, begin_mask, end_mask);

    EXPECT_EQ(strided_slice->get_output_partial_shape(0), PartialShape({4, 2, 0, 0, 1, 2, 4, 4}));
    EXPECT_THAT(get_shape_symbols(strided_slice->get_output_partial_shape(0)),
                ElementsAre(symbols[0], nullptr, nullptr, nullptr, nullptr, nullptr, symbols[6], symbols[7]));
}

TEST(type_prop, strided_slice_ignore_end_mask_stride_neg_1) {
    auto shape = PartialShape{4, 4, 4, 4, 4, 4, 4, 4};
    auto symbols = set_shape_symbols(shape);
    auto data = std::make_shared<ov::op::v0::Parameter>(element::f32, shape);
    auto begin = ov::op::v0::Constant::create(element::i64, Shape{shape.size()}, {0, 2, 4, 10, -1, -2, -4, -10});
    auto end = ov::op::v0::Constant::create(element::i64, Shape{shape.size()}, {1, 3, 4, 5, -1, -3, -4, -5});
    auto stride = ov::op::v0::Constant::create(element::i64, Shape{shape.size()}, {-1, -1, -1, -1, -1, -1, -1, -1});

    auto begin_mask = std::vector<int64_t>(shape.size(), 0);
    auto end_mask = std::vector<int64_t>(shape.size(), 1);

    auto strided_slice = std::make_shared<op::v1::StridedSlice>(data, begin, end, stride, begin_mask, end_mask);

    EXPECT_EQ(strided_slice->get_output_partial_shape(0), PartialShape({1, 3, 4, 4, 4, 3, 1, 1}));
    EXPECT_THAT(get_shape_symbols(strided_slice->get_output_partial_shape(0)),
                ElementsAre(nullptr, nullptr, symbols[2], symbols[3], symbols[4], nullptr, nullptr, nullptr));
}

TEST(type_prop, strided_slice_ignore_begin_end_masks_variadic_stride) {
    auto shape = PartialShape{4, 4, 4, 4, 4, 4, 4, 4};
    auto symbols = set_shape_symbols(shape);
    auto data = std::make_shared<ov::op::v0::Parameter>(element::f32, shape);
    auto begin = ov::op::v0::Constant::create(element::i64, Shape{shape.size()}, {0, 2, 4, 10, -1, -2, -4, -10});
    auto end = ov::op::v0::Constant::create(element::i64, Shape{shape.size()}, {1, 3, 4, 5, -1, -3, -4, -5});
    auto stride = ov::op::v0::Constant::create(element::i64, Shape{shape.size()}, {1, 2, 3, 10, -1, -2, -3, -10});

    auto mask = std::vector<int64_t>(shape.size(), 1);

    auto strided_slice = std::make_shared<op::v1::StridedSlice>(data, begin, end, stride, mask, mask);

    EXPECT_EQ(strided_slice->get_output_partial_shape(0), PartialShape({4, 2, 2, 1, 4, 2, 2, 1}));
    EXPECT_THAT(get_shape_symbols(strided_slice->get_output_partial_shape(0)),
                ElementsAre(symbols[0], nullptr, nullptr, nullptr, symbols[4], nullptr, nullptr, nullptr));
}

TEST(type_prop, strided_slice_end_over_dimension_size) {
    auto shape = PartialShape{3, 3, 3, 3, 3, 3, 3, 3};
    auto symbols = set_shape_symbols(shape);
    auto data = std::make_shared<ov::op::v0::Parameter>(element::f32, shape);
    auto begin = ov::op::v0::Constant::create(element::i64, Shape{shape.size()}, {0, 0, 0, 0, 1, 1, 1, 1});
    auto end = ov::op::v0::Constant::create(element::i64, Shape{shape.size()}, {1, 2, 3, 4, 1, 2, 3, 4});
    auto stride = ov::op::v0::Constant::create(element::i64, Shape{shape.size()}, {1});

    auto mask = std::vector<int64_t>(shape.size(), 0);

    auto strided_slice = std::make_shared<op::v1::StridedSlice>(data, begin, end, stride, mask, mask);

    EXPECT_EQ(strided_slice->get_output_partial_shape(0), PartialShape({1, 2, 3, 3, 0, 1, 2, 2}));
    EXPECT_THAT(get_shape_symbols(strided_slice->get_output_partial_shape(0)),
                ElementsAre(nullptr, nullptr, symbols[2], symbols[3], nullptr, nullptr, nullptr, nullptr));
}

TEST(type_prop, strided_slice_begin_over_dimension_size) {
    auto shape = PartialShape{3, 3, 3, 3, 3, 3, 3, 3};
    set_shape_symbols(shape);
    auto data = std::make_shared<ov::op::v0::Parameter>(element::f32, shape);
    auto begin = ov::op::v0::Constant::create(element::i64, Shape{shape.size()}, {-1, -1, -1, -1, -2, -2, -2, -2});
    auto end = ov::op::v0::Constant::create(element::i64, Shape{shape.size()}, {1, 2, 3, 4, 1, 2, 3, 4});
    auto stride = ov::op::v0::Constant::create(element::i64, Shape{shape.size()}, {1});

    auto mask = std::vector<int64_t>(shape.size(), 0);

    auto strided_slice = std::make_shared<op::v1::StridedSlice>(data, begin, end, stride, mask, mask);

    EXPECT_EQ(strided_slice->get_output_partial_shape(0), PartialShape({0, 0, 1, 1, 0, 1, 2, 2}));
    EXPECT_THAT(get_shape_symbols(strided_slice->get_output_partial_shape(0)), Each(nullptr));
}

TEST(type_prop, strided_slice_end_is_shape_of_with_bounds) {
    auto shape = PartialShape{1, {5, 7}};
    set_shape_symbols(shape);
    const auto p_end = std::make_shared<ov::op::v0::Parameter>(element::i64, shape);
    const auto shape_of_end = std::make_shared<op::v0::ShapeOf>(p_end);

    auto data = ov::op::v0::Constant::create(element::i64, Shape{1, 10}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 0});
    auto begin = ov::op::v0::Constant::create(element::i64, Shape{2}, {0, 0});
    auto stride = ov::op::v0::Constant::create(element::i64, Shape{2}, {1, 1});

    auto mask = std::vector<int64_t>(2, 0);

    auto strided_slice = std::make_shared<op::v1::StridedSlice>(data, begin, shape_of_end, stride, mask, mask);

    EXPECT_EQ(strided_slice->get_output_partial_shape(0), PartialShape({1, {5, 7}}));
    EXPECT_THAT(get_shape_symbols(strided_slice->get_output_partial_shape(0)), Each(nullptr));
}

TEST(type_prop, strided_slice_begin_is_shape_of_with_bounds) {
    auto shape = PartialShape{0, {3, 5}};
    set_shape_symbols(shape);
    const auto p_begin = std::make_shared<ov::op::v0::Parameter>(element::i64, shape);
    const auto shape_of_begin = std::make_shared<op::v0::ShapeOf>(p_begin);

    auto data = ov::op::v0::Constant::create(element::i64, Shape{1, 10}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 0});
    auto end = ov::op::v0::Constant::create(element::i64, Shape{2}, {1, 7});
    auto stride = ov::op::v0::Constant::create(element::i64, Shape{2}, {1, 1});

    auto mask = std::vector<int64_t>(2, 0);

    auto strided_slice = std::make_shared<op::v1::StridedSlice>(data, shape_of_begin, end, stride, mask, mask);

    EXPECT_EQ(strided_slice->get_output_partial_shape(0), PartialShape({1, {2, 4}}));
    EXPECT_THAT(get_shape_symbols(strided_slice->get_output_partial_shape(0)), Each(nullptr));
}

TEST(type_prop, strided_slice_begin_end_is_shape_of_with_bounds) {
    auto begin_shape = PartialShape{0, {3, 5}};
    auto end_shape = PartialShape{2, {6, 7}};
    set_shape_symbols(begin_shape);
    set_shape_symbols(end_shape);
    const auto p_begin = std::make_shared<ov::op::v0::Parameter>(element::i64, begin_shape);
    const auto p_end = std::make_shared<ov::op::v0::Parameter>(element::i64, end_shape);
    const auto shape_of_begin = std::make_shared<op::v0::ShapeOf>(p_begin);
    const auto shape_of_end = std::make_shared<op::v0::ShapeOf>(p_end);

    auto data = ov::op::v0::Constant::create(element::i64, Shape{1, 10}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 0});
    auto stride = ov::op::v0::Constant::create(element::i64, Shape{2}, {1, 1});

    auto mask = std::vector<int64_t>(2, 0);

    auto strided_slice = std::make_shared<op::v1::StridedSlice>(data, shape_of_begin, shape_of_end, stride, mask, mask);

    EXPECT_EQ(strided_slice->get_output_partial_shape(0), PartialShape({1, {1, 4}}));
    EXPECT_THAT(get_shape_symbols(strided_slice->get_output_partial_shape(0)), Each(nullptr));
}

TEST(type_prop, strided_slice_out_of_bounds_different_stride) {
    auto data = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{5, 5, 5, 5, 5});
    const auto data_rank_size = data->get_partial_shape().size();
    auto begin = ov::op::v0::Constant::create(element::i64, Shape{data_rank_size}, {5, 5, 5, 5, 5});
    auto end = ov::op::v0::Constant::create(element::i64, Shape{data_rank_size}, {5, 5, 5, 5, 5});
    auto stride = ov::op::v0::Constant::create(element::i64, Shape{data_rank_size}, {1, 2, 5, -2, -5});

    const auto mask = std::vector<int64_t>(data_rank_size, 0);

    auto strided_slice = std::make_shared<op::v1::StridedSlice>(data, begin, end, stride, mask, mask);

    EXPECT_EQ(strided_slice->get_output_partial_shape(0), PartialShape({0, 0, 0, 0, 0}));
}

TEST(type_prop, strided_slice_reverse_end_is_int64_min) {
    auto data = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{{0, 20}, -1});
    const auto data_rank_size = data->get_partial_shape().size();
    auto begin = ov::op::v0::Constant::create(element::i64, Shape{data_rank_size}, {20, 20});
    auto end =
        ov::op::v0::Constant::create(element::i64, Shape{data_rank_size}, std::vector<int64_t>{INT64_MIN, INT64_MIN});
    auto stride = ov::op::v0::Constant::create(element::i64, Shape{data_rank_size}, std::vector<int64_t>{-1, -1});

    const auto mask = std::vector<int64_t>(data_rank_size, 0);

    auto ss = std::make_shared<op::v1::StridedSlice>(data, begin, end, stride, mask, mask);

    EXPECT_EQ(ss->get_output_partial_shape(0), PartialShape({{0, 20}, {0, 21}}));
}

TEST(type_prop, strided_slice_dynamic_value_and_symbol_propagation) {
    // Use evaluate upper,lower and symbols
    auto marked_0 = Dimension(3, 5);
    auto symbol = std::make_shared<Symbol>();
    marked_0.set_symbol(symbol);
    PartialShape target_0 = PartialShape{marked_0, 4};

    auto param = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{1});
    auto param_0 = std::make_shared<ov::op::v0::Parameter>(element::f32, target_0);
    auto shape_0 = std::make_shared<op::v0::ShapeOf>(param_0);

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
    EXPECT_THAT(get_shape_symbols(output_shape), ElementsAre(symbol));
}

TEST(type_prop, strided_slice_use_default_ctor) {
    const auto zero_mask = std::vector<int64_t>(3, 0);

    auto data = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{10, 11, 12});
    auto begin = ov::op::v0::Constant::create(element::i64, Shape{4}, {0, 0, 0, 0});
    auto end = ov::op::v0::Constant::create(element::i64, Shape{4}, {1, 5, 20, 20});
    auto stride = ov::op::v0::Constant::create(element::i64, Shape{4}, {1, 1, 1, 1});

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

TEST(type_prop, strided_slice_inf_dim_start_from_last_N_to_end) {
    auto data = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{1, 256, -1});
    auto start = ov::op::v0::Constant::create(element::i64, Shape{3}, {0, 0, -7});
    auto stop = ov::op::v0::Constant::create(element::i64, Shape{3}, std::vector<int64_t>{0, 0, INT64_MAX});
    auto step = ov::op::v0::Constant::create(element::i64, Shape{3}, {1, 1, 1});

    const auto slice = std::make_shared<op::v1::StridedSlice>(data,
                                                              start,
                                                              stop,
                                                              step,
                                                              std::vector<int64_t>{1, 1, 0},
                                                              std::vector<int64_t>{1, 1, 0});

    EXPECT_EQ(slice->get_output_partial_shape(0), PartialShape({1, 256, {0, 7}}));
}

TEST(type_prop, strided_slice_different_ranks) {
    auto data = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{1, 2, 3, 4});
    auto start = ov::op::v0::Constant::create(element::i64, Shape{1}, {0});
    auto stop = ov::op::v0::Constant::create(element::i64, Shape{1}, std::vector<int64_t>{INT64_MAX});

    const auto slice = std::make_shared<op::v1::StridedSlice>(data,
                                                              start,
                                                              stop,
                                                              std::vector<int64_t>{1, 1, 1, 1, 1},
                                                              std::vector<int64_t>{0, 0, 0, 0, 0});

    EXPECT_EQ(slice->get_output_partial_shape(0), PartialShape({1, 2, 3, 4}));
}

TEST(type_prop, strided_slice_different_ranks_long_masks) {
    auto data = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{1, 2, 3, 4});
    auto start = ov::op::v0::Constant::create(element::i64, Shape{4}, {0, 0, 0, 0});
    auto stop = ov::op::v0::Constant::create(element::i64, Shape{4}, std::vector<int64_t>{2, 2, 2, 2});

    const auto slice = std::make_shared<op::v1::StridedSlice>(data,
                                                              start,
                                                              stop,
                                                              std::vector<int64_t>{1, 1, 0, 1, 1},
                                                              std::vector<int64_t>{0, 0, 1, 0, 0});
    EXPECT_EQ(slice->get_output_partial_shape(0), PartialShape({1, 2, 3, 2}));
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
                                 {{8, 40}, 200, {100, 200}, 3},  // input_shape
                                 {4},                            // begin shape
                                 {4},                            // end shape
                                 {4},                            // strides shape
                                 {0, 0, 0, 0},                   // begin mask
                                 {0, 0, 0, 0},                   // end mask
                                 {0, 0, 0, 0},                   // new axis mask
                                 {1, 0, 0, 0},                   // shrink axis mask
                                 {0, 0, 0, 0},                   // ellipsis mask
                                 {{0, 200}, {0, 200}, {0, 3}},   // reference shape
                                 element::f32                    // reference type
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
                                 "begin_strides_are_dynamic_rank_and_ellipsis_mask_present",
                                 {3, 5, 4},                // input_shape
                                 PartialShape::dynamic(),  // begin shape
                                 {3},                      // end shape
                                 {3},                      // strides shape
                                 {0, 0, 1, 0},             // begin mask
                                 {0, 0, 0, 0},             // end mask
                                 {0, 0, 0, 0},             // new axis mask
                                 {0, 0, 0, 0},             // shrink axis mask
                                 {0, 1, 0, 0},             // ellipsis mask
                                 {{0, 3}, 5, {0, 4}},      // reference shape
                                 element::f32              // reference type
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

using StridedSliceIntervalParams =
    std::tuple<ov::PartialShape, ov::PartialShape, ov::PartialShape, int64_t, int64_t, int64_t, ov::PartialShape>;

class StridedSliceIntervalTest : public TypePropOpTest<op::v1::StridedSlice>,
                                 public WithParamInterface<StridedSliceIntervalParams> {
protected:
    void SetUp() override {
        std::tie(data_shape, begin_shape, end_shape, begin_offset, end_offset, step, exp_shape) = GetParam();
    }

    ov::PartialShape data_shape, begin_shape, end_shape, exp_shape;
    int64_t begin_offset, end_offset, step;
};

INSTANTIATE_TEST_SUITE_P(type_prop,
                         StridedSliceIntervalTest,
                         Values(StridedSliceIntervalParams({1024}, {{0, 20}}, {{10, 20}}, 0, 0, 1, {{0, 20}}),
                                StridedSliceIntervalParams({1024}, {{0, 20}}, {{10, 20}}, 0, 0, 1, {{0, 20}}),
                                StridedSliceIntervalParams({1024}, {{0, 20}}, {{10, 20}}, 10, 0, 1, {{0, 20}}),
                                StridedSliceIntervalParams({-1}, {{0, 20}}, {{10, 20}}, 10, 0, 1, {{0, 20}}),
                                StridedSliceIntervalParams({1024}, {{10, 20}}, {{0, 20}}, 0, 10, 1, {{0, 1013}}),
                                StridedSliceIntervalParams({1024}, {{0, 10}}, {{0, 5}}, 0, 10, 1, {{1004, 1019}}),
                                StridedSliceIntervalParams({{120, 1024}}, {{0, 10}}, {{0, 5}}, 0, 10, 1, {{100, 1019}}),
                                StridedSliceIntervalParams({1024}, {{0, 1030}}, {{0, 2000}}, 1025, 10, 1, {{0, 1024}}),
                                StridedSliceIntervalParams({1024}, {{0, 10}}, {{10, 20}}, 10, 0, 1, {{0, 20}}),
                                StridedSliceIntervalParams({1024}, {{1, 12}}, {{0, 18}}, 10, 0, 2, {{0, 9}}),
                                StridedSliceIntervalParams({1024}, {{10, 20}}, {{0, 20}}, 0, 10, 2, {{0, 507}}),
                                StridedSliceIntervalParams({{100, 1024}}, {{10, 20}}, {{0, 20}}, 0, 10, 2, {{0, 507}}),
                                StridedSliceIntervalParams({1024}, {10}, {30}, 0, 0, 2, {10}),
                                StridedSliceIntervalParams({1024}, {{10, 15}}, {30}, 0, 0, 2, {{8, 10}}),
                                StridedSliceIntervalParams({1024}, {{10, 15}}, {{30, 40}}, 0, 0, 2, {{8, 15}}),
                                StridedSliceIntervalParams({{20, 1024}}, {{10, 15}}, {{30, 40}}, 0, 0, 2, {{3, 15}}),
                                // reverse stride
                                StridedSliceIntervalParams({1024}, {{0, 20}}, {{10, 20}}, 10, 0, -1, {{0, 1013}}),
                                StridedSliceIntervalParams({-1}, {{0, 20}}, {{10, 20}}, 10, 0, -1, {-1}),
                                StridedSliceIntervalParams({1024}, {{10, 20}}, {{0, 20}}, 0, 10, -1, {{0, 20}}),
                                StridedSliceIntervalParams({1024}, {30}, {10}, 35, 40, -1, {25}),
                                StridedSliceIntervalParams({1024}, {{0, 2000}}, {{0, 1030}}, 10, 1026, -1, {{0, 1024}}),
                                StridedSliceIntervalParams({1024}, {30}, {10}, 0, 0, -2, {10}),
                                StridedSliceIntervalParams({1024}, {{20, 30}}, {10}, 0, 0, -2, {{5, 10}}),
                                StridedSliceIntervalParams({1024}, {{20, 30}}, {10}, 40, 0, -2, {{497, 502}}),
                                StridedSliceIntervalParams({{100, 1024}}, {{20, 30}}, {10}, 40, 0, -2, {{35, 502}}),
                                StridedSliceIntervalParams({1024}, {{20, 30}}, {{10, 15}}, 0, 0, -2, {{3, 10}}),
                                StridedSliceIntervalParams({{10, 1024}}, {{20, 30}}, {{10, 15}}, 0, 0, -2, {{0, 10}})));

TEST_P(StridedSliceIntervalTest, begin_end_as_interval) {
    using namespace ov::opset9;

    const auto p_begin = std::make_shared<Parameter>(element::i64, begin_shape);
    const auto shape_of_begin = std::make_shared<ShapeOf>(p_begin);
    const auto begin =
        std::make_shared<Subtract>(shape_of_begin, Constant::create(element::i64, Shape{1}, {begin_offset}));

    const auto p_end = std::make_shared<Parameter>(element::i64, end_shape);
    const auto shape_of_end = std::make_shared<ShapeOf>(p_end);
    const auto end = std::make_shared<Subtract>(shape_of_end, Constant::create(element::i64, Shape{1}, {end_offset}));

    const auto data = std::make_shared<Parameter>(element::f32, data_shape);
    const auto stride = Constant::create(element::i64, Shape{1}, {step});
    const auto mask = std::vector<int64_t>{0};

    const auto op = make_op(data, begin, end, stride, mask, mask);

    EXPECT_EQ(op->get_output_partial_shape(0), exp_shape);
}
