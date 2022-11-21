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

    auto begin_mask = std::vector<int64_t>(shape.size(), 0);
    auto end_mask = std::vector<int64_t>(shape.size(), 0);

    auto ss = std::make_shared<op::v1::StridedSlice>(data, begin, end, stride, begin_mask, end_mask);

    EXPECT_EQ(ss->get_output_partial_shape(0), PartialShape({3, 3, 5}));
    EXPECT_THAT(get_shape_labels(ss->get_output_partial_shape(0)), ElementsAre(10, ov::no_label, 12));
}

TEST(type_prop, strided_slice_to_out_of_upper_bound) {
    auto shape = PartialShape{3, 3, 3, 3, 3, 3, 3, 3};
    set_shape_labels(shape, 10);
    auto data = std::make_shared<op::Parameter>(element::f32, shape);
    auto begin = op::Constant::create(element::i64, Shape{shape.size()}, {0, 0, 0, 0, 1, 1, 1, 1});
    auto end = op::Constant::create(element::i64, Shape{shape.size()}, {1, 2, 3, 4, 1, 2, 3, 4});
    auto stride = op::Constant::create(element::i64, Shape{shape.size()}, {1});

    auto begin_mask = std::vector<int64_t>(shape.size(), 0);
    auto end_mask = std::vector<int64_t>(shape.size(), 0);

    auto ss = std::make_shared<op::v1::StridedSlice>(data, begin, end, stride, begin_mask, end_mask);

    EXPECT_EQ(ss->get_output_partial_shape(0), PartialShape({1, 2, 3, 3, 0, 1, 2, 2}));
    EXPECT_THAT(
        get_shape_labels(ss->get_output_partial_shape(0)),
        ElementsAre(ov::no_label, ov::no_label, 12, 13, ov::no_label, ov::no_label, ov::no_label, ov::no_label));
}

TEST(type_prop, strided_slice_to_out_of_upper_bound_negative_begin) {
    auto shape = PartialShape{3, 3, 3, 3, 3, 3, 3, 3};
    set_shape_labels(shape, 10);
    auto data = std::make_shared<op::Parameter>(element::f32, shape);
    auto begin = op::Constant::create(element::i64, Shape{shape.size()}, {-1, -1, -1, -1, -2, -2, -2, -2});
    auto end = op::Constant::create(element::i64, Shape{shape.size()}, {1, 2, 3, 4, 1, 2, 3, 4});
    auto stride = op::Constant::create(element::i64, Shape{shape.size()}, {1});

    auto begin_mask = std::vector<int64_t>(shape.size(), 0);
    auto end_mask = std::vector<int64_t>(shape.size(), 0);

    auto ss = std::make_shared<op::v1::StridedSlice>(data, begin, end, stride, begin_mask, end_mask);

    EXPECT_EQ(ss->get_output_partial_shape(0), PartialShape({0, 0, 1, 1, 0, 1, 2, 2}));
    EXPECT_THAT(get_shape_labels(ss->get_output_partial_shape(0)), Each(ov::no_label));
}

TEST(type_prop, strided_slice_dynamic_value_and_label_propagation) {
    Dimension marked_0 = Dimension(3);
    ov::DimensionTracker::set_label(marked_0, 10);
    PartialShape target_0 = PartialShape{marked_0, 4};

    auto param = std::make_shared<op::Parameter>(element::f32, Shape{1});
    auto param_0 = std::make_shared<op::Parameter>(element::f32, target_0);
    auto shape_0 = std::make_shared<op::ShapeOf>(param_0);

    const auto& et = element::i64;
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

    auto bc = std::make_shared<op::v1::Broadcast>(param, slice);
    ASSERT_EQ(bc->get_shape(), (Shape{3}));

    const auto& output_shape = bc->get_output_partial_shape(0);
    ASSERT_EQ(ov::DimensionTracker::get_label(output_shape[0]), 10);
}

TEST(type_prop, strided_slice_default_shape_inference) {
    auto slice = new op::v1::StridedSlice;
    slice->set_begin_mask({0, 0, 0});
    slice->set_end_mask({0, 0, 0});
    slice->set_new_axis_mask({1, 0, 0});
    slice->set_shrink_axis_mask({0, 0, 0, 1});
    slice->set_ellipsis_mask_mask({0, 0, 0});
    std::vector<ov::PartialShape> in = {{10, 11, 12}, {4}, {4}, {4}}, out = {PartialShape()};
    int64_t begin_data[] = {0, 0, 0, 0}, end_data[] = {1, 1, 5, 1}, stride_data[] = {1, 1, 1, 1};
    const std::map<size_t, std::shared_ptr<ngraph::runtime::HostTensor>> const_data = {
        {1, std::make_shared<ov::HostTensor>(element::i64, Shape{4}, begin_data)},
        {2, std::make_shared<ov::HostTensor>(element::i64, Shape{4}, end_data)},
        {3, std::make_shared<ov::HostTensor>(element::i64, Shape{4}, stride_data)}};
    ov::op::v1::shape_infer(slice, in, out, const_data);
    ASSERT_EQ(out[0], PartialShape({1, 1, 5}));
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
