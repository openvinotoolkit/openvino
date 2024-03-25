// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/non_max_suppression.hpp"

#include "common_test_utils/test_assertions.hpp"
#include "common_test_utils/type_prop.hpp"
#include "openvino/op/constant.hpp"

using namespace std;
using namespace ov;
using namespace testing;

template <class TOp>
class NMSCommonTest : public TypePropOpTest<TOp> {};

TYPED_TEST_SUITE_P(NMSCommonTest);

TYPED_TEST_P(NMSCommonTest, incorrect_boxes_rank) {
    const auto boxes = make_shared<op::v0::Parameter>(element::f32, Shape{1, 2, 3, 4});
    const auto scores = make_shared<op::v0::Parameter>(element::f32, Shape{1, 2, 3});

    OV_EXPECT_THROW(ignore = this->make_op(boxes, scores),
                    NodeValidationFailure,
                    HasSubstr("Expected a 3D tensor for the 'boxes' input"));
}

TYPED_TEST_P(NMSCommonTest, incorrect_scores_rank) {
    const auto boxes = make_shared<op::v0::Parameter>(element::f32, Shape{1, 2, 3});
    const auto scores = make_shared<op::v0::Parameter>(element::f32, Shape{1, 2});

    OV_EXPECT_THROW(ignore = this->make_op(boxes, scores),
                    NodeValidationFailure,
                    HasSubstr("Expected a 3D tensor for the 'scores' input"));
}

TYPED_TEST_P(NMSCommonTest, incorrect_scheme_num_batches) {
    const auto boxes = make_shared<op::v0::Parameter>(element::f32, Shape{1, 2, 3});
    const auto scores = make_shared<op::v0::Parameter>(element::f32, Shape{2, 2, 3});

    OV_EXPECT_THROW(ignore = this->make_op(boxes, scores),
                    NodeValidationFailure,
                    HasSubstr("The first dimension of both 'boxes' and 'scores' must match"));
}

TYPED_TEST_P(NMSCommonTest, incorrect_scheme_num_boxes) {
    const auto boxes = make_shared<op::v0::Parameter>(element::f32, Shape{1, 2, 3});
    const auto scores = make_shared<op::v0::Parameter>(element::f32, Shape{1, 2, 3});

    OV_EXPECT_THROW(ignore = this->make_op(boxes, scores),
                    NodeValidationFailure,
                    HasSubstr("'boxes' and 'scores' input shapes must match at the second and third "
                              "dimension respectively"));
}

REGISTER_TYPED_TEST_SUITE_P(NMSCommonTest,
                            incorrect_boxes_rank,
                            incorrect_scores_rank,
                            incorrect_scheme_num_batches,
                            incorrect_scheme_num_boxes);

using NMSCommonTypes = testing::Types<op::v1::NonMaxSuppression,
                                      op::v3::NonMaxSuppression,
                                      op::v4::NonMaxSuppression,
                                      op::v5::NonMaxSuppression,
                                      op::v9::NonMaxSuppression>;
INSTANTIATE_TYPED_TEST_SUITE_P(type_prop, NMSCommonTest, NMSCommonTypes);

template <class TOp>
using NMSNonDynamicOutputTest = NMSCommonTest<TOp>;
TYPED_TEST_SUITE_P(NMSNonDynamicOutputTest);

TYPED_TEST_P(NMSNonDynamicOutputTest, scalar_inputs_check) {
    const auto boxes = make_shared<op::v0::Parameter>(element::f32, Shape{1, 2, 4});
    const auto scores = make_shared<op::v0::Parameter>(element::f32, Shape{1, 2, 2});

    const auto scalar = make_shared<op::v0::Parameter>(element::f32, Shape{});
    const auto non_scalar = make_shared<op::v0::Parameter>(element::f32, Shape{1});

    OV_EXPECT_THROW(ignore = this->make_op(boxes, scores, non_scalar, scalar, scalar),
                    NodeValidationFailure,
                    HasSubstr("Expected a scalar for the 'max_output_boxes_per_class' input"));

    OV_EXPECT_THROW(ignore = this->make_op(boxes, scores, scalar, non_scalar, scalar),
                    NodeValidationFailure,
                    HasSubstr("Expected a scalar for the 'iou_threshold' input"));

    OV_EXPECT_THROW(ignore = this->make_op(boxes, scores, scalar, scalar, non_scalar),
                    NodeValidationFailure,
                    HasSubstr("Expected a scalar for the 'score_threshold' input"));
}

TYPED_TEST_P(NMSNonDynamicOutputTest, static_boxes_and_scores_default_others) {
    const auto boxes = make_shared<op::v0::Parameter>(element::f32, Shape{1, 2, 4});
    const auto scores = make_shared<op::v0::Parameter>(element::f32, Shape{1, 2, 2});

    const auto op = this->make_op(boxes, scores);

    EXPECT_EQ(op->get_output_size(), 1);
    EXPECT_EQ(op->get_output_element_type(0), element::i64);
    EXPECT_EQ(op->get_output_partial_shape(0), PartialShape({0, 3}));
}

TYPED_TEST_P(NMSNonDynamicOutputTest, num_boxes_gt_max_out_boxes) {
    const auto boxes = make_shared<op::v0::Parameter>(element::f32, Shape{1, 6, 4});
    const auto scores = make_shared<op::v0::Parameter>(element::f32, Shape{1, 1, 6});
    const auto max_output_boxes_per_class = op::v0::Constant::create(element::i32, Shape{}, {3});
    const auto iou_threshold = make_shared<op::v0::Parameter>(element::f32, Shape{});
    const auto score_threshold = make_shared<op::v0::Parameter>(element::f32, Shape{});

    const auto op = this->make_op(boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold);

    EXPECT_EQ(op->get_output_size(), 1);
    EXPECT_EQ(op->get_output_element_type(0), element::i64);
    EXPECT_EQ(op->get_output_partial_shape(0), PartialShape({3, 3}));
}

TYPED_TEST_P(NMSNonDynamicOutputTest, num_boxes_lt_max_out_boxes) {
    const auto boxes = make_shared<op::v0::Parameter>(element::f32, Shape{1, 1, 4});
    const auto scores = make_shared<op::v0::Parameter>(element::f32, Shape{1, 1, 1});
    const auto max_output_boxes_per_class = op::v0::Constant::create(element::i16, Shape{}, {3});
    const auto iou_threshold = make_shared<op::v0::Parameter>(element::f32, Shape{});
    const auto score_threshold = make_shared<op::v0::Parameter>(element::f32, Shape{});

    const auto op = this->make_op(boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold);

    EXPECT_EQ(op->get_output_size(), 1);
    EXPECT_EQ(op->get_output_element_type(0), element::i64);
    EXPECT_EQ(op->get_output_partial_shape(0), PartialShape({1, 3}));
}

TYPED_TEST_P(NMSNonDynamicOutputTest, interval_shapes_boxes_and_scores_with_symbols) {
    auto boxes_shape = PartialShape{{1, 2}, {2, 5}, {0, 4}};
    auto scores_shape = PartialShape{{0, 2}, {1, 2}, {0, 6}};
    set_shape_symbols(boxes_shape);
    set_shape_symbols(scores_shape);

    const auto boxes = make_shared<op::v0::Parameter>(element::f32, boxes_shape);
    const auto scores = make_shared<op::v0::Parameter>(element::f32, scores_shape);
    const auto max_output_boxes_per_class = op::v0::Constant::create(element::i16, Shape{}, {3});
    const auto iou_threshold = make_shared<op::v0::Parameter>(element::f32, Shape{});
    const auto score_threshold = make_shared<op::v0::Parameter>(element::f32, Shape{});

    const auto op = this->make_op(boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold);
    EXPECT_EQ(op->get_output_size(), 1);
    EXPECT_EQ(op->get_output_element_type(0), element::i64);
    EXPECT_EQ(op->get_output_partial_shape(0), PartialShape({-1, 3}));
    EXPECT_THAT(get_shape_symbols(op->get_output_partial_shape(0)), Each(nullptr));
}

TYPED_TEST_P(NMSNonDynamicOutputTest, interval_boxes_scores_dynamic_rank) {
    auto boxes_shape = PartialShape{{1, 2}, {2, 5}, {0, 4}};
    set_shape_symbols(boxes_shape);

    const auto boxes = make_shared<op::v0::Parameter>(element::f32, boxes_shape);
    const auto scores = make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());
    const auto max_output_boxes_per_class = op::v0::Constant::create(element::i16, Shape{}, {3});
    const auto iou_threshold = make_shared<op::v0::Parameter>(element::f32, Shape{});
    const auto score_threshold = make_shared<op::v0::Parameter>(element::f32, Shape{});

    const auto op = this->make_op(boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold);
    EXPECT_EQ(op->get_output_size(), 1);
    EXPECT_EQ(op->get_output_element_type(0), element::i64);
    EXPECT_EQ(op->get_output_partial_shape(0), PartialShape({-1, 3}));
    EXPECT_THAT(get_shape_symbols(op->get_output_partial_shape(0)), Each(nullptr));
}

TYPED_TEST_P(NMSNonDynamicOutputTest, dynamic_rank_boxes_and_scores) {
    const auto boxes = make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());
    const auto scores = make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());
    const auto max_output_boxes_per_class = op::v0::Constant::create(element::i16, Shape{}, {3});
    const auto iou_threshold = make_shared<op::v0::Parameter>(element::f32, Shape{});
    const auto score_threshold = make_shared<op::v0::Parameter>(element::f32, Shape{});

    const auto op = this->make_op(boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold);
    EXPECT_EQ(op->get_output_size(), 1);
    EXPECT_EQ(op->get_output_element_type(0), element::i64);
    EXPECT_EQ(op->get_output_partial_shape(0), PartialShape({-1, 3}));
}

REGISTER_TYPED_TEST_SUITE_P(NMSNonDynamicOutputTest,
                            scalar_inputs_check,
                            static_boxes_and_scores_default_others,
                            num_boxes_gt_max_out_boxes,
                            num_boxes_lt_max_out_boxes,
                            interval_shapes_boxes_and_scores_with_symbols,
                            interval_boxes_scores_dynamic_rank,
                            dynamic_rank_boxes_and_scores);
using NMSNonDynamicOutputTypes =
    testing::Types<op::v1::NonMaxSuppression, op::v3::NonMaxSuppression, op::v4::NonMaxSuppression>;
INSTANTIATE_TYPED_TEST_SUITE_P(type_prop, NMSNonDynamicOutputTest, NMSNonDynamicOutputTypes);

// ------------------------------ V3 ------------------------------
TEST(type_prop, nms_v3_output_shape_i32) {
    const auto boxes = make_shared<op::v0::Parameter>(element::f32, Shape{1, 1, 4});
    const auto scores = make_shared<op::v0::Parameter>(element::f32, Shape{1, 1, 1});
    const auto max_output_boxes_per_class = op::v0::Constant::create(element::i16, Shape{}, {3});
    const auto iou_threshold = make_shared<op::v0::Parameter>(element::f32, Shape{});
    const auto score_threshold = make_shared<op::v0::Parameter>(element::f32, Shape{});

    const auto nms = make_shared<op::v3::NonMaxSuppression>(boxes,
                                                            scores,
                                                            max_output_boxes_per_class,
                                                            iou_threshold,
                                                            score_threshold,
                                                            op::v3::NonMaxSuppression::BoxEncodingType::CORNER,
                                                            true,
                                                            element::i32);

    ASSERT_EQ(nms->get_element_type(), element::i32);
    ASSERT_EQ(nms->get_shape(), (Shape{1, 3}));
}

// ------------------------------ V4 ------------------------------
TEST(type_prop, nms_v4_output_shape_i32) {
    const auto boxes = make_shared<op::v0::Parameter>(element::f32, Shape{2, 7, 4});
    const auto scores = make_shared<op::v0::Parameter>(element::f32, Shape{2, 5, 7});
    const auto max_output_boxes_per_class = op::v0::Constant::create(element::i16, Shape{}, {3});
    const auto iou_threshold = make_shared<op::v0::Parameter>(element::f32, Shape{});
    const auto score_threshold = make_shared<op::v0::Parameter>(element::f32, Shape{});

    const auto nms = make_shared<op::v4::NonMaxSuppression>(boxes,
                                                            scores,
                                                            max_output_boxes_per_class,
                                                            iou_threshold,
                                                            score_threshold,
                                                            op::v3::NonMaxSuppression::BoxEncodingType::CORNER,
                                                            true,
                                                            element::i32);

    ASSERT_EQ(nms->get_element_type(), element::i32);
    ASSERT_EQ(nms->get_shape(), (Shape{30, 3}));
}

template <class TOp>
using NMSDynamicOutputTest = NMSCommonTest<TOp>;
TYPED_TEST_SUITE_P(NMSDynamicOutputTest);

TYPED_TEST_P(NMSDynamicOutputTest, scalar_inputs_check) {
    const auto boxes = make_shared<op::v0::Parameter>(element::f32, Shape{1, 2, 4});
    const auto scores = make_shared<op::v0::Parameter>(element::f32, Shape{1, 2, 2});

    const auto scalar = make_shared<op::v0::Parameter>(element::f32, Shape{});
    const auto non_0d_or_1d = make_shared<op::v0::Parameter>(element::f32, Shape{2});

    OV_EXPECT_THROW(ignore = this->make_op(boxes, scores, non_0d_or_1d, scalar, scalar),
                    NodeValidationFailure,
                    HasSubstr("Expected 0D or 1D tensor for the 'max_output_boxes_per_class' input"));

    OV_EXPECT_THROW(ignore = this->make_op(boxes, scores, scalar, non_0d_or_1d, scalar),
                    NodeValidationFailure,
                    HasSubstr("Expected 0D or 1D tensor for the 'iou_threshold' input"));

    OV_EXPECT_THROW(ignore = this->make_op(boxes, scores, scalar, scalar, non_0d_or_1d),
                    NodeValidationFailure,
                    HasSubstr("Expected 0D or 1D tensor for the 'score_threshold' input"));

    OV_EXPECT_THROW(ignore = this->make_op(boxes, scores, scalar, scalar, scalar, non_0d_or_1d),
                    NodeValidationFailure,
                    HasSubstr("Expected 0D or 1D tensor for the 'soft_nms_sigma' input"));
}

TYPED_TEST_P(NMSDynamicOutputTest, boxes_scores_static_other_defaults) {
    const auto boxes = make_shared<op::v0::Parameter>(element::f32, Shape{5, 2, 4});
    const auto scores = make_shared<op::v0::Parameter>(element::f32, Shape{5, 3, 2});
    const auto op = this->make_op(boxes, scores);

    EXPECT_THAT(op->outputs(),
                ElementsAre(Property("Indicies type", &Output<Node>::get_element_type, element::i64),
                            Property("Scores type", &Output<Node>::get_element_type, element::f32),
                            Property("Outputs type", &Output<Node>::get_element_type, element::i64)));
    EXPECT_THAT(op->outputs(),
                ElementsAre(Property("Indicies shape", &Output<Node>::get_partial_shape, PartialShape({-1, 3})),
                            Property("Scores shape", &Output<Node>::get_partial_shape, PartialShape({-1, 3})),
                            Property("Outputs shape", &Output<Node>::get_partial_shape, PartialShape({1}))));
}

TYPED_TEST_P(NMSDynamicOutputTest, num_boxes_gt_max_out_boxes) {
    auto boxes_shape = PartialShape{2, 7, 4};
    auto scores_shape = PartialShape{2, 5, 7};
    set_shape_symbols(boxes_shape);
    set_shape_symbols(scores_shape);

    const auto boxes = make_shared<op::v0::Parameter>(element::f32, boxes_shape);
    const auto scores = make_shared<op::v0::Parameter>(element::f32, scores_shape);
    const auto max_output_boxes_per_class = op::v0::Constant::create(element::i32, Shape{}, {3});
    const auto iou_threshold = make_shared<op::v0::Parameter>(element::f32, Shape{});
    const auto score_threshold = make_shared<op::v0::Parameter>(element::f32, Shape{});

    const auto op = this->make_op(boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold);

    EXPECT_THAT(op->outputs(),
                ElementsAre(Property("Indicies type", &Output<Node>::get_element_type, element::i64),
                            Property("Scores type", &Output<Node>::get_element_type, element::f32),
                            Property("Outputs type", &Output<Node>::get_element_type, element::i64)));
    EXPECT_THAT(op->outputs(),
                ElementsAre(Property("Indicies shape", &Output<Node>::get_partial_shape, PartialShape({{0, 30}, 3})),
                            Property("Scores shape", &Output<Node>::get_partial_shape, PartialShape({{0, 30}, 3})),
                            Property("Outputs shape", &Output<Node>::get_partial_shape, PartialShape({1}))));
    EXPECT_THAT(get_shape_symbols(op->get_output_partial_shape(0)), Each(nullptr));
    EXPECT_THAT(get_shape_symbols(op->get_output_partial_shape(1)), Each(nullptr));
    EXPECT_THAT(get_shape_symbols(op->get_output_partial_shape(2)), Each(nullptr));
}

TYPED_TEST_P(NMSDynamicOutputTest, num_boxes_lt_max_out_boxes) {
    const auto boxes = make_shared<op::v0::Parameter>(element::f32, Shape{2, 7, 4});
    const auto scores = make_shared<op::v0::Parameter>(element::f32, Shape{2, 5, 7});
    const auto max_output_boxes_per_class = op::v0::Constant::create(element::i16, Shape{}, {1000});
    const auto iou_threshold = make_shared<op::v0::Parameter>(element::f32, Shape{});
    const auto score_threshold = make_shared<op::v0::Parameter>(element::f32, Shape{});

    const auto op = this->make_op(boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold);

    EXPECT_THAT(op->outputs(),
                ElementsAre(Property("Indicies type", &Output<Node>::get_element_type, element::i64),
                            Property("Scores type", &Output<Node>::get_element_type, element::f32),
                            Property("Outputs type", &Output<Node>::get_element_type, element::i64)));
    EXPECT_THAT(op->outputs(),
                ElementsAre(Property("Indicies shape", &Output<Node>::get_partial_shape, PartialShape({{0, 70}, 3})),
                            Property("Scores shape", &Output<Node>::get_partial_shape, PartialShape({{0, 70}, 3})),
                            Property("Outputs shape", &Output<Node>::get_partial_shape, PartialShape({1}))));
}

TYPED_TEST_P(NMSDynamicOutputTest, max_out_boxes_is_zero) {
    const auto boxes = make_shared<op::v0::Parameter>(element::f32, Shape{2, 7, 4});
    const auto scores = make_shared<op::v0::Parameter>(element::f32, Shape{2, 5, 7});
    const auto max_output_boxes_per_class = op::v0::Constant::create(element::i16, Shape{}, {0});
    const auto iou_threshold = make_shared<op::v0::Parameter>(element::f32, Shape{});
    const auto score_threshold = make_shared<op::v0::Parameter>(element::f32, Shape{});

    const auto op = this->make_op(boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold);

    EXPECT_THAT(op->outputs(),
                ElementsAre(Property("Indicies type", &Output<Node>::get_element_type, element::i64),
                            Property("Scores type", &Output<Node>::get_element_type, element::f32),
                            Property("Outputs type", &Output<Node>::get_element_type, element::i64)));
    EXPECT_THAT(op->outputs(),
                ElementsAre(Property("Indicies shape", &Output<Node>::get_partial_shape, PartialShape({0, 3})),
                            Property("Scores shape", &Output<Node>::get_partial_shape, PartialShape({0, 3})),
                            Property("Outputs shape", &Output<Node>::get_partial_shape, PartialShape({1}))));
}

TYPED_TEST_P(NMSDynamicOutputTest, interval_shapes_symbols) {
    auto boxes_shape = PartialShape{{0, 2}, {0, 7}, 4};
    auto scores_shape = PartialShape{{0, 2}, {0, 5}, {1, 7}};
    set_shape_symbols(boxes_shape);
    set_shape_symbols(scores_shape);

    const auto boxes = make_shared<op::v0::Parameter>(element::f32, boxes_shape);
    const auto scores = make_shared<op::v0::Parameter>(element::f32, scores_shape);
    const auto max_output_boxes_per_class = op::v0::Constant::create(element::i16, Shape{}, {1000});
    const auto iou_threshold = make_shared<op::v0::Parameter>(element::f32, Shape{});
    const auto score_threshold = make_shared<op::v0::Parameter>(element::f32, Shape{});

    const auto op = this->make_op(boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold);

    EXPECT_THAT(op->outputs(),
                ElementsAre(Property("Indicies type", &Output<Node>::get_element_type, element::i64),
                            Property("Scores type", &Output<Node>::get_element_type, element::f32),
                            Property("Outputs type", &Output<Node>::get_element_type, element::i64)));
    EXPECT_THAT(op->outputs(),
                ElementsAre(Property("Indicies shape", &Output<Node>::get_partial_shape, PartialShape({{0, 70}, 3})),
                            Property("Scores shape", &Output<Node>::get_partial_shape, PartialShape({{0, 70}, 3})),
                            Property("Outputs shape", &Output<Node>::get_partial_shape, PartialShape({1}))));
    EXPECT_THAT(get_shape_symbols(op->get_output_partial_shape(0)), Each(nullptr));
    EXPECT_THAT(get_shape_symbols(op->get_output_partial_shape(1)), Each(nullptr));
    EXPECT_THAT(get_shape_symbols(op->get_output_partial_shape(2)), Each(nullptr));
}

TYPED_TEST_P(NMSDynamicOutputTest, num_box_dynamic_dim_max_boxes_per_class_as_const) {
    auto boxes_shape = PartialShape{2, -1, 4};
    auto scores_shape = PartialShape{2, {0, 5}, {1, 7}};
    set_shape_symbols(boxes_shape);
    set_shape_symbols(scores_shape);

    const auto boxes = make_shared<op::v0::Parameter>(element::f32, boxes_shape);
    const auto scores = make_shared<op::v0::Parameter>(element::f32, scores_shape);
    const auto max_output_boxes_per_class = op::v0::Constant::create(element::i16, Shape{}, {5});
    const auto iou_threshold = make_shared<op::v0::Parameter>(element::f32, Shape{});
    const auto score_threshold = make_shared<op::v0::Parameter>(element::f32, Shape{});

    const auto op = this->make_op(boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold);

    EXPECT_THAT(op->outputs(),
                ElementsAre(Property("Indicies shape", &Output<Node>::get_partial_shape, PartialShape({-1, 3})),
                            Property("Scores shape", &Output<Node>::get_partial_shape, PartialShape({-1, 3})),
                            Property("Outputs shape", &Output<Node>::get_partial_shape, PartialShape({1}))));
    EXPECT_THAT(get_shape_symbols(op->get_output_partial_shape(0)), Each(nullptr));
    EXPECT_THAT(get_shape_symbols(op->get_output_partial_shape(1)), Each(nullptr));
    EXPECT_THAT(get_shape_symbols(op->get_output_partial_shape(2)), Each(nullptr));
}

TYPED_TEST_P(NMSDynamicOutputTest, output_shape_i32) {
    const auto boxes = make_shared<op::v0::Parameter>(element::f32, Shape{2, 7, 4});
    const auto scores = make_shared<op::v0::Parameter>(element::f32, Shape{2, 5, 7});
    const auto max_output_boxes_per_class = op::v0::Constant::create(element::i16, Shape{}, {3});
    const auto iou_threshold = make_shared<op::v0::Parameter>(element::f32, Shape{});
    const auto score_threshold = make_shared<op::v0::Parameter>(element::f32, Shape{});

    const auto op = this->make_op(boxes,
                                  scores,
                                  max_output_boxes_per_class,
                                  iou_threshold,
                                  score_threshold,
                                  typename TypeParam::BoxEncodingType(0),
                                  true,
                                  element::i32);

    EXPECT_THAT(op->outputs(),
                ElementsAre(Property("Indicies type", &Output<Node>::get_element_type, element::i32),
                            Property("Scores type", &Output<Node>::get_element_type, element::f32),
                            Property("Outputs type", &Output<Node>::get_element_type, element::i32)));
    EXPECT_THAT(op->outputs(),
                ElementsAre(Property("Indicies shape", &Output<Node>::get_partial_shape, PartialShape({{0, 30}, 3})),
                            Property("Scores shape", &Output<Node>::get_partial_shape, PartialShape({{0, 30}, 3})),
                            Property("Outputs shape", &Output<Node>::get_partial_shape, PartialShape({1}))));
}

TYPED_TEST_P(NMSDynamicOutputTest, dynamic_boxes_and_scores) {
    const auto boxes = make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());
    const auto scores = make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());
    const auto max_output_boxes_per_class = op::v0::Constant::create(element::i16, Shape{}, {3});
    const auto iou_threshold = make_shared<op::v0::Parameter>(element::f32, Shape{});
    const auto score_threshold = make_shared<op::v0::Parameter>(element::f32, Shape{});

    const auto op = this->make_op(boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold);

    EXPECT_THAT(op->outputs(),
                ElementsAre(Property("Indicies type", &Output<Node>::get_element_type, element::i64),
                            Property("Scores type", &Output<Node>::get_element_type, element::f32),
                            Property("Outputs type", &Output<Node>::get_element_type, element::i64)));
    EXPECT_THAT(op->outputs(),
                ElementsAre(Property("Indicies shape", &Output<Node>::get_partial_shape, PartialShape({-1, 3})),
                            Property("Scores shape", &Output<Node>::get_partial_shape, PartialShape({-1, 3})),
                            Property("Outputs shape", &Output<Node>::get_partial_shape, PartialShape({1}))));
}

TYPED_TEST_P(NMSDynamicOutputTest, dynamic_types) {
    const auto boxes = make_shared<op::v0::Parameter>(element::dynamic, Shape{5, 2, 4});
    const auto scores = make_shared<op::v0::Parameter>(element::dynamic, Shape{5, 3, 2});

    const auto op = this->make_op(boxes, scores);

    EXPECT_THAT(op->outputs(),
                ElementsAre(Property("Indicies type", &Output<Node>::get_element_type, element::i64),
                            Property("Scores type", &Output<Node>::get_element_type, element::f32),
                            Property("Outputs type", &Output<Node>::get_element_type, element::i64)));
    EXPECT_THAT(op->outputs(),
                ElementsAre(Property("Indicies shape", &Output<Node>::get_partial_shape, PartialShape({-1, 3})),
                            Property("Scores shape", &Output<Node>::get_partial_shape, PartialShape({-1, 3})),
                            Property("Outputs shape", &Output<Node>::get_partial_shape, PartialShape({1}))));
}

TYPED_TEST_P(NMSDynamicOutputTest, scores_shape_is_dynamic_rank) {
    const auto boxes = make_shared<op::v0::Parameter>(element::dynamic, Shape{5, 2, 4});
    const auto scores = make_shared<op::v0::Parameter>(element::dynamic, PartialShape::dynamic());
    const auto max_output_boxes_per_class = op::v0::Constant::create(element::i16, Shape{}, {3});
    const auto iou_threshold = make_shared<op::v0::Parameter>(element::f32, Shape{});
    const auto score_threshold = make_shared<op::v0::Parameter>(element::f32, Shape{});

    const auto op = this->make_op(boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold);

    EXPECT_THAT(op->outputs(),
                ElementsAre(Property("Indicies type", &Output<Node>::get_element_type, element::i64),
                            Property("Scores type", &Output<Node>::get_element_type, element::f32),
                            Property("Outputs type", &Output<Node>::get_element_type, element::i64)));
    EXPECT_THAT(op->outputs(),
                ElementsAre(Property("Indicies shape", &Output<Node>::get_partial_shape, PartialShape({-1, 3})),
                            Property("Scores shape", &Output<Node>::get_partial_shape, PartialShape({-1, 3})),
                            Property("Outputs shape", &Output<Node>::get_partial_shape, PartialShape({1}))));
}

REGISTER_TYPED_TEST_SUITE_P(NMSDynamicOutputTest,
                            scalar_inputs_check,
                            boxes_scores_static_other_defaults,
                            num_boxes_gt_max_out_boxes,
                            num_boxes_lt_max_out_boxes,
                            max_out_boxes_is_zero,
                            interval_shapes_symbols,
                            num_box_dynamic_dim_max_boxes_per_class_as_const,
                            output_shape_i32,
                            dynamic_boxes_and_scores,
                            dynamic_types,
                            scores_shape_is_dynamic_rank);
using NMSDynamicOutputTypes = testing::Types<op::v5::NonMaxSuppression, op::v9::NonMaxSuppression>;
INSTANTIATE_TYPED_TEST_SUITE_P(type_prop, NMSDynamicOutputTest, NMSDynamicOutputTypes);
