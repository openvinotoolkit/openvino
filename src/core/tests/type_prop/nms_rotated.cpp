// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/nms_rotated.hpp"

#include "common_test_utils/test_assertions.hpp"
#include "common_test_utils/type_prop.hpp"
#include "openvino/op/constant.hpp"

using namespace std;
using namespace ov;
using namespace testing;

template <class TOp>
class NMSRotatedCommonTest : public TypePropOpTest<TOp> {};

TYPED_TEST_SUITE_P(NMSRotatedCommonTest);

TYPED_TEST_P(NMSRotatedCommonTest, incorrect_boxes_rank) {
    const auto boxes = make_shared<op::v0::Parameter>(element::f32, Shape{1, 2, 5, 4});
    const auto scores = make_shared<op::v0::Parameter>(element::f32, Shape{1, 2, 3});
    const auto scalar_int = make_shared<op::v0::Parameter>(element::i32, Shape{});
    const auto scalar_fp = make_shared<op::v0::Parameter>(element::f32, Shape{});

    OV_EXPECT_THROW(ignore = this->make_op(boxes, scores, scalar_int, scalar_fp, scalar_fp),
                    NodeValidationFailure,
                    HasSubstr("Expected a 3D tensor for the 'boxes' input"));
}

TYPED_TEST_P(NMSRotatedCommonTest, incorrect_scores_rank) {
    const auto boxes = make_shared<op::v0::Parameter>(element::f32, Shape{1, 2, 5});
    const auto scores = make_shared<op::v0::Parameter>(element::f32, Shape{1, 2});
    const auto scalar_int = make_shared<op::v0::Parameter>(element::i32, Shape{});
    const auto scalar_fp = make_shared<op::v0::Parameter>(element::f32, Shape{});

    OV_EXPECT_THROW(ignore = this->make_op(boxes, scores, scalar_int, scalar_fp, scalar_fp),
                    NodeValidationFailure,
                    HasSubstr("Expected a 3D tensor for the 'scores' input"));
}

TYPED_TEST_P(NMSRotatedCommonTest, incorrect_scheme_num_batches) {
    const auto boxes = make_shared<op::v0::Parameter>(element::f32, Shape{1, 2, 5});
    const auto scores = make_shared<op::v0::Parameter>(element::f32, Shape{2, 2, 3});
    const auto scalar_int = make_shared<op::v0::Parameter>(element::i32, Shape{});
    const auto scalar_fp = make_shared<op::v0::Parameter>(element::f32, Shape{});

    OV_EXPECT_THROW(ignore = this->make_op(boxes, scores, scalar_int, scalar_fp, scalar_fp),
                    NodeValidationFailure,
                    HasSubstr("The first dimension of both 'boxes' and 'scores' must match"));
}

TYPED_TEST_P(NMSRotatedCommonTest, incorrect_scheme_num_boxes) {
    const auto boxes = make_shared<op::v0::Parameter>(element::f32, Shape{1, 2, 5});
    const auto scores = make_shared<op::v0::Parameter>(element::f32, Shape{1, 2, 3});
    const auto scalar_int = make_shared<op::v0::Parameter>(element::i32, Shape{});
    const auto scalar_fp = make_shared<op::v0::Parameter>(element::f32, Shape{});

    OV_EXPECT_THROW(ignore = this->make_op(boxes, scores, scalar_int, scalar_fp, scalar_fp),
                    NodeValidationFailure,
                    HasSubstr("'boxes' and 'scores' input shapes must match at the second and third "
                              "dimension respectively"));
}

TYPED_TEST_P(NMSRotatedCommonTest, incorrect_boxes_last_dim) {
    const auto boxes = make_shared<op::v0::Parameter>(element::f32, Shape{1, 2, 3});
    const auto scores = make_shared<op::v0::Parameter>(element::f32, Shape{1, 2, 2});
    const auto scalar_int = make_shared<op::v0::Parameter>(element::i32, Shape{});
    const auto scalar_fp = make_shared<op::v0::Parameter>(element::f32, Shape{});

    OV_EXPECT_THROW(ignore = this->make_op(boxes, scores, scalar_int, scalar_fp, scalar_fp),
                    NodeValidationFailure,
                    HasSubstr("The last dimension of the 'boxes' input must be equal to 5"));
}

TYPED_TEST_P(NMSRotatedCommonTest, input_types_check) {
    const auto param_fp = make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());
    const auto param_int = make_shared<op::v0::Parameter>(element::i32, PartialShape::dynamic());

    OV_EXPECT_THROW(ignore = this->make_op(param_int, param_fp, param_int, param_fp, param_fp),
                    NodeValidationFailure,
                    HasSubstr("Expected floating point type as element type for the input at: 0"));

    OV_EXPECT_THROW(ignore = this->make_op(param_fp, param_int, param_int, param_fp, param_fp),
                    NodeValidationFailure,
                    HasSubstr("Expected floating point type as element type for the input at: 1"));

    OV_EXPECT_THROW(ignore = this->make_op(param_fp, param_fp, param_fp, param_fp, param_fp),
                    NodeValidationFailure,
                    HasSubstr("Expected integer type as element type for the input at: 2"));

    OV_EXPECT_THROW(ignore = this->make_op(param_fp, param_fp, param_int, param_int, param_fp),
                    NodeValidationFailure,
                    HasSubstr("Expected floating point type as element type for the input at: 3"));

    OV_EXPECT_THROW(ignore = this->make_op(param_fp, param_fp, param_int, param_fp, param_int),
                    NodeValidationFailure,
                    HasSubstr("Expected floating point type as element type for the input at: 4"));
}

TYPED_TEST_P(NMSRotatedCommonTest, output_type_attr_check) {
    const auto param_fp = make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());
    const auto param_int = make_shared<op::v0::Parameter>(element::i32, PartialShape::dynamic());

    OV_EXPECT_THROW(
        ignore = this->make_op(param_fp, param_fp, param_int, param_fp, param_fp, true, element::f16),
        NodeValidationFailure,
        HasSubstr("The `output_type` attribute (related to the first and third output) must be i32 or i64"));
}

REGISTER_TYPED_TEST_SUITE_P(NMSRotatedCommonTest,
                            incorrect_boxes_rank,
                            incorrect_scores_rank,
                            incorrect_scheme_num_batches,
                            incorrect_scheme_num_boxes,
                            incorrect_boxes_last_dim,
                            input_types_check,
                            output_type_attr_check);

using NMSRotatedCommonTypes = testing::Types<op::v13::NMSRotated>;
INSTANTIATE_TYPED_TEST_SUITE_P(type_prop, NMSRotatedCommonTest, NMSRotatedCommonTypes);

template <class TOp>
using NMSRotatedDynamicOutputTest = NMSRotatedCommonTest<TOp>;
TYPED_TEST_SUITE_P(NMSRotatedDynamicOutputTest);

TYPED_TEST_P(NMSRotatedDynamicOutputTest, scalar_inputs_check) {
    const auto boxes = make_shared<op::v0::Parameter>(element::f32, Shape{1, 2, 5});
    const auto scores = make_shared<op::v0::Parameter>(element::f32, Shape{1, 2, 2});
    const auto max_output_boxes_per_class = op::v0::Constant::create(element::i32, Shape{}, {1000});

    const auto scalar_fp = make_shared<op::v0::Parameter>(element::f32, Shape{});
    const auto non_0d_or_1d = make_shared<op::v0::Parameter>(element::f32, Shape{2});

    OV_EXPECT_THROW(ignore = this->make_op(boxes, scores, non_0d_or_1d, scalar_fp, scalar_fp),
                    NodeValidationFailure,
                    HasSubstr("Expected 0D or 1D tensor for the 'max_output_boxes_per_class' input"));

    OV_EXPECT_THROW(ignore = this->make_op(boxes, scores, max_output_boxes_per_class, non_0d_or_1d, scalar_fp),
                    NodeValidationFailure,
                    HasSubstr("Expected 0D or 1D tensor for the 'iou_threshold' input"));

    OV_EXPECT_THROW(ignore = this->make_op(boxes, scores, max_output_boxes_per_class, scalar_fp, non_0d_or_1d),
                    NodeValidationFailure,
                    HasSubstr("Expected 0D or 1D tensor for the 'score_threshold' input"));
}

TYPED_TEST_P(NMSRotatedDynamicOutputTest, boxes_scores_static_max_out_param) {
    const auto boxes = make_shared<op::v0::Parameter>(element::f32, Shape{5, 2, 5});
    const auto scores = make_shared<op::v0::Parameter>(element::f32, Shape{5, 3, 2});
    const auto scalar_int = make_shared<op::v0::Parameter>(element::i32, Shape{});
    const auto scalar_fp = op::v0::Constant::create(element::f32, Shape{}, {0.5});

    const auto op = this->make_op(boxes, scores, scalar_int, scalar_fp, scalar_fp);

    EXPECT_THAT(op->outputs(),
                ElementsAre(Property("Indicies type", &Output<Node>::get_element_type, element::i64),
                            Property("Scores type", &Output<Node>::get_element_type, element::f32),
                            Property("Outputs type", &Output<Node>::get_element_type, element::i64)));
    EXPECT_THAT(op->outputs(),
                ElementsAre(Property("Indicies shape", &Output<Node>::get_partial_shape, PartialShape({-1, 3})),
                            Property("Scores shape", &Output<Node>::get_partial_shape, PartialShape({-1, 3})),
                            Property("Outputs shape", &Output<Node>::get_partial_shape, PartialShape({1}))));
}

TYPED_TEST_P(NMSRotatedDynamicOutputTest, num_boxes_gt_max_out_boxes) {
    auto boxes_shape = PartialShape{2, 7, 5};
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

TYPED_TEST_P(NMSRotatedDynamicOutputTest, num_boxes_lt_max_out_boxes) {
    const auto boxes = make_shared<op::v0::Parameter>(element::f32, Shape{2, 7, 5});
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

TYPED_TEST_P(NMSRotatedDynamicOutputTest, max_out_boxes_is_zero) {
    const auto boxes = make_shared<op::v0::Parameter>(element::f32, Shape{2, 7, 5});
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

TYPED_TEST_P(NMSRotatedDynamicOutputTest, interval_shapes_labels) {
    auto boxes_shape = PartialShape{{0, 2}, {0, 7}, 5};
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

TYPED_TEST_P(NMSRotatedDynamicOutputTest, num_box_dynamic_dim_max_boxes_per_class_as_const) {
    auto boxes_shape = PartialShape{2, -1, 5};
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

TYPED_TEST_P(NMSRotatedDynamicOutputTest, output_shape_i32) {
    const auto boxes = make_shared<op::v0::Parameter>(element::f32, Shape{2, 7, 5});
    const auto scores = make_shared<op::v0::Parameter>(element::f32, Shape{2, 5, 7});
    const auto max_output_boxes_per_class = op::v0::Constant::create(element::i16, Shape{}, {3});
    const auto iou_threshold = make_shared<op::v0::Parameter>(element::f32, Shape{});
    const auto score_threshold = make_shared<op::v0::Parameter>(element::f32, Shape{});

    const auto op =
        this->make_op(boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold, true, element::i32);

    EXPECT_THAT(op->outputs(),
                ElementsAre(Property("Indicies type", &Output<Node>::get_element_type, element::i32),
                            Property("Scores type", &Output<Node>::get_element_type, element::f32),
                            Property("Outputs type", &Output<Node>::get_element_type, element::i32)));
    EXPECT_THAT(op->outputs(),
                ElementsAre(Property("Indicies shape", &Output<Node>::get_partial_shape, PartialShape({{0, 30}, 3})),
                            Property("Scores shape", &Output<Node>::get_partial_shape, PartialShape({{0, 30}, 3})),
                            Property("Outputs shape", &Output<Node>::get_partial_shape, PartialShape({1}))));
}

TYPED_TEST_P(NMSRotatedDynamicOutputTest, dynamic_boxes_and_scores) {
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

TYPED_TEST_P(NMSRotatedDynamicOutputTest, dynamic_types) {
    const auto boxes = make_shared<op::v0::Parameter>(element::dynamic, Shape{5, 2, 5});
    const auto scores = make_shared<op::v0::Parameter>(element::dynamic, Shape{5, 3, 2});
    const auto scalar_int = make_shared<op::v0::Parameter>(element::i32, Shape{});
    const auto scalar_fp = make_shared<op::v0::Parameter>(element::f32, Shape{});

    const auto op = this->make_op(boxes, scores, scalar_int, scalar_fp, scalar_fp);
    EXPECT_THAT(op->outputs(),
                ElementsAre(Property("Indicies type", &Output<Node>::get_element_type, element::i64),
                            Property("Scores type", &Output<Node>::get_element_type, element::f32),
                            Property("Outputs type", &Output<Node>::get_element_type, element::i64)));
    EXPECT_THAT(op->outputs(),
                ElementsAre(Property("Indicies shape", &Output<Node>::get_partial_shape, PartialShape({-1, 3})),
                            Property("Scores shape", &Output<Node>::get_partial_shape, PartialShape({-1, 3})),
                            Property("Outputs shape", &Output<Node>::get_partial_shape, PartialShape({1}))));
}

TYPED_TEST_P(NMSRotatedDynamicOutputTest, scores_shape_is_dynamic_rank) {
    const auto boxes = make_shared<op::v0::Parameter>(element::dynamic, Shape{5, 2, 5});
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

REGISTER_TYPED_TEST_SUITE_P(NMSRotatedDynamicOutputTest,
                            scalar_inputs_check,
                            boxes_scores_static_max_out_param,
                            num_boxes_gt_max_out_boxes,
                            num_boxes_lt_max_out_boxes,
                            max_out_boxes_is_zero,
                            interval_shapes_labels,
                            num_box_dynamic_dim_max_boxes_per_class_as_const,
                            output_shape_i32,
                            dynamic_boxes_and_scores,
                            dynamic_types,
                            scores_shape_is_dynamic_rank);

using NMSRotatedDynamicOutputTypes = testing::Types<op::v13::NMSRotated>;
INSTANTIATE_TYPED_TEST_SUITE_P(type_prop, NMSRotatedDynamicOutputTest, NMSRotatedDynamicOutputTypes);
