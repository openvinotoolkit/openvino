// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "util/type_prop.hpp"

using namespace std;
using namespace ngraph;

TEST(type_prop, matrix_nms_incorrect_boxes_rank)
{
    try
    {
        const auto boxes = make_shared<op::Parameter>(element::f32, Shape{1, 2, 3, 4});
        const auto scores = make_shared<op::Parameter>(element::f32, Shape{1, 2, 3});

        make_shared<op::v8::MatrixNms>(boxes, scores);
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "Expected a 3D tensor for the 'boxes' input");
    }
}

TEST(type_prop, matrix_nms_incorrect_scores_rank)
{
    try
    {
        const auto boxes = make_shared<op::Parameter>(element::f32, Shape{1, 2, 3});
        const auto scores = make_shared<op::Parameter>(element::f32, Shape{1, 2});

        make_shared<op::v8::MatrixNms>(boxes, scores);
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "Expected a 3D tensor for the 'scores' input");
    }
}

TEST(type_prop, matrix_nms_incorrect_scheme_num_batches)
{
    try
    {
        const auto boxes = make_shared<op::Parameter>(element::f32, Shape{1, 2, 3});
        const auto scores = make_shared<op::Parameter>(element::f32, Shape{2, 2, 3});

        make_shared<op::v8::MatrixNms>(boxes, scores);
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             "The first dimension of both 'boxes' and 'scores' must match");
    }
}

TEST(type_prop, matrix_nms_incorrect_scheme_num_boxes)
{
    try
    {
        const auto boxes = make_shared<op::Parameter>(element::f32, Shape{1, 2, 3});
        const auto scores = make_shared<op::Parameter>(element::f32, Shape{1, 2, 3});

        make_shared<op::v8::MatrixNms>(boxes, scores);
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             "'boxes' and 'scores' input shapes must match at the second and third "
                             "dimension respectively");
    }
}

TEST(type_prop, matrix_nms_scalar_inputs_check)
{
    const auto boxes = make_shared<op::Parameter>(element::f32, Shape{1, 2, 4});
    const auto scores = make_shared<op::Parameter>(element::f32, Shape{1, 2, 2});

    const auto scalar = make_shared<op::Parameter>(element::f32, Shape{});
    const auto non_0d_or_1d = make_shared<op::Parameter>(element::f32, Shape{2});

    try
    {
        make_shared<op::v8::MatrixNms>(boxes, scores, non_0d_or_1d, scalar, scalar);
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             "Expected 0D or 1D tensor for the 'max_output_boxes_per_class' input");
    }

    try
    {
        make_shared<op::v8::MatrixNms>(boxes, scores, scalar, non_0d_or_1d, scalar);
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             "Expected 0D or 1D tensor for the 'iou_threshold' input");
    }

    try
    {
        make_shared<op::v8::MatrixNms>(boxes, scores, scalar, scalar, non_0d_or_1d);
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             "Expected 0D or 1D tensor for the 'score_threshold' input");
    }

    try
    {
        make_shared<op::v8::MatrixNms>(boxes, scores, scalar, scalar, scalar, non_0d_or_1d);
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             "Expected 0D or 1D tensor for the 'soft_nms_sigma' input");
    }
}

TEST(type_prop, matrix_nms_output_shape)
{
    const auto boxes = make_shared<op::Parameter>(element::f32, Shape{5, 2, 4});
    const auto scores = make_shared<op::Parameter>(element::f32, Shape{5, 3, 2});

    const auto nms = make_shared<op::v8::MatrixNms>(boxes, scores);

    ASSERT_TRUE(
        nms->get_output_partial_shape(0).same_scheme(PartialShape{Dimension::dynamic(), 3}));
    ASSERT_TRUE(
        nms->get_output_partial_shape(1).same_scheme(PartialShape{Dimension::dynamic(), 3}));

    EXPECT_EQ(nms->get_output_shape(2), (Shape{1}));
}

TEST(type_prop, matrix_nms_output_shape_2)
{
    const auto boxes = make_shared<op::Parameter>(element::f32, Shape{2, 7, 4});
    const auto scores = make_shared<op::Parameter>(element::f32, Shape{2, 5, 7});
    const auto max_output_boxes_per_class = op::Constant::create(element::i32, Shape{}, {3});
    const auto iou_threshold = make_shared<op::Parameter>(element::f32, Shape{});
    const auto score_threshold = make_shared<op::Parameter>(element::f32, Shape{});

    const auto nms = make_shared<op::v8::MatrixNms>(
        boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold);

    ASSERT_EQ(nms->get_output_element_type(0), element::i64);
    ASSERT_EQ(nms->get_output_element_type(1), element::f32);
    ASSERT_EQ(nms->get_output_element_type(2), element::i64);

    EXPECT_EQ(nms->get_output_partial_shape(0), PartialShape({Dimension(0, 30), Dimension(3)}));
    EXPECT_EQ(nms->get_output_partial_shape(1), PartialShape({Dimension(0, 30), Dimension(3)}));
    EXPECT_EQ(nms->get_output_shape(2), (Shape{1}));
}

TEST(type_prop, matrix_nms_output_shape_3)
{
    const auto boxes = make_shared<op::Parameter>(element::f32, Shape{2, 7, 4});
    const auto scores = make_shared<op::Parameter>(element::f32, Shape{2, 5, 7});
    const auto max_output_boxes_per_class = op::Constant::create(element::i16, Shape{}, {1000});
    const auto iou_threshold = make_shared<op::Parameter>(element::f32, Shape{});
    const auto score_threshold = make_shared<op::Parameter>(element::f32, Shape{});

    const auto nms = make_shared<op::v8::MatrixNms>(
        boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold);

    ASSERT_EQ(nms->get_output_element_type(0), element::i64);
    ASSERT_EQ(nms->get_output_element_type(1), element::f32);
    ASSERT_EQ(nms->get_output_element_type(2), element::i64);
    EXPECT_EQ(nms->get_output_partial_shape(0), PartialShape({Dimension(0, 70), Dimension(3)}));
    EXPECT_EQ(nms->get_output_partial_shape(1), PartialShape({Dimension(0, 70), Dimension(3)}));
    EXPECT_EQ(nms->get_output_shape(2), (Shape{1}));
}

TEST(type_prop, matrix_nms_output_shape_i32)
{
    const auto boxes = make_shared<op::Parameter>(element::f32, Shape{2, 7, 4});
    const auto scores = make_shared<op::Parameter>(element::f32, Shape{2, 5, 7});
    const auto max_output_boxes_per_class = op::Constant::create(element::i16, Shape{}, {3});
    const auto iou_threshold = make_shared<op::Parameter>(element::f32, Shape{});
    const auto score_threshold = make_shared<op::Parameter>(element::f32, Shape{});

    const auto nms =
        make_shared<op::v8::MatrixNms>(boxes,
                                               scores,
                                               max_output_boxes_per_class,
                                               iou_threshold,
                                               score_threshold,
                                               op::v8::MatrixNms::BoxEncodingType::CORNER,
                                               true,
                                               element::i32);

    ASSERT_EQ(nms->get_output_element_type(0), element::i32);
    ASSERT_EQ(nms->get_output_element_type(1), element::f32);
    ASSERT_EQ(nms->get_output_element_type(2), element::i32);

    EXPECT_EQ(nms->get_output_partial_shape(0), PartialShape({Dimension(0, 30), Dimension(3)}));
    EXPECT_EQ(nms->get_output_partial_shape(1), PartialShape({Dimension(0, 30), Dimension(3)}));
    EXPECT_EQ(nms->get_output_shape(2), (Shape{1}));
}

TEST(type_prop, matrix_nms_dynamic_boxes_and_scores)
{
    const auto boxes = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    const auto scores = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    const auto max_output_boxes_per_class = op::Constant::create(element::i16, Shape{}, {3});
    const auto iou_threshold = make_shared<op::Parameter>(element::f32, Shape{});
    const auto score_threshold = make_shared<op::Parameter>(element::f32, Shape{});

    const auto nms = make_shared<op::v8::MatrixNms>(
        boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold);

    ASSERT_EQ(nms->get_output_element_type(0), element::i64);
    ASSERT_EQ(nms->get_output_element_type(1), element::f32);
    ASSERT_EQ(nms->get_output_element_type(2), element::i64);
    EXPECT_EQ(nms->get_output_partial_shape(0), PartialShape({Dimension::dynamic(), 3}));
    EXPECT_EQ(nms->get_output_partial_shape(1), PartialShape({Dimension::dynamic(), 3}));
    EXPECT_EQ(nms->get_output_shape(2), (Shape{1}));
}
