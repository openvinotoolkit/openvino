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

// ------------------------------ V1 ------------------------------

TEST(type_prop, nms_incorrect_boxes_rank)
{
    try
    {
        const auto boxes = make_shared<op::Parameter>(element::f32, Shape{1, 2, 3, 4});
        const auto scores = make_shared<op::Parameter>(element::f32, Shape{1, 2, 3});

        make_shared<op::v1::NonMaxSuppression>(boxes, scores);
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "Expected a 3D tensor for the 'boxes' input");
    }
}

TEST(type_prop, nms_incorrect_scores_rank)
{
    try
    {
        const auto boxes = make_shared<op::Parameter>(element::f32, Shape{1, 2, 3});
        const auto scores = make_shared<op::Parameter>(element::f32, Shape{1, 2});

        make_shared<op::v1::NonMaxSuppression>(boxes, scores);
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "Expected a 3D tensor for the 'scores' input");
    }
}

TEST(type_prop, nms_incorrect_scheme_num_batches)
{
    try
    {
        const auto boxes = make_shared<op::Parameter>(element::f32, Shape{1, 2, 3});
        const auto scores = make_shared<op::Parameter>(element::f32, Shape{2, 2, 3});

        make_shared<op::v1::NonMaxSuppression>(boxes, scores);
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             "The first dimension of both 'boxes' and 'scores' must match");
    }
}

TEST(type_prop, nms_incorrect_scheme_num_boxes)
{
    try
    {
        const auto boxes = make_shared<op::Parameter>(element::f32, Shape{1, 2, 3});
        const auto scores = make_shared<op::Parameter>(element::f32, Shape{1, 2, 3});

        make_shared<op::v1::NonMaxSuppression>(boxes, scores);
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             "'boxes' and 'scores' input shapes must match at the second and third "
                             "dimension respectively");
    }
}

TEST(type_prop, nms_scalar_inputs_check)
{
    const auto boxes = make_shared<op::Parameter>(element::f32, Shape{1, 2, 4});
    const auto scores = make_shared<op::Parameter>(element::f32, Shape{1, 2, 2});

    const auto scalar = make_shared<op::Parameter>(element::f32, Shape{});
    const auto non_scalar = make_shared<op::Parameter>(element::f32, Shape{1});

    try
    {
        make_shared<op::v1::NonMaxSuppression>(boxes, scores, non_scalar, scalar, scalar);
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             "Expected a scalar for the 'max_output_boxes_per_class' input");
    }

    try
    {
        make_shared<op::v1::NonMaxSuppression>(boxes, scores, scalar, non_scalar, scalar);
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "Expected a scalar for the 'iou_threshold' input");
    }

    try
    {
        make_shared<op::v1::NonMaxSuppression>(boxes, scores, scalar, scalar, non_scalar);
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "Expected a scalar for the 'score_threshold' input");
    }
}

TEST(type_prop, nms_output_shape)
{
    const auto boxes = make_shared<op::Parameter>(element::f32, Shape{1, 2, 4});
    const auto scores = make_shared<op::Parameter>(element::f32, Shape{1, 2, 2});

    const auto nms = make_shared<op::v1::NonMaxSuppression>(boxes, scores);
    const auto nms_out_ps = nms->get_output_partial_shape(0);

    EXPECT_TRUE(nms_out_ps.rank().is_static());
    EXPECT_EQ(nms_out_ps.rank().get_length(), 2);
    EXPECT_EQ(nms_out_ps[1].get_length(), 3);
}

TEST(type_prop, nms_output_shape_2)
{
    const auto boxes = make_shared<op::Parameter>(element::f32, Shape{1, 6, 4});
    const auto scores = make_shared<op::Parameter>(element::f32, Shape{1, 1, 6});
    const auto max_output_boxes_per_class = op::Constant::create(element::i32, Shape{}, {3});
    const auto iou_threshold = make_shared<op::Parameter>(element::f32, Shape{});
    const auto score_threshold = make_shared<op::Parameter>(element::f32, Shape{});

    const auto nms = make_shared<op::v1::NonMaxSuppression>(
        boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold);

    ASSERT_EQ(nms->get_element_type(), element::i64);
    ASSERT_EQ(nms->get_shape(), (Shape{3, 3}));
}

TEST(type_prop, nms_output_shape_3)
{
    const auto boxes = make_shared<op::Parameter>(element::f32, Shape{1, 1, 4});
    const auto scores = make_shared<op::Parameter>(element::f32, Shape{1, 1, 1});
    const auto max_output_boxes_per_class = op::Constant::create(element::i16, Shape{}, {3});
    const auto iou_threshold = make_shared<op::Parameter>(element::f32, Shape{});
    const auto score_threshold = make_shared<op::Parameter>(element::f32, Shape{});

    const auto nms = make_shared<op::v1::NonMaxSuppression>(
        boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold);

    ASSERT_EQ(nms->get_element_type(), element::i64);
    ASSERT_EQ(nms->get_shape(), (Shape{1, 3}));
}

TEST(type_prop, nms_dynamic_boxes_and_scores)
{
    const auto boxes = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    const auto scores = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    const auto max_output_boxes_per_class = op::Constant::create(element::i16, Shape{}, {3});
    const auto iou_threshold = make_shared<op::Parameter>(element::f32, Shape{});
    const auto score_threshold = make_shared<op::Parameter>(element::f32, Shape{});

    const auto nms = make_shared<op::v1::NonMaxSuppression>(
        boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold);

    ASSERT_EQ(nms->get_element_type(), element::i64);
    ASSERT_TRUE(
        nms->get_output_partial_shape(0).same_scheme(PartialShape{Dimension::dynamic(), 3}));
}

// ------------------------------ V3 ------------------------------

TEST(type_prop, nms_v3_incorrect_boxes_rank)
{
    try
    {
        const auto boxes = make_shared<op::Parameter>(element::f32, Shape{1, 2, 3, 4});
        const auto scores = make_shared<op::Parameter>(element::f32, Shape{1, 2, 3});

        make_shared<op::v3::NonMaxSuppression>(boxes, scores);
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "Expected a 3D tensor for the 'boxes' input");
    }
}

TEST(type_prop, nms_v3_incorrect_scores_rank)
{
    try
    {
        const auto boxes = make_shared<op::Parameter>(element::f32, Shape{1, 2, 3});
        const auto scores = make_shared<op::Parameter>(element::f32, Shape{1, 2});

        make_shared<op::v3::NonMaxSuppression>(boxes, scores);
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "Expected a 3D tensor for the 'scores' input");
    }
}

TEST(type_prop, nms_v3_incorrect_scheme_num_batches)
{
    try
    {
        const auto boxes = make_shared<op::Parameter>(element::f32, Shape{1, 2, 3});
        const auto scores = make_shared<op::Parameter>(element::f32, Shape{2, 2, 3});

        make_shared<op::v3::NonMaxSuppression>(boxes, scores);
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             "The first dimension of both 'boxes' and 'scores' must match");
    }
}

TEST(type_prop, nms_v3_incorrect_scheme_num_boxes)
{
    try
    {
        const auto boxes = make_shared<op::Parameter>(element::f32, Shape{1, 2, 3});
        const auto scores = make_shared<op::Parameter>(element::f32, Shape{1, 2, 3});

        make_shared<op::v3::NonMaxSuppression>(boxes, scores);
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             "'boxes' and 'scores' input shapes must match at the second and third "
                             "dimension respectively");
    }
}

TEST(type_prop, nms_v3_scalar_inputs_check)
{
    const auto boxes = make_shared<op::Parameter>(element::f32, Shape{1, 2, 4});
    const auto scores = make_shared<op::Parameter>(element::f32, Shape{1, 2, 2});

    const auto scalar = make_shared<op::Parameter>(element::f32, Shape{});
    const auto non_scalar = make_shared<op::Parameter>(element::f32, Shape{1});

    try
    {
        make_shared<op::v3::NonMaxSuppression>(boxes, scores, non_scalar, scalar, scalar);
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             "Expected a scalar for the 'max_output_boxes_per_class' input");
    }

    try
    {
        make_shared<op::v3::NonMaxSuppression>(boxes, scores, scalar, non_scalar, scalar);
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "Expected a scalar for the 'iou_threshold' input");
    }

    try
    {
        make_shared<op::v3::NonMaxSuppression>(boxes, scores, scalar, scalar, non_scalar);
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "Expected a scalar for the 'score_threshold' input");
    }
}

TEST(type_prop, nms_v3_output_shape)
{
    const auto boxes = make_shared<op::Parameter>(element::f32, Shape{1, 2, 4});
    const auto scores = make_shared<op::Parameter>(element::f32, Shape{1, 2, 2});

    const auto nms = make_shared<op::v3::NonMaxSuppression>(boxes, scores);
    const auto nms_out_ps = nms->get_output_partial_shape(0);

    EXPECT_TRUE(nms_out_ps.rank().is_static());
    EXPECT_EQ(nms_out_ps.rank().get_length(), 2);
    EXPECT_EQ(nms_out_ps[1].get_length(), 3);
}

TEST(type_prop, nms_v3_output_shape_2)
{
    const auto boxes = make_shared<op::Parameter>(element::f32, Shape{1, 6, 4});
    const auto scores = make_shared<op::Parameter>(element::f32, Shape{1, 1, 6});
    const auto max_output_boxes_per_class = op::Constant::create(element::i32, Shape{}, {3});
    const auto iou_threshold = make_shared<op::Parameter>(element::f32, Shape{});
    const auto score_threshold = make_shared<op::Parameter>(element::f32, Shape{});

    const auto nms = make_shared<op::v3::NonMaxSuppression>(
        boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold);

    ASSERT_EQ(nms->get_element_type(), element::i64);
    ASSERT_EQ(nms->get_shape(), (Shape{3, 3}));
}

TEST(type_prop, nms_v3_output_shape_3)
{
    const auto boxes = make_shared<op::Parameter>(element::f32, Shape{1, 1, 4});
    const auto scores = make_shared<op::Parameter>(element::f32, Shape{1, 1, 1});
    const auto max_output_boxes_per_class = op::Constant::create(element::i16, Shape{}, {3});
    const auto iou_threshold = make_shared<op::Parameter>(element::f32, Shape{});
    const auto score_threshold = make_shared<op::Parameter>(element::f32, Shape{});

    const auto nms = make_shared<op::v3::NonMaxSuppression>(
        boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold);

    ASSERT_EQ(nms->get_element_type(), element::i64);
    ASSERT_EQ(nms->get_shape(), (Shape{1, 3}));
}

TEST(type_prop, nms_v3_output_shape_i32)
{
    const auto boxes = make_shared<op::Parameter>(element::f32, Shape{1, 1, 4});
    const auto scores = make_shared<op::Parameter>(element::f32, Shape{1, 1, 1});
    const auto max_output_boxes_per_class = op::Constant::create(element::i16, Shape{}, {3});
    const auto iou_threshold = make_shared<op::Parameter>(element::f32, Shape{});
    const auto score_threshold = make_shared<op::Parameter>(element::f32, Shape{});

    const auto nms =
        make_shared<op::v3::NonMaxSuppression>(boxes,
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

TEST(type_prop, nms_v3_dynamic_boxes_and_scores)
{
    const auto boxes = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    const auto scores = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    const auto max_output_boxes_per_class = op::Constant::create(element::i16, Shape{}, {3});
    const auto iou_threshold = make_shared<op::Parameter>(element::f32, Shape{});
    const auto score_threshold = make_shared<op::Parameter>(element::f32, Shape{});

    const auto nms = make_shared<op::v3::NonMaxSuppression>(
        boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold);

    ASSERT_EQ(nms->get_element_type(), element::i64);
    ASSERT_TRUE(
        nms->get_output_partial_shape(0).same_scheme(PartialShape{Dimension::dynamic(), 3}));
}

// ------------------------------ V4 ------------------------------

TEST(type_prop, nms_v4_incorrect_boxes_rank)
{
    try
    {
        const auto boxes = make_shared<op::Parameter>(element::f32, Shape{1, 2, 3, 4});
        const auto scores = make_shared<op::Parameter>(element::f32, Shape{1, 2, 3});

        make_shared<op::v4::NonMaxSuppression>(boxes, scores);
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "Expected a 3D tensor for the 'boxes' input");
    }
}

TEST(type_prop, nms_v4_incorrect_scores_rank)
{
    try
    {
        const auto boxes = make_shared<op::Parameter>(element::f32, Shape{1, 2, 3});
        const auto scores = make_shared<op::Parameter>(element::f32, Shape{1, 2});

        make_shared<op::v4::NonMaxSuppression>(boxes, scores);
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "Expected a 3D tensor for the 'scores' input");
    }
}

TEST(type_prop, nms_v4_incorrect_scheme_num_batches)
{
    try
    {
        const auto boxes = make_shared<op::Parameter>(element::f32, Shape{1, 2, 3});
        const auto scores = make_shared<op::Parameter>(element::f32, Shape{2, 2, 3});

        make_shared<op::v4::NonMaxSuppression>(boxes, scores);
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             "The first dimension of both 'boxes' and 'scores' must match");
    }
}

TEST(type_prop, nms_v4_incorrect_scheme_num_boxes)
{
    try
    {
        const auto boxes = make_shared<op::Parameter>(element::f32, Shape{1, 2, 3});
        const auto scores = make_shared<op::Parameter>(element::f32, Shape{1, 2, 3});

        make_shared<op::v4::NonMaxSuppression>(boxes, scores);
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             "'boxes' and 'scores' input shapes must match at the second and third "
                             "dimension respectively");
    }
}

TEST(type_prop, nms_v4_scalar_inputs_check)
{
    const auto boxes = make_shared<op::Parameter>(element::f32, Shape{1, 2, 4});
    const auto scores = make_shared<op::Parameter>(element::f32, Shape{1, 2, 2});

    const auto scalar = make_shared<op::Parameter>(element::f32, Shape{});
    const auto non_scalar = make_shared<op::Parameter>(element::f32, Shape{1});

    try
    {
        make_shared<op::v4::NonMaxSuppression>(boxes, scores, non_scalar, scalar, scalar);
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             "Expected a scalar for the 'max_output_boxes_per_class' input");
    }

    try
    {
        make_shared<op::v4::NonMaxSuppression>(boxes, scores, scalar, non_scalar, scalar);
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "Expected a scalar for the 'iou_threshold' input");
    }

    try
    {
        make_shared<op::v4::NonMaxSuppression>(boxes, scores, scalar, scalar, non_scalar);
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "Expected a scalar for the 'score_threshold' input");
    }
}

TEST(type_prop, nms_v4_output_shape)
{
    const auto boxes = make_shared<op::Parameter>(element::f32, Shape{5, 2, 4});
    const auto scores = make_shared<op::Parameter>(element::f32, Shape{5, 3, 2});

    const auto nms = make_shared<op::v4::NonMaxSuppression>(boxes, scores);
    const auto nms_out_ps = nms->get_output_partial_shape(0);

    EXPECT_TRUE(nms_out_ps.rank().is_static());
    EXPECT_EQ(nms_out_ps.rank().get_length(), 2);
    EXPECT_EQ(nms->get_shape(), (Shape{0, 3}));
}

TEST(type_prop, nms_v4_output_shape_2)
{
    const auto boxes = make_shared<op::Parameter>(element::f32, Shape{2, 7, 4});
    const auto scores = make_shared<op::Parameter>(element::f32, Shape{2, 5, 7});
    const auto max_output_boxes_per_class = op::Constant::create(element::i32, Shape{}, {3});
    const auto iou_threshold = make_shared<op::Parameter>(element::f32, Shape{});
    const auto score_threshold = make_shared<op::Parameter>(element::f32, Shape{});

    const auto nms = make_shared<op::v4::NonMaxSuppression>(
        boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold);

    ASSERT_EQ(nms->get_element_type(), element::i64);
    ASSERT_EQ(nms->get_shape(), (Shape{2 * 5 * 3, 3}));
}

TEST(type_prop, nms_v4_output_shape_3)
{
    const auto boxes = make_shared<op::Parameter>(element::f32, Shape{2, 7, 4});
    const auto scores = make_shared<op::Parameter>(element::f32, Shape{2, 5, 7});
    const auto max_output_boxes_per_class = op::Constant::create(element::i16, Shape{}, {1000});
    const auto iou_threshold = make_shared<op::Parameter>(element::f32, Shape{});
    const auto score_threshold = make_shared<op::Parameter>(element::f32, Shape{});

    const auto nms = make_shared<op::v4::NonMaxSuppression>(
        boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold);

    ASSERT_EQ(nms->get_element_type(), element::i64);
    ASSERT_EQ(nms->get_shape(), (Shape{2 * 5 * 7, 3}));
}

TEST(type_prop, nms_v4_output_shape_i32)
{
    const auto boxes = make_shared<op::Parameter>(element::f32, Shape{2, 7, 4});
    const auto scores = make_shared<op::Parameter>(element::f32, Shape{2, 5, 7});
    const auto max_output_boxes_per_class = op::Constant::create(element::i16, Shape{}, {3});
    const auto iou_threshold = make_shared<op::Parameter>(element::f32, Shape{});
    const auto score_threshold = make_shared<op::Parameter>(element::f32, Shape{});

    const auto nms =
        make_shared<op::v4::NonMaxSuppression>(boxes,
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

TEST(type_prop, nms_v4_dynamic_boxes_and_scores)
{
    const auto boxes = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    const auto scores = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    const auto max_output_boxes_per_class = op::Constant::create(element::i16, Shape{}, {3});
    const auto iou_threshold = make_shared<op::Parameter>(element::f32, Shape{});
    const auto score_threshold = make_shared<op::Parameter>(element::f32, Shape{});

    const auto nms = make_shared<op::v4::NonMaxSuppression>(
        boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold);

    ASSERT_EQ(nms->get_element_type(), element::i64);
    ASSERT_TRUE(
        nms->get_output_partial_shape(0).same_scheme(PartialShape{Dimension::dynamic(), 3}));
}
