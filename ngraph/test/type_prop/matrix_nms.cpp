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
        const auto boxes = make_shared<op::Parameter>(element::f32, Shape{1, 2, 4});
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
        const auto boxes = make_shared<op::Parameter>(element::f32, Shape{1, 2, 4});
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
        const auto boxes = make_shared<op::Parameter>(element::f32, Shape{1, 2, 4});
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

TEST(type_prop, matrix_nms_incorrect_boxes_rank2)
{
    try
    {
        const auto boxes = make_shared<op::Parameter>(element::f32, Shape{1, 2, 3});
        const auto scores = make_shared<op::Parameter>(element::f32, Shape{2, 2, 2});

        make_shared<op::v8::MatrixNms>(boxes, scores);
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             "The third dimension of the 'boxes' must be 4");
    }
}

TEST(type_prop, matrix_nms_incorrect_output_type)
{
    try
    {
        const auto boxes = make_shared<op::Parameter>(element::f32, Shape{1, 2, 4});
        const auto scores = make_shared<op::Parameter>(element::f32, Shape{1, 2, 2});

        make_shared<op::v8::MatrixNms>(boxes, scores, ngraph::op::util::NmsBase::SortResultType::NONE, true, ngraph::element::f32);
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             "Output type must be i32 or i64");
    }
}

TEST(type_prop, matrix_nms_incorrect_nms_topk)
{
    try
    {
        const auto boxes = make_shared<op::Parameter>(element::f32, Shape{1, 2, 4});
        const auto scores = make_shared<op::Parameter>(element::f32, Shape{1, 2, 2});

        make_shared<op::v8::MatrixNms>(boxes, scores, ngraph::op::util::NmsBase::SortResultType::NONE, true, ngraph::element::i32, 0.0f, -2);
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             "The 'nms_top_k' must be great or equal -1");
    }
}

TEST(type_prop, matrix_nms_incorrect_keep_topk)
{
    try
    {
        const auto boxes = make_shared<op::Parameter>(element::f32, Shape{1, 2, 4});
        const auto scores = make_shared<op::Parameter>(element::f32, Shape{1, 2, 2});

        make_shared<op::v8::MatrixNms>(boxes, scores, ngraph::op::util::NmsBase::SortResultType::NONE, true, ngraph::element::i32, 0.0f, -1, -2);
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             "The 'keep_top_k' must be great or equal -1");
    }
}

TEST(type_prop, matrix_nms_incorrect_background_class)
{
    try
    {
        const auto boxes = make_shared<op::Parameter>(element::f32, Shape{1, 2, 4});
        const auto scores = make_shared<op::Parameter>(element::f32, Shape{1, 2, 2});

        make_shared<op::v8::MatrixNms>(boxes, scores, ngraph::op::util::NmsBase::SortResultType::NONE, true, ngraph::element::i32, 0.0f, -1, -1, -2);
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             "The 'background_class' must be great or equal -1");
    }
}

TEST(type_prop, matrix_nms_output_shape_1dim_dynamic)
{
    const auto boxes = make_shared<op::Parameter>(element::f32, Shape{5, 2, 4});
    const auto scores = make_shared<op::Parameter>(element::f32, Shape{5, 3, 2});

    const auto nms = make_shared<op::v8::MatrixNms>(boxes, scores);

    ASSERT_TRUE(
        nms->get_output_partial_shape(0).same_scheme(PartialShape{Dimension::dynamic(), 6}));
    ASSERT_TRUE(
        nms->get_output_partial_shape(1).same_scheme(PartialShape{Dimension::dynamic(), 1}));

    EXPECT_EQ(nms->get_output_shape(2), (Shape{5}));
}

TEST(type_prop, matrix_nms_output_shape_1dim_max_out)
{
    const auto boxes = make_shared<op::Parameter>(element::f32, Shape{2, 7, 4});
    const auto scores = make_shared<op::Parameter>(element::f32, Shape{2, 5, 7});

    const auto nms = make_shared<op::v8::MatrixNms>(boxes, scores);

    ASSERT_EQ(nms->get_output_element_type(0), element::f32);
    ASSERT_EQ(nms->get_output_element_type(1), element::i64);
    ASSERT_EQ(nms->get_output_element_type(2), element::i64);

    // batch * class * box
    EXPECT_EQ(nms->get_output_partial_shape(0), PartialShape({Dimension(0, 2 * 5 * 7), Dimension(6)}));
    EXPECT_EQ(nms->get_output_partial_shape(1), PartialShape({Dimension(0, 2 * 5 * 7), 1}));
    EXPECT_EQ(nms->get_output_shape(2), (Shape{2}));
}

TEST(type_prop, matrix_nms_output_shape_1dim_nms_topk)
{
    const auto boxes = make_shared<op::Parameter>(element::f32, Shape{2, 7, 4});
    const auto scores = make_shared<op::Parameter>(element::f32, Shape{2, 5, 7});

    const auto nms = make_shared<op::v8::MatrixNms>(
        boxes, scores, op::v8::MatrixNms::SortResultType::CLASSID, true, ngraph::element::i64, 0.0f, 3);

    ASSERT_EQ(nms->get_output_element_type(0), element::f32);
    ASSERT_EQ(nms->get_output_element_type(1), element::i64);
    ASSERT_EQ(nms->get_output_element_type(2), element::i64);
    // batch * class * min(nms_topk, box)
    EXPECT_EQ(nms->get_output_partial_shape(0), PartialShape({Dimension(0, 2 * 5 * 3), Dimension(6)}));
    EXPECT_EQ(nms->get_output_partial_shape(1), PartialShape({Dimension(0, 2 * 5 * 3), 1}));
    EXPECT_EQ(nms->get_output_shape(2), (Shape{2}));
}

TEST(type_prop, matrix_nms_output_shape_1dim_keep_topk)
{
    const auto boxes = make_shared<op::Parameter>(element::f32, Shape{2, 7, 4});
    const auto scores = make_shared<op::Parameter>(element::f32, Shape{2, 5, 7});

    const auto nms = make_shared<op::v8::MatrixNms>(
        boxes, scores, op::v8::MatrixNms::SortResultType::CLASSID, true, ngraph::element::i64, 0.0f, 3, 8);

    ASSERT_EQ(nms->get_output_element_type(0), element::f32);
    ASSERT_EQ(nms->get_output_element_type(1), element::i64);
    ASSERT_EQ(nms->get_output_element_type(2), element::i64);
    // batch * min(keep_topk, class * box))
    EXPECT_EQ(nms->get_output_partial_shape(0), PartialShape({Dimension(0, 2 * 8), Dimension(6)}));
    EXPECT_EQ(nms->get_output_partial_shape(1), PartialShape({Dimension(0, 2 * 8), 1}));
    EXPECT_EQ(nms->get_output_shape(2), (Shape{2}));
}

TEST(type_prop, matrix_nms_output_shape_i32)
{
    const auto boxes = make_shared<op::Parameter>(element::f32, Shape{2, 7, 4});
    const auto scores = make_shared<op::Parameter>(element::f32, Shape{2, 5, 7});

    const auto nms = make_shared<op::v8::MatrixNms>(
        boxes, scores, op::v8::MatrixNms::SortResultType::CLASSID, true, ngraph::element::i32);

    ASSERT_EQ(nms->get_output_element_type(0), element::f32);
    ASSERT_EQ(nms->get_output_element_type(1), element::i32);
    ASSERT_EQ(nms->get_output_element_type(2), element::i32);
    // batch * class * box
    EXPECT_EQ(nms->get_output_partial_shape(0), PartialShape({Dimension(0, 2 * 5 * 7), Dimension(6)}));
    EXPECT_EQ(nms->get_output_partial_shape(1), PartialShape({Dimension(0, 2 * 5 * 7), 1}));
    EXPECT_EQ(nms->get_output_shape(2), (Shape{2}));
}

TEST(type_prop, matrix_nms_dynamic_boxes_and_scores)
{
    const auto boxes = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    const auto scores = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());

    const auto nms = make_shared<op::v8::MatrixNms>(boxes, scores);

    ASSERT_EQ(nms->get_output_element_type(0), element::f32);
    ASSERT_EQ(nms->get_output_element_type(1), element::i64);
    ASSERT_EQ(nms->get_output_element_type(2), element::i64);
    EXPECT_EQ(nms->get_output_partial_shape(0), PartialShape({Dimension::dynamic(), 6}));
    EXPECT_EQ(nms->get_output_partial_shape(1), PartialShape({Dimension::dynamic(), 1}));
    EXPECT_EQ(nms->get_output_partial_shape(2), PartialShape({Dimension::dynamic()}));
}
