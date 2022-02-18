// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "util/type_prop.hpp"

using namespace std;
using namespace ngraph;

TEST(type_prop, multiclass_nms_incorrect_boxes_rank) {
    try {
        const auto boxes = make_shared<op::Parameter>(element::f32, Shape{1, 2, 3, 4});
        const auto scores = make_shared<op::Parameter>(element::f32, Shape{1, 2, 3});

        const auto unused = make_shared<op::v8::MulticlassNms>(boxes, scores, op::v8::MulticlassNms::Attributes());
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), "Expected a 3D tensor for the 'boxes' input");
    }
}

TEST(type_prop, multiclass_nms_incorrect_scores_rank) {
    try {
        const auto boxes = make_shared<op::Parameter>(element::f32, Shape{1, 2, 4});
        const auto scores = make_shared<op::Parameter>(element::f32, Shape{1, 2});

        const auto unused = make_shared<op::v8::MulticlassNms>(boxes, scores, op::v8::MulticlassNms::Attributes());
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), "Expected a 3D tensor for the 'scores' input");
    }
}

TEST(type_prop, multiclass_nms_incorrect_scheme_num_batches) {
    try {
        const auto boxes = make_shared<op::Parameter>(element::f32, Shape{1, 2, 4});
        const auto scores = make_shared<op::Parameter>(element::f32, Shape{2, 2, 3});

        const auto unused = make_shared<op::v8::MulticlassNms>(boxes, scores, op::v8::MulticlassNms::Attributes());
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), "The first dimension of both 'boxes' and 'scores' must match");
    }
}

TEST(type_prop, multiclass_nms_incorrect_scheme_num_boxes) {
    try {
        const auto boxes = make_shared<op::Parameter>(element::f32, Shape{1, 2, 4});
        const auto scores = make_shared<op::Parameter>(element::f32, Shape{1, 2, 3});

        const auto unused = make_shared<op::v8::MulticlassNms>(boxes, scores, op::v8::MulticlassNms::Attributes());
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(),
                             "'boxes' and 'scores' input shapes must match at the second and third "
                             "dimension respectively");
    }
}

TEST(type_prop, multiclass_nms_incorrect_boxes_rank2) {
    try {
        const auto boxes = make_shared<op::Parameter>(element::f32, Shape{1, 2, 3});
        const auto scores = make_shared<op::Parameter>(element::f32, Shape{2, 2, 2});

        const auto unused = make_shared<op::v8::MulticlassNms>(boxes, scores, op::v8::MulticlassNms::Attributes());
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), "The third dimension of the 'boxes' must be 4");
    }
}

TEST(type_prop, multiclass_nms_incorrect_output_type) {
    try {
        const auto boxes = make_shared<op::Parameter>(element::f32, Shape{1, 2, 4});
        const auto scores = make_shared<op::Parameter>(element::f32, Shape{1, 2, 2});
        op::v8::MulticlassNms::Attributes attrs;
        attrs.output_type = ngraph::element::f32;

        const auto unused = make_shared<op::v8::MulticlassNms>(boxes, scores, attrs);
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), "Output type must be i32 or i64");
    }
}

TEST(type_prop, multiclass_nms_incorrect_nms_topk) {
    try {
        const auto boxes = make_shared<op::Parameter>(element::f32, Shape{1, 2, 4});
        const auto scores = make_shared<op::Parameter>(element::f32, Shape{1, 2, 2});
        op::v8::MulticlassNms::Attributes attrs;
        attrs.nms_top_k = -2;

        const auto unused = make_shared<op::v8::MulticlassNms>(boxes, scores, attrs);
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), "The 'nms_top_k' must be great or equal -1");
    }
}

TEST(type_prop, multiclass_nms_incorrect_keep_topk) {
    try {
        const auto boxes = make_shared<op::Parameter>(element::f32, Shape{1, 2, 4});
        const auto scores = make_shared<op::Parameter>(element::f32, Shape{1, 2, 2});
        op::v8::MulticlassNms::Attributes attrs;
        attrs.keep_top_k = -2;

        const auto unused = make_shared<op::v8::MulticlassNms>(boxes, scores, attrs);
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), "The 'keep_top_k' must be great or equal -1");
    }
}

TEST(type_prop, multiclass_nms_incorrect_background_class) {
    try {
        const auto boxes = make_shared<op::Parameter>(element::f32, Shape{1, 2, 4});
        const auto scores = make_shared<op::Parameter>(element::f32, Shape{1, 2, 2});
        op::v8::MulticlassNms::Attributes attrs;
        attrs.background_class = -2;

        const auto unused = make_shared<op::v8::MulticlassNms>(boxes, scores, attrs);
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), "The 'background_class' must be great or equal -1");
    }
}

TEST(type_prop, multiclass_nms_incorrect_eta) {
    try {
        const auto boxes = make_shared<op::Parameter>(element::f32, Shape{1, 2, 4});
        const auto scores = make_shared<op::Parameter>(element::f32, Shape{1, 2, 2});
        op::v8::MulticlassNms::Attributes attrs;
        attrs.nms_eta = 2.0f;

        const auto unused = make_shared<op::v8::MulticlassNms>(boxes, scores, attrs);
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), "The 'nms_eta' must be in close range [0, 1.0]");
    }
}

TEST(type_prop, multiclass_nms_incorrect_input_type) {
    try {
        const auto boxes = make_shared<op::Parameter>(element::f16, Shape{1, 2, 4});
        const auto scores = make_shared<op::Parameter>(element::f32, Shape{1, 2, 2});

        const auto unused = make_shared<op::v8::MulticlassNms>(boxes, scores, op::v8::MulticlassNms::Attributes());
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), "Expected 'boxes', 'scores' type is same.");
    }
}

TEST(type_prop, multiclass_nms_output_shape_1dim_dynamic) {
    const auto boxes = make_shared<op::Parameter>(element::f32, Shape{5, 2, 4});
    const auto scores = make_shared<op::Parameter>(element::f32, Shape{5, 3, 2});

    const auto nms = make_shared<op::v8::MulticlassNms>(boxes, scores, op::v8::MulticlassNms::Attributes());

    ASSERT_TRUE(nms->get_output_partial_shape(0).same_scheme(PartialShape{Dimension::dynamic(), 6}));
    ASSERT_TRUE(nms->get_output_partial_shape(1).same_scheme(PartialShape{Dimension::dynamic(), 1}));

    EXPECT_EQ(nms->get_output_shape(2), (Shape{5}));
}

TEST(type_prop, multiclass_nms_output_shape_1dim_max_out) {
    const auto boxes = make_shared<op::Parameter>(element::f32, Shape{2, 7, 4});
    const auto scores = make_shared<op::Parameter>(element::f32, Shape{2, 5, 7});

    const auto nms = make_shared<op::v8::MulticlassNms>(boxes, scores, op::v8::MulticlassNms::Attributes());

    ASSERT_EQ(nms->get_output_element_type(0), element::f32);
    ASSERT_EQ(nms->get_output_element_type(1), element::i64);
    ASSERT_EQ(nms->get_output_element_type(2), element::i64);

    // batch * class * box
    EXPECT_EQ(nms->get_output_partial_shape(0), PartialShape({Dimension(0, 2 * 5 * 7), Dimension(6)}));
    EXPECT_EQ(nms->get_output_partial_shape(1), PartialShape({Dimension(0, 2 * 5 * 7), 1}));
    EXPECT_EQ(nms->get_output_shape(2), (Shape{2}));
}

TEST(type_prop, multiclass_nms_output_shape_1dim_nms_topk) {
    const auto boxes = make_shared<op::Parameter>(element::f32, Shape{2, 7, 4});
    const auto scores = make_shared<op::Parameter>(element::f32, Shape{2, 5, 7});
    op::v8::MulticlassNms::Attributes attrs;
    attrs.nms_top_k = 3;

    const auto nms = make_shared<op::v8::MulticlassNms>(boxes, scores, attrs);

    ASSERT_EQ(nms->get_output_element_type(0), element::f32);
    ASSERT_EQ(nms->get_output_element_type(1), element::i64);
    ASSERT_EQ(nms->get_output_element_type(2), element::i64);
    // batch * class * min(nms_topk, box)
    EXPECT_EQ(nms->get_output_partial_shape(0), PartialShape({Dimension(0, 2 * 5 * 3), Dimension(6)}));
    EXPECT_EQ(nms->get_output_partial_shape(1), PartialShape({Dimension(0, 2 * 5 * 3), 1}));
    EXPECT_EQ(nms->get_output_shape(2), (Shape{2}));
}

TEST(type_prop, multiclass_nms_output_shape_1dim_keep_topk) {
    const auto boxes = make_shared<op::Parameter>(element::f32, Shape{2, 7, 4});
    const auto scores = make_shared<op::Parameter>(element::f32, Shape{2, 5, 7});
    op::v8::MulticlassNms::Attributes attrs;
    attrs.nms_top_k = 3;
    attrs.keep_top_k = 8;

    const auto nms = make_shared<op::v8::MulticlassNms>(boxes, scores, attrs);

    ASSERT_EQ(nms->get_output_element_type(0), element::f32);
    ASSERT_EQ(nms->get_output_element_type(1), element::i64);
    ASSERT_EQ(nms->get_output_element_type(2), element::i64);
    // batch * min(keep_topk, class * box))
    EXPECT_EQ(nms->get_output_partial_shape(0), PartialShape({Dimension(0, 2 * 8), Dimension(6)}));
    EXPECT_EQ(nms->get_output_partial_shape(1), PartialShape({Dimension(0, 2 * 8), 1}));
    EXPECT_EQ(nms->get_output_shape(2), (Shape{2}));
}

TEST(type_prop, multiclass_nms_input_f16) {
    const auto boxes = make_shared<op::Parameter>(element::f16, Shape{2, 7, 4});
    const auto scores = make_shared<op::Parameter>(element::f16, Shape{2, 5, 7});

    const auto nms = make_shared<op::v8::MulticlassNms>(boxes, scores, op::v8::MulticlassNms::Attributes());

    ASSERT_EQ(nms->get_output_element_type(0), element::f16);
    ASSERT_EQ(nms->get_output_element_type(1), element::i64);
    ASSERT_EQ(nms->get_output_element_type(2), element::i64);
    // batch * class * box
    EXPECT_EQ(nms->get_output_partial_shape(0), PartialShape({Dimension(0, 2 * 5 * 7), Dimension(6)}));
    EXPECT_EQ(nms->get_output_partial_shape(1), PartialShape({Dimension(0, 2 * 5 * 7), 1}));
    EXPECT_EQ(nms->get_output_shape(2), (Shape{2}));
}

TEST(type_prop, multiclass_nms_output_shape_i32) {
    const auto boxes = make_shared<op::Parameter>(element::f32, Shape{2, 7, 4});
    const auto scores = make_shared<op::Parameter>(element::f32, Shape{2, 5, 7});
    op::v8::MulticlassNms::Attributes attrs;
    attrs.output_type = ngraph::element::i32;

    const auto nms = make_shared<op::v8::MulticlassNms>(boxes, scores, attrs);

    ASSERT_EQ(nms->get_output_element_type(0), element::f32);
    ASSERT_EQ(nms->get_output_element_type(1), element::i32);
    ASSERT_EQ(nms->get_output_element_type(2), element::i32);
    // batch * class * box
    EXPECT_EQ(nms->get_output_partial_shape(0), PartialShape({Dimension(0, 2 * 5 * 7), Dimension(6)}));
    EXPECT_EQ(nms->get_output_partial_shape(1), PartialShape({Dimension(0, 2 * 5 * 7), 1}));
    EXPECT_EQ(nms->get_output_shape(2), (Shape{2}));
}

TEST(type_prop, multiclass_nms_dynamic_boxes_and_scores) {
    const auto boxes = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    const auto scores = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());

    const auto nms = make_shared<op::v8::MulticlassNms>(boxes, scores, op::v8::MulticlassNms::Attributes());

    ASSERT_EQ(nms->get_output_element_type(0), element::f32);
    ASSERT_EQ(nms->get_output_element_type(1), element::i64);
    ASSERT_EQ(nms->get_output_element_type(2), element::i64);
    EXPECT_EQ(nms->get_output_partial_shape(0), PartialShape({Dimension::dynamic(), 6}));
    EXPECT_EQ(nms->get_output_partial_shape(1), PartialShape({Dimension::dynamic(), 1}));
    EXPECT_EQ(nms->get_output_partial_shape(2), PartialShape({Dimension::dynamic()}));
}
