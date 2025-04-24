// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/matrix_nms.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/test_assertions.hpp"
#include "common_test_utils/type_prop.hpp"

using namespace std;
using namespace ov;
using namespace testing;

class TypePropMatrixNmsV8Test : public TypePropOpTest<op::v8::MatrixNms> {
protected:
    using Attributes = op::v8::MatrixNms::Attributes;
};

TEST_F(TypePropMatrixNmsV8Test, incorrect_boxes_rank) {
    const auto boxes = make_shared<op::v0::Parameter>(element::f32, Shape{1, 2, 3, 4});
    const auto scores = make_shared<op::v0::Parameter>(element::f32, Shape{1, 2, 3});
    OV_EXPECT_THROW(ignore = make_op(boxes, scores, Attributes()),
                    NodeValidationFailure,
                    HasSubstr("Expected a 3D tensor for the 'boxes' input"));
}

TEST_F(TypePropMatrixNmsV8Test, incorrect_scores_rank) {
    const auto boxes = make_shared<op::v0::Parameter>(element::f32, Shape{1, 2, 4});
    const auto scores = make_shared<op::v0::Parameter>(element::f32, Shape{1, 2});

    OV_EXPECT_THROW(ignore = make_op(boxes, scores, Attributes()),
                    NodeValidationFailure,
                    HasSubstr("Expected a 3D tensor for the 'scores' input"));
}

TEST_F(TypePropMatrixNmsV8Test, incorrect_scheme_num_batches) {
    const auto boxes = make_shared<op::v0::Parameter>(element::f32, Shape{1, 2, 4});
    const auto scores = make_shared<op::v0::Parameter>(element::f32, Shape{2, 2, 3});

    OV_EXPECT_THROW(ignore = make_op(boxes, scores, Attributes()),
                    NodeValidationFailure,
                    HasSubstr("The first dimension of both 'boxes' and 'scores' must match"));
}

TEST_F(TypePropMatrixNmsV8Test, incorrect_scheme_num_boxes) {
    const auto boxes = make_shared<op::v0::Parameter>(element::f32, Shape{1, 2, 4});
    const auto scores = make_shared<op::v0::Parameter>(element::f32, Shape{1, 2, 3});

    OV_EXPECT_THROW(ignore = make_op(boxes, scores, Attributes()),
                    NodeValidationFailure,
                    HasSubstr("'boxes' and 'scores' input shapes must match at the second and third "
                              "dimension respectively"));
}

TEST_F(TypePropMatrixNmsV8Test, incorrect_boxes_rank2) {
    const auto boxes = make_shared<op::v0::Parameter>(element::f32, Shape{1, 2, 3});
    const auto scores = make_shared<op::v0::Parameter>(element::f32, Shape{1, 2, 2});

    OV_EXPECT_THROW(ignore = make_op(boxes, scores, Attributes()),
                    NodeValidationFailure,
                    HasSubstr("The last dimension of the 'boxes' input must be equal to 4"));
}

TEST_F(TypePropMatrixNmsV8Test, incorrect_output_type) {
    const auto boxes = make_shared<op::v0::Parameter>(element::f32, Shape{1, 2, 4});
    const auto scores = make_shared<op::v0::Parameter>(element::f32, Shape{1, 2, 2});
    Attributes attrs;
    attrs.output_type = element::f32;

    OV_EXPECT_THROW(ignore = make_op(boxes, scores, attrs),
                    NodeValidationFailure,
                    HasSubstr("Output type must be i32 or i64"));
}

TEST_F(TypePropMatrixNmsV8Test, incorrect_nms_topk) {
    const auto boxes = make_shared<op::v0::Parameter>(element::f32, Shape{1, 2, 4});
    const auto scores = make_shared<op::v0::Parameter>(element::f32, Shape{1, 2, 2});
    Attributes attrs;
    attrs.nms_top_k = -2;

    OV_EXPECT_THROW(ignore = make_op(boxes, scores, attrs),
                    NodeValidationFailure,
                    HasSubstr("The 'nms_top_k' must be great or equal -1"));
}

TEST_F(TypePropMatrixNmsV8Test, incorrect_keep_topk) {
    const auto boxes = make_shared<op::v0::Parameter>(element::f32, Shape{1, 2, 4});
    const auto scores = make_shared<op::v0::Parameter>(element::f32, Shape{1, 2, 2});
    Attributes attrs;
    attrs.keep_top_k = -2;

    OV_EXPECT_THROW(ignore = make_op(boxes, scores, attrs),
                    NodeValidationFailure,
                    HasSubstr("The 'keep_top_k' must be great or equal -1"));
}

TEST_F(TypePropMatrixNmsV8Test, incorrect_background_class) {
    const auto boxes = make_shared<op::v0::Parameter>(element::f32, Shape{1, 2, 4});
    const auto scores = make_shared<op::v0::Parameter>(element::f32, Shape{1, 2, 2});
    Attributes attrs;
    attrs.background_class = -2;

    OV_EXPECT_THROW(ignore = make_op(boxes, scores, attrs),
                    NodeValidationFailure,
                    HasSubstr("The 'background_class' must be great or equal -1"));
}

TEST_F(TypePropMatrixNmsV8Test, incorrect_input_type) {
    const auto boxes = make_shared<op::v0::Parameter>(element::f16, Shape{1, 2, 4});
    const auto scores = make_shared<op::v0::Parameter>(element::f32, Shape{1, 2, 2});

    OV_EXPECT_THROW(ignore = make_op(boxes, scores, Attributes()),
                    NodeValidationFailure,
                    HasSubstr("Expected 'boxes', 'scores' type is same."));
}

TEST_F(TypePropMatrixNmsV8Test, default_ctor) {
    const auto boxes = make_shared<op::v0::Parameter>(element::f32, Shape{5, 2, 4});
    const auto scores = make_shared<op::v0::Parameter>(element::f32, Shape{5, 3, 2});
    Attributes attrs;
    attrs.nms_top_k = 3;

    const auto nms = make_op();
    nms->set_arguments(OutputVector{boxes, scores});
    nms->set_output_type(element::f16);
    nms->set_attrs(std::move(attrs));
    nms->validate_and_infer_types();

    EXPECT_EQ(nms->get_output_partial_shape(0), (PartialShape{{0, 30}, 6}));
    EXPECT_EQ(nms->get_output_partial_shape(1), (PartialShape{{0, 30}, 1}));
    EXPECT_EQ(nms->get_output_shape(2), (Shape{5}));
}

TEST_F(TypePropMatrixNmsV8Test, output_shape_1dim_dynamic) {
    const auto boxes = make_shared<op::v0::Parameter>(element::f32, Shape{5, 2, 4});
    const auto scores = make_shared<op::v0::Parameter>(element::f32, Shape{5, 3, 2});

    const auto nms = make_op(boxes, scores, Attributes());

    EXPECT_EQ(nms->get_output_partial_shape(0), (PartialShape{{0, 30}, 6}));
    EXPECT_EQ(nms->get_output_partial_shape(1), (PartialShape{{0, 30}, 1}));

    EXPECT_EQ(nms->get_output_shape(2), (Shape{5}));
}

TEST_F(TypePropMatrixNmsV8Test, output_shape_1dim_max_out) {
    const auto boxes = make_shared<op::v0::Parameter>(element::f32, Shape{2, 7, 4});
    const auto scores = make_shared<op::v0::Parameter>(element::f32, Shape{2, 5, 7});

    const auto nms = make_op(boxes, scores, Attributes());

    ASSERT_EQ(nms->get_output_element_type(0), element::f32);
    ASSERT_EQ(nms->get_output_element_type(1), element::i64);
    ASSERT_EQ(nms->get_output_element_type(2), element::i64);

    // batch * class * box
    EXPECT_EQ(nms->get_output_partial_shape(0), PartialShape({Dimension(0, 2 * 5 * 7), Dimension(6)}));
    EXPECT_EQ(nms->get_output_partial_shape(1), PartialShape({Dimension(0, 2 * 5 * 7), 1}));
    EXPECT_EQ(nms->get_output_shape(2), (Shape{2}));
}

TEST_F(TypePropMatrixNmsV8Test, output_shape_1dim_nms_topk) {
    const auto boxes = make_shared<op::v0::Parameter>(element::f32, Shape{2, 7, 4});
    const auto scores = make_shared<op::v0::Parameter>(element::f32, Shape{2, 5, 7});
    op::v8::MatrixNms::Attributes attrs;
    attrs.nms_top_k = 3;

    const auto nms = make_op(boxes, scores, attrs);

    ASSERT_EQ(nms->get_output_element_type(0), element::f32);
    ASSERT_EQ(nms->get_output_element_type(1), element::i64);
    ASSERT_EQ(nms->get_output_element_type(2), element::i64);
    // batch * class * min(nms_topk, box)
    EXPECT_EQ(nms->get_output_partial_shape(0), PartialShape({Dimension(0, 2 * 5 * 3), Dimension(6)}));
    EXPECT_EQ(nms->get_output_partial_shape(1), PartialShape({Dimension(0, 2 * 5 * 3), 1}));
    EXPECT_EQ(nms->get_output_shape(2), (Shape{2}));
}

TEST_F(TypePropMatrixNmsV8Test, output_shape_1dim_keep_topk) {
    const auto boxes = make_shared<op::v0::Parameter>(element::f32, Shape{2, 7, 4});
    const auto scores = make_shared<op::v0::Parameter>(element::f32, Shape{2, 5, 7});
    op::v8::MatrixNms::Attributes attrs;
    attrs.nms_top_k = 3;
    attrs.keep_top_k = 8;

    const auto nms = make_op(boxes, scores, attrs);

    ASSERT_EQ(nms->get_output_element_type(0), element::f32);
    ASSERT_EQ(nms->get_output_element_type(1), element::i64);
    ASSERT_EQ(nms->get_output_element_type(2), element::i64);
    // batch * min(keep_topk, class * box))
    EXPECT_EQ(nms->get_output_partial_shape(0), PartialShape({Dimension(0, 2 * 8), Dimension(6)}));
    EXPECT_EQ(nms->get_output_partial_shape(1), PartialShape({Dimension(0, 2 * 8), 1}));
    EXPECT_EQ(nms->get_output_shape(2), (Shape{2}));
}

TEST_F(TypePropMatrixNmsV8Test, input_f16) {
    const auto boxes = make_shared<op::v0::Parameter>(element::f16, Shape{2, 7, 4});
    const auto scores = make_shared<op::v0::Parameter>(element::f16, Shape{2, 5, 7});

    const auto nms = make_op(boxes, scores, Attributes());

    ASSERT_EQ(nms->get_output_element_type(0), element::f16);
    ASSERT_EQ(nms->get_output_element_type(1), element::i64);
    ASSERT_EQ(nms->get_output_element_type(2), element::i64);
    // batch * class * box
    EXPECT_EQ(nms->get_output_partial_shape(0), PartialShape({Dimension(0, 2 * 5 * 7), Dimension(6)}));
    EXPECT_EQ(nms->get_output_partial_shape(1), PartialShape({Dimension(0, 2 * 5 * 7), 1}));
    EXPECT_EQ(nms->get_output_shape(2), (Shape{2}));
}

TEST_F(TypePropMatrixNmsV8Test, output_shape_i32) {
    const auto boxes = make_shared<op::v0::Parameter>(element::f32, Shape{2, 7, 4});
    const auto scores = make_shared<op::v0::Parameter>(element::f32, Shape{2, 5, 7});
    op::v8::MatrixNms::Attributes attrs;
    attrs.output_type = ov::element::i32;

    const auto nms = make_op(boxes, scores, attrs);

    ASSERT_EQ(nms->get_output_element_type(0), element::f32);
    ASSERT_EQ(nms->get_output_element_type(1), element::i32);
    ASSERT_EQ(nms->get_output_element_type(2), element::i32);
    // batch * class * box
    EXPECT_EQ(nms->get_output_partial_shape(0), PartialShape({Dimension(0, 2 * 5 * 7), Dimension(6)}));
    EXPECT_EQ(nms->get_output_partial_shape(1), PartialShape({Dimension(0, 2 * 5 * 7), 1}));
    EXPECT_EQ(nms->get_output_shape(2), (Shape{2}));
}

TEST_F(TypePropMatrixNmsV8Test, dynamic_boxes_and_scores) {
    const auto boxes = make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());
    const auto scores = make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());

    const auto nms = make_op(boxes, scores, Attributes());

    ASSERT_EQ(nms->get_output_element_type(0), element::f32);
    ASSERT_EQ(nms->get_output_element_type(1), element::i64);
    ASSERT_EQ(nms->get_output_element_type(2), element::i64);
    EXPECT_EQ(nms->get_output_partial_shape(0), PartialShape({Dimension::dynamic(), 6}));
    EXPECT_EQ(nms->get_output_partial_shape(1), PartialShape({Dimension::dynamic(), 1}));
    EXPECT_EQ(nms->get_output_partial_shape(2), PartialShape({Dimension::dynamic()}));
}
