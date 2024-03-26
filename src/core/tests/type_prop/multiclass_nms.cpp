// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/multiclass_nms.hpp"

#include "common_test_utils/test_assertions.hpp"
#include "common_test_utils/type_prop.hpp"

using namespace std;
using namespace ov;
using namespace testing;

template <typename T>
class type_prop : public testing::Test {
protected:
    using Attributes = op::util::MulticlassNmsBase::Attributes;

    Attributes attrs;
};

using MulticlassNmsTypes = testing::Types<op::v8::MulticlassNms, op::v9::MulticlassNms>;
TYPED_TEST_SUITE(type_prop, MulticlassNmsTypes);

TYPED_TEST(type_prop, multiclass_nms_incorrect_boxes_rank) {
    const auto boxes = make_shared<op::v0::Parameter>(element::f32, Shape{1, 2, 3, 4});
    const auto scores = make_shared<op::v0::Parameter>(element::f32, Shape{1, 2, 3});

    OV_EXPECT_THROW(ignore = make_shared<TypeParam>(boxes, scores, this->attrs),
                    NodeValidationFailure,
                    HasSubstr("Expected a 3D tensor for the 'boxes' input"));
}

TEST(type_prop2, multiclass_nms_incorrect_boxes_rank) {
    const auto boxes = make_shared<op::v0::Parameter>(element::f32, Shape{1, 2, 3, 4});
    const auto scores = make_shared<op::v0::Parameter>(element::f32, Shape{1, 2, 3});
    const auto roisnum = make_shared<op::v0::Parameter>(element::i32, Shape{1});

    OV_EXPECT_THROW(
        ignore = make_shared<op::v9::MulticlassNms>(boxes, scores, op::util::MulticlassNmsBase::Attributes()),
        NodeValidationFailure,
        HasSubstr("Expected a 3D tensor for the 'boxes' input"));
}

TYPED_TEST(type_prop, multiclass_nms_incorrect_scores_rank) {
    const auto boxes = make_shared<op::v0::Parameter>(element::f32, Shape{1, 2, 4});
    const auto scores = make_shared<op::v0::Parameter>(element::f32, Shape{1});

    OV_EXPECT_THROW(ignore = make_shared<TypeParam>(boxes, scores, this->attrs),
                    NodeValidationFailure,
                    HasSubstr("Expected a 3D tensor for the 'scores' input"));
}

TEST(type_prop2, multiclass_nms_incorrect_scores_rank2) {
    const auto boxes = make_shared<op::v0::Parameter>(element::f32, Shape{1, 2, 4});
    const auto scores = make_shared<op::v0::Parameter>(element::f32, Shape{1, 2});

    OV_EXPECT_THROW(
        ignore = make_shared<op::v9::MulticlassNms>(boxes, scores, op::util::MulticlassNmsBase::Attributes()),
        NodeValidationFailure,
        HasSubstr("Expected a 3D tensor for the 'scores' input"));
}

TEST(type_prop2, multiclass_nms_incorrect_roisnum_rank) {
    const auto boxes = make_shared<op::v0::Parameter>(element::f32, Shape{2, 3, 4});
    const auto scores = make_shared<op::v0::Parameter>(element::f32, Shape{2, 3});
    const auto roisnum = make_shared<op::v0::Parameter>(element::i32, Shape{1, 2});

    OV_EXPECT_THROW(
        ignore = make_shared<op::v9::MulticlassNms>(boxes, scores, roisnum, op::util::MulticlassNmsBase::Attributes()),
        NodeValidationFailure,
        HasSubstr("Expected a 1D tensor for the 'roisnum' input"));
}

TYPED_TEST(type_prop, multiclass_nms_incorrect_scheme_num_batches) {
    const auto boxes = make_shared<op::v0::Parameter>(element::f32, Shape{1, 2, 4});
    const auto scores = make_shared<op::v0::Parameter>(element::f32, Shape{2, 2, 3});

    OV_EXPECT_THROW(ignore = make_shared<TypeParam>(boxes, scores, this->attrs),
                    NodeValidationFailure,
                    HasSubstr("The first dimension of both 'boxes' and 'scores' must match"));
}

TEST(type_prop2, multiclass_nms_incorrect_scheme_num_classes) {
    const auto boxes = make_shared<op::v0::Parameter>(element::f32, Shape{2, 3, 4});
    const auto scores = make_shared<op::v0::Parameter>(element::f32, Shape{1, 3});
    const auto roisnum = make_shared<op::v0::Parameter>(element::i32, Shape{1});

    OV_EXPECT_THROW(
        ignore = make_shared<op::v9::MulticlassNms>(boxes, scores, roisnum, op::util::MulticlassNmsBase::Attributes()),
        NodeValidationFailure,
        HasSubstr("The first dimension of both 'boxes' and 'scores' must match"));
}

TYPED_TEST(type_prop, multiclass_nms_incorrect_scheme_num_boxes) {
    const auto boxes = make_shared<op::v0::Parameter>(element::f32, Shape{1, 2, 4});
    const auto scores = make_shared<op::v0::Parameter>(element::f32, Shape{1, 2, 3});

    OV_EXPECT_THROW(ignore = make_shared<TypeParam>(boxes, scores, this->attrs),
                    NodeValidationFailure,
                    HasSubstr("'boxes' and 'scores' input shapes must match at the second and third "
                              "dimension respectively"));
}

TEST(type_prop2, multiclass_nms_incorrect_scheme_num_boxes) {
    const auto boxes = make_shared<op::v0::Parameter>(element::f32, Shape{1, 2, 4});
    const auto scores = make_shared<op::v0::Parameter>(element::f32, Shape{1, 3});
    const auto roisnum = make_shared<op::v0::Parameter>(element::i32, Shape{1});

    OV_EXPECT_THROW(
        ignore = make_shared<op::v9::MulticlassNms>(boxes, scores, roisnum, op::util::MulticlassNmsBase::Attributes()),
        NodeValidationFailure,
        HasSubstr("'boxes' and 'scores' input shapes must match at the second dimension respectively"));
}

TEST(type_prop2, multiclass_nms_incorrect_scheme_num_boxes2) {
    const auto boxes = make_shared<op::v0::Parameter>(element::f32, Shape{1, 2, 4});
    const auto scores = make_shared<op::v0::Parameter>(element::f32, Shape{3, 2});
    const auto roisnum = make_shared<op::v0::Parameter>(element::i32, Shape{1});

    OV_EXPECT_THROW(
        ignore = make_shared<op::v9::MulticlassNms>(boxes, scores, roisnum, op::util::MulticlassNmsBase::Attributes()),
        NodeValidationFailure,
        HasSubstr("The first dimension of both 'boxes' and 'scores' must match"));
}

TYPED_TEST(type_prop, multiclass_nms_incorrect_boxes_rank2) {
    const auto boxes = make_shared<op::v0::Parameter>(element::f32, Shape{2, 2, 3});
    const auto scores = make_shared<op::v0::Parameter>(element::f32, Shape{2, 2, 2});

    OV_EXPECT_THROW(ignore = make_shared<TypeParam>(boxes, scores, this->attrs),
                    NodeValidationFailure,
                    HasSubstr("The last dimension of the 'boxes' input must be equal to 4"));
}

TYPED_TEST(type_prop, multiclass_nms_incorrect_output_type) {
    const auto boxes = make_shared<op::v0::Parameter>(element::f32, Shape{1, 2, 4});
    const auto scores = make_shared<op::v0::Parameter>(element::f32, Shape{1, 2, 2});
    this->attrs.output_type = element::f32;

    OV_EXPECT_THROW(ignore = make_shared<TypeParam>(boxes, scores, this->attrs),
                    NodeValidationFailure,
                    HasSubstr("Output type must be i32 or i64"));
}

TYPED_TEST(type_prop, multiclass_nms_incorrect_nms_topk) {
    const auto boxes = make_shared<op::v0::Parameter>(element::f32, Shape{1, 2, 4});
    const auto scores = make_shared<op::v0::Parameter>(element::f32, Shape{1, 2, 2});
    this->attrs.nms_top_k = -2;

    OV_EXPECT_THROW(ignore = make_shared<TypeParam>(boxes, scores, this->attrs),
                    NodeValidationFailure,
                    HasSubstr("The 'nms_top_k' must be great or equal -1"));
}

TYPED_TEST(type_prop, multiclass_nms_incorrect_keep_topk) {
    const auto boxes = make_shared<op::v0::Parameter>(element::f32, Shape{1, 2, 4});
    const auto scores = make_shared<op::v0::Parameter>(element::f32, Shape{1, 2, 2});
    this->attrs.keep_top_k = -2;

    OV_EXPECT_THROW(ignore = make_shared<TypeParam>(boxes, scores, this->attrs),
                    NodeValidationFailure,
                    HasSubstr("The 'keep_top_k' must be great or equal -1"));
}

TYPED_TEST(type_prop, multiclass_nms_incorrect_background_class) {
    const auto boxes = make_shared<op::v0::Parameter>(element::f32, Shape{1, 2, 4});
    const auto scores = make_shared<op::v0::Parameter>(element::f32, Shape{1, 2, 2});
    this->attrs.background_class = -2;

    OV_EXPECT_THROW(ignore = make_shared<TypeParam>(boxes, scores, this->attrs),
                    NodeValidationFailure,
                    HasSubstr("The 'background_class' must be great or equal -1"));
}

TYPED_TEST(type_prop, multiclass_nms_incorrect_eta) {
    const auto boxes = make_shared<op::v0::Parameter>(element::f32, Shape{1, 2, 4});
    const auto scores = make_shared<op::v0::Parameter>(element::f32, Shape{1, 2, 2});
    this->attrs.nms_eta = 2.0f;

    OV_EXPECT_THROW(ignore = make_shared<TypeParam>(boxes, scores, this->attrs),
                    NodeValidationFailure,
                    HasSubstr("The 'nms_eta' must be in close range [0, 1.0]"));
}

TYPED_TEST(type_prop, multiclass_nms_incorrect_input_type) {
    const auto boxes = make_shared<op::v0::Parameter>(element::f16, Shape{1, 2, 4});
    const auto scores = make_shared<op::v0::Parameter>(element::f32, Shape{1, 2, 2});

    OV_EXPECT_THROW(ignore = make_shared<TypeParam>(boxes, scores, this->attrs),
                    NodeValidationFailure,
                    HasSubstr("Expected 'boxes', 'scores' type is same."));
}

TYPED_TEST(type_prop, multiclass_nms_default_ctor) {
    const auto boxes = make_shared<op::v0::Parameter>(element::f32, Shape{2, 7, 4});
    const auto scores = make_shared<op::v0::Parameter>(element::f32, Shape{2, 5, 7});
    this->attrs.nms_top_k = 3;

    const auto nms = make_shared<TypeParam>();
    nms->set_arguments(OutputVector{boxes, scores});
    nms->set_attrs(this->attrs);
    nms->validate_and_infer_types();

    EXPECT_EQ(nms->get_output_partial_shape(0), (PartialShape{{0, 30}, 6}));
    EXPECT_EQ(nms->get_output_partial_shape(1), (PartialShape{{0, 30}, 1}));
    EXPECT_EQ(nms->get_output_partial_shape(2), (PartialShape{2}));
}

TYPED_TEST(type_prop, multiclass_nms_output_shape_1dim_dynamic) {
    auto boxes_shape = PartialShape{5, 2, 4};
    auto scores_shape = PartialShape{5, 3, 2};
    auto symbols = set_shape_symbols(boxes_shape);
    set_shape_symbols(scores_shape);

    const auto boxes = make_shared<op::v0::Parameter>(element::f32, boxes_shape);
    const auto scores = make_shared<op::v0::Parameter>(element::f32, scores_shape);

    const auto nms = make_shared<TypeParam>(boxes, scores, this->attrs);

    EXPECT_EQ(nms->get_output_partial_shape(0), (PartialShape{{0, 30}, 6}));
    EXPECT_EQ(nms->get_output_partial_shape(1), (PartialShape{{0, 30}, 1}));
    EXPECT_EQ(nms->get_output_partial_shape(2), (PartialShape{5}));
    EXPECT_THAT(get_shape_symbols(nms->get_output_partial_shape(0)), Each(nullptr));
    EXPECT_THAT(get_shape_symbols(nms->get_output_partial_shape(1)), Each(nullptr));
    EXPECT_THAT(get_shape_symbols(nms->get_output_partial_shape(2)), ElementsAre(symbols[0]));
}

TYPED_TEST(type_prop, multiclass_nms_output_shape_1dim_max_out) {
    const auto boxes = make_shared<op::v0::Parameter>(element::f32, Shape{2, 7, 4});
    const auto scores = make_shared<op::v0::Parameter>(element::f32, Shape{2, 5, 7});

    const auto nms = make_shared<TypeParam>(boxes, scores, this->attrs);

    ASSERT_EQ(nms->get_output_element_type(0), element::f32);
    ASSERT_EQ(nms->get_output_element_type(1), element::i64);
    ASSERT_EQ(nms->get_output_element_type(2), element::i64);

    // batch * class * box
    EXPECT_EQ(nms->get_output_partial_shape(0), PartialShape({Dimension(0, 2 * 5 * 7), Dimension(6)}));
    EXPECT_EQ(nms->get_output_partial_shape(1), PartialShape({Dimension(0, 2 * 5 * 7), 1}));
    EXPECT_EQ(nms->get_output_shape(2), (Shape{2}));
}

TYPED_TEST(type_prop, multiclass_nms_output_shape_1dim_nms_topk) {
    const auto boxes = make_shared<op::v0::Parameter>(element::f32, Shape{2, 7, 4});
    const auto scores = make_shared<op::v0::Parameter>(element::f32, Shape{2, 5, 7});
    this->attrs.nms_top_k = 3;

    const auto nms = make_shared<TypeParam>(boxes, scores, this->attrs);

    ASSERT_EQ(nms->get_output_element_type(0), element::f32);
    ASSERT_EQ(nms->get_output_element_type(1), element::i64);
    ASSERT_EQ(nms->get_output_element_type(2), element::i64);
    // batch * class * min(nms_topk, box)
    EXPECT_EQ(nms->get_output_partial_shape(0), PartialShape({Dimension(0, 2 * 5 * 3), Dimension(6)}));
    EXPECT_EQ(nms->get_output_partial_shape(1), PartialShape({Dimension(0, 2 * 5 * 3), 1}));
    EXPECT_EQ(nms->get_output_shape(2), (Shape{2}));
}

TYPED_TEST(type_prop, multiclass_nms_output_shape_1dim_keep_topk) {
    const auto boxes = make_shared<op::v0::Parameter>(element::f32, Shape{2, 7, 4});
    const auto scores = make_shared<op::v0::Parameter>(element::f32, Shape{2, 5, 7});
    this->attrs.nms_top_k = 3;
    this->attrs.keep_top_k = 8;

    const auto nms = make_shared<TypeParam>(boxes, scores, this->attrs);

    ASSERT_EQ(nms->get_output_element_type(0), element::f32);
    ASSERT_EQ(nms->get_output_element_type(1), element::i64);
    ASSERT_EQ(nms->get_output_element_type(2), element::i64);
    // batch * min(keep_topk, class * box))
    EXPECT_EQ(nms->get_output_partial_shape(0), PartialShape({Dimension(0, 2 * 8), Dimension(6)}));
    EXPECT_EQ(nms->get_output_partial_shape(1), PartialShape({Dimension(0, 2 * 8), 1}));
    EXPECT_EQ(nms->get_output_shape(2), (Shape{2}));
}

TYPED_TEST(type_prop, multiclass_nms_input_f16) {
    const auto boxes = make_shared<op::v0::Parameter>(element::f16, Shape{2, 7, 4});
    const auto scores = make_shared<op::v0::Parameter>(element::f16, Shape{2, 5, 7});

    const auto nms = make_shared<TypeParam>(boxes, scores, this->attrs);

    ASSERT_EQ(nms->get_output_element_type(0), element::f16);
    ASSERT_EQ(nms->get_output_element_type(1), element::i64);
    ASSERT_EQ(nms->get_output_element_type(2), element::i64);
    // batch * class * box
    EXPECT_EQ(nms->get_output_partial_shape(0), PartialShape({Dimension(0, 2 * 5 * 7), Dimension(6)}));
    EXPECT_EQ(nms->get_output_partial_shape(1), PartialShape({Dimension(0, 2 * 5 * 7), 1}));
    EXPECT_EQ(nms->get_output_shape(2), (Shape{2}));
}

TYPED_TEST(type_prop, multiclass_nms_output_shape_i32) {
    const auto boxes = make_shared<op::v0::Parameter>(element::f32, Shape{2, 7, 4});
    const auto scores = make_shared<op::v0::Parameter>(element::f32, Shape{2, 5, 7});
    this->attrs.output_type = element::i32;

    const auto nms = make_shared<TypeParam>(boxes, scores, this->attrs);

    ASSERT_EQ(nms->get_output_element_type(0), element::f32);
    ASSERT_EQ(nms->get_output_element_type(1), element::i32);
    ASSERT_EQ(nms->get_output_element_type(2), element::i32);
    // batch * class * box
    EXPECT_EQ(nms->get_output_partial_shape(0), PartialShape({Dimension(0, 2 * 5 * 7), Dimension(6)}));
    EXPECT_EQ(nms->get_output_partial_shape(1), PartialShape({Dimension(0, 2 * 5 * 7), 1}));
    EXPECT_EQ(nms->get_output_shape(2), (Shape{2}));
}

TYPED_TEST(type_prop, multiclass_nms_dynamic_boxes_and_scores) {
    const auto boxes = make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());
    const auto scores = make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());

    const auto nms = make_shared<TypeParam>(boxes, scores, this->attrs);

    ASSERT_EQ(nms->get_output_element_type(0), element::f32);
    ASSERT_EQ(nms->get_output_element_type(1), element::i64);
    ASSERT_EQ(nms->get_output_element_type(2), element::i64);
    EXPECT_EQ(nms->get_output_partial_shape(0), PartialShape({Dimension::dynamic(), 6}));
    EXPECT_EQ(nms->get_output_partial_shape(1), PartialShape({Dimension::dynamic(), 1}));
    EXPECT_EQ(nms->get_output_partial_shape(2), PartialShape({Dimension::dynamic()}));
}

TEST(type_prop2, multiclass_nms_dynamic_boxes_and_scores) {
    const auto boxes = make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());
    const auto scores = make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());
    const auto roisnum = make_shared<op::v0::Parameter>(element::i32, PartialShape::dynamic());

    const auto nms =
        make_shared<op::v9::MulticlassNms>(boxes, scores, roisnum, ov::op::util::MulticlassNmsBase::Attributes());

    ASSERT_EQ(nms->get_output_element_type(0), element::f32);
    ASSERT_EQ(nms->get_output_element_type(1), element::i64);
    ASSERT_EQ(nms->get_output_element_type(2), element::i64);
    EXPECT_EQ(nms->get_output_partial_shape(0), PartialShape({Dimension::dynamic(), 6}));
    EXPECT_EQ(nms->get_output_partial_shape(1), PartialShape({Dimension::dynamic(), 1}));
    EXPECT_EQ(nms->get_output_partial_shape(2), PartialShape({Dimension::dynamic()}));
}

TEST(type_prop2, multiclass_nms_interval_shapes_and_labels) {
    auto boxes_shape = PartialShape{2, 7, 4};
    auto scores_shape = PartialShape{2, 7};
    auto roisnum_shape = PartialShape{4};
    set_shape_symbols(boxes_shape);
    set_shape_symbols(scores_shape);
    auto symbols = set_shape_symbols(roisnum_shape);

    const auto boxes = make_shared<op::v0::Parameter>(element::f32, boxes_shape);
    const auto scores = make_shared<op::v0::Parameter>(element::f32, scores_shape);
    const auto roisnum = make_shared<op::v0::Parameter>(element::i32, roisnum_shape);

    const auto nms =
        make_shared<op::v9::MulticlassNms>(boxes, scores, roisnum, ov::op::util::MulticlassNmsBase::Attributes());

    ASSERT_EQ(nms->get_output_element_type(0), element::f32);
    ASSERT_EQ(nms->get_output_element_type(1), element::i64);
    ASSERT_EQ(nms->get_output_element_type(2), element::i64);
    EXPECT_EQ(nms->get_output_partial_shape(0), PartialShape({{0, 56}, 6}));
    EXPECT_EQ(nms->get_output_partial_shape(1), PartialShape({{0, 56}, 1}));
    EXPECT_EQ(nms->get_output_partial_shape(2), PartialShape({4}));
    EXPECT_THAT(get_shape_symbols(nms->get_output_partial_shape(0)), Each(nullptr));
    EXPECT_THAT(get_shape_symbols(nms->get_output_partial_shape(1)), Each(nullptr));
    EXPECT_THAT(get_shape_symbols(nms->get_output_partial_shape(2)), ElementsAre(symbols[0]));
}

TEST(type_prop2, multiclass_nms_static_shapes) {
    const auto boxes = make_shared<op::v0::Parameter>(element::f32, PartialShape{3, 7, 4});
    const auto scores = make_shared<op::v0::Parameter>(element::f32, PartialShape{3, 7});
    const auto roisnum = make_shared<op::v0::Parameter>(element::i32, PartialShape{4});
    ov::op::util::MulticlassNmsBase::Attributes attrs;
    attrs.nms_top_k = 3;
    attrs.keep_top_k = 8;

    const auto nms = make_shared<op::v9::MulticlassNms>(boxes, scores, roisnum, attrs);

    EXPECT_EQ(nms->get_output_partial_shape(0), PartialShape({{0, 32}, Dimension(6)}));
    EXPECT_EQ(nms->get_output_partial_shape(1), PartialShape({{0, 32}, 1}));
    EXPECT_EQ(nms->get_output_partial_shape(2), PartialShape({4}));
}

TYPED_TEST(type_prop, multiclass_nms_dynamic_boxes_and_scores2) {
    const auto boxes =
        make_shared<op::v0::Parameter>(element::f32, PartialShape({Dimension::dynamic(), Dimension::dynamic(), 4}));
    const auto scores = make_shared<op::v0::Parameter>(
        element::f32,
        PartialShape({Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic()}));

    const auto nms = make_shared<TypeParam>(boxes, scores, this->attrs);

    ASSERT_EQ(nms->get_output_element_type(0), element::f32);
    ASSERT_EQ(nms->get_output_element_type(1), element::i64);
    ASSERT_EQ(nms->get_output_element_type(2), element::i64);
    EXPECT_EQ(nms->get_output_partial_shape(0), PartialShape({Dimension::dynamic(), 6}));
    EXPECT_EQ(nms->get_output_partial_shape(1), PartialShape({Dimension::dynamic(), 1}));
    EXPECT_EQ(nms->get_output_partial_shape(2), PartialShape({Dimension::dynamic()}));
}

TEST(type_prop2, multiclass_nms_dynamic_boxes_and_scores2) {
    const auto boxes =
        make_shared<op::v0::Parameter>(element::f32, PartialShape({Dimension::dynamic(), Dimension::dynamic(), 4}));
    const auto scores =
        make_shared<op::v0::Parameter>(element::f32, PartialShape({Dimension::dynamic(), Dimension::dynamic()}));
    const auto roisnum = make_shared<op::v0::Parameter>(element::i32, PartialShape({Dimension::dynamic()}));

    const auto nms =
        make_shared<op::v9::MulticlassNms>(boxes, scores, roisnum, ov::op::util::MulticlassNmsBase::Attributes());

    ASSERT_EQ(nms->get_output_element_type(0), element::f32);
    ASSERT_EQ(nms->get_output_element_type(1), element::i64);
    ASSERT_EQ(nms->get_output_element_type(2), element::i64);
    EXPECT_EQ(nms->get_output_partial_shape(0), PartialShape({Dimension::dynamic(), 6}));
    EXPECT_EQ(nms->get_output_partial_shape(1), PartialShape({Dimension::dynamic(), 1}));
    EXPECT_EQ(nms->get_output_partial_shape(2), PartialShape({Dimension::dynamic()}));
}
