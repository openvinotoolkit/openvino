// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/proposal.hpp"

#include "common_test_utils/test_assertions.hpp"
#include "common_test_utils/type_prop.hpp"

using namespace std;
using namespace ov;
using namespace testing;

// ------------------------------ V0 ------------------------------

TEST(type_prop, proposal_v0_invalid_class_probs_rank) {
    op::v0::Proposal::Attributes attrs;
    auto class_probs = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 2, 3});
    auto class_bbox_deltas = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 2, 3, 4});
    auto image_shape = make_shared<ov::op::v0::Parameter>(element::f32, Shape{3});

    OV_EXPECT_THROW(std::ignore = make_shared<op::v0::Proposal>(class_probs, class_bbox_deltas, image_shape, attrs),
                    NodeValidationFailure,
                    HasSubstr("Proposal layer shape class_probs should be rank 4 compatible"));
}

TEST(type_prop, proposal_v0_invalid_anchor_count) {
    op::v0::Proposal::Attributes attrs;
    auto class_probs = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 2, 3, 4});
    auto class_bbox_deltas = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 2, 3, 4});
    auto image_shape = make_shared<ov::op::v0::Parameter>(element::f32, Shape{3});

    OV_EXPECT_THROW(std::ignore = make_shared<op::v0::Proposal>(class_probs, class_bbox_deltas, image_shape, attrs),
                    NodeValidationFailure,
                    HasSubstr("Anchor number inconsistent between"));
}

TEST(type_prop, proposal_v0_invalid_class_bbox_deltas_rank) {
    op::v0::Proposal::Attributes attrs;
    auto class_probs = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 2, 3, 4});
    auto class_bbox_deltas = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 2, 3});
    auto image_shape = make_shared<ov::op::v0::Parameter>(element::f32, Shape{3});

    OV_EXPECT_THROW(std::ignore = make_shared<op::v0::Proposal>(class_probs, class_bbox_deltas, image_shape, attrs),
                    NodeValidationFailure,
                    HasSubstr("Proposal layer shape bbox_deltas should be rank 4 compatible"));
}

TEST(type_prop, proposal_v0_invalid_image_shape_rank) {
    op::v0::Proposal::Attributes attrs;
    auto class_probs = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 2, 3, 4});
    auto class_bbox_deltas = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 4, 3, 4});
    auto image_shape = make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 1});

    OV_EXPECT_THROW(std::ignore = make_shared<op::v0::Proposal>(class_probs, class_bbox_deltas, image_shape, attrs),
                    NodeValidationFailure,
                    HasSubstr("Image_shape must be 1-D tensor"));
}

TEST(type_prop, proposal_v0_invalid_image_shape_size) {
    op::v0::Proposal::Attributes attrs;
    auto class_probs = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 2, 3, 4});
    auto class_bbox_deltas = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 4, 3, 4});
    auto image_shape = make_shared<ov::op::v0::Parameter>(element::f32, Shape{5});

    OV_EXPECT_THROW(std::ignore = make_shared<op::v0::Proposal>(class_probs, class_bbox_deltas, image_shape, attrs),
                    NodeValidationFailure,
                    HasSubstr("Image_shape must be 1-D tensor and has got 3 or 4 elements (image_shape_shape[0]"));
}

TEST(type_prop, proposal_v0_default_ctor) {
    op::v0::Proposal::Attributes attrs;
    attrs.base_size = 1;
    attrs.pre_nms_topn = 20;
    attrs.post_nms_topn = 200;
    const size_t batch_size = 7;

    auto class_probs = make_shared<ov::op::v0::Parameter>(element::f16, Shape{batch_size, 12, 34, 62});
    auto class_bbox_deltas = make_shared<ov::op::v0::Parameter>(element::f16, Shape{batch_size, 24, 34, 62});
    auto image_shape = make_shared<ov::op::v0::Parameter>(element::f16, Shape{3});

    auto op = make_shared<op::v0::Proposal>();
    op->set_arguments(OutputVector{class_probs, class_bbox_deltas, image_shape});
    op->set_attrs(attrs);
    op->validate_and_infer_types();

    EXPECT_EQ(op->get_input_size(), 3);
    EXPECT_EQ(op->get_output_size(), 1);
    EXPECT_EQ(op->get_output_element_type(0), element::f16);
    EXPECT_EQ(op->get_output_shape(0), (Shape{batch_size * attrs.post_nms_topn, 5}));
}

TEST(type_prop, proposal_v0_shape_infer) {
    op::v0::Proposal::Attributes attrs;
    attrs.base_size = 1;
    attrs.pre_nms_topn = 20;
    attrs.post_nms_topn = 200;
    const size_t batch_size = 7;

    auto class_probs = make_shared<ov::op::v0::Parameter>(element::bf16, Shape{batch_size, 12, 34, 62});
    auto class_bbox_deltas = make_shared<ov::op::v0::Parameter>(element::bf16, Shape{batch_size, 24, 34, 62});
    auto image_shape = make_shared<ov::op::v0::Parameter>(element::bf16, Shape{3});
    auto op = make_shared<op::v0::Proposal>(class_probs, class_bbox_deltas, image_shape, attrs);

    EXPECT_EQ(op->get_output_element_type(0), element::bf16);
    EXPECT_EQ(op->get_output_shape(0), (Shape{batch_size * attrs.post_nms_topn, 5}));
}

TEST(type_prop, proposal_v0_dynamic_class_probs_dim1_batch_size_infer) {
    op::v0::Proposal::Attributes attrs;
    attrs.post_nms_topn = 1;
    const auto batch_size = Dimension(2);

    auto class_props_shape = PartialShape{-1, 2, 3, 4};
    auto class_bbox_shape = PartialShape{batch_size, 4, {0, 3}, {1, 4}};
    auto symbols = set_shape_symbols(class_props_shape);
    set_shape_symbols(class_bbox_shape);

    auto class_probs = make_shared<ov::op::v0::Parameter>(element::f32, class_props_shape);
    auto class_bbox_deltas = make_shared<ov::op::v0::Parameter>(element::f32, class_bbox_shape);
    auto image_shape = make_shared<ov::op::v0::Parameter>(element::f32, Shape{3});

    auto op = make_shared<op::v0::Proposal>(class_probs, class_bbox_deltas, image_shape, attrs);

    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{batch_size * attrs.post_nms_topn, 5}));
    EXPECT_THAT(get_shape_symbols(op->get_output_partial_shape(0)), ElementsAre(symbols[0], nullptr));
}

TEST(type_prop, proposal_v0_dynamic_bbox_deltas_dim1_batch_size_infer) {
    op::v0::Proposal::Attributes attrs;
    attrs.post_nms_topn = 1;
    const auto batch_size = Dimension(2);

    auto class_props_shape = PartialShape{batch_size, 2, {1, 3}, {1, 4}};
    auto class_bbox_shape = PartialShape{-1, 4, 3, 4};
    auto symbols = set_shape_symbols(class_props_shape);

    auto class_probs = make_shared<ov::op::v0::Parameter>(element::f64, class_props_shape);
    auto class_bbox_deltas = make_shared<ov::op::v0::Parameter>(element::f64, class_bbox_shape);
    auto image_shape = make_shared<ov::op::v0::Parameter>(element::f64, Shape{3});

    auto op = make_shared<op::v0::Proposal>(class_probs, class_bbox_deltas, image_shape, attrs);

    EXPECT_EQ(op->get_output_element_type(0), element::f64);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{batch_size * attrs.post_nms_topn, 5}));
    EXPECT_THAT(get_shape_symbols(op->get_output_partial_shape(0)), ElementsAre(symbols[0], nullptr));
}

TEST(type_prop, proposal_v0_dynamic_class_probs_bbox_deltas_dim1_batch_size_infer) {
    op::v0::Proposal::Attributes attrs;
    attrs.post_nms_topn = 1;
    auto class_probs = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, 2, 3, 4});
    auto class_bbox_deltas = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, 4, 3, 4});
    auto image_shape = make_shared<ov::op::v0::Parameter>(element::f32, Shape{3});

    auto op = make_shared<op::v0::Proposal>(class_probs, class_bbox_deltas, image_shape, attrs);

    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{-1, 5}));
    EXPECT_THAT(get_shape_symbols(op->get_output_partial_shape(0)), Each(nullptr));
}

TEST(type_prop, proposal_v0_dynamic_range_class_probs_bbox_deltas_dim1_batch_size_infer) {
    op::v0::Proposal::Attributes attrs;
    attrs.post_nms_topn = 2;

    auto class_props_shape = PartialShape{{8, 14}, 2, 3, 4};
    auto class_bbox_shape = PartialShape{{10, 15}, 4, {0, 3}, {1, 4}};
    set_shape_symbols(class_props_shape);
    set_shape_symbols(class_bbox_shape);

    auto class_probs = make_shared<ov::op::v0::Parameter>(element::f32, class_props_shape);
    auto class_bbox_deltas = make_shared<ov::op::v0::Parameter>(element::f32, class_bbox_shape);
    auto image_shape = make_shared<ov::op::v0::Parameter>(element::f32, Shape{3});

    auto op = make_shared<op::v0::Proposal>(class_probs, class_bbox_deltas, image_shape, attrs);
    EXPECT_EQ(op->get_output_partial_shape(0),
              (PartialShape{Dimension(10 * attrs.post_nms_topn, 14 * attrs.post_nms_topn), 5}));
    EXPECT_THAT(get_shape_symbols(op->get_output_partial_shape(0)), Each(nullptr));
}

TEST(type_prop, proposal_v0_dynamic_image_shape_shape_infer) {
    op::v0::Proposal::Attributes attrs;
    attrs.base_size = 2;
    attrs.pre_nms_topn = 20;
    attrs.post_nms_topn = 200;
    const auto batch_size = Dimension(7);

    auto class_props_shape = PartialShape{batch_size, 12, 34, 62};
    auto class_bbox_shape = PartialShape{batch_size, 24, 34, 62};
    set_shape_symbols(class_props_shape);
    set_shape_symbols(class_bbox_shape);

    auto class_probs = make_shared<ov::op::v0::Parameter>(element::f32, class_props_shape);
    auto class_bbox_deltas = make_shared<ov::op::v0::Parameter>(element::f32, class_bbox_shape);
    auto image_shape = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic());

    auto op = make_shared<op::v0::Proposal>(class_probs, class_bbox_deltas, image_shape, attrs);

    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{batch_size * attrs.post_nms_topn, 5}));
    EXPECT_THAT(get_shape_symbols(op->get_output_partial_shape(0)), Each(nullptr));
}

TEST(type_prop, proposal_v0_class_probs_dynamic_rank_but_batch_shape_defined_in_bbox) {
    op::v0::Proposal::Attributes attrs;
    attrs.post_nms_topn = 2;
    const auto batch_size = Dimension(7);

    auto class_probs = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic());
    auto class_bbox_deltas = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{batch_size, 24, 32, 32});
    auto image_shape = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic(1));

    auto op = make_shared<op::v0::Proposal>(class_probs, class_bbox_deltas, image_shape, attrs);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{batch_size * attrs.post_nms_topn, 5}));
}

TEST(type_prop, proposal_v0_bbox_dynamic_rank_but_batch_defined_in_class_probs) {
    op::v0::Proposal::Attributes attrs;
    attrs.post_nms_topn = 2;
    const auto batch_size = Dimension(7);

    auto class_probs = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{batch_size, 24, 32, 32});
    auto class_bbox_deltas = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic());
    auto image_shape = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic(1));

    auto op = make_shared<op::v0::Proposal>(class_probs, class_bbox_deltas, image_shape, attrs);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{batch_size * attrs.post_nms_topn, 5}));
}

TEST(type_prop, proposal_v0_everything_dynamic_shape_infer) {
    op::v0::Proposal::Attributes attrs;
    attrs.post_nms_topn = 1;
    auto class_probs = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic(4));
    auto class_bbox_deltas = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic(4));
    auto image_shape = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic(1));

    auto op = make_shared<op::v0::Proposal>(class_probs, class_bbox_deltas, image_shape, attrs);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{-1, 5}));
}

TEST(type_prop, proposal_v0_everything_dynamic_class_probs_dynamic_rank_shape_infer) {
    op::v0::Proposal::Attributes attrs;
    attrs.post_nms_topn = 1;
    auto class_probs = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic());
    auto class_bbox_deltas = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic(4));
    auto image_shape = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic(1));

    auto op = make_shared<op::v0::Proposal>(class_probs, class_bbox_deltas, image_shape, attrs);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{-1, 5}));
}

TEST(type_prop, proposal_v0_everything_dynamic_class_probs_bbox_deltas_dynamic_rank_shape_infer) {
    op::v0::Proposal::Attributes attrs;
    attrs.post_nms_topn = 1;
    auto class_probs = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic(Rank::dynamic()));
    auto class_bbox_deltas = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic(Rank::dynamic()));
    auto image_shape = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic(1));

    auto op = make_shared<op::v0::Proposal>(class_probs, class_bbox_deltas, image_shape, attrs);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{-1, 5}));
}

TEST(type_prop, proposal_v0_invalid_class_probs_dynamic) {
    op::v0::Proposal::Attributes attrs;
    auto class_probs = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic(3));
    auto class_bbox_deltas = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 4, 3, 4});
    auto image_shape = make_shared<ov::op::v0::Parameter>(element::f32, Shape{5});

    OV_EXPECT_THROW(std::ignore = make_shared<op::v0::Proposal>(class_probs, class_bbox_deltas, image_shape, attrs),
                    NodeValidationFailure,
                    HasSubstr("Proposal layer shape class_probs should be rank 4 compatible"));
}

TEST(type_prop, proposal_v0_invalid_bbox_deltas_dynamic) {
    op::v0::Proposal::Attributes attrs;
    auto class_probs = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 2, 3, 4});
    auto class_bbox_deltas = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic(3));
    auto image_shape = make_shared<ov::op::v0::Parameter>(element::f32, Shape{5});

    OV_EXPECT_THROW(std::ignore = make_shared<op::v0::Proposal>(class_probs, class_bbox_deltas, image_shape, attrs),
                    NodeValidationFailure,
                    HasSubstr("Proposal layer shape bbox_deltas should be rank 4 compatible"));
}

TEST(type_prop, proposal_v0_invalid_image_shape_dynamic) {
    op::v0::Proposal::Attributes attrs;
    auto class_probs = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 2, 3, 4});
    auto class_bbox_deltas = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 4, 3, 4});
    auto image_shape = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic(0));

    OV_EXPECT_THROW(std::ignore = make_shared<op::v0::Proposal>(class_probs, class_bbox_deltas, image_shape, attrs),
                    NodeValidationFailure,
                    HasSubstr("Image_shape must be 1-D tensor"));
}

TEST(type_prop, proposal_v0_invalid_class_probs_type) {
    op::v0::Proposal::Attributes attrs;
    auto class_probs = make_shared<ov::op::v0::Parameter>(element::i32, Shape{1, 2, 3, 4});
    auto class_bbox_deltas = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 4, 3, 4});
    auto image_shape = make_shared<ov::op::v0::Parameter>(element::f32, Shape{3});

    OV_EXPECT_THROW(std::ignore = make_shared<op::v0::Proposal>(class_probs, class_bbox_deltas, image_shape, attrs),
                    NodeValidationFailure,
                    HasSubstr("Proposal layer input class_probs should have floating point type"));
}

TEST(type_prop, proposal_v0_invalid_bbox_deltas_type) {
    op::v0::Proposal::Attributes attrs;
    auto class_probs = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 2, 3, 4});
    auto class_bbox_deltas = make_shared<ov::op::v0::Parameter>(element::i32, Shape{1, 4, 3, 4});
    auto image_shape = make_shared<ov::op::v0::Parameter>(element::f32, Shape{3});

    OV_EXPECT_THROW(std::ignore = make_shared<op::v0::Proposal>(class_probs, class_bbox_deltas, image_shape, attrs),
                    NodeValidationFailure,
                    HasSubstr("Proposal layer input bbox_deltas should have floating point type"));
}

TEST(type_prop, proposal_v0_invalid_image_shape_type) {
    op::v0::Proposal::Attributes attrs;
    auto class_probs = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 2, 3, 4});
    auto class_bbox_deltas = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 4, 3, 4});
    auto image_shape = make_shared<ov::op::v0::Parameter>(element::i32, Shape{3});

    OV_EXPECT_THROW(std::ignore = make_shared<op::v0::Proposal>(class_probs, class_bbox_deltas, image_shape, attrs),
                    NodeValidationFailure,
                    HasSubstr("Proposal layer input image_shape should have floating point type"));
}

// ------------------------------ V4 ------------------------------

TEST(type_prop, proposal_v4_invalid_class_probs_rank) {
    op::v0::Proposal::Attributes attrs;
    auto class_probs = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 2, 3});
    auto class_bbox_deltas = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 2, 3, 4});
    auto image_shape = make_shared<ov::op::v0::Parameter>(element::f32, Shape{3});

    OV_EXPECT_THROW(std::ignore = make_shared<op::v0::Proposal>(class_probs, class_bbox_deltas, image_shape, attrs),
                    NodeValidationFailure,
                    HasSubstr("Proposal layer shape class_probs should be rank 4 compatible"));
}

TEST(type_prop, proposal_v4_invalid_class_bbox_deltas_rank) {
    op::v0::Proposal::Attributes attrs;
    auto class_probs = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 2, 3, 4});
    auto class_bbox_deltas = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 2, 3});
    auto image_shape = make_shared<ov::op::v0::Parameter>(element::f32, Shape{3});

    OV_EXPECT_THROW(std::ignore = make_shared<op::v0::Proposal>(class_probs, class_bbox_deltas, image_shape, attrs),
                    NodeValidationFailure,
                    HasSubstr("Proposal layer shape bbox_deltas should be rank 4 compatible"));
}

TEST(type_prop, proposal_v4_invalid_image_shape_rank) {
    op::v0::Proposal::Attributes attrs;
    auto class_probs = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 2, 3, 4});
    auto class_bbox_deltas = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 4, 3, 4});
    auto image_shape = make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 1});

    OV_EXPECT_THROW(std::ignore = make_shared<op::v0::Proposal>(class_probs, class_bbox_deltas, image_shape, attrs),
                    NodeValidationFailure,
                    HasSubstr("Image_shape must be 1-D tensor"));
}

TEST(type_prop, proposal_v4_invalid_image_shape_size) {
    op::v0::Proposal::Attributes attrs;
    auto class_probs = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 2, 3, 4});
    auto class_bbox_deltas = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 4, 3, 4});
    auto image_shape = make_shared<ov::op::v0::Parameter>(element::f32, Shape{5});

    OV_EXPECT_THROW(std::ignore = make_shared<op::v0::Proposal>(class_probs, class_bbox_deltas, image_shape, attrs),
                    NodeValidationFailure,
                    HasSubstr("Image_shape must be 1-D tensor and has got 3 or 4 elements (image_shape_shape[0]"));
}

TEST(type_prop, proposal_v4_default_ctor) {
    op::v0::Proposal::Attributes attrs;
    attrs.base_size = 1;
    attrs.pre_nms_topn = 20;
    attrs.post_nms_topn = 200;
    const size_t batch_size = 7;

    auto class_probs = make_shared<ov::op::v0::Parameter>(element::f16, Shape{batch_size, 12, 34, 62});
    auto class_bbox_deltas = make_shared<ov::op::v0::Parameter>(element::f16, Shape{batch_size, 24, 34, 62});
    auto image_shape = make_shared<ov::op::v0::Parameter>(element::f16, Shape{3});

    auto op = make_shared<op::v4::Proposal>();
    op->set_arguments(OutputVector{class_probs, class_bbox_deltas, image_shape});
    op->set_attrs(attrs);
    op->validate_and_infer_types();

    EXPECT_EQ(op->get_input_size(), 3);
    EXPECT_EQ(op->get_output_size(), 2);

    EXPECT_THAT(op->outputs(), Each(Property("Element type", &Output<Node>::get_element_type, element::f16)));
    EXPECT_EQ(op->get_output_shape(0), (Shape{batch_size * attrs.post_nms_topn, 5}));
    EXPECT_EQ(op->get_output_shape(1), (Shape{batch_size * attrs.post_nms_topn}));
}

TEST(type_prop, proposal_v4_shape_infer) {
    op::v0::Proposal::Attributes attrs;
    attrs.base_size = 1;
    attrs.pre_nms_topn = 20;
    attrs.post_nms_topn = 200;
    const size_t batch_size = 7;

    auto class_probs = make_shared<ov::op::v0::Parameter>(element::f32, Shape{batch_size, 12, 34, 62});
    auto class_bbox_deltas = make_shared<ov::op::v0::Parameter>(element::f32, Shape{batch_size, 24, 34, 62});
    auto image_shape = make_shared<ov::op::v0::Parameter>(element::f32, Shape{3});
    auto op = make_shared<op::v4::Proposal>(class_probs, class_bbox_deltas, image_shape, attrs);

    EXPECT_THAT(op->outputs(), Each(Property("Element type", &Output<Node>::get_element_type, element::f32)));
    EXPECT_EQ(op->get_output_shape(0), (Shape{batch_size * attrs.post_nms_topn, 5}));
    EXPECT_EQ(op->get_output_shape(1), (Shape{batch_size * attrs.post_nms_topn}));
}

TEST(type_prop, proposal_v4_dynamic_class_probs_dim1_batch_size_infer) {
    op::v0::Proposal::Attributes attrs;
    attrs.post_nms_topn = 1;
    const auto batch_size = Dimension(2);

    auto class_props_shape = PartialShape{-1, 2, 3, 4};
    auto class_bbox_shape = PartialShape{batch_size, 4, {0, 3}, {1, 4}};
    auto symbols = set_shape_symbols(class_props_shape);
    set_shape_symbols(class_bbox_shape);

    auto class_probs = make_shared<ov::op::v0::Parameter>(element::f64, class_props_shape);
    auto class_bbox_deltas = make_shared<ov::op::v0::Parameter>(element::f64, class_bbox_shape);
    auto image_shape = make_shared<ov::op::v0::Parameter>(element::f64, Shape{3});

    auto op = make_shared<op::v4::Proposal>(class_probs, class_bbox_deltas, image_shape, attrs);

    EXPECT_THAT(op->outputs(), Each(Property("Element type", &Output<Node>::get_element_type, element::f64)));
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{batch_size * attrs.post_nms_topn, 5}));
    EXPECT_THAT(get_shape_symbols(op->get_output_partial_shape(0)), ElementsAre(symbols[0], nullptr));

    EXPECT_EQ(op->get_output_partial_shape(1), (PartialShape{batch_size * attrs.post_nms_topn}));
    EXPECT_THAT(get_shape_symbols(op->get_output_partial_shape(1)), ElementsAre(symbols[0]));
}

TEST(type_prop, proposal_v4_dynamic_bbox_deltas_dim1_batch_size_infer) {
    op::v0::Proposal::Attributes attrs;
    attrs.post_nms_topn = 1;
    const auto batch_size = Dimension(2);

    auto class_props_shape = PartialShape{batch_size, 2, 3, 4};
    auto class_bbox_shape = PartialShape{-1, 4, {0, 3}, {1, 4}};
    auto symbols = set_shape_symbols(class_props_shape);

    auto class_probs = make_shared<ov::op::v0::Parameter>(element::bf16, class_props_shape);
    auto class_bbox_deltas = make_shared<ov::op::v0::Parameter>(element::bf16, class_bbox_shape);
    auto image_shape = make_shared<ov::op::v0::Parameter>(element::bf16, Shape{3});

    auto op = make_shared<op::v4::Proposal>(class_probs, class_bbox_deltas, image_shape, attrs);

    EXPECT_THAT(op->outputs(), Each(Property("Element type", &Output<Node>::get_element_type, element::bf16)));
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{batch_size * attrs.post_nms_topn, 5}));
    EXPECT_THAT(get_shape_symbols(op->get_output_partial_shape(0)), ElementsAre(symbols[0], nullptr));

    EXPECT_EQ(op->get_output_partial_shape(1), (PartialShape{batch_size * attrs.post_nms_topn}));
    EXPECT_THAT(get_shape_symbols(op->get_output_partial_shape(1)), ElementsAre(symbols[0]));
}

TEST(type_prop, proposal_v4_dynamic_class_probs_bbox_deltas_dim1_batch_size_infer) {
    op::v0::Proposal::Attributes attrs;
    attrs.post_nms_topn = 1;

    auto class_props_shape = PartialShape{-1, 2, 3, 4};
    auto class_bbox_shape = PartialShape{-1, 4, 3, 4};
    auto symbols = set_shape_symbols(class_bbox_shape);

    auto class_probs = make_shared<ov::op::v0::Parameter>(element::f32, class_props_shape);
    auto class_bbox_deltas = make_shared<ov::op::v0::Parameter>(element::f32, class_bbox_shape);
    auto image_shape = make_shared<ov::op::v0::Parameter>(element::f32, Shape{3});

    auto op = make_shared<op::v4::Proposal>(class_probs, class_bbox_deltas, image_shape, attrs);

    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{-1, 5}));
    EXPECT_THAT(get_shape_symbols(op->get_output_partial_shape(0)), ElementsAre(symbols[0], nullptr));

    EXPECT_EQ(op->get_output_partial_shape(1), (PartialShape{-1}));
    EXPECT_THAT(get_shape_symbols(op->get_output_partial_shape(1)), ElementsAre(symbols[0]));
}

TEST(type_prop, proposal_v4_dynamic_image_shape_shape_infer) {
    op::v0::Proposal::Attributes attrs;
    attrs.base_size = 1;
    attrs.pre_nms_topn = 20;
    attrs.post_nms_topn = 200;
    const auto batch_size = Dimension(7);

    auto class_props_shape = PartialShape{batch_size, 2, 3, 4};
    auto class_bbox_shape = PartialShape{batch_size, 4, {0, 3}, {1, 4}};
    set_shape_symbols(class_props_shape);
    set_shape_symbols(class_bbox_shape);

    auto class_probs = make_shared<ov::op::v0::Parameter>(element::f32, class_props_shape);
    auto class_bbox_deltas = make_shared<ov::op::v0::Parameter>(element::f32, class_bbox_shape);
    auto image_shape = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic());

    auto op = make_shared<op::v4::Proposal>(class_probs, class_bbox_deltas, image_shape, attrs);

    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{batch_size * attrs.post_nms_topn, 5}));
    EXPECT_THAT(get_shape_symbols(op->get_output_partial_shape(0)), Each(nullptr));

    EXPECT_EQ(op->get_output_partial_shape(1), (PartialShape{batch_size * attrs.post_nms_topn}));
    EXPECT_THAT(get_shape_symbols(op->get_output_partial_shape(1)), ElementsAre(nullptr));
}

TEST(type_prop, proposal_v4_everything_dynamic_shape_infer) {
    op::v0::Proposal::Attributes attrs;
    attrs.post_nms_topn = 1;
    auto class_probs = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic(4));
    auto class_bbox_deltas = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic(4));
    auto image_shape = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic(1));

    auto op = make_shared<op::v4::Proposal>(class_probs, class_bbox_deltas, image_shape, attrs);

    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{-1, 5}));
    EXPECT_EQ(op->get_output_partial_shape(1), (PartialShape{-1}));
}

TEST(type_prop, proposal_v4_everything_dynamic_class_probs_dynamic_rank_shape_infer) {
    op::v0::Proposal::Attributes attrs;
    attrs.post_nms_topn = 1;
    auto class_probs = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic(Rank::dynamic()));
    auto class_bbox_deltas = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic(4));
    auto image_shape = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic(1));

    auto op = make_shared<op::v4::Proposal>(class_probs, class_bbox_deltas, image_shape, attrs);

    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{-1, 5}));
    EXPECT_EQ(op->get_output_partial_shape(1), (PartialShape{-1}));
}

TEST(type_prop, proposal_v4_everything_dynamic_class_probs_bbox_deltas_dynamic_rank_shape_infer) {
    op::v0::Proposal::Attributes attrs;
    attrs.post_nms_topn = 1;
    auto class_probs = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic(Rank::dynamic()));
    auto class_bbox_deltas = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic(Rank::dynamic()));
    auto image_shape = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic(1));

    auto op = make_shared<op::v4::Proposal>(class_probs, class_bbox_deltas, image_shape, attrs);

    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{-1, 5}));
    EXPECT_EQ(op->get_output_partial_shape(1), (PartialShape{-1}));
}

TEST(type_prop, proposal_v4_dynamic_range_class_probs_bbox_deltas_dim1_batch_size_infer) {
    op::v0::Proposal::Attributes attrs;
    attrs.post_nms_topn = 2;

    auto class_props_shape = PartialShape{{8, 14}, 2, 3, 4};
    auto class_bbox_shape = PartialShape{{10, 15}, 4, {0, 3}, {1, 4}};
    set_shape_symbols(class_props_shape);
    set_shape_symbols(class_bbox_shape);

    auto class_probs = make_shared<ov::op::v0::Parameter>(element::f32, class_props_shape);
    auto class_bbox_deltas = make_shared<ov::op::v0::Parameter>(element::f32, class_bbox_shape);
    auto image_shape = make_shared<ov::op::v0::Parameter>(element::f32, Shape{3});

    auto op = make_shared<op::v4::Proposal>(class_probs, class_bbox_deltas, image_shape, attrs);

    EXPECT_EQ(op->get_output_partial_shape(0),
              (PartialShape{Dimension(10 * attrs.post_nms_topn, 14 * attrs.post_nms_topn), 5}));
    EXPECT_THAT(get_shape_symbols(op->get_output_partial_shape(0)), Each(nullptr));

    EXPECT_EQ(op->get_output_partial_shape(1),
              (PartialShape{Dimension(10 * attrs.post_nms_topn, 14 * attrs.post_nms_topn)}));
    EXPECT_THAT(get_shape_symbols(op->get_output_partial_shape(1)), Each(nullptr));
}

TEST(type_prop, proposal_v4_class_dynamic_rank_but_batch_shape_defined_in_bbox) {
    op::v0::Proposal::Attributes attrs;
    attrs.post_nms_topn = 1;
    const auto batch_size = Dimension(7);

    auto class_probs = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic());
    auto class_bbox_deltas = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{batch_size, 24, 32, 32});
    auto image_shape = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic(1));

    auto op = make_shared<op::v4::Proposal>(class_probs, class_bbox_deltas, image_shape, attrs);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{batch_size * attrs.post_nms_topn, 5}));
    EXPECT_EQ(op->get_output_partial_shape(1), (PartialShape{batch_size * attrs.post_nms_topn}));
}

TEST(type_prop, proposal_v4_bbox_dynamic_rank_but_batch_defined_in_class_probs) {
    op::v0::Proposal::Attributes attrs;
    attrs.post_nms_topn = 1;
    const auto batch_size = Dimension(10);

    auto class_probs = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{batch_size, 24, 32, 32});
    auto class_bbox_deltas = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic());
    auto image_shape = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic(1));

    auto op = make_shared<op::v4::Proposal>(class_probs, class_bbox_deltas, image_shape, attrs);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{batch_size * attrs.post_nms_topn, 5}));
    EXPECT_EQ(op->get_output_partial_shape(1), (PartialShape{batch_size * attrs.post_nms_topn}));
}

TEST(type_prop, proposal_v4_invalid_class_probs_dynamic) {
    op::v0::Proposal::Attributes attrs;
    auto class_probs = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic(3));
    auto class_bbox_deltas = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 4, 3, 4});
    auto image_shape = make_shared<ov::op::v0::Parameter>(element::f32, Shape{5});

    OV_EXPECT_THROW(std::ignore = make_shared<op::v4::Proposal>(class_probs, class_bbox_deltas, image_shape, attrs),
                    NodeValidationFailure,
                    HasSubstr("Proposal layer shape class_probs should be rank 4 compatible"));
}

TEST(type_prop, proposal_v4_invalid_bbox_deltas_dynamic) {
    op::v0::Proposal::Attributes attrs;
    auto class_probs = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 2, 3, 4});
    auto class_bbox_deltas = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic(3));
    auto image_shape = make_shared<ov::op::v0::Parameter>(element::f32, Shape{5});

    OV_EXPECT_THROW(std::ignore = make_shared<op::v4::Proposal>(class_probs, class_bbox_deltas, image_shape, attrs),
                    NodeValidationFailure,
                    HasSubstr("Proposal layer shape bbox_deltas should be rank 4 compatible"));
}

TEST(type_prop, proposal_v4_invalid_image_shape_dynamic) {
    op::v0::Proposal::Attributes attrs;
    auto class_probs = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 2, 3, 4});
    auto class_bbox_deltas = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 4, 3, 4});
    auto image_shape = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic(0));

    OV_EXPECT_THROW(std::ignore = make_shared<op::v4::Proposal>(class_probs, class_bbox_deltas, image_shape, attrs),
                    NodeValidationFailure,
                    HasSubstr("Image_shape must be 1-D tensor"));
}

TEST(type_prop, proposal_v4_invalid_class_probs_type) {
    op::v0::Proposal::Attributes attrs;
    auto class_probs = make_shared<ov::op::v0::Parameter>(element::i32, Shape{1, 2, 3, 4});
    auto class_bbox_deltas = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 4, 3, 4});
    auto image_shape = make_shared<ov::op::v0::Parameter>(element::f32, Shape{3});

    OV_EXPECT_THROW(std::ignore = make_shared<op::v4::Proposal>(class_probs, class_bbox_deltas, image_shape, attrs),
                    NodeValidationFailure,
                    HasSubstr("Proposal layer input class_probs should have floating point type"));
}

TEST(type_prop, proposal_v4_invalid_bbox_deltas_type) {
    op::v0::Proposal::Attributes attrs;
    auto class_probs = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 2, 3, 4});
    auto class_bbox_deltas = make_shared<ov::op::v0::Parameter>(element::i32, Shape{1, 4, 3, 4});
    auto image_shape = make_shared<ov::op::v0::Parameter>(element::f32, Shape{3});

    OV_EXPECT_THROW(std::ignore = make_shared<op::v4::Proposal>(class_probs, class_bbox_deltas, image_shape, attrs),
                    NodeValidationFailure,
                    HasSubstr("Proposal layer input bbox_deltas should have floating point type"));
}

TEST(type_prop, proposal_v4_invalid_image_shape_type) {
    op::v0::Proposal::Attributes attrs;
    auto class_probs = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 2, 3, 4});
    auto class_bbox_deltas = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 4, 3, 4});
    auto image_shape = make_shared<ov::op::v0::Parameter>(element::i32, Shape{3});

    OV_EXPECT_THROW(std::ignore = make_shared<op::v4::Proposal>(class_probs, class_bbox_deltas, image_shape, attrs),
                    NodeValidationFailure,
                    HasSubstr("Proposal layer input image_shape should have floating point type"));
}
