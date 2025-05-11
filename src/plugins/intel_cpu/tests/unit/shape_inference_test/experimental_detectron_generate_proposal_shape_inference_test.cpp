// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/test_assertions.hpp"
#include "experimental_detectron_generate_proposals_shape_inference.hpp"
#include "utils.hpp"

using namespace ov;
using namespace ov::intel_cpu;
using namespace testing;

class ExperimentalDetectronGenerateProposalsSingleImageV6StaticShapeInferenceTest
    : public OpStaticShapeInferenceTest<op::v6::ExperimentalDetectronGenerateProposalsSingleImage> {
protected:
    void SetUp() override {
        output_shapes.resize(2);
    }

    static op_type::Attributes make_attrs(int64_t post_nms_count) {
        return {0.0f, 0.0f, post_nms_count, 0};
    }
};

TEST_F(ExperimentalDetectronGenerateProposalsSingleImageV6StaticShapeInferenceTest, default_ctor) {
    op = make_op();
    op->set_attrs({0.0f, 0.0f, 100, 0});

    input_shapes = StaticShapeVector{{3}, {12, 4}, {3, 12, 15}, {5, 12, 15}};
    output_shapes = shape_inference(op.get(), input_shapes);

    EXPECT_EQ(output_shapes, StaticShapeVector({{100, 4}, {100}}));
}

TEST_F(ExperimentalDetectronGenerateProposalsSingleImageV6StaticShapeInferenceTest, inputs_dynamic_rank) {
    const auto im_info = std::make_shared<op::v0::Parameter>(element::f16, PartialShape::dynamic());
    const auto anchors = std::make_shared<op::v0::Parameter>(element::f16, PartialShape::dynamic());
    const auto deltas = std::make_shared<op::v0::Parameter>(element::f16, PartialShape::dynamic());
    const auto scores = std::make_shared<op::v0::Parameter>(element::f16, PartialShape::dynamic());
    op = make_op(im_info, anchors, deltas, scores, make_attrs(100));

    input_shapes = StaticShapeVector{{3}, {12, 4}, {3, 12, 15}, {5, 12, 15}};
    output_shapes = shape_inference(op.get(), input_shapes);

    EXPECT_EQ(output_shapes, StaticShapeVector({{100, 4}, {100}}));
}

TEST_F(ExperimentalDetectronGenerateProposalsSingleImageV6StaticShapeInferenceTest, inputs_static_rank) {
    const auto im_info = std::make_shared<op::v0::Parameter>(element::f16, PartialShape::dynamic(1));
    const auto anchors = std::make_shared<op::v0::Parameter>(element::f16, PartialShape::dynamic(2));
    const auto deltas = std::make_shared<op::v0::Parameter>(element::f16, PartialShape::dynamic(3));
    const auto scores = std::make_shared<op::v0::Parameter>(element::f16, PartialShape::dynamic(3));
    op = make_op(im_info, anchors, deltas, scores, make_attrs(1000));

    input_shapes = StaticShapeVector{{3}, {12, 4}, {3, 120, 15}, {5, 120, 15}};
    output_shapes = shape_inference(op.get(), input_shapes);

    EXPECT_EQ(output_shapes, StaticShapeVector({{1000, 4}, {1000}}));
}

TEST_F(ExperimentalDetectronGenerateProposalsSingleImageV6StaticShapeInferenceTest, im_info_bad_dimension) {
    const auto im_info = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(1));
    const auto anchors = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());
    const auto deltas = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());
    const auto scores = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(3));
    op = make_op(im_info, anchors, deltas, scores, make_attrs(40));

    input_shapes = StaticShapeVector{{4}, {12, 4}, {3, 120, 15}, {5, 120, 15}};
    OV_EXPECT_THROW(shape_inference(op.get(), input_shapes),
                    NodeValidationFailure,
                    HasSubstr("The 'input_im_info' shape is expected to be a compatible with [3]"));
}

TEST_F(ExperimentalDetectronGenerateProposalsSingleImageV6StaticShapeInferenceTest, deltas_not_3d) {
    const auto im_info = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(1));
    const auto anchors = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(2));
    const auto deltas = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());
    const auto scores = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(3));
    op = make_op(im_info, anchors, deltas, scores, make_attrs(40));

    input_shapes = StaticShapeVector{{3}, {12, 4}, {3, 120, 15, 1}, {5, 120, 15}};
    OV_EXPECT_THROW(shape_inference(op.get(), input_shapes),
                    NodeValidationFailure,
                    HasSubstr("The 'input_deltas' input is expected to be a 3D"));
}
