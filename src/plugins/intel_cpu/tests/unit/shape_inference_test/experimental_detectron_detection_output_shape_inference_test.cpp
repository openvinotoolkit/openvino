// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/test_assertions.hpp"
#include "utils.hpp"

using namespace ov;
using namespace ov::intel_cpu;
using namespace testing;

class ExperimentalDetectronDetectionOutputV6StaticShapeInferenceTest
    : public OpStaticShapeInferenceTest<op::v6::ExperimentalDetectronDetectionOutput> {
protected:
    using Attrs = op::v6::ExperimentalDetectronDetectionOutput::Attributes;

    void SetUp() override {
        output_shapes.resize(2);
    }

    static Attrs make_attrs() {
        return {.05f, .5f, 4.1352f, 10, 20, 5, false, {10.0f, 10.0f, 5.0f, 5.0f}};
    }
};

TEST_F(ExperimentalDetectronDetectionOutputV6StaticShapeInferenceTest, default_ctor) {
    op = make_op();
    op->set_attrs({.05f, .5f, 4.1352f, 12, 20, 7, false, {10.0f, 10.0f, 5.0f, 5.0f}});

    input_shapes = StaticShapeVector{{10, 4}, {10, 48}, {10, 12}, {1, 3}};
    output_shapes = shape_inference(op.get(), input_shapes);

    EXPECT_EQ(output_shapes, StaticShapeVector({{7, 4}, {7}, {7}}));
}

TEST_F(ExperimentalDetectronDetectionOutputV6StaticShapeInferenceTest, inputs_dynamic_rank) {
    const auto rois = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());
    const auto deltas = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());
    const auto scores = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());
    const auto im_info = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());
    op = make_op(rois, deltas, scores, im_info, make_attrs());

    input_shapes = StaticShapeVector{{10, 4}, {10, 40}, {10, 10}, {1, 3}};
    output_shapes = shape_inference(op.get(), input_shapes);

    EXPECT_EQ(output_shapes, StaticShapeVector({{5, 4}, {5}, {5}}));
}

TEST_F(ExperimentalDetectronDetectionOutputV6StaticShapeInferenceTest, inputs_static_rank) {
    const auto rois = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(2));
    const auto deltas = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(2));
    const auto scores = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(2));
    const auto im_info = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(2));
    op = make_op(rois, deltas, scores, im_info, make_attrs());

    input_shapes = StaticShapeVector{{10, 4}, {10, 40}, {10, 10}, {1, 3}};
    output_shapes = shape_inference(op.get(), input_shapes);

    EXPECT_EQ(output_shapes, StaticShapeVector({{5, 4}, {5}, {5}}));
}

TEST_F(ExperimentalDetectronDetectionOutputV6StaticShapeInferenceTest, im_info_bad_dimension) {
    const auto rois = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());
    const auto deltas = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(2));
    const auto scores = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(2));
    const auto im_info = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());
    op = make_op(rois, deltas, scores, im_info, make_attrs());

    input_shapes = StaticShapeVector{{10, 4}, {10, 40}, {10, 10}, {3}};
    OV_EXPECT_THROW(shape_inference(op.get(), input_shapes),
                    NodeValidationFailure,
                    HasSubstr("Input image info shape must be compatible with [1,3]"));
}

TEST_F(ExperimentalDetectronDetectionOutputV6StaticShapeInferenceTest, deltas_not_2d) {
    const auto rois = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(2));
    const auto deltas = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(2));
    const auto scores = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());
    const auto im_info = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());
    op = make_op(rois, deltas, scores, im_info, make_attrs());

    input_shapes = StaticShapeVector{{10, 4}, {10, 40, 1}, {10, 10}, {1, 3}};
    OV_EXPECT_THROW(shape_inference(op.get(), input_shapes),
                    NodeValidationFailure,
                    HasSubstr("Input deltas rank must be equal to 2"));
}

TEST_F(ExperimentalDetectronDetectionOutputV6StaticShapeInferenceTest, rois_1st_dim_not_compatible) {
    const auto rois = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());
    const auto deltas = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());
    const auto scores = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());
    const auto im_info = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(2));
    op = make_op(rois, deltas, scores, im_info, make_attrs());

    input_shapes = StaticShapeVector{{9, 4}, {10, 40}, {10, 10}, {1, 3}};
    OV_EXPECT_THROW(
        shape_inference(op.get(), input_shapes),
        NodeValidationFailure,
        HasSubstr("The first dimension of inputs 'input_rois', 'input_deltas', 'input_scores' must be the compatible"));
}
