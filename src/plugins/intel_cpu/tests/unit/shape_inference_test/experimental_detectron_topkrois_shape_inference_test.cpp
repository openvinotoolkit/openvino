// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/test_assertions.hpp"
#include "experimental_detectron_topkrois_shape_inference.hpp"
#include "utils.hpp"

using namespace ov;
using namespace ov::intel_cpu;
using namespace testing;

class ExperimentalDetectronTopKROIsV6StaticShapeInferenceTest
    : public OpStaticShapeInferenceTest<op::v6::ExperimentalDetectronTopKROIs> {
protected:
    void SetUp() override {
        output_shapes.resize(1);
    }
};

TEST_F(ExperimentalDetectronTopKROIsV6StaticShapeInferenceTest, default_ctor) {
    op = make_op();
    op->set_max_rois(100);

    input_shapes = StaticShapeVector{{12, 4}, {12}};
    output_shapes = shape_inference(op.get(), input_shapes);

    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes.front(), StaticShape({100, 4}));
}

TEST_F(ExperimentalDetectronTopKROIsV6StaticShapeInferenceTest, inputs_dynamic_rank) {
    const auto input_rois = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());
    const auto rois_probs = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());
    op = make_op(input_rois, rois_probs, 5);

    input_shapes = StaticShapeVector{{10, 4}, {10}};
    output_shapes = shape_inference(op.get(), input_shapes);

    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes.front(), StaticShape({5, 4}));
}

TEST_F(ExperimentalDetectronTopKROIsV6StaticShapeInferenceTest, inputs_static_rank) {
    const auto input_rois = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(2));
    const auto rois_probs = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(1));
    op = make_op(input_rois, rois_probs, 15);

    input_shapes = StaticShapeVector{{100, 4}, {100}};
    output_shapes = shape_inference(op.get(), input_shapes);

    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes.front(), StaticShape({15, 4}));
}

TEST_F(ExperimentalDetectronTopKROIsV6StaticShapeInferenceTest, input_rois_not_2d) {
    const auto input_rois = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());
    const auto rois_probs = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(1));
    op = make_op(input_rois, rois_probs, 5);

    input_shapes = StaticShapeVector{{10, 4, 10}, {10}};

    OV_EXPECT_THROW(shape_inference(op.get(), input_shapes),
                    NodeValidationFailure,
                    HasSubstr("The 'input_rois' input is expected to be a 2D."));
}

TEST_F(ExperimentalDetectronTopKROIsV6StaticShapeInferenceTest, rois_prob_not_1d) {
    const auto input_rois = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(2));
    const auto rois_probs = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());
    op = make_op(input_rois, rois_probs, 5);

    input_shapes = StaticShapeVector{{10, 4}, {10, 2}};

    OV_EXPECT_THROW(shape_inference(op.get(), input_shapes),
                    NodeValidationFailure,
                    HasSubstr("The 'rois_probs' input is expected to be a 1D."));
}

TEST_F(ExperimentalDetectronTopKROIsV6StaticShapeInferenceTest, input_rois_second_dim_is_not_4) {
    const auto input_rois = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());
    const auto rois_probs = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(1));
    op = make_op(input_rois, rois_probs, 5);

    input_shapes = StaticShapeVector{{10, 5}, {10}};

    OV_EXPECT_THROW(shape_inference(op.get(), input_shapes),
                    NodeValidationFailure,
                    HasSubstr("The second dimension of 'input_rois' should be 4."));
}
