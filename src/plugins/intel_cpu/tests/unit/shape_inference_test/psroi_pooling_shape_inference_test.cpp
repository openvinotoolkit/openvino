// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gmock/gmock.h>

#include "common_test_utils/test_assertions.hpp"
#include "openvino/opsets/opset11.hpp"
#include "utils.hpp"

using namespace ov;
using namespace ov::intel_cpu;
using namespace testing;

class PSROIPoolingV0StaticShapeInferenceTest : public OpStaticShapeInferenceTest<op::v0::PSROIPooling> {
protected:
    void SetUp() override {
        output_shapes.resize(1);
    }

    float scale = 0.45f;
    size_t group = 3;
    int bins_x = 4;
    int bins_y = 3;
};

TEST_F(PSROIPoolingV0StaticShapeInferenceTest, default_ctor_avg_mode) {
    op = make_op();
    op->set_output_dim(5);
    op->set_group_size(3);
    op->set_spatial_scale(scale);
    op->set_mode("average");

    input_shapes = StaticShapeVector{{1, 45, 10, 10}, {3, 5}};
    auto shape_infer = make_shape_inference(op);
    output_shapes = shape_inference(op.get(), input_shapes);

    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes.front(), StaticShape({3, 5, 3, 3}));
}

TEST_F(PSROIPoolingV0StaticShapeInferenceTest, default_ctor_bilinear_mode) {
    op = make_op();
    op->set_output_dim(5);
    op->set_group_size(8);
    op->set_spatial_bins_x(5);
    op->set_spatial_bins_y(3);
    op->set_spatial_scale(scale);
    op->set_mode("bilinear");

    input_shapes = StaticShapeVector{{1, 75, 10, 10}, {2, 5}};
    auto shape_infer = make_shape_inference(op);
    output_shapes = shape_inference(op.get(), input_shapes);

    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes.front(), StaticShape({2, 5, 8, 8}));
}

TEST_F(PSROIPoolingV0StaticShapeInferenceTest, inputs_dynamic_rank) {
    const auto feat = std::make_shared<op::v0::Parameter>(element::f64, PartialShape::dynamic());
    const auto rois = std::make_shared<op::v0::Parameter>(element::f64, PartialShape::dynamic());

    op = make_op(feat, rois, 4, group, scale, 0, 0, "average");

    input_shapes = StaticShapeVector{{2, 36, 100, 100}, {10, 5}};
    output_shapes = shape_inference(op.get(), input_shapes);

    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes.front(), StaticShape({10, 4, 3, 3}));
}

TEST_F(PSROIPoolingV0StaticShapeInferenceTest, inputs_static_rank) {
    const auto feat = std::make_shared<op::v0::Parameter>(element::f64, PartialShape::dynamic(4));
    const auto rois = std::make_shared<op::v0::Parameter>(element::f64, PartialShape::dynamic(2));

    op = make_op(feat, rois, 2, 1, scale, bins_x, bins_y, "bilinear");

    input_shapes = StaticShapeVector{{2, 24, 20, 100}, {1, 5}};
    output_shapes = shape_inference(op.get(), input_shapes);

    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes.front(), StaticShape({1, 2, 1, 1}));
}

TEST_F(PSROIPoolingV0StaticShapeInferenceTest, invalid_rois_batch_size) {
    const auto feat = std::make_shared<op::v0::Parameter>(element::f64, PartialShape::dynamic(4));
    const auto rois = std::make_shared<op::v0::Parameter>(element::f64, PartialShape::dynamic());

    op = make_op(feat, rois, 2, 1, scale, bins_x, bins_y, "bilinear");

    input_shapes = StaticShapeVector{{2, 24, 20, 100}, {1, 6}};

    OV_EXPECT_THROW(shape_inference(op.get(), input_shapes),
                    NodeValidationFailure,
                    HasSubstr("The second dimension of ROIs input should contain batch id and box coordinates. This "
                              "dimension is expected to be equal to 5"));
}
