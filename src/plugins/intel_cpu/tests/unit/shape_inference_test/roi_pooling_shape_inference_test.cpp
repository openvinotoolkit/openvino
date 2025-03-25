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

class ROIPoolingV0StaticShapeInferenceTest : public OpStaticShapeInferenceTest<op::v0::ROIPooling> {
protected:
    void SetUp() override {
        output_shapes.resize(1);
    }
};

TEST_F(ROIPoolingV0StaticShapeInferenceTest, default_ctor) {
    op = make_op();
    op->set_output_roi({3, 3});
    op->set_method("max");
    op->set_spatial_scale(0.34f);

    input_shapes = StaticShapeVector{{1, 5, 10, 10}, {2, 5}};
    auto shape_infer = make_shape_inference(op);
    output_shapes = shape_inference(op.get(), input_shapes);

    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes.front(), StaticShape({2, 5, 3, 3}));
}

TEST_F(ROIPoolingV0StaticShapeInferenceTest, inputs_dynamic_rank) {
    const auto feat = std::make_shared<op::v0::Parameter>(element::f64, PartialShape::dynamic());
    const auto rois = std::make_shared<op::v0::Parameter>(element::f64, PartialShape::dynamic());

    op = make_op(feat, rois, ov::Shape{5, 5}, 0.9f);

    input_shapes = StaticShapeVector{{2, 3, 100, 100}, {10, 5}};
    output_shapes = shape_inference(op.get(), input_shapes);

    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes.front(), StaticShape({10, 3, 5, 5}));
}

TEST_F(ROIPoolingV0StaticShapeInferenceTest, inputs_static_rank) {
    const auto feat = std::make_shared<op::v0::Parameter>(element::f64, PartialShape::dynamic(4));
    const auto rois = std::make_shared<op::v0::Parameter>(element::f64, PartialShape::dynamic(2));

    op = make_op(feat, rois, ov::Shape{7, 5}, 1.9f, "max");

    input_shapes = StaticShapeVector{{2, 3, 20, 100}, {10, 5}};
    output_shapes = shape_inference(op.get(), input_shapes);

    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes.front(), StaticShape({10, 3, 7, 5}));
}

TEST_F(ROIPoolingV0StaticShapeInferenceTest, invalid_rois_batch_size) {
    const auto feat = std::make_shared<op::v0::Parameter>(element::f64, PartialShape::dynamic(4));
    const auto rois = std::make_shared<op::v0::Parameter>(element::f64, PartialShape::dynamic());

    op = make_op(feat, rois, ov::Shape{7, 5}, 1.9f, "max");

    input_shapes = StaticShapeVector{{2, 3, 20, 100}, {10, 6}};

    OV_EXPECT_THROW(shape_inference(op.get(), input_shapes),
                    NodeValidationFailure,
                    HasSubstr("The second dimension of ROIs input should contain batch id and box coordinates. This "
                              "dimension is expected to be equal to 5"));
}
