// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gmock/gmock.h>

#include "openvino/op/matrix_nms.hpp"
#include "utils.hpp"

using namespace ov;
using namespace ov::intel_cpu;
using namespace testing;

class StaticShapeInferenceMatrixNmsV8Test : public OpStaticShapeInferenceTest<op::v8::MatrixNms> {
protected:
    using Attributes = op::v8::MatrixNms::Attributes;
    Attributes attrs;
};

TEST_F(StaticShapeInferenceMatrixNmsV8Test, default_ctor_no_args) {
    attrs.keep_top_k = 4;
    op = make_op();
    op->set_attrs(attrs);

    input_shapes = StaticShapeVector{{5, 2, 4}, {5, 3, 2}};
    output_shapes = shape_inference(op.get(), input_shapes);

    EXPECT_EQ(output_shapes, StaticShapeVector({{20, 6}, {20, 1}, {5}}));
}

TEST_F(StaticShapeInferenceMatrixNmsV8Test, inputs_static_rank) {
    const auto boxes = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(3));
    const auto scores = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(3));

    op = make_op(boxes, scores, attrs);

    input_shapes = StaticShapeVector{{3, 2, 4}, {3, 3, 2}};
    output_shapes = shape_inference(op.get(), input_shapes);

    EXPECT_EQ(output_shapes, StaticShapeVector({{18, 6}, {18, 1}, {3}}));
}

TEST_F(StaticShapeInferenceMatrixNmsV8Test, all_inputs_are_dynamic) {
    const auto boxes = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());
    const auto scores = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());

    op = make_op(boxes, scores, attrs);

    input_shapes = StaticShapeVector{{5, 2, 4}, {5, 3, 2}};
    output_shapes = shape_inference(op.get(), input_shapes);

    EXPECT_EQ(output_shapes, StaticShapeVector({{30, 6}, {30, 1}, {5}}));
}
