// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/test_assertions.hpp"
#include "histc_shape_inference.hpp"
#include "utils.hpp"

using namespace ov;
using namespace ov::intel_cpu;
using namespace testing;

class HistcV17StaticShapeInferenceTest : public OpStaticShapeInferenceTest<op::v17::Histc> {
    void SetUp() override {
        output_shapes.resize(1);
    }
};

TEST_F(HistcV17StaticShapeInferenceTest, default_bins) {
    const auto data = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());
    op = make_op(data);

    input_shapes = StaticShapeVector{{2, 3, 4}};
    output_shapes = shape_inference(op.get(), input_shapes);

    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes.front(), StaticShape({100}));
}

TEST_F(HistcV17StaticShapeInferenceTest, custom_bins) {
    const auto data = std::make_shared<op::v0::Parameter>(element::f16, PartialShape::dynamic());
    op = make_op(data, 7, -1.0, 1.0);

    input_shapes = StaticShapeVector{{6}};
    output_shapes = shape_inference(op.get(), input_shapes);

    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes.front(), StaticShape({7}));
}

TEST_F(HistcV17StaticShapeInferenceTest, empty_input) {
    const auto data = std::make_shared<op::v0::Parameter>(element::f64, PartialShape::dynamic());
    op = make_op(data, 3, 0.0, 0.0);

    input_shapes = StaticShapeVector{{0}};
    output_shapes = shape_inference(op.get(), input_shapes);

    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes.front(), StaticShape({3}));
}
