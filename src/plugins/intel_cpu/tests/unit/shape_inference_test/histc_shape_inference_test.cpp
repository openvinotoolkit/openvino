// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/test_assertions.hpp"
#include "openvino/op/histc.hpp"
#include "utils.hpp"

using namespace ov;
using namespace ov::intel_cpu;
using ov::op::v0::Parameter;

class HistcShapeInferenceTest : public OpStaticShapeInferenceTest<op::v17::Histc> {};

TEST_F(HistcShapeInferenceTest, output_is_bins_for_any_input) {
    const auto data = std::make_shared<Parameter>(element::f32, PartialShape::dynamic());
    const auto op = make_op(data, 7, 0.0, 1.0);
    for (const auto& in : {StaticShape{2, 3, 4}, StaticShape{}, StaticShape{0}}) {
        const auto output_shapes = shape_inference(op.get(), StaticShapeVector{in});
        ASSERT_EQ(output_shapes.size(), 1);
        EXPECT_EQ(output_shapes.front(), StaticShape({7}));
    }
}

TEST_F(HistcShapeInferenceTest, bins_read_at_inference) {
    const auto data = std::make_shared<Parameter>(element::f32, PartialShape::dynamic());
    const auto op = make_op(data, 7, 0.0, 1.0);
    op->set_bins(1);
    EXPECT_EQ(shape_inference(op.get(), StaticShapeVector{{3}}).front(), StaticShape({1}));
    op->set_bins(0);
    OV_EXPECT_THROW(shape_inference(op.get(), StaticShapeVector{{3}}),
                    NodeValidationFailure,
                    testing::HasSubstr("'bins' must be positive"));
}
