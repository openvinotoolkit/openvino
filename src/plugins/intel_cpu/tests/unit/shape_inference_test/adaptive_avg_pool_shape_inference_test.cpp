// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gmock/gmock.h>

#include "common_test_utils/test_assertions.hpp"
#include "openvino/opsets/opset10.hpp"
#include "utils.hpp"

using namespace ov;
using namespace ov::intel_cpu;
using namespace testing;

class AdaptiveAvgPoolV8StaticShapeInferenceTest : public OpStaticShapeInferenceTest<op::v8::AdaptiveAvgPool> {
protected:
    void SetUp() override {
        output_shapes.resize(1);
    }
};

TEST_F(AdaptiveAvgPoolV8StaticShapeInferenceTest, default_ctor) {
    int32_t spatial_dims[] = {10, 20};
    const std::unordered_map<size_t, ov::Tensor> const_data{{1, {element::i32, ov::Shape{2}, spatial_dims}}};

    op = make_op();
    input_shapes = StaticShapeVector{{1, 3, 1, 2}, {2}};
    output_shapes = shape_inference(op.get(), input_shapes, const_data);

    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes.front(), StaticShape({1, 3, 10, 20}));
}

TEST_F(AdaptiveAvgPoolV8StaticShapeInferenceTest, out_spatial_dims_as_constant) {
    const auto data = std::make_shared<op::v0::Parameter>(element::f64, PartialShape{-1, -1, -2});
    const auto out_shape = op::v0::Constant::create<int64_t>(element::i64, ov::Shape{1}, {17});

    op = make_op(data, out_shape);

    input_shapes = StaticShapeVector{{1, 3, 10}, {1}};
    output_shapes = shape_inference(op.get(), input_shapes);

    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes.front(), StaticShape({1, 3, 17}));
}

TEST_F(AdaptiveAvgPoolV8StaticShapeInferenceTest, out_spatial_dims_in_const_map) {
    const auto data = std::make_shared<op::v0::Parameter>(element::f64, PartialShape::dynamic());
    const auto out_shape = std::make_shared<op::v0::Parameter>(element::i32, PartialShape::dynamic());

    op = make_op(data, out_shape);

    int32_t spatial_dims[] = {9, 8, 7};
    const std::unordered_map<size_t, ov::Tensor> const_data{{1, {element::i32, ov::Shape{3}, spatial_dims}}};

    input_shapes = StaticShapeVector{{1, 3, 10, 2, 4}, {3}};
    output_shapes = shape_inference(op.get(), input_shapes, const_data);

    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes.front(), StaticShape({1, 3, 9, 8, 7}));
}

TEST_F(AdaptiveAvgPoolV8StaticShapeInferenceTest, out_spatial_dims_in_const_map_has_wrong_length) {
    const auto data = std::make_shared<op::v0::Parameter>(element::f64, PartialShape::dynamic());
    const auto out_shape = std::make_shared<op::v0::Parameter>(element::i32, PartialShape::dynamic());

    op = make_op(data, out_shape);

    int32_t spatial_dims[] = {9, 8};
    const std::unordered_map<size_t, ov::Tensor> const_data{{1, {element::i32, ov::Shape{2}, spatial_dims}}};

    input_shapes = StaticShapeVector{{1, 3, 10, 2, 4}, {3}};
    OV_EXPECT_THROW(shape_inference(op.get(), input_shapes, const_data),
                    ov::NodeValidationFailure,
                    HasSubstr("Number of spatial dimensions is not compatible with input data rank"));
}
