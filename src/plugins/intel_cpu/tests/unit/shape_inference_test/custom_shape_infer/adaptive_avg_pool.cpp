// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gmock/gmock.h>

#include "common_test_utils/test_assertions.hpp"
#include "custom_shape_infer.hpp"
#include <ngraph/opsets/opset8.hpp>

using namespace ov;
using namespace ov::intel_cpu;
using namespace testing;

class AdaptiveAvgPoolV8CpuShapeInferenceTest : public unit_test::OpCpuShapeInferenceTest<op::v8::AdaptiveAvgPool> {
protected:
    void SetUp() override {
        output_shapes.resize(0);
    }
};

TEST_F(AdaptiveAvgPoolV8CpuShapeInferenceTest, default_ctor) {
    int32_t spatial_dims[] = {10, 20};
    const std::map<size_t, HostTensorPtr> const_data{
        {1, std::make_shared<HostTensor>(element::i32, ov::Shape{2}, spatial_dims)}};

    op = make_op();
    input_shapes = unit_test::ShapeVector{{1, 3, 1, 2}, {2}};
    output_shapes.push_back(StaticShape{1, 3, 10, 20});

    // implementation depends on some output information of the op
    op->set_output_type(0, element::i32, {-1, -1, -1, -1});
    unit_test::cpu_test_shape_infer(op.get(), input_shapes, output_shapes, const_data);
}

TEST_F(AdaptiveAvgPoolV8CpuShapeInferenceTest, out_spatial_dims_as_constant) {
    const auto data = std::make_shared<op::v0::Parameter>(element::f64, PartialShape{-1, -1, -2});
    const auto out_shape = op::v0::Constant::create<int64_t>(element::i64, ov::Shape{1}, {17});

    op = make_op(data, out_shape);

    input_shapes = unit_test::ShapeVector{{1, 3, 10}, {1}};

    output_shapes.push_back(StaticShape{1, 3, 17});
    unit_test::cpu_test_shape_infer(op.get(), input_shapes, output_shapes);
}

TEST_F(AdaptiveAvgPoolV8CpuShapeInferenceTest, out_spatial_dims_in_const_map) {
    const auto data = std::make_shared<op::v0::Parameter>(element::f64, PartialShape::dynamic());
    const auto out_shape = std::make_shared<op::v0::Parameter>(element::i32, PartialShape::dynamic());

    op = make_op(data, out_shape);

    int32_t spatial_dims[] = {9, 8, 7};
    const std::map<size_t, HostTensorPtr> const_data{
        {1, std::make_shared<HostTensor>(element::i32, ov::Shape{3}, spatial_dims)}};

    input_shapes = unit_test::ShapeVector{{1, 3, 10, 2, 4}, {3}};

    output_shapes.push_back(StaticShape{1, 3, 9, 8, 7});
    unit_test::cpu_test_shape_infer(op.get(), input_shapes, output_shapes, const_data);
}

TEST_F(AdaptiveAvgPoolV8CpuShapeInferenceTest, out_spatial_dims_in_const_map_has_wrong_length) {
    const auto data = std::make_shared<op::v0::Parameter>(element::f64, PartialShape::dynamic());
    const auto out_shape = std::make_shared<op::v0::Parameter>(element::i32, PartialShape::dynamic());

    op = make_op(data, out_shape);

    int32_t spatial_dims[] = {9, 8};
    const std::map<size_t, HostTensorPtr> const_data{
        {1, std::make_shared<HostTensor>(element::i32, ov::Shape{2}, spatial_dims)}};

    input_shapes = unit_test::ShapeVector{{1, 3, 10, 2, 4}, {3}};
    // TODO ,implementation should throw exception
    // ASSERT_THROW(unit_test::cpu_test_shape_infer(op.get(), input_shapes, output_shapes, const_data),
    //             InferenceEngine::GeneralError);
}
