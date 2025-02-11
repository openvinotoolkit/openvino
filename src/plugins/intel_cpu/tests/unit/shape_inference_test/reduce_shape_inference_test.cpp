// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gmock/gmock.h>

#include "common_test_utils/test_assertions.hpp"
#include "utils.hpp"
#include "reduce_shape_inference.hpp"

using namespace ov;
using namespace ov::intel_cpu;
using namespace testing;

template <class TOp>
class ReduceStaticShapeInferenceTest : public OpStaticShapeInferenceTest<TOp> {
protected:
    void SetUp() override {
        this->output_shapes = StaticShapeVector(1);
    }
};

TYPED_TEST_SUITE_P(ReduceStaticShapeInferenceTest);

TYPED_TEST_P(ReduceStaticShapeInferenceTest, default_ctor) {
    this->op = this->make_op();
    this->op->set_keep_dims(true);
    this->input_shapes = StaticShapeVector{{1, 6, 7, 8, 4}, {3}};

    int32_t axes_val[] = {0, 1, 3};
    const auto constant_data = std::unordered_map<size_t, ov::Tensor>{{1, {element::i32, ov::Shape{3}, axes_val}}};
    this->output_shapes = shape_inference(this->op.get(), this->input_shapes, constant_data);

    EXPECT_EQ(this->output_shapes.size(), 1);
    EXPECT_EQ(this->output_shapes.front(), StaticShape({1, 1, 7, 1, 4}));
}

TYPED_TEST_P(ReduceStaticShapeInferenceTest, axes_constant) {
    const auto data = std::make_shared<op::v0::Parameter>(element::dynamic, PartialShape{-1, -1, -1, -1});
    const auto axes = std::make_shared<op::v0::Constant>(element::i32, ov::Shape{2}, std::vector<int32_t>{1, 3});

    this->op = this->make_op(data, axes, false);
    this->input_shapes = {StaticShape{3, 6, 5, 8}, StaticShape{2}};

    this->output_shapes = shape_inference(this->op.get(), this->input_shapes);

    EXPECT_EQ(this->output_shapes.size(), 1);
    EXPECT_EQ(this->output_shapes.front(), StaticShape({3, 5}));
}

TYPED_TEST_P(ReduceStaticShapeInferenceTest, axes_param) {
    const auto data = std::make_shared<op::v0::Parameter>(element::dynamic, PartialShape{-1, -1, -1, -1});
    const auto axes = std::make_shared<op::v0::Parameter>(element::i32, ov::Shape{2});

    this->op = this->make_op(data, axes, false);
    this->input_shapes = {StaticShape{3, 6, 5, 8}, StaticShape{2}};

    int32_t axes_val[] = {1, 3};
    const auto constant_data = std::unordered_map<size_t, ov::Tensor>{{1, {element::i32, ov::Shape{2}, axes_val}}};
    this->output_shapes = shape_inference(this->op.get(), this->input_shapes, constant_data);

    EXPECT_EQ(this->output_shapes.size(), 1);
    EXPECT_EQ(this->output_shapes.front(), StaticShape({3, 5}));
}

TYPED_TEST_P(ReduceStaticShapeInferenceTest, axes_constant_keep_dims) {
    const auto data = std::make_shared<op::v0::Parameter>(element::dynamic, PartialShape{-1, -1, -1, -1});
    const auto axes = std::make_shared<op::v0::Constant>(element::i32, ov::Shape{2}, std::vector<int32_t>{1, 3});

    this->op = this->make_op(data, axes, true);
    this->input_shapes = {StaticShape{3, 6, 5, 8}, StaticShape{2}};

    this->output_shapes = shape_inference(this->op.get(), this->input_shapes);

    EXPECT_EQ(this->output_shapes.size(), 1);
    EXPECT_EQ(this->output_shapes.front(), StaticShape({3, 1, 5, 1}));
}

TYPED_TEST_P(ReduceStaticShapeInferenceTest, axes_param_keep_dims) {
    const auto data = std::make_shared<op::v0::Parameter>(element::dynamic, PartialShape{-1, -1, -1, -1});
    const auto axes = std::make_shared<op::v0::Parameter>(element::i32, ov::Shape{2});

    this->op = this->make_op(data, axes, true);
    this->input_shapes = {StaticShape{3, 6, 5, 8}, StaticShape{2}};

    int32_t axes_val[] = {1, 3};
    const auto constant_data = std::unordered_map<size_t, ov::Tensor>{{1, {element::i32, ov::Shape{2}, axes_val}}};
    this->output_shapes = shape_inference(this->op.get(), this->input_shapes, constant_data);

    EXPECT_EQ(this->output_shapes.size(), 1);
    EXPECT_EQ(this->output_shapes.front(), StaticShape({3, 1, 5, 1}));
}

REGISTER_TYPED_TEST_SUITE_P(ReduceStaticShapeInferenceTest,
                            default_ctor,
                            axes_constant,
                            axes_param,
                            axes_param_keep_dims,
                            axes_constant_keep_dims);

using ReduceOpTypes =
    Types<op::v4::ReduceL1, op::v4::ReduceL2, op::v1::ReduceMax, op::v1::ReduceMean, op::v1::ReduceMin, op::v1::ReduceProd, op::v1::ReduceSum,
          op::v1::ReduceLogicalAnd, op::v1::ReduceLogicalOr>;
INSTANTIATE_TYPED_TEST_SUITE_P(shape_inference, ReduceStaticShapeInferenceTest, ReduceOpTypes);
