// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "openvino/opsets/opset10.hpp"
#include "utils.hpp"

using namespace ov;
using namespace ov::intel_cpu;
using namespace ov::opset10;
using namespace testing;

class SpaceToBatchV1StaticShapeInferenceTest : public OpStaticShapeInferenceTest<op::v1::SpaceToBatch> {
protected:
    void SetUp() override {
        output_shapes.resize(1);
    }

    std::shared_ptr<op_type> make_space_to_batch_dynamic() {
        const auto data = std::make_shared<Parameter>(element::f32, PartialShape::dynamic());
        const auto block = std::make_shared<Parameter>(element::i32, PartialShape::dynamic());
        const auto pads_begin = std::make_shared<Parameter>(element::i32, PartialShape::dynamic());
        const auto pads_end = std::make_shared<Parameter>(element::i32, PartialShape::dynamic());

        return make_op(data, block, pads_begin, pads_end);
    }
};

TEST_F(SpaceToBatchV1StaticShapeInferenceTest, default_ctor) {
    const auto op = make_op();

    int32_t block_val[] = {1, 6, 5, 1, 16};
    int32_t pads_begin_val[] = {0, 2, 0, 0, 0};
    int32_t pads_end_val[] = {0, 2, 1, 0, 0};

    const auto constant_data = std::unordered_map<size_t, ov::Tensor>{{1, {element::i32, ov::Shape{5}, block_val}},
                                                                      {2, {element::i32, ov::Shape{5}, pads_begin_val}},
                                                                      {3, {element::i32, ov::Shape{5}, pads_end_val}}};

    input_shapes = {{2, 32, 64, 128, 256}, {5}, {5}, {5}};
    output_shapes = shape_inference(op.get(), input_shapes, constant_data);
    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes[0], (StaticShape{2 * 6 * 5 * 16, (32 + 2 + 2) / 6, (64 + 1) / 5, 128, 256 / 16}));
}

TEST_F(SpaceToBatchV1StaticShapeInferenceTest, blocks_pads_as_constants) {
    const auto data = std::make_shared<Parameter>(element::f32, PartialShape{-1, -1, -1, -1});
    const auto block_shape =
        std::make_shared<Constant>(element::i64, ov::Shape{4}, std::vector<int64_t>{1, 12, 100, 2});
    const auto pads_begin = std::make_shared<Constant>(element::i64, ov::Shape{4}, std::vector<int64_t>{0, 3, 38, 1});
    const auto pads_end = std::make_shared<Constant>(element::i64, ov::Shape{4}, std::vector<int64_t>{0, 5, 38, 0});

    const auto op = make_op(data, block_shape, pads_begin, pads_end);

    input_shapes = {{2, 100, 1024, 3}, {4}, {4}, {4}};
    output_shapes = shape_inference(op.get(), input_shapes);

    EXPECT_EQ(output_shapes[0],
              (StaticShape{2 * 12 * 100 * 2, (100 + 3 + 5) / 12, (1024 + 38 + 38) / 100, (3 + 1) / 2}));
}

TEST_F(SpaceToBatchV1StaticShapeInferenceTest, blocks_pads_in_constant_map) {
    const auto op = make_space_to_batch_dynamic();

    int32_t block_val[] = {1, 6, 5, 1, 16};
    int32_t pads_begin_val[] = {0, 2, 0, 0, 0};
    int32_t pads_end_val[] = {0, 2, 1, 0, 0};

    const auto constant_data = std::unordered_map<size_t, ov::Tensor>{{1, {element::i32, ov::Shape{5}, block_val}},
                                                                      {2, {element::i32, ov::Shape{5}, pads_begin_val}},
                                                                      {3, {element::i32, ov::Shape{5}, pads_end_val}}};

    input_shapes = {{2, 32, 64, 128, 256}, {5}, {5}, {5}};
    output_shapes = shape_inference(op.get(), input_shapes, constant_data);
    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes[0], (StaticShape{2 * 6 * 5 * 16, (32 + 2 + 2) / 6, (64 + 1) / 5, 128, 256 / 16}));
}

TEST_F(SpaceToBatchV1StaticShapeInferenceTest, throw_no_data_const_map) {
    const auto op = make_space_to_batch_dynamic();

    input_shapes = {{2, 32, 64, 128, 256}, {5}, {5}, {5}};
    EXPECT_THROW(shape_inference(op.get(), input_shapes), NodeValidationFailure);
}

TEST_F(SpaceToBatchV1StaticShapeInferenceTest, exception_missing_pads_data_in_const_map) {
    const auto op = make_space_to_batch_dynamic();

    int32_t block_val[] = {1, 6, 5, 1, 16};
    const auto constant_data = std::unordered_map<size_t, ov::Tensor>{{1, {element::i32, ov::Shape{5}, block_val}}};

    input_shapes = {{2, 32, 64, 128, 256}, {5}, {5}, {5}};

    EXPECT_THROW(shape_inference(op.get(), input_shapes), NodeValidationFailure);
}
