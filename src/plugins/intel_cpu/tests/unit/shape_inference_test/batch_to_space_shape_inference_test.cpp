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

class BatchToSpaceV1StaticShapeInferenceTest : public OpStaticShapeInferenceTest<op::v1::BatchToSpace> {
protected:
    void SetUp() override {
        output_shapes.resize(1);
    }

    std::shared_ptr<op_type> make_batch_to_space_dynamic() {
        const auto data = std::make_shared<Parameter>(element::f32, PartialShape::dynamic());
        const auto block = std::make_shared<Parameter>(element::i32, PartialShape::dynamic());
        const auto crops_begin = std::make_shared<Parameter>(element::i32, PartialShape::dynamic());
        const auto crops_end = std::make_shared<Parameter>(element::i32, PartialShape::dynamic());

        return make_op(data, block, crops_begin, crops_end);
    }
};

TEST_F(BatchToSpaceV1StaticShapeInferenceTest, default_ctor) {
    const auto op = make_op();

    int32_t block_val[] = {1, 6, 5, 1, 16};
    int32_t crops_begin_val[] = {0, 2, 0, 0, 0};
    int32_t crops_end_val[] = {0, 2, 1, 0, 0};

    const auto constant_data =
        std::unordered_map<size_t, ov::Tensor>{{1, {element::i32, ov::Shape{5}, block_val}},
                                               {2, {element::i32, ov::Shape{5}, crops_begin_val}},
                                               {3, {element::i32, ov::Shape{5}, crops_end_val}}};

    input_shapes = {{960, 6, 13, 128, 16}, {5}, {5}, {5}};
    output_shapes = shape_inference(op.get(), input_shapes, constant_data);
    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes[0], (StaticShape{960 / (6 * 5 * 16), 6 * 6 - 2 - 2, 13 * 5 - 1, 128, 16 * 16}));
}

TEST_F(BatchToSpaceV1StaticShapeInferenceTest, blocks_crops_in_constant_map) {
    op = make_batch_to_space_dynamic();

    int32_t block_val[] = {1, 6, 5, 1, 16};
    int32_t crops_begin_val[] = {0, 2, 0, 0, 0};
    int32_t crops_end_val[] = {0, 2, 1, 0, 0};

    const auto constant_data =
        std::unordered_map<size_t, ov::Tensor>{{1, {element::i32, ov::Shape{5}, block_val}},
                                               {2, {element::i32, ov::Shape{5}, crops_begin_val}},
                                               {3, {element::i32, ov::Shape{5}, crops_end_val}}};

    input_shapes = {{960, 6, 13, 128, 16}, {5}, {5}, {5}};

    output_shapes = shape_inference(op.get(), input_shapes, constant_data);
    EXPECT_EQ(output_shapes[0], (StaticShape{960 / (6 * 5 * 16), 6 * 6 - 2 - 2, 13 * 5 - 1, 128, 16 * 16}));
}

TEST_F(BatchToSpaceV1StaticShapeInferenceTest, blocs_crops_as_constants) {
    auto data = std::make_shared<Parameter>(element::f32, PartialShape{-1, -1, -1, -1});
    auto block_shape = std::make_shared<Constant>(element::i64, ov::Shape{4}, std::vector<int64_t>{1, 10, 5, 1});
    auto crops_begin = std::make_shared<Constant>(element::i64, ov::Shape{4}, std::vector<int64_t>{0, 3, 1, 0});
    auto crops_end = std::make_shared<Constant>(element::i64, ov::Shape{4}, std::vector<int64_t>{0, 3, 0, 0});

    op = make_op(data, block_shape, crops_begin, crops_end);
    input_shapes = {{100, 7, 13, 3}, {4}, {4}, {4}};
    output_shapes = shape_inference(op.get(), input_shapes);

    EXPECT_EQ(output_shapes[0], (StaticShape{100 / (10 * 5), 7 * 10 - 3 - 3, 13 * 5 - 1, 3}));
}

TEST_F(BatchToSpaceV1StaticShapeInferenceTest, missing_tensor_data) {
    auto op = make_batch_to_space_dynamic();

    int32_t block_val[] = {1, 6, 5, 1, 16};
    int32_t crops_end_val[] = {0, 2, 1, 0, 0};

    const auto constant_data = std::unordered_map<size_t, ov::Tensor>{{1, {element::i32, ov::Shape{5}, block_val}},
                                                                      {3, {element::i32, ov::Shape{5}, crops_end_val}}};

    input_shapes = {{960, 6, 13, 128, 16}, {5}, {5}, {5}};

    EXPECT_THROW(shape_inference(op.get(), input_shapes, constant_data), NodeValidationFailure);
}
