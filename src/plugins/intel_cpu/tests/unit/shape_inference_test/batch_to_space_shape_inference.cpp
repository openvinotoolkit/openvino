// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <openvino/core/coordinate_diff.hpp>
#include <openvino/op/batch_to_space.hpp>
#include <openvino/op/parameter.hpp>
#include <openvino/op/constant.hpp>
#include <utils/shape_inference/shape_inference.hpp>
#include <utils/shape_inference/static_shape.hpp>

using namespace ov;
using namespace ov::intel_cpu;

static std::shared_ptr<op::v1::BatchToSpace> make_batch_to_space(
    PartialShape data_shape = PartialShape::dynamic(ov::Rank(2)),
    PartialShape block_shape = PartialShape::dynamic(),
    PartialShape crops_begin_shape = PartialShape::dynamic(),
    PartialShape crops_end_shape = PartialShape::dynamic()) {
    auto data = std::make_shared<ov::op::v0::Parameter>(element::f32, data_shape);
    auto block = std::make_shared<ov::op::v0::Parameter>(element::i32, block_shape);
    auto crops_begin = std::make_shared<ov::op::v0::Parameter>(element::i32, crops_begin_shape);
    auto crops_end = std::make_shared<ov::op::v0::Parameter>(element::i32, crops_end_shape);

    const auto batch_to_space = std::make_shared<op::v1::BatchToSpace>(data, block, crops_begin, crops_end);
    return batch_to_space;
}

TEST(StaticShapeInferenceTest, BatchToSpaceWithHostTensorData) {
    auto space_to_batch = make_batch_to_space();
    int32_t block_val[] = {1, 6, 5, 1, 16};
    int32_t pads_begin_val[] = {0, 2, 0, 0, 0};
    int32_t pads_end_val[] = {0, 2, 1, 0, 0};
    auto block = std::make_shared<ngraph::runtime::HostTensor>(ngraph::element::Type_t::i32, ov::Shape{5}, block_val);
    auto crops_begin = std::make_shared<ngraph::runtime::HostTensor>(element::i32, ov::Shape{5}, pads_begin_val);
    auto crops_end = std::make_shared<ngraph::runtime::HostTensor>(element::i32, ov::Shape{5}, pads_end_val);

    const std::vector<StaticShape> input_shapes = {{960, 6, 13, 128, 16}, {5}, {5}, {5}};
    std::vector<StaticShape> output_shapes = {{}};

    std::map<size_t, std::shared_ptr<ngraph::runtime::HostTensor>> constant_data;
    constant_data[1] = block;
    constant_data[2] = crops_begin;
    constant_data[3] = crops_end;

    shape_inference(space_to_batch.get(), input_shapes, output_shapes, constant_data);
    ASSERT_EQ(output_shapes[0], (StaticShape{960 / (6 * 5 * 16), 6 * 6 - 2 - 2, 13 * 5 - 1, 128, 16 * 16}));
}

TEST(StaticShapeInferenceTest, BatchToSpaceWithMissingTensorData) {
    auto batch_to_space = make_batch_to_space();
    int32_t block_val[] = {1, 6, 5, 1, 16};
    int32_t pads_end_val[] = {0, 2, 1, 0, 0};
    auto block = std::make_shared<ngraph::runtime::HostTensor>(ngraph::element::Type_t::i32, ov::Shape{5}, block_val);
    auto crops_end = std::make_shared<ngraph::runtime::HostTensor>(element::i32, ov::Shape{5}, pads_end_val);

    const std::vector<StaticShape> input_shapes = {{960, 6, 13, 128, 16}, {5}, {5}, {5}};
    std::vector<StaticShape> output_shapes = {{}};

    std::map<size_t, std::shared_ptr<ngraph::runtime::HostTensor>> constant_data;
    constant_data[1] = block;
    constant_data[3] = crops_end;

    EXPECT_THROW(shape_inference(batch_to_space.get(), input_shapes, output_shapes, constant_data), NodeValidationFailure);
}

TEST(StaticShapeInferenceTest, batch_to_space_output_with_const_inputs) {
    auto data = std::make_shared<ov::op::v0::Parameter>(element::f32, ov::PartialShape{-1, -1, -1, -1});
    auto block_shape = std::make_shared<ov::op::v0::Constant>(element::i64, ov::Shape{4}, std::vector<int64_t>{1, 10, 5, 1});
    auto crops_begin = std::make_shared<ov::op::v0::Constant>(element::i64, ov::Shape{4}, std::vector<int64_t>{0, 3, 1, 0});
    auto crops_end = std::make_shared<ov::op::v0::Constant>(element::i64, ov::Shape{4}, std::vector<int64_t>{0, 3, 0, 0});
    const auto batch_to_space = std::make_shared<ov::op::v1::BatchToSpace>(data, block_shape, crops_begin, crops_end);
    std::vector<StaticShape> input_shapes = {{100, 7, 13, 3}, {4}, {4}, {4}};
    std::vector<StaticShape> output_shapes = {{}};
    shape_inference(batch_to_space.get(), input_shapes, output_shapes);

    ASSERT_EQ(output_shapes[0], (StaticShape{100 / (10 * 5), 7 * 10 - 3 - 3, 13 * 5 - 1, 3}));
}
