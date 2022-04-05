// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <openvino/core/coordinate_diff.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/parameter.hpp>
#include <openvino/op/space_to_batch.hpp>
#include <utils/shape_inference/shape_inference.hpp>
#include <utils/shape_inference/static_shape.hpp>

using namespace ov;
using namespace ov::intel_cpu;

static std::shared_ptr<op::v1::SpaceToBatch> build_space_to_batch(
    PartialShape data_shape = PartialShape::dynamic(ov::Rank(2)),
    PartialShape block_shape = PartialShape::dynamic(),
    PartialShape pads_begin_shape = PartialShape::dynamic(),
    PartialShape pad_end_shape = PartialShape::dynamic()) {
    auto data = std::make_shared<ov::op::v0::Parameter>(element::f32, data_shape);
    auto block = std::make_shared<ov::op::v0::Parameter>(element::i32, block_shape);
    auto pads_begin = std::make_shared<ov::op::v0::Parameter>(element::i32, pads_begin_shape);
    auto pads_end = std::make_shared<ov::op::v0::Parameter>(element::i32, pad_end_shape);

    auto space_to_batch = std::make_shared<op::v1::SpaceToBatch>(data, block, pads_begin, pads_end);
    return space_to_batch;
}

TEST(StaticShapeInferenceTest, SpaceToBatchTest) {
    auto space_to_batch = build_space_to_batch();
    int32_t block_val[] = {1, 6, 5, 1, 16};
    int32_t pads_begin_val[] = {0, 2, 0, 0, 0};
    int32_t pads_end_val[] = {0, 2, 1, 0, 0};
    auto block = std::make_shared<ngraph::runtime::HostTensor>(ngraph::element::Type_t::i32, ov::Shape{5}, block_val);
    auto pads_begin = std::make_shared<ngraph::runtime::HostTensor>(element::i32, ov::Shape{5}, pads_begin_val);
    auto pads_end = std::make_shared<ngraph::runtime::HostTensor>(element::i32, ov::Shape{5}, pads_end_val);

    const std::vector<StaticShape> input_shapes = {{2, 32, 64, 128, 256}, {5}, {5}, {5}};
    std::vector<StaticShape> output_shapes = {{}};

    std::map<size_t, std::shared_ptr<ngraph::runtime::HostTensor>> constant_data;
    constant_data[1] = block;
    constant_data[2] = pads_begin;
    constant_data[3] = pads_end;

    shape_inference(space_to_batch.get(), input_shapes, output_shapes, constant_data);
    ASSERT_EQ(output_shapes[0], (StaticShape{2 * 6 * 5 * 16, (32 + 2 + 2) / 6, (64 + 1) / 5, 128, 256 / 16}));
}

TEST(StaticShapeInferenceTest, SpaceToBatchThrowExceptionWithoutHostTensorData) {
    auto space_to_batch = build_space_to_batch();

    std::map<size_t, std::shared_ptr<ngraph::runtime::HostTensor>> constant_data;
    const std::vector<StaticShape> input_shapes = {{2, 32, 64, 128, 256}, {5}, {5}, {5}};
    std::vector<StaticShape> output_shapes = {{}};

    EXPECT_THROW(shape_inference(space_to_batch.get(), input_shapes, output_shapes), NodeValidationFailure);
}

TEST(StaticShapeInferenceTest, SpaceToBatchThrowExceptionWithMissingPadsHostTensorData) {
    auto space_to_batch = build_space_to_batch();

    int32_t block_val[] = {1, 6, 5, 1, 16};
    auto block = std::make_shared<ngraph::runtime::HostTensor>(ngraph::element::Type_t::i32, ov::Shape{5}, block_val);

    std::map<size_t, std::shared_ptr<ngraph::runtime::HostTensor>> constant_data;
    constant_data[1] = block;

    const std::vector<StaticShape> input_shapes = {{2, 32, 64, 128, 256}, {5}, {5}, {5}};
    std::vector<StaticShape> output_shapes = {{}};

    EXPECT_THROW(shape_inference(space_to_batch.get(), input_shapes, output_shapes), NodeValidationFailure);
}

TEST(StaticShapeInferenceTest, space_to_batch_output_with_const_inputs) {
    auto data = std::make_shared<ov::op::v0::Parameter>(element::f32, ov::PartialShape{-1, -1, -1, -1});
    auto block_shape = std::make_shared<ov::op::v0::Constant>(element::i64, ov::Shape{4}, std::vector<int64_t>{1, 12, 100, 2});
    auto pads_begin = std::make_shared<ov::op::v0::Constant>(element::i64, ov::Shape{4}, std::vector<int64_t>{0, 3, 38, 1});
    auto pads_end = std::make_shared<ov::op::v0::Constant>(element::i64, ov::Shape{4}, std::vector<int64_t>{0, 5, 38, 0});
    const auto space_to_batch = std::make_shared<ov::op::v1::SpaceToBatch>(data, block_shape, pads_begin, pads_end);
    std::vector<StaticShape> input_shapes = {{2, 100, 1024, 3}, {4}, {4}, {4}};
    std::vector<StaticShape> output_shapes = {{}};
    shape_inference(space_to_batch.get(), input_shapes, output_shapes);

    ASSERT_EQ(output_shapes[0], (StaticShape{2 * 12 * 100 * 2, (100 + 3 + 5) / 12, (1024 + 38 + 38) / 100, (3 + 1) / 2}));
}
