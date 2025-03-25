// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "utils.hpp"

using namespace ov;
using namespace ov::intel_cpu;

TEST(StaticShapeInferenceTest, ScatterUpdate_3D_axis_1) {
    auto data_param = std::make_shared<ov::op::v0::Parameter>(element::i32, PartialShape{-1, -1, -1});
    auto indices_param = std::make_shared<ov::op::v0::Parameter>(element::i32, PartialShape{-1, -1});
    auto updates_param = std::make_shared<ov::op::v0::Parameter>(element::i32, PartialShape{-1, -1, -1, -1});
    auto axis_param = std::make_shared<ov::op::v0::Parameter>(element::i32, PartialShape{-1});

    auto scatter_update = std::make_shared<op::v3::ScatterUpdate>(data_param, indices_param, updates_param, axis_param);

    int32_t axis_val[] = {1};
    std::unordered_map<size_t, ov::Tensor> constant_data;
    constant_data[3] = ov::Tensor(element::Type_t::i32, ov::Shape{1}, axis_val);
    std::vector<StaticShape> input_shapes = {StaticShape{2, 3, 4},
                                             StaticShape{2, 1},
                                             StaticShape{2, 2, 1, 4},
                                             StaticShape{1}},
                             output_shapes = {StaticShape{}};
    output_shapes = shape_inference(scatter_update.get(), input_shapes, constant_data);
    EXPECT_EQ(output_shapes[0], StaticShape({2, 3, 4}));
}

TEST(StaticShapeInferenceTest, ScatterUpdate_4D_axis_2) {
    auto data_param = std::make_shared<ov::op::v0::Parameter>(element::i32, PartialShape{-1, -1, -1, -1});
    auto indices_param = std::make_shared<ov::op::v0::Parameter>(element::i32, PartialShape{-1, -1});
    auto updates_param = std::make_shared<ov::op::v0::Parameter>(element::i32, PartialShape{-1, -1, -1, -1, -1});
    auto axis_param = std::make_shared<ov::op::v0::Parameter>(element::i32, PartialShape{-1});

    auto scatter_update = std::make_shared<op::v3::ScatterUpdate>(data_param, indices_param, updates_param, axis_param);

    int32_t axis_val[] = {2};
    std::unordered_map<size_t, ov::Tensor> constant_data;
    constant_data[3] = ov::Tensor(element::Type_t::i32, ov::Shape{1}, axis_val);
    std::vector<StaticShape> input_shapes = {StaticShape{1000, 256, 10, 15},
                                             StaticShape{125, 20},
                                             StaticShape{1000, 125, 20, 10, 15},
                                             StaticShape{1}},
                             output_shapes = {StaticShape{}};
    output_shapes = shape_inference(scatter_update.get(), input_shapes, constant_data);
    EXPECT_EQ(output_shapes[0], StaticShape({1000, 256, 10, 15}));
}

TEST(StaticShapeInferenceTest, ScatterUpdate_4D_incompatible_axis) {
    auto data_param = std::make_shared<ov::op::v0::Parameter>(element::i32, PartialShape{-1, -1, -1, -1});
    auto indices_param = std::make_shared<ov::op::v0::Parameter>(element::i32, PartialShape{-1, -1});
    auto updates_param = std::make_shared<ov::op::v0::Parameter>(element::i32, PartialShape{-1, -1, -1, -1, -1});
    auto axis_param = std::make_shared<ov::op::v0::Parameter>(element::i32, PartialShape{-1});

    auto scatter_update = std::make_shared<op::v3::ScatterUpdate>(data_param, indices_param, updates_param, axis_param);

    int32_t axis_val[] = {1};
    std::unordered_map<size_t, ov::Tensor> constant_data;
    constant_data[3] = ov::Tensor(element::Type_t::i32, ov::Shape{1}, axis_val);
    std::vector<StaticShape> input_shapes = {StaticShape{1000, 256, 10, 15},
                                             StaticShape{125, 20},
                                             StaticShape{1000, 125, 20, 10, 15},
                                             StaticShape{1}},
                             output_shapes = {StaticShape{}};
    output_shapes = shape_inference(scatter_update.get(), input_shapes, constant_data);
    EXPECT_EQ(output_shapes[0], StaticShape({1000, 256, 10, 15}));
}

TEST(StaticShapeInferenceTest, ScatterUpdate_axis_as_const) {
    auto data_param = std::make_shared<ov::op::v0::Parameter>(element::i32, PartialShape{-1, -1, -1, -1});
    auto indices_param = std::make_shared<ov::op::v0::Parameter>(element::i32, PartialShape{-1, -1});
    auto updates_param = std::make_shared<ov::op::v0::Parameter>(element::i32, PartialShape{-1, -1, -1, -1, -1});
    auto axis_const = std::make_shared<ov::op::v0::Constant>(element::i32, ov::Shape{1}, std::vector<int32_t>{1});

    auto scatter_update = std::make_shared<op::v3::ScatterUpdate>(data_param, indices_param, updates_param, axis_const);

    std::vector<StaticShape> input_shapes = {StaticShape{1000, 256, 10, 15},
                                             StaticShape{125, 20},
                                             StaticShape{1000, 125, 20, 10, 15},
                                             StaticShape{1}};
    const auto output_shapes = shape_inference(scatter_update.get(), input_shapes);
    EXPECT_EQ(output_shapes[0], StaticShape({1000, 256, 10, 15}));
}

TEST(StaticShapeInferenceTest, ScatterUpdate_dynamic_rank) {
    auto data_param = std::make_shared<ov::op::v0::Parameter>(element::i32, PartialShape::dynamic());
    auto indices_param = std::make_shared<ov::op::v0::Parameter>(element::i32, PartialShape::dynamic());
    auto updates_param = std::make_shared<ov::op::v0::Parameter>(element::i32, PartialShape::dynamic());
    auto axis_param = std::make_shared<ov::op::v0::Parameter>(element::i32, PartialShape::dynamic());

    auto scatter_update = std::make_shared<op::v3::ScatterUpdate>(data_param, indices_param, updates_param, axis_param);

    int32_t axis_val[] = {1};
    std::unordered_map<size_t, ov::Tensor> constant_data;
    constant_data[3] = ov::Tensor(element::Type_t::i32, ov::Shape{1}, axis_val);
    std::vector<StaticShape> input_shapes = {StaticShape{1000, 256, 10, 15},
                                             StaticShape{125, 20},
                                             StaticShape{1000, 125, 20, 10, 15},
                                             StaticShape{1}},
                             output_shapes = {StaticShape{}};
    output_shapes = shape_inference(scatter_update.get(), input_shapes, constant_data);
    EXPECT_EQ(output_shapes[0], StaticShape({1000, 256, 10, 15}));
}

TEST(StaticShapeInferenceTest, ScatterUpdate_params_dynamic_rank_incorrect_updates_shape) {
    auto data_param = std::make_shared<ov::op::v0::Parameter>(element::i32, PartialShape::dynamic());
    auto indices_param = std::make_shared<ov::op::v0::Parameter>(element::i32, PartialShape::dynamic());
    auto updates_param = std::make_shared<ov::op::v0::Parameter>(element::i32, PartialShape::dynamic());
    auto axis_param = std::make_shared<ov::op::v0::Parameter>(element::i32, PartialShape::dynamic());

    auto scatter_update = std::make_shared<op::v3::ScatterUpdate>(data_param, indices_param, updates_param, axis_param);

    int32_t axis_val[] = {1};
    std::unordered_map<size_t, ov::Tensor> constant_data;
    constant_data[3] = ov::Tensor(element::Type_t::i32, ov::Shape{1}, axis_val);

    // Incorrect rank of the third input shape
    std::vector<StaticShape> input_shapes = {StaticShape{1000, 256, 10, 15},
                                             StaticShape{125, 20, 1, 1, 1},
                                             StaticShape{1000, 125, 20, 10},
                                             StaticShape{1}},
                             output_shapes = {StaticShape{}};

    // ScatterUpdate shape_inference is implemented by usage of entryFirstPassthrough, no additional checks
    output_shapes = shape_inference(scatter_update.get(), input_shapes, constant_data);
    EXPECT_EQ(output_shapes[0], StaticShape({1000, 256, 10, 15}));
}
