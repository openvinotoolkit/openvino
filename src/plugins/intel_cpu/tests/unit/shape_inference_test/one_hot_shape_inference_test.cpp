// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gmock/gmock.h>

#include "common_test_utils/test_assertions.hpp"
#include "one_hot_shape_inference.hpp"
#include "utils.hpp"

using namespace ov;
using namespace ov::intel_cpu;
using namespace testing;

TEST(StaticShapeInferenceTest, OneHotTestConstantInput) {
    auto indices = std::make_shared<op::v0::Parameter>(element::i64, PartialShape{-1});
    auto depth = op::v0::Constant::create(element::i64, ov::Shape{}, {2});
    auto on_value = op::v0::Constant::create(element::u32, ov::Shape{}, {5});
    auto off_value = op::v0::Constant::create(element::u32, ov::Shape{}, {10});
    int64_t axis = -1;
    auto ont_hot = std::make_shared<op::v1::OneHot>(indices, depth, on_value, off_value, axis);
    // Test StaticShape
    std::vector<StaticShape> static_input_shapes = {StaticShape{3}, StaticShape{}, StaticShape{}, StaticShape{}};
    const auto static_output_shapes = shape_inference(ont_hot.get(), static_input_shapes);
    ASSERT_EQ(static_output_shapes[0], (StaticShape{3, 2}));
}

TEST(StaticShapeInferenceTest, OneHotTestConstantMap) {
    auto indices = std::make_shared<op::v0::Parameter>(element::i64, PartialShape{-1});
    auto depth = std::make_shared<op::v0::Parameter>(element::i64, ov::Shape{});
    auto on_param = std::make_shared<op::v0::Parameter>(element::i32, ov::Shape{});
    auto off_param = std::make_shared<op::v0::Parameter>(element::i32, ov::Shape{});
    int64_t axis = -1;
    auto ont_hot = std::make_shared<op::v1::OneHot>(indices, depth, on_param, off_param, axis);

    int64_t depth_value[] = {2};
    int32_t on_value[] = {1};
    int32_t off_value[] = {0};

    const auto constant_data = std::unordered_map<size_t, ov::Tensor>{{1, {element::i64, ov::Shape{}, depth_value}},
                                                                      {2, {element::i32, ov::Shape{}, on_value}},
                                                                      {1, {element::i32, ov::Shape{}, off_value}}};

    std::vector<StaticShape> static_input_shapes = {StaticShape{3}, StaticShape{}, StaticShape{}, StaticShape{}};
    const auto static_output_shapes = shape_inference(ont_hot.get(), static_input_shapes, constant_data);
    EXPECT_EQ(static_output_shapes[0], (StaticShape{3, 2}));
}

TEST(StaticShapeInferenceTest, OneHotTestConstantMapDefaultCtor) {
    auto ont_hot = std::make_shared<op::v1::OneHot>();
    ont_hot->set_axis(-1);

    int64_t depth_value[] = {2};
    int32_t on_value[] = {1};
    int32_t off_value[] = {0};

    const auto constant_data = std::unordered_map<size_t, ov::Tensor>{{1, {element::i64, ov::Shape{}, depth_value}},
                                                                      {2, {element::i32, ov::Shape{}, on_value}},
                                                                      {1, {element::i32, ov::Shape{}, off_value}}};

    std::vector<StaticShape> static_input_shapes = {StaticShape{3}, StaticShape{}, StaticShape{}, StaticShape{}};

    const auto static_output_shapes = shape_inference(ont_hot.get(), static_input_shapes, constant_data);

    EXPECT_EQ(static_output_shapes[0], (StaticShape{3, 2}));
}

TEST(StaticShapeInferenceTest, OneHotTestConstantMapNegativeDepth) {
    auto indices = std::make_shared<op::v0::Parameter>(element::i64, PartialShape{-1});
    auto depth = std::make_shared<op::v0::Parameter>(element::i64, ov::Shape{});
    auto on_param = std::make_shared<op::v0::Parameter>(element::i32, ov::Shape{});
    auto off_param = std::make_shared<op::v0::Parameter>(element::i32, ov::Shape{});
    int64_t axis = -1;
    auto ont_hot = std::make_shared<op::v1::OneHot>(indices, depth, on_param, off_param, axis);

    int64_t depth_value[] = {-2};
    int32_t on_value[] = {1};
    int32_t off_value[] = {0};

    const auto constant_data = std::unordered_map<size_t, ov::Tensor>{{1, {element::i64, ov::Shape{}, depth_value}},
                                                                      {2, {element::i32, ov::Shape{}, on_value}},
                                                                      {1, {element::i32, ov::Shape{}, off_value}}};

    std::vector<StaticShape> static_input_shapes = {StaticShape{3}, StaticShape{}, StaticShape{}, StaticShape{}};

    OV_EXPECT_THROW(shape_inference(ont_hot.get(), static_input_shapes, constant_data),
                    ov::NodeValidationFailure,
                    HasSubstr("can't be negative"));
}
