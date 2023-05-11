// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "utils.hpp"

using namespace ov;
using namespace ov::intel_cpu;

TEST(StaticShapeInferenceTest, prior_box_clustered0) {
    op::v0::PriorBoxClustered::Attributes attrs;
    attrs.widths = {86.0f, 13.0f, 57.0f, 39.0f, 68.0f, 34.0f, 142.0f, 50.0f, 23.0};
    attrs.heights = {44.0f, 10.0f, 30.0f, 19.0f, 94.0f, 32.0f, 61.0f, 53.0f, 17.0f};
    attrs.clip = false;
    attrs.step = 16.0f;
    attrs.offset = 0.5f;
    attrs.variances = {0.1f, 0.1f, 0.2f, 0.2f};

    auto layer_shape = std::make_shared<ov::op::v0::Parameter>(element::i32, PartialShape{-1});
    auto image_shape = std::make_shared<ov::op::v0::Parameter>(element::i32, PartialShape{-1});
    auto op =
        std::make_shared<ov::op::v0::PriorBoxClustered>(layer_shape, image_shape, attrs);
    int32_t layer_data[] = {10, 19};
    int32_t image_data[] = {180, 320};
    const std::map<size_t, HostTensorPtr> const_data{
        {0, std::make_shared<HostTensor>(element::i32, ov::Shape{2}, layer_data)},
        {1, std::make_shared<HostTensor>(element::i32, ov::Shape{2}, image_data)},
    };

    std::vector<StaticShape> static_input_shapes = {StaticShape{2}, StaticShape{2}};
    std::vector<StaticShape> static_output_shapes = {StaticShape{}};
    shape_inference(op.get(), static_input_shapes, static_output_shapes, const_data);

    ASSERT_EQ(static_output_shapes[0], StaticShape({2, 6840}));
}

TEST(StaticShapeInferenceTest, prior_box_clustered1) {
    op::v0::PriorBoxClustered::Attributes attrs;
    attrs.widths = {4.0f, 2.0f, 3.2f};
    attrs.heights = {1.0f, 2.0f, 1.1f};

    auto layer_shape = std::make_shared<ov::op::v0::Parameter>(element::i32, PartialShape{-1});
    auto image_shape = std::make_shared<ov::op::v0::Parameter>(element::i32, PartialShape{-1});
    auto op =
        std::make_shared<ov::op::v0::PriorBoxClustered>(layer_shape, image_shape, attrs);
    int32_t layer_data[] = {19, 19};
    int32_t image_data[] = {300, 300};
    const std::map<size_t, HostTensorPtr> const_data{
        {0, std::make_shared<HostTensor>(element::i32, ov::Shape{2}, layer_data)},
        {1, std::make_shared<HostTensor>(element::i32, ov::Shape{2}, image_data)},
    };

    std::vector<StaticShape> static_input_shapes = {StaticShape{2}, StaticShape{2}};
    std::vector<StaticShape> static_output_shapes = {StaticShape{}};
    shape_inference(op.get(), static_input_shapes, static_output_shapes, const_data);

    ASSERT_EQ(static_output_shapes[0], StaticShape({2, 4332}));
}

