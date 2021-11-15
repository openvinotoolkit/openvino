// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <openvino/op/parameter.hpp>
#include <openvino/op/roll.hpp>
#include <utils/shape_inference/shape_inference.hpp>
#include <utils/shape_inference/static_shape.hpp>

using namespace ov;

TEST(StaticShapeInferenceTest, RollTest) {
    auto arg =
        std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{Dimension::dynamic(), Dimension::dynamic()});
    auto shift = std::make_shared<ov::op::v0::Parameter>(element::i64, PartialShape{Dimension::dynamic()});
    auto axes = std::make_shared<ov::op::v0::Parameter>(element::i32, PartialShape{Dimension::dynamic()});

    auto roll = std::make_shared<ov::op::v7::Roll>(arg, shift, axes);

    {
        int64_t shift_val[] = {5};
        auto shift_tensor =
            std::make_shared<ngraph::runtime::HostTensor>(ngraph::element::Type_t::i64, ov::Shape{}, shift_val);
        std::map<size_t, std::shared_ptr<ngraph::runtime::HostTensor>> constant_data;
        constant_data[1] = shift_tensor;

        const std::vector<ov::StaticShape> input_shapes = {ov::StaticShape{3, 3, 4, 2},
                                                           ov::StaticShape{},
                                                           ov::StaticShape{}};
        std::vector<ov::StaticShape> output_shapes = {ov::StaticShape{}};
        shape_inference(roll.get(), input_shapes, output_shapes, constant_data)
            ASSERT_EQ(output_shapes[0], input_shapes[0]);
    }

    {
        int32_t axes_val[] = {0, 1, -1};
        auto axes_tensor = std::make_shared<ngraph::runtime::HostTensor>(element::i32, ov::Shape{3}, axes_val);

        std::map<size_t, std::shared_ptr<ngraph::runtime::HostTensor>> constant_data;
        constant_data[2] = axes_tensor;

        const std::vector<ov::StaticShape> input_shapes = {ov::StaticShape{3, 3, 3},
                                                           ov::StaticShape{3},
                                                           ov::StaticShape{3}};
        std::vector<ov::StaticShape> output_shapes = {ov::StaticShape{}};
        shape_inference(roll.get(), input_shapes, output_shapes, constant_data)
            ASSERT_EQ(output_shapes[0], input_shapes[0]);
    }
}