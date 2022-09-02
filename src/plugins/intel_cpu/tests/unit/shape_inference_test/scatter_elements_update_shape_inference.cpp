// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <openvino/op/ops.hpp>
#include <openvino/op/parameter.hpp>
#include <scatter_elements_update_shape_inference.hpp>
#include <utils/shape_inference/shape_inference.hpp>
#include <utils/shape_inference/static_shape.hpp>

using namespace ov;
using namespace ov::intel_cpu;

TEST(StaticShapeInferenceTest, ScatterElementsUpdateTest) {
    auto data_shape = std::make_shared<ov::op::v0::Parameter>(element::i32, PartialShape{-1, -1, -1, -1});
    auto indices_shape = std::make_shared<ov::op::v0::Parameter>(element::i32, PartialShape{-1, -1, -1, -1});
    auto updates_shape = std::make_shared<ov::op::v0::Parameter>(element::i32, PartialShape{-1, -1, -1, -1});
    auto axis_shape = std::make_shared<ov::op::v0::Parameter>(element::i32, PartialShape::dynamic());

    auto scatter_elements =
        std::make_shared<op::v3::ScatterElementsUpdate>(data_shape, indices_shape, updates_shape, axis_shape);

    int32_t axis_shape_val[] = {2};
    std::map<size_t, std::shared_ptr<ngraph::runtime::HostTensor>> constant_data;
    constant_data[3] =
        std::make_shared<ngraph::runtime::HostTensor>(ngraph::element::Type_t::i32, Shape{1}, axis_shape_val);
    std::vector<StaticShape> input_shapes = {StaticShape{1000, 256, 7, 7},
                                             StaticShape{125, 20, 7, 6},
                                             StaticShape{125, 20, 7, 6},
                                             StaticShape{1}},
                             output_shapes = {StaticShape{}};
    shape_inference(scatter_elements.get(), input_shapes, output_shapes, constant_data);

    ASSERT_EQ(output_shapes[0], StaticShape({1000, 256, 7, 7}));
}