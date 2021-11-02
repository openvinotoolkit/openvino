// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <openvino/op/ops.hpp>
#include <openvino/op/parameter.hpp>
#include <topk_shape_inference.hpp>

#include "utils/shape_inference/static_shape.hpp"

using namespace ov;

static std::shared_ptr<op::v3::TopK> build_topk(PartialShape data_shape = PartialShape::dynamic(),
                                                int64_t axis = 1,
                                                int k_value = -1) {
    std::shared_ptr<op::Op> k;
    const auto data = std::make_shared<op::v0::Parameter>(element::f32, data_shape);
    if (k_value >= 0)
        k = std::static_pointer_cast<op::Op>(op::v0::Constant::create(element::i64, ov::Shape{}, {2}));
    else
        k = std::static_pointer_cast<op::Op>(std::make_shared<op::v0::Parameter>(element::i64, ov::PartialShape{}));
    return std::make_shared<op::v3::TopK>(data, k, axis, "max", "value");
}

TEST(StaticShapeInferenceTest, TopKv3) {
    const auto topk = build_topk(PartialShape::dynamic(), 1, 2);

    std::vector<PartialShape> input_shapes = {PartialShape{1, 10, 100}, PartialShape{}};
    std::vector<PartialShape> output_shapes = {PartialShape{}, PartialShape{}};
    shape_infer(topk.get(), input_shapes, output_shapes);

    ASSERT_EQ(output_shapes[0], PartialShape({1, 2, 100}));
    ASSERT_EQ(output_shapes[1], PartialShape({1, 2, 100}));

    std::vector<StaticShape> static_input_shapes = {StaticShape{1, 10, 100}, StaticShape{}};
    std::vector<StaticShape> static_output_shapes = {StaticShape{}, StaticShape{}};
    shape_infer(topk.get(), static_input_shapes, static_output_shapes);

    ASSERT_EQ(static_output_shapes[0], StaticShape({1, 2, 100}));
    ASSERT_EQ(static_output_shapes[1], StaticShape({1, 2, 100}));
}

TEST(StaticShapeInferenceTest, TopKv3_Dynamic) {
    const auto topk = build_topk();

    std::vector<PartialShape> input_shapes = {PartialShape{1, 10, 100}, PartialShape{}};
    std::vector<PartialShape> output_shapes = {PartialShape{}, PartialShape{}};
    shape_infer(topk.get(), input_shapes, output_shapes);
    ASSERT_EQ(output_shapes[0], PartialShape({1, Dimension(0, 10), 100}));
    ASSERT_EQ(output_shapes[1], PartialShape({1, Dimension(0, 10), 100}));
}

TEST(StaticShapeInferenceTest, TopKv3_StaticNoConstMap) {
    const auto topk = build_topk();

    std::vector<StaticShape> static_input_shapes = {StaticShape{1, 10, 100}, StaticShape{}};
    std::vector<StaticShape> static_output_shapes = {StaticShape{}, StaticShape{}};
    EXPECT_THROW(shape_infer(topk.get(), static_input_shapes, static_output_shapes), NodeValidationFailure);
}

TEST(StaticShapeInferenceTest, TopKv3_StaticWithConstMap) {
    const auto topk = build_topk();

    std::vector<StaticShape> static_input_shapes = {StaticShape{1, 10, 100}, StaticShape{}};
    std::vector<StaticShape> static_output_shapes = {StaticShape{}, StaticShape{}};

    auto k_value = std::make_shared<ngraph::runtime::HostTensor>(ngraph::element::Type_t::i32, ov::Shape({}));
    k_value->get_data_ptr<ngraph::element::Type_t::i32>()[0] = 2;
    shape_infer(topk.get(), static_input_shapes, static_output_shapes, {{1, k_value}});
    ASSERT_EQ(static_output_shapes[0], StaticShape({1, 2, 100}));
    ASSERT_EQ(static_output_shapes[1], StaticShape({1, 2, 100}));
}
