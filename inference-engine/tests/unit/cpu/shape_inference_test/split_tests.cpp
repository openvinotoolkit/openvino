// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <openvino/op/ops.hpp>
#include <openvino/op/parameter.hpp>
#include <split_shape_inference.hpp>

#include "utils/shape_inference/static_shape.hpp"

using namespace ov;

template <ngraph::element::Type_t ET, typename T = typename element_type_traits<ET>::value_type>
static std::shared_ptr<ngraph::runtime::HostTensor> make_host_tensor(ov::Shape shape, std::initializer_list<T> values) {
    auto tensor = std::make_shared<ngraph::runtime::HostTensor>(ET, shape);

    T* ptr = tensor->get_data_ptr<T>();
    int i = 0;
    auto size = shape_size(shape);
    for (auto& v : values) {
        if (i >= size)
            break;
        ptr[i++] = v;
    }
    return tensor;
}

static std::shared_ptr<op::v1::Split> build_split(PartialShape data_shape, int64_t axis_value, size_t num_splits) {
    std::shared_ptr<op::Op> axis;
    const auto data = std::make_shared<op::v0::Parameter>(element::f32, data_shape);
    if (axis_value >= 0)
        axis = std::static_pointer_cast<op::Op>(op::v0::Constant::create(element::i64, ov::Shape{}, {axis_value}));
    else
        axis = std::static_pointer_cast<op::Op>(std::make_shared<op::v0::Parameter>(element::i64, ov::PartialShape{}));

    return std::make_shared<op::v1::Split>(data, axis, num_splits);
}

TEST(StaticShapeInferenceTest, SplitV1) {
    const auto axis_value = 1;
    const auto num_splits = 3;
    const auto op = build_split(PartialShape({2, 3, 4}), axis_value, num_splits);

    EXPECT_EQ(op->outputs().size(), num_splits);
    for (size_t i = 0; i < num_splits; ++i)
        EXPECT_EQ(op->get_output_shape(i), (ov::Shape{2, 1, 4}));

    std::vector<StaticShape> static_input_shapes = {StaticShape{2, 3, 4}, StaticShape{}};
    std::vector<StaticShape> static_output_shapes = {};
    shape_infer(op.get(), static_input_shapes, static_output_shapes);

    EXPECT_EQ(static_output_shapes.size(), num_splits);
    for (size_t i = 0; i < num_splits; ++i)
        EXPECT_EQ(static_output_shapes[i], (ov::StaticShape{2, 1, 4}));
}

TEST(StaticShapeInferenceTest, SplitV1_Dynamic) {
    const auto num_splits = 4;
    const auto op = build_split(PartialShape({2, 8, 4}), -1, num_splits);

    EXPECT_EQ(op->outputs().size(), num_splits);
    for (size_t i = 0; i < num_splits; ++i)
        EXPECT_EQ(op->get_output_partial_shape(i), (ov::PartialShape::dynamic(ov::Rank(3))));
}

TEST(StaticShapeInferenceTest, SplitV1_StaticNoConstMap) {
    const auto num_splits = 4;
    const auto op = build_split(PartialShape({2, 8, 4}), -1, num_splits);

    std::vector<StaticShape> static_input_shapes = {StaticShape{2, 8, 4}, StaticShape{}};
    std::vector<StaticShape> static_output_shapes = {};
    EXPECT_THROW(shape_infer(op.get(), static_input_shapes, static_output_shapes), NodeValidationFailure);
}

TEST(StaticShapeInferenceTest, SplitV1_StaticWithConstMap) {
    const auto num_splits = 4;
    const auto op = build_split(PartialShape({2, 8, 4}), -1, num_splits);

    std::vector<StaticShape> static_input_shapes = {StaticShape{2, 8, 4}, StaticShape{}};
    std::vector<StaticShape> static_output_shapes = {};

    auto axis_value = make_host_tensor<ngraph::element::Type_t::i32>(ov::Shape({}), {2});

    shape_infer(op.get(), static_input_shapes, static_output_shapes, {{1, axis_value}});

    EXPECT_EQ(static_output_shapes.size(), num_splits);
    for (size_t i = 0; i < num_splits; ++i)
        EXPECT_EQ(static_output_shapes[i], (ov::StaticShape{2, 8, 1}));
}
