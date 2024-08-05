// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/test_assertions.hpp"
#include "openvino/opsets/opset10.hpp"
#include "utils.hpp"

using namespace ov;
using namespace ov::opset10;
using namespace ov::intel_cpu;
using namespace testing;

class ScatterElementsUpdateV3StaticShapeInferenceTest
    : public OpStaticShapeInferenceTest<op::v3::ScatterElementsUpdate> {
protected:
    void SetUp() override {
        output_shapes.resize(1);
    }
};

TEST_F(ScatterElementsUpdateV3StaticShapeInferenceTest, default_ctor) {
    const auto op = make_op();

    int32_t axis = 1;
    const auto const_data = std::unordered_map<size_t, ov::Tensor>{{3, {element::i32, Shape{1}, &axis}}};

    input_shapes = ShapeVector{{1000, 256, 10, 13}, {25, 125, 3, 1}, {25, 125, 3, 1}, {1}};
    const auto output_shapes = shape_inference(op.get(), input_shapes, const_data);

    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes[0], StaticShape({1000, 256, 10, 13}));
}

TEST_F(ScatterElementsUpdateV3StaticShapeInferenceTest, correct_inputs_axis_as_constant) {
    const auto d = std::make_shared<Parameter>(element::i32, PartialShape{-1, -1, -1, -1});
    const auto i = std::make_shared<Parameter>(element::i32, PartialShape{-1, -1, -1, -1});
    const auto u = std::make_shared<Parameter>(element::i32, PartialShape{-1, -1, -1, -1});
    const auto a = std::make_shared<Constant>(element::i64, Shape{}, -2);

    const auto op = make_op(d, i, u, a);

    input_shapes = ShapeVector{{2, 5, 10, 15}, {2, 1, 10, 15}, {2, 1, 10, 15}, {}};
    output_shapes = shape_inference(op.get(), input_shapes);

    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes[0], StaticShape({2, 5, 10, 15}));
}

TEST_F(ScatterElementsUpdateV3StaticShapeInferenceTest, params_are_dynamic_rank_axis_in_const_map) {
    const auto d = std::make_shared<Parameter>(element::i32, PartialShape::dynamic());
    const auto i = std::make_shared<Parameter>(element::i32, PartialShape::dynamic());
    const auto u = std::make_shared<Parameter>(element::i32, PartialShape::dynamic());
    const auto a = std::make_shared<Parameter>(element::u32, PartialShape::dynamic());

    const auto op = make_op(d, i, u, a);

    uint32_t axis = 2;
    const auto const_data = std::unordered_map<size_t, ov::Tensor>{{3, {element::u32, Shape{}, &axis}}};

    input_shapes = ShapeVector{{5000, 256, 10, 15}, {30, 25, 3, 3}, {30, 25, 3, 3}, {}};
    const auto output_shapes = shape_inference(op.get(), input_shapes, const_data);

    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes[0], StaticShape({5000, 256, 10, 15}));
}

TEST_F(ScatterElementsUpdateV3StaticShapeInferenceTest, incorrect_axis_value) {
    const auto d = std::make_shared<Parameter>(element::i32, PartialShape::dynamic());
    const auto i = std::make_shared<Parameter>(element::i32, PartialShape::dynamic());
    const auto u = std::make_shared<Parameter>(element::i32, PartialShape::dynamic());
    const auto a = std::make_shared<Parameter>(element::u32, PartialShape::dynamic());

    const auto op = make_op(d, i, u, a);

    uint32_t axis = 4;
    const auto const_data = std::unordered_map<size_t, ov::Tensor>{{3, {element::u32, Shape{}, &axis}}};

    input_shapes = ShapeVector{{5000, 256, 10, 15}, {30, 25, 3, 3}, {30, 25, 3, 3}, {}};
    OV_EXPECT_THROW(shape_inference(op.get(), input_shapes, const_data),
                    AssertFailure,
                    HasSubstr("Axis 4 out of the tensor rank range [-4, 3]"));
}
