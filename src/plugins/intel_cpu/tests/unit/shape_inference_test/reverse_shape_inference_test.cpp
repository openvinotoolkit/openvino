// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/test_assertions.hpp"
#include "openvino/op/reverse.hpp"
#include "openvino/opsets/opset1.hpp"
#include "utils.hpp"

using namespace ov;
using namespace ov::intel_cpu;
using namespace ov::opset1;
using namespace testing;

class ReverseV1StaticShapeInferenceTest : public OpStaticShapeInferenceTest<op::v1::Reverse> {
protected:
    void SetUp() override {
        output_shapes.resize(1);

        data = std::make_shared<Parameter>(element::f32, PartialShape::dynamic());
        axes = std::make_shared<Parameter>(element::i32, PartialShape::dynamic());
    }

    std::shared_ptr<Parameter> data, axes;
};

TEST_F(ReverseV1StaticShapeInferenceTest, axes_index_as_constant) {
    auto op = make_op(data, Constant::create(element::i16, ov::Shape{4}, {-1000, 1, 2, 2}), Reverse::Mode::INDEX);

    input_shapes = StaticShapeVector{{4, 3, 2, 4}, {4}};
    output_shapes = shape_inference(op.get(), input_shapes);

    EXPECT_EQ(output_shapes[0], StaticShape({4, 3, 2, 4}));
}

TEST_F(ReverseV1StaticShapeInferenceTest, axes_index_in_constant_data) {
    auto op = make_op(data, axes, Reverse::Mode::INDEX);

    input_shapes = StaticShapeVector{{4, 3, 2, 4}, {4}};
    int8_t axes_val[] = {-1, 2, 1};
    auto const_data = std::unordered_map<size_t, ov::Tensor>{{1, {element::i8, ov::Shape{3}, axes_val}}};
    const auto output_shapes = shape_inference(op.get(), input_shapes, const_data);

    EXPECT_EQ(output_shapes[0], StaticShape({4, 3, 2, 4}));
}

TEST_F(ReverseV1StaticShapeInferenceTest, axes_mask_as_constant) {
    auto op = make_op(data,
                      Constant::create(element::boolean, ov::Shape{4}, {true, false, false, true}),
                      Reverse::Mode::MASK);

    input_shapes = StaticShapeVector{{4, 3, 2, 4}, {4}};

    output_shapes = shape_inference(op.get(), input_shapes);

    EXPECT_EQ(output_shapes[0], StaticShape({4, 3, 2, 4}));
}

TEST_F(ReverseV1StaticShapeInferenceTest, axes_mask_in_constant_data) {
    auto op =
        make_op(data, std::make_shared<Parameter>(element::boolean, PartialShape::dynamic()), Reverse::Mode::MASK);

    input_shapes = StaticShapeVector{{4, 3, 2, 4}, {4}};
    bool axes_val[] = {true, true, false, false};
    auto const_data = std::unordered_map<size_t, ov::Tensor>{{1, {element::boolean, ov::Shape{4}, axes_val}}};
    const auto output_shapes = shape_inference(op.get(), input_shapes, const_data);

    EXPECT_EQ(output_shapes[0], StaticShape({4, 3, 2, 4}));
}

TEST_F(ReverseV1StaticShapeInferenceTest, invalid_axes_mask_length) {
    auto op =
        make_op(data, Constant::create(element::boolean, ov::Shape{3}, {false, false, true}), Reverse::Mode::MASK);

    input_shapes = StaticShapeVector{{1, 2, 4, 3}, {3}};
    OV_EXPECT_THROW(shape_inference(op.get(), input_shapes),
                    NodeValidationFailure,
                    HasSubstr("The number of elements in the reversed_axes tensor (3) must match the input data tensor "
                              "rank (4) in 'mask' mode"));
}

TEST_F(ReverseV1StaticShapeInferenceTest, axes_index_out_of_data_rank) {
    auto op = make_op(data, Constant::create(element::u8, ov::Shape{3}, {0, 20, 3}), Reverse::Mode::INDEX);

    input_shapes = StaticShapeVector{{1, 2, 4, 3}, {3}};
    OV_EXPECT_THROW(shape_inference(op.get(), input_shapes),
                    NodeValidationFailure,
                    HasSubstr("Some of the provided axes (AxisSet{0, 3, 20}) are out of bounds (input rank: 4)"));
}

TEST_F(ReverseV1StaticShapeInferenceTest, default_ctor) {
    auto op = make_op();
    op->set_mode(Reverse::Mode::INDEX);

    input_shapes = StaticShapeVector{{11, 2, 3}, {3}};

    int64_t axes_val[] = {-1, 2, 0};
    auto const_data = std::unordered_map<size_t, ov::Tensor>{{1, {element::i64, ov::Shape{3}, axes_val}}};
    const auto output_shapes = shape_inference(op.get(), input_shapes, const_data);

    EXPECT_EQ(output_shapes[0], StaticShape({11, 2, 3}));
}
