// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gmock/gmock.h>

#include "common_test_utils/test_assertions.hpp"
#include "openvino/opsets/opset12.hpp"
#include "utils.hpp"

using namespace ov;
using namespace ov::intel_cpu;
using namespace testing;

class RandomUniformV8StaticShapeInferenceTest : public OpStaticShapeInferenceTest<op::v8::RandomUniform> {
protected:
    void SetUp() override {}

    uint64_t global_seed = 120, op_seed = 100;
};

TEST_F(RandomUniformV8StaticShapeInferenceTest, default_ctor_no_args) {
    op = make_op();
    op->set_out_type(element::i32);
    op->set_global_seed(global_seed);
    op->set_op_seed(op_seed);

    int32_t min = 10, max = 15;
    int64_t shape[] = {2, 4, 12, 13};

    const auto const_data = std::unordered_map<size_t, Tensor>{{0, {element::i64, ov::Shape{4}, shape}},
                                                               {1, {element::i32, ov::Shape{1}, &min}},
                                                               {2, {element::i32, ov::Shape{}, &max}}};

    input_shapes = StaticShapeVector{{4}, {1}, {}};
    output_shapes = shape_inference(op.get(), input_shapes, const_data);

    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes.front(), StaticShape({2, 4, 12, 13}));
}

TEST_F(RandomUniformV8StaticShapeInferenceTest, all_inputs_dynamic_rank) {
    const auto out_shape = std::make_shared<op::v0::Parameter>(element::i64, PartialShape::dynamic());
    const auto min_val = std::make_shared<op::v0::Parameter>(element::i64, PartialShape::dynamic());
    const auto max_val = std::make_shared<op::v0::Parameter>(element::i64, PartialShape::dynamic());

    op = make_op(out_shape, min_val, max_val, element::i64, global_seed, op_seed);

    int64_t min = 1, max = 15;
    int64_t shape[] = {2, 4, 12, 13, 2};

    const auto const_data = std::unordered_map<size_t, Tensor>{{0, {element::i64, ov::Shape{5}, shape}},
                                                               {1, {element::i64, ov::Shape{}, &min}},
                                                               {2, {element::i64, ov::Shape{}, &max}}};

    input_shapes = StaticShapeVector{{5}, {}, {}};
    output_shapes = shape_inference(op.get(), input_shapes, const_data);

    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes.front(), StaticShape({2, 4, 12, 13, 2}));
}

TEST_F(RandomUniformV8StaticShapeInferenceTest, all_inputs_static_rank) {
    const auto out_shape = std::make_shared<op::v0::Parameter>(element::i32, PartialShape::dynamic(1));
    const auto min_val = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(1));
    const auto max_val = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(1));

    op = make_op(out_shape, min_val, max_val, element::f32, global_seed, op_seed);

    float min = 1., max = 15.;
    int32_t shape[] = {12, 13, 2};

    const auto const_data = std::unordered_map<size_t, Tensor>{{0, {element::i32, ov::Shape{3}, shape}},
                                                               {1, {element::f32, ov::Shape{1}, &min}},
                                                               {2, {element::f32, ov::Shape{1}, &max}}};

    input_shapes = StaticShapeVector{{3}, {}, {}};
    output_shapes = shape_inference(op.get(), input_shapes, const_data);

    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes.front(), StaticShape({12, 13, 2}));
}

TEST_F(RandomUniformV8StaticShapeInferenceTest, all_inputs_as_const) {
    const auto out_shape = op::v0::Constant::create(element::i32, ov::Shape{6}, {2, 1, 3, 5, 1, 7});
    const auto min_val = op::v0::Constant::create(element::f16, ov::Shape{}, {2});
    const auto max_val = op::v0::Constant::create(element::f16, ov::Shape{1}, {16});

    op = make_op(out_shape, min_val, max_val, element::f16, global_seed, op_seed);

    input_shapes = StaticShapeVector{{6}, {}, {1}};
    output_shapes = shape_inference(op.get(), input_shapes);

    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes.front(), StaticShape({2, 1, 3, 5, 1, 7}));
}

TEST_F(RandomUniformV8StaticShapeInferenceTest, some_inputs_are_const_some_dynamic) {
    const auto out_shape = std::make_shared<op::v0::Parameter>(element::i32, PartialShape::dynamic());
    const auto min_val = op::v0::Constant::create(element::f32, ov::Shape{}, {2});
    const auto max_val = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());

    op = make_op(out_shape, min_val, max_val, element::f32, global_seed, op_seed);

    float max = 15.;
    int32_t shape[] = {12, 13, 2};

    const auto const_data = std::unordered_map<size_t, Tensor>{{0, {element::i32, ov::Shape{3}, shape}},
                                                               {2, {element::f32, ov::Shape{1}, &max}}};

    input_shapes = StaticShapeVector{{3}, {}, {}};
    output_shapes = shape_inference(op.get(), input_shapes, const_data);

    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes.front(), StaticShape({12, 13, 2}));
}

TEST_F(RandomUniformV8StaticShapeInferenceTest, min_not_lt_max) {
    const auto out_shape = std::make_shared<op::v0::Parameter>(element::i32, PartialShape::dynamic());
    const auto min_val = op::v0::Constant::create(element::i64, ov::Shape{}, {2});
    const auto max_val = std::make_shared<op::v0::Parameter>(element::i64, PartialShape::dynamic());

    op = make_op(out_shape, min_val, max_val, element::i64, global_seed, op_seed);

    int64_t max = 2;
    int32_t shape[] = {12, 13, 2};

    const auto const_data = std::unordered_map<size_t, Tensor>{{0, {element::i32, ov::Shape{3}, shape}},
                                                               {2, {element::i64, ov::Shape{1}, &max}}};

    input_shapes = StaticShapeVector{{3}, {}, {}};
    OV_EXPECT_THROW(shape_inference(op.get(), input_shapes, const_data),
                    NodeValidationFailure,
                    HasSubstr("Min value must be less than max value. Got min value:"));
}

TEST_F(RandomUniformV8StaticShapeInferenceTest, out_shape_input_not_rank_1) {
    const auto out_shape = std::make_shared<op::v0::Parameter>(element::i32, PartialShape::dynamic());
    const auto min_val = op::v0::Constant::create(element::i64, ov::Shape{}, {2});
    const auto max_val = std::make_shared<op::v0::Parameter>(element::i64, PartialShape::dynamic());

    op = make_op(out_shape, min_val, max_val, element::i64, global_seed, op_seed);

    int64_t max = 20;
    int32_t shape[] = {12, 13, 2};

    const auto const_data = std::unordered_map<size_t, Tensor>{{0, {element::i32, ov::Shape{3}, shape}},
                                                               {2, {element::i64, ov::Shape{1}, &max}}};

    input_shapes = StaticShapeVector{{3, 1}, {}, {}};
    OV_EXPECT_THROW(shape_inference(op.get(), input_shapes, const_data),
                    NodeValidationFailure,
                    HasSubstr("The rank of the tensor defining output shape must be equal to 1"));
}

TEST_F(RandomUniformV8StaticShapeInferenceTest, all_inputs_dynamic_no_const_data) {
    const auto out_shape = std::make_shared<op::v0::Parameter>(element::i32, PartialShape::dynamic());
    const auto min_val = std::make_shared<op::v0::Parameter>(element::i64, PartialShape::dynamic());
    const auto max_val = std::make_shared<op::v0::Parameter>(element::i64, PartialShape::dynamic());

    op = make_op(out_shape, min_val, max_val, element::i64, global_seed, op_seed);

    input_shapes = StaticShapeVector{{3}, {}, {}};
    OV_EXPECT_THROW(shape_inference(op.get(), input_shapes),
                    NodeValidationFailure,
                    HasSubstr("Static shape inference lacks constant data on port"));
}
