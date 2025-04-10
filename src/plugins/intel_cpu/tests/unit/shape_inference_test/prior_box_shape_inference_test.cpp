// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gmock/gmock.h>

#include "common_test_utils/test_assertions.hpp"
#include "openvino/opsets/opset11.hpp"
#include "utils.hpp"

using namespace ov;
using namespace ov::intel_cpu;
using namespace testing;

class PriorBoxV8StaticShapeInferenceTest : public OpStaticShapeInferenceTest<op::v8::PriorBox> {
protected:
    void SetUp() override {
        output_shapes.resize(1);

        attrs.min_size = {2.0f, 3.0f};
        attrs.aspect_ratio = {1.5f, 2.0f, 2.5f};
        attrs.scale_all_sizes = false;
    }

    typename op_type::Attributes attrs;
};

TEST_F(PriorBoxV8StaticShapeInferenceTest, default_ctor_no_args) {
    op = make_op();
    op->set_attrs(attrs);

    int32_t out_size[] = {2, 5};
    input_shapes = StaticShapeVector{{2}, {2}};

    output_shapes = shape_inference(op.get(), input_shapes, {{0, {element::i32, ov::Shape{2}, out_size}}});

    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes.front(), StaticShape({2, 200}));
}

TEST_F(PriorBoxV8StaticShapeInferenceTest, all_inputs_dynamic_rank) {
    const auto out_size = std::make_shared<op::v0::Parameter>(element::i32, PartialShape::dynamic());
    const auto img_size = std::make_shared<op::v0::Parameter>(element::i32, PartialShape::dynamic());

    op = make_op(out_size, img_size, attrs);

    int32_t output_size[] = {2, 5};

    input_shapes = StaticShapeVector{{2}, {2}};
    output_shapes = shape_inference(op.get(), input_shapes, {{0, {element::i32, ov::Shape{2}, output_size}}});

    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes[0], (StaticShape{2, 200}));
}

TEST_F(PriorBoxV8StaticShapeInferenceTest, all_inputs_static_rank) {
    const auto out_size = std::make_shared<op::v0::Parameter>(element::i32, PartialShape::dynamic(1));
    const auto img_size = std::make_shared<op::v0::Parameter>(element::i32, PartialShape::dynamic(1));

    op = make_op(out_size, img_size, attrs);

    int32_t output_size[] = {5, 2};

    input_shapes = StaticShapeVector{{2}, {2}};
    output_shapes = shape_inference(op.get(), input_shapes, {{0, {element::i32, ov::Shape{2}, output_size}}});

    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes[0], (StaticShape{2, 200}));
}

TEST_F(PriorBoxV8StaticShapeInferenceTest, out_size_constant) {
    const auto out_size = op::v0::Constant::create(element::i32, ov::Shape{2}, {4, 6});
    const auto img_size = std::make_shared<op::v0::Parameter>(element::i32, PartialShape::dynamic(1));

    op = make_op(out_size, img_size, attrs);

    input_shapes = StaticShapeVector{{2}, {2}};
    output_shapes = shape_inference(op.get(), input_shapes);

    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes[0], (StaticShape{2, 480}));
}

TEST_F(PriorBoxV8StaticShapeInferenceTest, all_inputs_constants) {
    const auto out_size = op::v0::Constant::create(element::i32, ov::Shape{2}, {12, 16});
    const auto img_size = op::v0::Constant::create(element::i32, ov::Shape{2}, {50, 50});

    op = make_op(out_size, img_size, attrs);

    input_shapes = StaticShapeVector{{2}, {2}};
    output_shapes = shape_inference(op.get(), input_shapes);

    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes[0], (StaticShape{2, 3840}));
}

TEST_F(PriorBoxV8StaticShapeInferenceTest, invalid_number_of_elements_in_out_size) {
    const auto out_size = std::make_shared<op::v0::Parameter>(element::i64, PartialShape::dynamic(1));
    const auto img_size = std::make_shared<op::v0::Parameter>(element::i64, PartialShape::dynamic(1));

    op = make_op(out_size, img_size, attrs);

    int64_t output_size[] = {5, 2, 1};
    input_shapes = StaticShapeVector{{2}, {2}};

    OV_EXPECT_THROW(shape_inference(op.get(), input_shapes, {{0, {element::i64, ov::Shape{3}, output_size}}}),
                    NodeValidationFailure,
                    HasSubstr("Output size must have two elements"));
}

TEST_F(PriorBoxV8StaticShapeInferenceTest, invalid_input_ranks) {
    const auto out_size = std::make_shared<op::v0::Parameter>(element::i64, PartialShape::dynamic(1));
    const auto img_size = std::make_shared<op::v0::Parameter>(element::i64, PartialShape::dynamic(1));

    op = make_op(out_size, img_size, attrs);

    int64_t output_size[] = {5, 2, 1};
    input_shapes = StaticShapeVector{{2, 1}, {2}};

    OV_EXPECT_THROW(shape_inference(op.get(), input_shapes, {{0, {element::i64, ov::Shape{3}, output_size}}}),
                    NodeValidationFailure,
                    HasSubstr("output size input rank 2 must match image shape input rank 1"));
}
