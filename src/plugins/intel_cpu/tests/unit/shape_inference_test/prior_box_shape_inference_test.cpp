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
    input_shapes = ShapeVector{{2}, {2}};

    shape_inference(op.get(),
                    input_shapes,
                    output_shapes,
                    {{0, std::make_shared<HostTensor>(element::i32, ov::Shape{2}, out_size)}});

    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes.front(), StaticShape({2, 200}));

    // TODO should support v8::PriorBox
    // unit_test::cus_usual_shape_infer(op.get(),
    //         input_shapes,
    //         output_shapes,
    //         {{0, std::make_shared<HostTensor>(element::i32, ov::Shape{2}, out_size)}});
}

TEST_F(PriorBoxV8StaticShapeInferenceTest, all_inputs_dynamic_rank) {
    const auto out_size = std::make_shared<op::v0::Parameter>(element::i32, PartialShape::dynamic());
    const auto img_size = std::make_shared<op::v0::Parameter>(element::i32, PartialShape::dynamic());

    op = make_op(out_size, img_size, attrs);

    int32_t output_size[] = {2, 5};

    input_shapes = ShapeVector{{2}, {2}};
    shape_inference(op.get(),
                    input_shapes,
                    output_shapes,
                    {{0, std::make_shared<HostTensor>(element::i32, ov::Shape{2}, output_size)}});

    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes[0], (StaticShape{2, 200}));

    // TODO should support v8::PriorBox
    // unit_test::cus_usual_shape_infer(op.get(),
    //         input_shapes,
    //         output_shapes,
    //         {{0, std::make_shared<HostTensor>(element::i32, ov::Shape{2}, output_size)}});
}

TEST_F(PriorBoxV8StaticShapeInferenceTest, all_inputs_static_rank) {
    const auto out_size = std::make_shared<op::v0::Parameter>(element::i32, PartialShape::dynamic(1));
    const auto img_size = std::make_shared<op::v0::Parameter>(element::i32, PartialShape::dynamic(1));

    op = make_op(out_size, img_size, attrs);

    int32_t output_size[] = {5, 2};

    input_shapes = ShapeVector{{2}, {2}};
    shape_inference(op.get(),
                    input_shapes,
                    output_shapes,
                    {{0, std::make_shared<HostTensor>(element::i32, ov::Shape{2}, output_size)}});

    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes[0], (StaticShape{2, 200}));

    // TODO should support v8::PriorBox
    // unit_test::cus_usual_shape_infer(op.get(),
    //                input_shapes,
    //                output_shapes,
    //                {{0, std::make_shared<HostTensor>(element::i32, ov::Shape{2}, output_size)}});
}

TEST_F(PriorBoxV8StaticShapeInferenceTest, out_size_constant) {
    const auto out_size = op::v0::Constant::create(element::i32, ov::Shape{2}, {4, 6});
    const auto img_size = std::make_shared<op::v0::Parameter>(element::i32, PartialShape::dynamic(1));

    op = make_op(out_size, img_size, attrs);

    input_shapes = ShapeVector{{2}, {2}};
    shape_inference(op.get(), input_shapes, output_shapes);

    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes[0], (StaticShape{2, 480}));
    // TODO should support v8::PriorBox
    // unit_test::cus_usual_shape_infer(op.get(), input_shapes, output_shapes);
}

TEST_F(PriorBoxV8StaticShapeInferenceTest, all_inputs_constants) {
    const auto out_size = op::v0::Constant::create(element::i32, ov::Shape{2}, {12, 16});
    const auto img_size = op::v0::Constant::create(element::i32, ov::Shape{2}, {50, 50});

    op = make_op(out_size, img_size, attrs);

    input_shapes = ShapeVector{{2}, {2}};
    shape_inference(op.get(), input_shapes, output_shapes);

    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes[0], (StaticShape{2, 3840}));

    // TODO should support v8::PriorBox
    // unit_test::cus_usual_shape_infer(op.get(), input_shapes, output_shapes);
}

TEST_F(PriorBoxV8StaticShapeInferenceTest, invalid_number_of_elements_in_out_size) {
    const auto out_size = std::make_shared<op::v0::Parameter>(element::i64, PartialShape::dynamic(1));
    const auto img_size = std::make_shared<op::v0::Parameter>(element::i64, PartialShape::dynamic(1));

    op = make_op(out_size, img_size, attrs);

    int64_t output_size[] = {5, 2, 1};
    input_shapes = ShapeVector{{2}, {2}};

    OV_EXPECT_THROW(shape_inference(op.get(),
                                    input_shapes,
                                    output_shapes,
                                    {{0, std::make_shared<HostTensor>(element::i64, ov::Shape{3}, output_size)}}),
                    NodeValidationFailure,
                    HasSubstr("Output size must have two elements"));

    // TODO should support v8::PriorBox
}

TEST_F(PriorBoxV8StaticShapeInferenceTest, invalid_input_ranks) {
    const auto out_size = std::make_shared<op::v0::Parameter>(element::i64, PartialShape::dynamic(1));
    const auto img_size = std::make_shared<op::v0::Parameter>(element::i64, PartialShape::dynamic(1));

    op = make_op(out_size, img_size, attrs);

    int64_t output_size[] = {5, 2, 1};
    input_shapes = ShapeVector{{2, 1}, {2}};

    OV_EXPECT_THROW(shape_inference(op.get(),
                                    input_shapes,
                                    output_shapes,
                                    {{0, std::make_shared<HostTensor>(element::i64, ov::Shape{3}, output_size)}}),
                    NodeValidationFailure,
                    HasSubstr("output size input rank 2 must match image shape input rank 1"));

    // TODO should support v8::PriorBox
}

TEST(StaticShapeInferenceTest, prior_box0) {
    op::v0::PriorBox::Attributes attrs;
    attrs.min_size = {16.0f};
    attrs.max_size = {38.46f};
    attrs.aspect_ratio = {2.0f};
    attrs.clip = false;
    attrs.flip = true;
    attrs.step = 16.0f;
    attrs.offset = 0.5f;
    attrs.scale_all_sizes = true;
    attrs.density = {};
    attrs.fixed_ratio = {};
    attrs.fixed_size = {};
    attrs.variance = {0.1f, 0.1f, 0.2f, 0.2f};

    auto layer_shape = std::make_shared<ov::op::v0::Parameter>(element::i32, PartialShape{-1});
    auto image_shape = std::make_shared<ov::op::v0::Parameter>(element::i32, PartialShape{-1});
    auto op =
        std::make_shared<ov::op::v0::PriorBox>(layer_shape, image_shape, attrs);
    int32_t layer_data[] = {24, 42};
    int32_t image_data[] = {384, 672};
    const std::map<size_t, HostTensorPtr> const_data{
        {0, std::make_shared<HostTensor>(element::i32, ov::Shape{2}, layer_data)},
        {1, std::make_shared<HostTensor>(element::i32, ov::Shape{2}, image_data)},
    };

    std::vector<StaticShape> static_input_shapes = {StaticShape{2}, StaticShape{2}};
    std::vector<StaticShape> static_output_shapes = {StaticShape{}};
    shape_inference(op.get(), static_input_shapes, static_output_shapes, const_data);

    ASSERT_EQ(static_output_shapes[0], StaticShape({2, 16128}));
    unit_test::cus_usual_shape_infer(op.get(), static_input_shapes, static_output_shapes, const_data);
}

TEST(StaticShapeInferenceTest, prior_box1) {
    op::v0::PriorBox::Attributes attrs;
    attrs.min_size = {2.0f, 3.0f};
    attrs.aspect_ratio = {1.5f, 2.0f, 2.5f};
    attrs.scale_all_sizes = false;

    auto layer_shape = std::make_shared<ov::op::v0::Parameter>(element::i32, PartialShape{-1});
    auto image_shape = std::make_shared<ov::op::v0::Parameter>(element::i32, PartialShape{-1});
    auto op =
        std::make_shared<ov::op::v0::PriorBox>(layer_shape, image_shape, attrs);
    int32_t layer_data[] = {32, 32};
    int32_t image_data[] = {300, 300};
    const std::map<size_t, HostTensorPtr> const_data{
        {0, std::make_shared<HostTensor>(element::i32, ov::Shape{2}, layer_data)},
        {1, std::make_shared<HostTensor>(element::i32, ov::Shape{2}, image_data)},
    };

    std::vector<StaticShape> static_input_shapes = {StaticShape{2}, StaticShape{2}};
    std::vector<StaticShape> static_output_shapes = {StaticShape{}};
    shape_inference(op.get(), static_input_shapes, static_output_shapes, const_data);

    ASSERT_EQ(static_output_shapes[0], StaticShape({2, 20480}));
    unit_test::cus_usual_shape_infer(op.get(), static_input_shapes, static_output_shapes, const_data);
}

TEST(StaticShapeInferenceTest, prior_box2) {
    op::v0::PriorBox::Attributes attrs;
    attrs.min_size = {2.0f, 3.0f};
    attrs.aspect_ratio = {1.5f, 2.0f, 2.5f};
    attrs.flip = true;
    attrs.scale_all_sizes = false;

    auto layer_shape = std::make_shared<ov::op::v0::Parameter>(element::i32, PartialShape{-1});
    auto image_shape = std::make_shared<ov::op::v0::Parameter>(element::i32, PartialShape{-1});
    auto op =
        std::make_shared<ov::op::v0::PriorBox>(layer_shape, image_shape, attrs);
    int32_t layer_data[] = {32, 32};
    int32_t image_data[] = {300, 300};
    const std::map<size_t, HostTensorPtr> const_data{
        {0, std::make_shared<HostTensor>(element::i32, ov::Shape{2}, layer_data)},
        {1, std::make_shared<HostTensor>(element::i32, ov::Shape{2}, image_data)},
    };

    std::vector<StaticShape> static_input_shapes = {StaticShape{2}, StaticShape{2}};
    std::vector<StaticShape> static_output_shapes = {StaticShape{}};
    shape_inference(op.get(), static_input_shapes, static_output_shapes, const_data);

    ASSERT_EQ(static_output_shapes[0], StaticShape({2, 32768}));
    unit_test::cus_usual_shape_infer(op.get(), static_input_shapes, static_output_shapes, const_data);
}

TEST(StaticShapeInferenceTest, prior_box3) {
    op::v0::PriorBox::Attributes attrs;
    attrs.min_size = {256.0f};
    attrs.max_size = {315.0f};
    attrs.aspect_ratio = {2.0f};
    attrs.flip = true;

    auto layer_shape = std::make_shared<ov::op::v0::Parameter>(element::i32, PartialShape{-1});
    auto image_shape = std::make_shared<ov::op::v0::Parameter>(element::i32, PartialShape{-1});
    auto op =
        std::make_shared<ov::op::v0::PriorBox>(layer_shape, image_shape, attrs);
    int32_t layer_data[] = {1, 1};
    int32_t image_data[] = {300, 300};
    const std::map<size_t, HostTensorPtr> const_data{
        {0, std::make_shared<HostTensor>(element::i32, ov::Shape{2}, layer_data)},
        {1, std::make_shared<HostTensor>(element::i32, ov::Shape{2}, image_data)},
    };

    std::vector<StaticShape> static_input_shapes = {StaticShape{2}, StaticShape{2}};
    std::vector<StaticShape> static_output_shapes = {StaticShape{}};
    shape_inference(op.get(), static_input_shapes, static_output_shapes, const_data);

    ASSERT_EQ(static_output_shapes[0], StaticShape({2, 16}));
    unit_test::cus_usual_shape_infer(op.get(), static_input_shapes, static_output_shapes, const_data);
}

TEST(StaticShapeInferenceTest, prior_box_v8_1) {
    op::v8::PriorBox::Attributes attrs;
    attrs.min_size = {2.0f, 3.0f};
    attrs.aspect_ratio = {1.5f, 2.0f, 2.5f};
    attrs.scale_all_sizes = false;
    attrs.min_max_aspect_ratios_order = true;

    auto layer_shape = std::make_shared<ov::op::v0::Parameter>(element::i32, PartialShape{-1});
    auto image_shape = std::make_shared<ov::op::v0::Parameter>(element::i32, PartialShape{-1});
    auto op =
        std::make_shared<ov::op::v8::PriorBox>(layer_shape, image_shape, attrs);
    int32_t layer_data[] = {32, 32};
    int32_t image_data[] = {300, 300};
    const std::map<size_t, HostTensorPtr> const_data{
        {0, std::make_shared<HostTensor>(element::i32, ov::Shape{2}, layer_data)},
        {1, std::make_shared<HostTensor>(element::i32, ov::Shape{2}, image_data)},
    };

    std::vector<StaticShape> static_input_shapes = {StaticShape{2}, StaticShape{2}};
    std::vector<StaticShape> static_output_shapes = {StaticShape{}};
    shape_inference(op.get(), static_input_shapes, static_output_shapes, const_data);

    ASSERT_EQ(static_output_shapes[0], StaticShape({2, 20480}));

    // TODO should support v8::PriorBox
    // unit_test::cus_usual_shape_infer(op.get(), static_input_shapes, static_output_shapes, const_data);
}

TEST(StaticShapeInferenceTest, prior_box_v8_2) {
    op::v8::PriorBox::Attributes attrs;
    attrs.min_size = {2.0f, 3.0f};
    attrs.aspect_ratio = {1.5f, 2.0f, 2.5f};
    attrs.flip = true;
    attrs.scale_all_sizes = false;
    attrs.min_max_aspect_ratios_order = false;

    auto layer_shape = std::make_shared<ov::op::v0::Parameter>(element::i32, PartialShape{-1});
    auto image_shape = std::make_shared<ov::op::v0::Parameter>(element::i32, PartialShape{-1});
    auto op =
        std::make_shared<ov::op::v8::PriorBox>(layer_shape, image_shape, attrs);
    int32_t layer_data[] = {32, 32};
    int32_t image_data[] = {300, 300};
    const std::map<size_t, HostTensorPtr> const_data{
        {0, std::make_shared<HostTensor>(element::i32, ov::Shape{2}, layer_data)},
        {1, std::make_shared<HostTensor>(element::i32, ov::Shape{2}, image_data)},
    };

    std::vector<StaticShape> static_input_shapes = {StaticShape{2}, StaticShape{2}};
    std::vector<StaticShape> static_output_shapes = {StaticShape{}};
    shape_inference(op.get(), static_input_shapes, static_output_shapes, const_data);

    ASSERT_EQ(static_output_shapes[0], StaticShape({2, 32768}));
    // should support v8::PriorBox
    // unit_test::cus_usual_shape_infer(op.get(), static_input_shapes, static_output_shapes, const_data);
}
