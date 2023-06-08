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

class PriorBoxClusteredV0StaticShapeInferenceTest : public OpStaticShapeInferenceTest<op::v0::PriorBoxClustered> {
protected:
    void SetUp() override {
        output_shapes.resize(1);

        attrs.widths = {2.0f, 3.0f};
        attrs.heights = {1.5f, 2.0f};
    }

    typename op_type::Attributes attrs;
};

TEST_F(PriorBoxClusteredV0StaticShapeInferenceTest, default_ctor_no_args) {
    op = make_op();
    op->set_attrs(attrs);

    int32_t out_size[] = {2, 5};
    input_shapes = ShapeVector{{2}, {2}};

    shape_inference(op.get(),
                    input_shapes,
                    output_shapes,
                    {{0, std::make_shared<HostTensor>(element::i32, ov::Shape{2}, out_size)}});

    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes.front(), StaticShape({2, 80}));
    unit_test::cpu_test_shape_infer(op.get(),
                    input_shapes,
                    output_shapes,
                    {{0, std::make_shared<HostTensor>(element::i32, ov::Shape{2}, out_size)}});
}

TEST_F(PriorBoxClusteredV0StaticShapeInferenceTest, all_inputs_dynamic_rank) {
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
    EXPECT_EQ(output_shapes[0], (StaticShape{2, 4 * 2 * 5 * 2}));
    unit_test::cpu_test_shape_infer(op.get(),
                    input_shapes,
                    output_shapes,
                    {{0, std::make_shared<HostTensor>(element::i32, ov::Shape{2}, output_size)}});
}

TEST_F(PriorBoxClusteredV0StaticShapeInferenceTest, all_inputs_static_rank) {
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
    EXPECT_EQ(output_shapes[0], (StaticShape{2, 4 * 5 * 2 * 2}));
    unit_test::cpu_test_shape_infer(op.get(),
                    input_shapes,
                    output_shapes,
                    {{0, std::make_shared<HostTensor>(element::i32, ov::Shape{2}, output_size)}});
}

TEST_F(PriorBoxClusteredV0StaticShapeInferenceTest, out_size_constant) {
    const auto out_size = op::v0::Constant::create(element::i32, ov::Shape{2}, {4, 6});
    const auto img_size = std::make_shared<op::v0::Parameter>(element::i32, PartialShape::dynamic(1));

    op = make_op(out_size, img_size, attrs);

    input_shapes = ShapeVector{{2}, {2}};
    shape_inference(op.get(), input_shapes, output_shapes);

    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes[0], (StaticShape{2, 4 * 4 * 6 * 2}));
    unit_test::cpu_test_shape_infer(op.get(), input_shapes, output_shapes);
}

TEST_F(PriorBoxClusteredV0StaticShapeInferenceTest, all_inputs_constants) {
    const auto out_size = op::v0::Constant::create(element::i32, ov::Shape{2}, {12, 16});
    const auto img_size = op::v0::Constant::create(element::i32, ov::Shape{2}, {50, 50});

    op = make_op(out_size, img_size, attrs);

    input_shapes = ShapeVector{{2}, {2}};
    shape_inference(op.get(), input_shapes, output_shapes);

    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes[0], (StaticShape{2, 4 * 12 * 16 * 2}));
    unit_test::cpu_test_shape_infer(op.get(), input_shapes, output_shapes);
}

TEST_F(PriorBoxClusteredV0StaticShapeInferenceTest, invalid_number_of_elements_in_out_size) {
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
    // TODO , implementation should throw execption
    // ASSERT_THROW(unit_test::cpu_test_shape_infer(op.get(),
    //                                 input_shapes,
    //                                 output_shapes,
    //                                 {{0, std::make_shared<HostTensor>(element::i64, ov::Shape{3}, output_size)}}),
    //                 InferenceEngine::GeneralError);
}

TEST_F(PriorBoxClusteredV0StaticShapeInferenceTest, invalid_input_ranks) {
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

    // TODO , implementation should throw execption
    // ASSERT_THROW(unit_test::cpu_test_shape_infer(op.get(),
    //                                input_shapes,
    //                                output_shapes,
    //                                {{0, std::make_shared<HostTensor>(element::i64, ov::Shape{3}, output_size)}}),
    //                InferenceEngine::GeneralError);
}

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
    unit_test::cpu_test_shape_infer(op.get(), static_input_shapes, static_output_shapes, const_data);
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
    unit_test::cpu_test_shape_infer(op.get(), static_input_shapes, static_output_shapes, const_data);
}

