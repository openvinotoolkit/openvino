// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/test_assertions.hpp"
#include "utils.hpp"

using namespace ov;
using namespace ov::intel_cpu;
using namespace testing;

TEST(StaticShapeInferenceTest, ReshapeTest) {
    auto data = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1});
    auto pattern = std::make_shared<ov::op::v0::Constant>(element::i32, ov::Shape{2}, std::vector<int32_t>{0, -1});

    auto reshape =
            std::make_shared<op::v1::Reshape>(data, pattern, true);

    std::vector<StaticShape> static_input_shapes = {StaticShape{3, 6, 5, 5}, StaticShape{2}},
            static_output_shapes = {StaticShape{}};
    shape_inference(reshape.get(), static_input_shapes, static_output_shapes);

    ASSERT_EQ(static_output_shapes[0], StaticShape({3, 150}));
    unit_test::cus_usual_shape_infer(reshape.get(), static_input_shapes, static_output_shapes);
}

TEST(StaticShapeInferenceTest, ReshapeEmptyTest) {
    auto data = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, 2, 2});
    auto pattern = std::make_shared<ov::op::v0::Constant>(element::i32, ov::Shape{2}, std::vector<int32_t>{0, 4});

    auto reshape =
            std::make_shared<op::v1::Reshape>(data, pattern, false);

    std::vector<StaticShape> static_input_shapes = {StaticShape{0, 2, 2}, StaticShape{2}},
            static_output_shapes = {StaticShape{}};
    shape_inference(reshape.get(), static_input_shapes, static_output_shapes);

    ASSERT_EQ(static_output_shapes[0], StaticShape({0, 4}));
    unit_test::cus_usual_shape_infer(reshape.get(), static_input_shapes, static_output_shapes);
}

using TestParams = std::tuple<ShapeVector,           // Input shapes
                              std::vector<int64_t>,  // Squeeze axes
                              StaticShape,           // Expected shape
                              bool                   // specal zero
                              >;

class ReshapeStaticShapeInferenceTest : public OpStaticShapeInferenceTest<op::v1::Reshape>,
                                        public WithParamInterface<TestParams> {
protected:
    void SetUp() override {
        std::tie(input_shapes, axes, exp_shape, specalZero) = GetParam();
        output_shapes = ShapeVector(1);
        arg = std::make_shared<op::v0::Parameter>(element::f32, input_shapes.front().get_shape());
    }

    std::vector<int64_t> axes;
    std::shared_ptr<op::v0::Parameter> arg;
    bool specalZero;
};

TEST_P(ReshapeStaticShapeInferenceTest, shape_inference_empty_const_map) {
    const auto axes_node = std::make_shared<op::v0::Constant>(element::i64, ov::Shape{axes.size()}, axes);
    const auto op = make_op(arg, axes_node, specalZero);

    shape_inference(op.get(), input_shapes, output_shapes);

    ASSERT_EQ(output_shapes.front(), exp_shape);
    unit_test::cus_usual_shape_infer(op.get(), input_shapes, output_shapes);
}

TEST_P(ReshapeStaticShapeInferenceTest, shape_inference_with_const_map) {
    const auto axes_node = std::make_shared<op::v0::Parameter>(element::i64, ov::Shape{1});
    const auto op = make_op(arg, axes_node, specalZero);

    const auto axes_const = std::make_shared<op::v0::Constant>(element::i64, ov::Shape{axes.size()}, axes);
    const auto axes_tensor = std::make_shared<ngraph::runtime::HostTensor>(axes_const);
    const std::map<size_t, std::shared_ptr<ngraph::runtime::HostTensor>>& constant_data = {{1, axes_tensor}};

    shape_inference(op.get(), input_shapes, output_shapes, constant_data);

    ASSERT_EQ(output_shapes.front(), exp_shape);
    unit_test::cus_usual_shape_infer(op.get(), input_shapes, output_shapes, constant_data);
}

INSTANTIATE_TEST_SUITE_P(
    multi_dim_shapes,
    ReshapeStaticShapeInferenceTest,
    Values(make_tuple(ShapeVector{{1, 2, 3, 1}, {2}}, std::vector<int64_t>{0, -1}, StaticShape({1, 6}), true),
           make_tuple(ShapeVector{{1, 2, 3, 8}, {4}}, std::vector<int64_t>{1, 2, 3, 8}, StaticShape({1, 2, 3, 8}), true),
           make_tuple(ShapeVector{{1, 2, 3, 8}, {4}}, std::vector<int64_t>{0, 2, 0, 8}, StaticShape({1, 2, 3, 8}), true),
           make_tuple(ShapeVector{{0, 2, 2}, {2}}, std::vector<int64_t>{0, -1}, StaticShape({0, 4}), true),
           make_tuple(ShapeVector{{0, 2, 2}, {2}}, std::vector<int64_t>{0, 4}, StaticShape({0, 4}), true),
           make_tuple(ShapeVector{{4, 0, 2}, {2}}, std::vector<int64_t>{-1, 0}, StaticShape({8, 0}), true),
           make_tuple(ShapeVector{{1, 2, 3, 8}, {4}}, std::vector<int64_t>{1, 2, 3, 8}, StaticShape({1, 2, 3, 8}), false),
           make_tuple(ShapeVector{{0, 2, 2}, {2}}, std::vector<int64_t>{3, 0}, StaticShape({3, 0}), false)),
        PrintToStringParamName());


using ReshapeCustomtaticShapeInferenceThrowExceptionTest = ReshapeStaticShapeInferenceTest;

TEST_P(ReshapeCustomtaticShapeInferenceThrowExceptionTest, shape_inference_with_const_map) {
    const auto axes_node = std::make_shared<op::v0::Parameter>(element::i64, ov::Shape{1});
    const auto op = make_op(arg, axes_node, specalZero);

    const auto axes_const = std::make_shared<op::v0::Constant>(element::i64, ov::Shape{axes.size()}, axes);
    const auto axes_tensor = std::make_shared<ngraph::runtime::HostTensor>(axes_const);
    const std::map<size_t, std::shared_ptr<ngraph::runtime::HostTensor>>& constant_data = {{1, axes_tensor}};

    OV_EXPECT_THROW(unit_test::cus_usual_shape_infer(op.get(), input_shapes, output_shapes, constant_data),
                    InferenceEngine::Unexpected,
                    HasSubstr("[cpu]reshape: the shape of input data conflicts with the reshape pattern"));
}

INSTANTIATE_TEST_SUITE_P(
    multi_dim_shapes,
    ReshapeCustomtaticShapeInferenceThrowExceptionTest,
    Values(make_tuple(ShapeVector{{1, 2, 3, 1}, {3}}, std::vector<int64_t>{0, -1, -1}, StaticShape({}), true),
           make_tuple(ShapeVector{{1, 2, 3, 8}, {4}}, std::vector<int64_t>{1, 2, 3, 6}, StaticShape({}), true),
           make_tuple(ShapeVector{{1, 2, 3, 8}, {4}}, std::vector<int64_t>{0, 3, 0, 8}, StaticShape({}), true),
           make_tuple(ShapeVector{{1, 2, 3, 8}, {4}}, std::vector<int64_t>{1, 2, 3, 6}, StaticShape({}), false),
           make_tuple(ShapeVector{{0, 2, 2}, {2}}, std::vector<int64_t>{3, 3}, StaticShape({}), false)),
        PrintToStringParamName());

TEST(StaticShapeInferenceTest, ShapeOf5DTest) {
    auto data = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1});

    auto shapeof =
            std::make_shared<op::v0::ShapeOf>(data);

    std::vector<StaticShape> static_input_shapes = {StaticShape{2, 3, 4, 5, 6}},
            static_output_shapes = {StaticShape{}};
    shape_inference(shapeof.get(), static_input_shapes, static_output_shapes);

    ASSERT_EQ(static_output_shapes[0], StaticShape({5}));
    unit_test::cus_usual_shape_infer(shapeof.get(), static_input_shapes, static_output_shapes);
}

TEST(StaticShapeInferenceTest, v3ShapeOf5DTest) {
    auto data = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1});

    auto shapeof =
            std::make_shared<op::v3::ShapeOf>(data);

    std::vector<StaticShape> static_input_shapes = {StaticShape{2, 3, 4, 5, 6}},
            static_output_shapes = {StaticShape{}};
    shape_inference(shapeof.get(), static_input_shapes, static_output_shapes);

    ASSERT_EQ(static_output_shapes[0], StaticShape({5}));
    unit_test::cus_usual_shape_infer(shapeof.get(), static_input_shapes, static_output_shapes);
}


TEST(StaticShapeInferenceTest, ShapeOf0DTest) {
    auto data = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{});

    auto shapeof =
            std::make_shared<op::v3::ShapeOf>(data);

    std::vector<StaticShape> static_input_shapes = {StaticShape{}},
            static_output_shapes = {StaticShape{}};
    shape_inference(shapeof.get(), static_input_shapes, static_output_shapes);

    ASSERT_EQ(static_output_shapes[0], StaticShape({}));
    // can't pass implementation don't support 0D shape input
    // unit_test::cus_usual_shape_infer(shapeof.get(), static_input_shapes, static_output_shapes);
}
