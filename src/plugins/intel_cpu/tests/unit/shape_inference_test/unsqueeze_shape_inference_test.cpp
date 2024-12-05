// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gmock/gmock.h>

#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "unsqueeze_shape_inference.hpp"
#include "utils.hpp"

using namespace ov;
using namespace ov::intel_cpu;
using namespace testing;

class UnsqueezeStaticShapeInferenceAssertTest : public OpStaticShapeInferenceTest<op::v0::Unsqueeze> {
protected:
    void SetUp() override {
        output_shapes = StaticShapeVector(1);
    }
};

TEST_F(UnsqueezeStaticShapeInferenceAssertTest, no_axes) {
    const auto arg = std::make_shared<op::v0::Parameter>(element::f64, PartialShape{-1, -1});
    const auto axes = std::make_shared<op::v0::Parameter>(element::i64, PartialShape{1});
    op = std::make_shared<op::v0::Unsqueeze>(arg, axes);

    input_shapes = StaticShapeVector{{5, 6}, axes->get_shape()};

    try {
        output_shapes = shape_inference(op.get(), input_shapes);
        FAIL() << "Axes nullptr not detected";
    } catch (const NodeValidationFailure& error) {
        EXPECT_THAT(error.what(), HasSubstr("Check 'constant != nullptr'"));
    } catch (...) {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST_F(UnsqueezeStaticShapeInferenceAssertTest, empty_axes) {
    const auto arg = std::make_shared<op::v0::Parameter>(element::f64, ov::Shape{5, 6});
    const auto axes = std::make_shared<op::v0::Constant>(element::i64, ov::Shape{0}, std::vector<int64_t>{});

    try {
        op = std::make_shared<op::v0::Unsqueeze>(arg, axes);
        FAIL() << "Empty axes not detected";
    } catch (const NodeValidationFailure& error) {
        EXPECT_THAT(error.what(), HasSubstr("'axes' input is mandatory"));
    } catch (...) {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

using TestParams = std::tuple<StaticShapeVector,     // Input shapes
                              std::vector<int64_t>,  // Unsqueeze axes
                              StaticShape            // Expected shape
                              >;

class UnsqueezeStaticShapeInferenceTest : public UnsqueezeStaticShapeInferenceAssertTest,
                                          public WithParamInterface<TestParams> {
protected:
    void SetUp() override {
        UnsqueezeStaticShapeInferenceAssertTest::SetUp();
        std::tie(input_shapes, axes, exp_shape) = GetParam();

        output_shapes = StaticShapeVector(1);
        arg = std::make_shared<op::v0::Parameter>(element::f32, input_shapes.front().get_shape());
    }

    std::vector<int64_t> axes;
    std::shared_ptr<op::v0::Parameter> arg;
};

INSTANTIATE_TEST_SUITE_P(1d_shapes,
                         UnsqueezeStaticShapeInferenceTest,
                         Values(make_tuple(StaticShapeVector{{0}, {1}}, std::vector<int64_t>{-1}, StaticShape({0, 1})),
                                make_tuple(StaticShapeVector{{0}, {1}}, std::vector<int64_t>{0}, StaticShape({1, 0})),
                                make_tuple(StaticShapeVector{{1}, {1}}, std::vector<int64_t>{1}, StaticShape({1, 1})),
                                make_tuple(StaticShapeVector{{2}, {1}}, std::vector<int64_t>{0}, StaticShape({1, 2})),
                                make_tuple(StaticShapeVector{{2}, {1}}, std::vector<int64_t>{1}, StaticShape({2, 1})),
                                make_tuple(StaticShapeVector{{2}, {1}}, std::vector<int64_t>{-1}, StaticShape({2, 1})),
                                make_tuple(StaticShapeVector{{2}, {1}}, std::vector<int64_t>{-2}, StaticShape({1, 2}))),
                         PrintToStringParamName());

INSTANTIATE_TEST_SUITE_P(
    multi_dim_shapes,
    UnsqueezeStaticShapeInferenceTest,
    Values(
        make_tuple(StaticShapeVector{{2, 3}, {2}}, std::vector<int64_t>{0, 3}, StaticShape({1, 2, 3, 1})),
        make_tuple(StaticShapeVector{{2, 4}, {2}}, std::vector<int64_t>{2, 1}, StaticShape({2, 1, 1, 4})),
        make_tuple(StaticShapeVector{{3, 2}, {3}}, std::vector<int64_t>{0, 2, 4}, StaticShape({1, 3, 1, 2, 1})),
        make_tuple(StaticShapeVector{{3, 2}, {3}}, std::vector<int64_t>{4, 2, 0}, StaticShape({1, 3, 1, 2, 1})),
        make_tuple(StaticShapeVector{{3, 2}, {3}}, std::vector<int64_t>{2, 0, 4}, StaticShape({1, 3, 1, 2, 1})),
        make_tuple(StaticShapeVector{{10, 0, 3}, {4}},
                   std::vector<int64_t>{1, -1, 3, -2},
                   StaticShape({10, 1, 0, 1, 3, 1, 1})),
        make_tuple(StaticShapeVector{{2, 6, 7, 8, 3}, {1}}, std::vector<int64_t>{0}, StaticShape({1, 2, 6, 7, 8, 3}))),
    PrintToStringParamName());

INSTANTIATE_TEST_SUITE_P(
    multi_dim_shapes_repeated_axis,
    UnsqueezeStaticShapeInferenceTest,
    Values(
        make_tuple(StaticShapeVector{{2, 3}, {2}}, std::vector<int64_t>{1, 1}, StaticShape({2, 1, 3})),
        make_tuple(StaticShapeVector{{3, 2}, {3}}, std::vector<int64_t>{1, -1, 1}, StaticShape({3, 1, 2, 1})),
        make_tuple(StaticShapeVector{{3, 2}, {3}}, std::vector<int64_t>{1, -1, 1, -1}, StaticShape({3, 1, 2, 1})),
        make_tuple(StaticShapeVector{{3, 2}, {3}}, std::vector<int64_t>{2, -1, 2, -1, 0}, StaticShape({1, 3, 1, 2, 1})),
        make_tuple(StaticShapeVector{{2, 6, 7, 8, 3}, {2}},
                   std::vector<int64_t>{-1, -1},
                   StaticShape({2, 6, 7, 8, 3, 1}))),
    PrintToStringParamName());

TEST_P(UnsqueezeStaticShapeInferenceTest, shape_inference_empty_const_map) {
    const auto axes_node = std::make_shared<op::v0::Constant>(element::i64, ov::Shape{axes.size()}, axes);
    op = std::make_shared<op::v0::Unsqueeze>(arg, axes_node);

    output_shapes = shape_inference(op.get(), input_shapes);

    ASSERT_EQ(output_shapes.front(), exp_shape);
}

TEST_P(UnsqueezeStaticShapeInferenceTest, shape_inference_with_const_map) {
    const auto axes_node = std::make_shared<op::v0::Parameter>(element::i64, ov::Shape{1});
    op = std::make_shared<op::v0::Unsqueeze>(arg, axes_node);

    const auto axes_tensor = ov::Tensor(element::i64, ov::Shape{axes.size()}, axes.data());
    const auto constant_data = std::unordered_map<size_t, ov::Tensor>{{1, axes_tensor}};

    output_shapes = shape_inference(op.get(), input_shapes, constant_data);

    ASSERT_EQ(output_shapes.front(), exp_shape);
}
