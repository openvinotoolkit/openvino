// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "squeeze_shape_inference.hpp"

#include <gmock/gmock.h>

#include "common_test_utils/test_assertions.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/squeeze.hpp"
#include "utils.hpp"

using namespace ov;
using namespace ov::intel_cpu;
using namespace testing;

namespace v0 {

class SqueezeV0StaticShapeInferenceAssertTest : public OpStaticShapeInferenceTest<op::v0::Squeeze> {
protected:
    void SetUp() override {
        output_shapes = StaticShapeVector(1);
    }
};

TEST_F(SqueezeV0StaticShapeInferenceAssertTest, no_axes) {
    const auto arg = std::make_shared<op::v0::Parameter>(element::f64, PartialShape{-1, -1});
    const auto axes = std::make_shared<op::v0::Parameter>(element::i64, PartialShape{1});
    const auto op = make_op(arg, axes);

    input_shapes = StaticShapeVector{{5, 6}, axes->get_shape()};

    OV_EXPECT_THROW(shape_inference(op.get(), input_shapes),
                    NodeValidationFailure,
                    HasSubstr("Check 'constant != nullptr'"));
}

TEST_F(SqueezeV0StaticShapeInferenceAssertTest, parameter_static_shape_axes_no_data) {
    const auto arg = std::make_shared<op::v0::Parameter>(element::f64, ov::Shape{2, 1, 3, 1});
    const auto axes = std::make_shared<op::v0::Parameter>(element::i64, ov::Shape{2});
    const auto op = make_op(arg, axes);

    input_shapes = StaticShapeVector{arg->get_shape(), axes->get_shape()};

    OV_EXPECT_THROW(shape_inference(op.get(), input_shapes),
                    NodeValidationFailure,
                    HasSubstr("Check 'constant != nullptr'"));
}

using TestParams = std::tuple<StaticShapeVector,     // Input shapes
                              std::vector<int64_t>,  // Squeeze axes
                              StaticShape            // Expected shape
                              >;

class SqueezeV0StaticShapeInferenceTest : public SqueezeV0StaticShapeInferenceAssertTest,
                                        public WithParamInterface<TestParams> {
protected:
    void SetUp() override {
        SqueezeV0StaticShapeInferenceAssertTest::SetUp();
        std::tie(input_shapes, axes, exp_shape) = GetParam();

        output_shapes = StaticShapeVector(1);
        arg = std::make_shared<op::v0::Parameter>(element::f32, input_shapes.front().get_shape());
    }

    std::vector<int64_t> axes;
    std::shared_ptr<op::v0::Parameter> arg;
};

INSTANTIATE_TEST_SUITE_P(1d_shapes,
                         SqueezeV0StaticShapeInferenceTest,
                         Values(make_tuple(StaticShapeVector{{1}, {1}}, std::vector<int64_t>{-1}, StaticShape({})),
                                make_tuple(StaticShapeVector{{6}, {1}}, std::vector<int64_t>{-1}, StaticShape({6})),
                                make_tuple(StaticShapeVector{{1}, {1}}, std::vector<int64_t>{0}, StaticShape({}))),
                         PrintToStringParamName());

INSTANTIATE_TEST_SUITE_P(
    multi_dim_shapes,
    SqueezeV0StaticShapeInferenceTest,
    Values(
        make_tuple(StaticShapeVector{{1, 2, 3, 1}, {2}}, std::vector<int64_t>{0, 3}, StaticShape({2, 3})),
        make_tuple(StaticShapeVector{{2, 1, 1, 4}, {2}}, std::vector<int64_t>{2, 1}, StaticShape({2, 4})),
        make_tuple(StaticShapeVector{{2, 1, 1, 4, 1}, {2}}, std::vector<int64_t>{0, 1, -2, -1}, StaticShape({2, 1, 4})),
        make_tuple(StaticShapeVector{{1, 3, 1, 2, 1}, {3}}, std::vector<int64_t>{0, 2, 4}, StaticShape({3, 2})),
        make_tuple(StaticShapeVector{{1, 3, 1, 2, 1}, {3}}, std::vector<int64_t>{4, 2, 0}, StaticShape({3, 2})),
        make_tuple(StaticShapeVector{{1, 3, 1, 2, 1}, {3}}, std::vector<int64_t>{2, 0, 4}, StaticShape({3, 2})),
        make_tuple(StaticShapeVector{{10, 1, 0, 1, 3, 1, 1}, {4}},
                   std::vector<int64_t>{1, -1, 3, -2},
                   StaticShape({10, 0, 3})),
        make_tuple(StaticShapeVector{{10, 1, 0, 1, 3, 1, 1}, {}}, std::vector<int64_t>{}, StaticShape({10, 0, 3})),
        make_tuple(StaticShapeVector{{2, 1, 7, 8, 3}, {1}}, std::vector<int64_t>{1}, StaticShape({2, 7, 8, 3}))),
    PrintToStringParamName());

INSTANTIATE_TEST_SUITE_P(
    multi_dim_shapes_repeated_axis,
    SqueezeV0StaticShapeInferenceTest,
    Values(
        make_tuple(StaticShapeVector{{2, 1, 3}, {2}}, std::vector<int64_t>{1, 1}, StaticShape({2, 3})),
        make_tuple(StaticShapeVector{{3, 1, 2, 1}, {3}}, std::vector<int64_t>{1, -1, 1}, StaticShape({3, 2})),
        make_tuple(StaticShapeVector{{3, 1, 2, 1}, {3}}, std::vector<int64_t>{1, -1, 1, -1}, StaticShape({3, 2})),
        make_tuple(StaticShapeVector{{1, 3, 1, 2, 1}, {3}}, std::vector<int64_t>{2, -1, 2, -1, 0}, StaticShape({3, 2})),
        make_tuple(StaticShapeVector{{2, 6, 7, 8, 1}, {2}}, std::vector<int64_t>{-1, -1}, StaticShape({2, 6, 7, 8}))),
    PrintToStringParamName());

TEST_P(SqueezeV0StaticShapeInferenceTest, shape_inference_empty_const_map) {
    const auto axes_node = std::make_shared<op::v0::Constant>(element::i64, ov::Shape{axes.size()}, axes);
    const auto op = make_op(arg, axes_node);

    output_shapes = shape_inference(op.get(), input_shapes);

    ASSERT_EQ(output_shapes.front(), exp_shape);
}

TEST_P(SqueezeV0StaticShapeInferenceTest, shape_inference_with_const_map) {
    const auto axes_node = std::make_shared<op::v0::Parameter>(element::i64, ov::PartialShape::dynamic());
    const auto op = make_op(arg, axes_node);

    const auto axes_tensor = axes.empty() ? ov::Tensor(element::i64, ov::Shape{axes.size()})
                                          : ov::Tensor(element::i64, ov::Shape{axes.size()}, axes.data());
    const auto constant_data = std::unordered_map<size_t, ov::Tensor>{{1, axes_tensor}};

    output_shapes = shape_inference(op.get(), input_shapes, constant_data);

    ASSERT_EQ(output_shapes.front(), exp_shape);
}

}   // namespace v0

namespace v15 {

class SqueezeV15StaticShapeInferenceAssertTest : public OpStaticShapeInferenceTest<op::v15::Squeeze> {
protected:
    void SetUp() override {
        output_shapes = StaticShapeVector(1);
    }
};

TEST_F(SqueezeV15StaticShapeInferenceAssertTest, no_axes) {
    const auto arg = std::make_shared<op::v0::Parameter>(element::f64, PartialShape{-1, -1});
    const auto axes = std::make_shared<op::v0::Parameter>(element::i64, PartialShape{1});
    const auto op = make_op(arg, axes);

    input_shapes = StaticShapeVector{{5, 6}, axes->get_shape()};

    OV_EXPECT_THROW(shape_inference(op.get(), input_shapes),
                    NodeValidationFailure,
                    HasSubstr("Check 'constant != nullptr'"));
}

TEST_F(SqueezeV15StaticShapeInferenceAssertTest, parameter_static_shape_axes_no_data) {
    const auto arg = std::make_shared<op::v0::Parameter>(element::f64, ov::Shape{2, 1, 3, 1});
    const auto axes = std::make_shared<op::v0::Parameter>(element::i64, ov::Shape{2});
    const auto op = make_op(arg, axes);

    input_shapes = StaticShapeVector{arg->get_shape(), axes->get_shape()};

    OV_EXPECT_THROW(shape_inference(op.get(), input_shapes),
                    NodeValidationFailure,
                    HasSubstr("Check 'constant != nullptr'"));
}

using TestParams = std::tuple<StaticShapeVector,     // Input shapes
                              std::vector<int64_t>,  // Squeeze axes
                              StaticShape            // Expected shape
                              >;

class SqueezeV15StaticShapeInferenceTest : public SqueezeV15StaticShapeInferenceAssertTest,
                                        public WithParamInterface<TestParams> {
protected:
    void SetUp() override {
        SqueezeV15StaticShapeInferenceAssertTest::SetUp();
        std::tie(input_shapes, axes, exp_shape) = GetParam();

        output_shapes = StaticShapeVector(1);
        arg = std::make_shared<op::v0::Parameter>(element::f32, input_shapes.front().get_shape());
    }

    std::vector<int64_t> axes;
    std::shared_ptr<op::v0::Parameter> arg;
};

INSTANTIATE_TEST_SUITE_P(1d_shapes,
                         SqueezeV15StaticShapeInferenceTest,
                         Values(make_tuple(StaticShapeVector{{1}, {1}}, std::vector<int64_t>{-1}, StaticShape({})),
                                make_tuple(StaticShapeVector{{6}, {1}}, std::vector<int64_t>{-1}, StaticShape({6})),
                                make_tuple(StaticShapeVector{{1}, {1}}, std::vector<int64_t>{0}, StaticShape({}))),
                         PrintToStringParamName());

INSTANTIATE_TEST_SUITE_P(
    multi_dim_shapes,
    SqueezeV15StaticShapeInferenceTest,
    Values(
        make_tuple(StaticShapeVector{{1, 2, 3, 1}, {2}}, std::vector<int64_t>{0, 3}, StaticShape({2, 3})),
        make_tuple(StaticShapeVector{{2, 1, 1, 4}, {2}}, std::vector<int64_t>{2, 1}, StaticShape({2, 4})),
        make_tuple(StaticShapeVector{{2, 1, 1, 4, 1}, {2}}, std::vector<int64_t>{0, 1, -2, -1}, StaticShape({2, 1, 4})),
        make_tuple(StaticShapeVector{{1, 3, 1, 2, 1}, {3}}, std::vector<int64_t>{0, 2, 4}, StaticShape({3, 2})),
        make_tuple(StaticShapeVector{{1, 3, 1, 2, 1}, {3}}, std::vector<int64_t>{4, 2, 0}, StaticShape({3, 2})),
        make_tuple(StaticShapeVector{{1, 3, 1, 2, 1}, {3}}, std::vector<int64_t>{2, 0, 4}, StaticShape({3, 2})),
        make_tuple(StaticShapeVector{{10, 1, 0, 1, 3, 1, 1}, {4}},
                   std::vector<int64_t>{1, -1, 3, -2},
                   StaticShape({10, 0, 3})),
        make_tuple(StaticShapeVector{{10, 1, 0, 1, 3, 1, 1}, {}}, std::vector<int64_t>{}, StaticShape({10, 0, 3})),
        make_tuple(StaticShapeVector{{2, 1, 7, 8, 3}, {1}}, std::vector<int64_t>{1}, StaticShape({2, 7, 8, 3}))),
    PrintToStringParamName());

INSTANTIATE_TEST_SUITE_P(
    multi_dim_shapes_repeated_axis,
    SqueezeV15StaticShapeInferenceTest,
    Values(
        make_tuple(StaticShapeVector{{2, 1, 3}, {2}}, std::vector<int64_t>{1, 1}, StaticShape({2, 3})),
        make_tuple(StaticShapeVector{{3, 1, 2, 1}, {3}}, std::vector<int64_t>{1, -1, 1}, StaticShape({3, 2})),
        make_tuple(StaticShapeVector{{3, 1, 2, 1}, {3}}, std::vector<int64_t>{1, -1, 1, -1}, StaticShape({3, 2})),
        make_tuple(StaticShapeVector{{1, 3, 1, 2, 1}, {3}}, std::vector<int64_t>{2, -1, 2, -1, 0}, StaticShape({3, 2})),
        make_tuple(StaticShapeVector{{2, 6, 7, 8, 1}, {2}}, std::vector<int64_t>{-1, -1}, StaticShape({2, 6, 7, 8}))),
    PrintToStringParamName());

TEST_P(SqueezeV15StaticShapeInferenceTest, shape_inference_empty_const_map) {
    const auto axes_node = std::make_shared<op::v0::Constant>(element::i64, ov::Shape{axes.size()}, axes);
    const auto op = make_op(arg, axes_node);

    output_shapes = shape_inference(op.get(), input_shapes);

    ASSERT_EQ(output_shapes.front(), exp_shape);
}

TEST_P(SqueezeV15StaticShapeInferenceTest, shape_inference_with_const_map) {
    const auto axes_node = std::make_shared<op::v0::Parameter>(element::i64, ov::PartialShape::dynamic());
    const auto op = make_op(arg, axes_node);

    const auto axes_tensor = axes.empty() ? ov::Tensor(element::i64, ov::Shape{axes.size()})
                                          : ov::Tensor(element::i64, ov::Shape{axes.size()}, axes.data());
    const auto constant_data = std::unordered_map<size_t, ov::Tensor>{{1, axes_tensor}};

    output_shapes = shape_inference(op.get(), input_shapes, constant_data);

    ASSERT_EQ(output_shapes.front(), exp_shape);
}

}   // namespace v15
