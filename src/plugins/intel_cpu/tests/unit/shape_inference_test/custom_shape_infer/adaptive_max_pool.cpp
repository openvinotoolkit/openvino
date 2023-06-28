// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gmock/gmock.h>

#include "common_test_utils/test_assertions.hpp"
#include "custom_shape_infer.hpp"
#include <ngraph/opsets/opset8.hpp>

using namespace ov;
using namespace ov::intel_cpu;
using namespace testing;

using AdaptiveMaxPoolV8TestParams = std::tuple<unit_test::ShapeVector, // Input shapes
                                               std::vector<int32_t>,   // output_shapes
                                               StaticShape             // Expected shape
                                               >;

class AdaptiveMaxPoolV8CpuShapeInferenceTest  : public unit_test::OpCpuShapeInferenceTest<op::v8::AdaptiveMaxPool>,
                                                public WithParamInterface<AdaptiveMaxPoolV8TestParams> {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<AdaptiveMaxPoolV8TestParams>& obj) {
        unit_test::ShapeVector tmp_input_shapes;
        std::vector<int32_t> tmp_axes;
        StaticShape tmp_exp_shape;
        std::tie(tmp_input_shapes, tmp_axes, tmp_exp_shape) = obj.param;
        std::ostringstream result;
        result << "IS" << CommonTestUtils::vec2str(tmp_input_shapes) << "_";
        result << "sd" << CommonTestUtils::vec2str(tmp_axes) << "_";
        result << "exp_shape" << tmp_exp_shape;
        return result.str();
    }

protected:
    void SetUp() override {
        std::tie(input_shapes, axes, exp_shape) = GetParam();
        output_shapes = unit_test::ShapeVector(0);
        output_shapes.push_back(exp_shape);
        output_shapes.push_back(exp_shape);
        arg = std::make_shared<op::v0::Parameter>(element::f64, input_shapes.front().get_shape());
    }

    std::vector<int32_t> axes;
    std::shared_ptr<op::v0::Parameter> arg;
    bool specalZero;
};

TEST_P(AdaptiveMaxPoolV8CpuShapeInferenceTest , shape_inference_empty_const_map) {
    const auto axes_node = std::make_shared<op::v0::Constant>(element::i32, ov::Shape{axes.size()}, axes);
    const auto op = make_op(arg, axes_node);

    unit_test::cpu_test_shape_infer(op.get(), input_shapes, output_shapes);
}

TEST_P(AdaptiveMaxPoolV8CpuShapeInferenceTest , shape_inference_with_const_map) {
    const auto axes_node = std::make_shared<op::v0::Parameter>(element::i32, PartialShape::dynamic());
    const auto op = make_op(arg, axes_node);

    const auto axes_const = std::make_shared<op::v0::Constant>(element::i32, ov::Shape{axes.size()}, axes);
    const auto axes_tensor = std::make_shared<ngraph::runtime::HostTensor>(axes_const);
    const std::map<size_t, std::shared_ptr<ngraph::runtime::HostTensor>>& constant_data = {{1, axes_tensor}};

    unit_test::cpu_test_shape_infer(op.get(), input_shapes, output_shapes, constant_data);
}

INSTANTIATE_TEST_SUITE_P(
    CpuShapeInfer,
    AdaptiveMaxPoolV8CpuShapeInferenceTest ,
    Values(make_tuple(unit_test::ShapeVector{{1, 3, 1, 2}, {2}}, std::vector<int32_t>{10, 20}, StaticShape({1, 3, 10, 20})),
           make_tuple(unit_test::ShapeVector{{1, 2, 10}, {1}}, std::vector<int32_t>{17}, StaticShape({1, 2, 17}))),
    AdaptiveMaxPoolV8CpuShapeInferenceTest::getTestCaseName);


using AdaptiveMaxPoolV8CpuShapeInferenceThrowExceptionTest = AdaptiveMaxPoolV8CpuShapeInferenceTest;

TEST_P(AdaptiveMaxPoolV8CpuShapeInferenceThrowExceptionTest, wrong_pattern) {
    const auto axes_node = std::make_shared<op::v0::Parameter>(element::i32, PartialShape::dynamic());
    const auto op = make_op(arg, axes_node);

    const auto axes_const = std::make_shared<op::v0::Constant>(element::i32, ov::Shape{axes.size()}, axes);
    const auto axes_tensor = std::make_shared<ngraph::runtime::HostTensor>(axes_const);
    const std::map<size_t, std::shared_ptr<ngraph::runtime::HostTensor>>& constant_data = {{1, axes_tensor}};

    //  OV_EXPECT_THROW(unit_test::cpu_test_shape_infer(op.get(), input_shapes, output_shapes, constant_data),
    //                  InferenceEngine::Unexpected,
    //                  HasSubstr(os.str()));

    // TODO ,implementation should throw exception
    // ASSERT_THROW(unit_test::cpu_test_shape_infer(op.get(), input_shapes, output_shapes, const_data),
    //             InferenceEngine::GeneralError);
}

INSTANTIATE_TEST_SUITE_P(
    CpuShapeInfer,
    AdaptiveMaxPoolV8CpuShapeInferenceThrowExceptionTest,
    Values(make_tuple(unit_test::ShapeVector{{1, 3, 10, 2, 4} , {3}}, std::vector<int32_t>{9, 8}, StaticShape({}))),
    AdaptiveMaxPoolV8CpuShapeInferenceThrowExceptionTest::getTestCaseName);


