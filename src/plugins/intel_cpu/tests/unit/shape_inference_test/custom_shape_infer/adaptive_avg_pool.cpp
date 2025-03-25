// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include "common_test_utils/test_assertions.hpp"
#include "custom_shape_infer.hpp"
#include "openvino/op/ops.hpp"
namespace ov {
namespace intel_cpu {
namespace unit_test {
namespace cpu_shape_infer {
using namespace ov;
using namespace ov::intel_cpu;
using namespace testing;

using AdaptiveAvgPoolV8TestParams = std::tuple<unit_test::ShapeVector, // Input shapes
                                               std::vector<int32_t>,   // output_shapes
                                               StaticShape             // Expected shape
                                               >;

class AdaptiveAvgPoolV8CpuShapeInferenceTest  : public unit_test::OpCpuShapeInferenceTest<op::v8::AdaptiveAvgPool>,
                                                public WithParamInterface<AdaptiveAvgPoolV8TestParams> {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<AdaptiveAvgPoolV8TestParams>& obj) {
        unit_test::ShapeVector tmp_input_shapes;
        std::vector<int32_t> tmp_axes;
        StaticShape tmp_exp_shape;
        std::tie(tmp_input_shapes, tmp_axes, tmp_exp_shape) = obj.param;
        std::ostringstream result;
        result << "IS" << ov::test::utils::vec2str(tmp_input_shapes) << "_";
        result << "sd" << ov::test::utils::vec2str(tmp_axes) << "_";
        result << "exp_shape" << tmp_exp_shape;
        return result.str();
    }

protected:
    void SetUp() override {
        std::tie(input_shapes, axes, exp_shape) = GetParam();
        output_shapes = unit_test::ShapeVector(0);
        output_shapes.push_back(exp_shape);
        arg = std::make_shared<op::v0::Parameter>(element::f64, input_shapes.front().get_shape());
    }
    std::vector<int32_t> axes;
    std::shared_ptr<op::v0::Parameter> arg;
    bool specalZero;
};

TEST_P(AdaptiveAvgPoolV8CpuShapeInferenceTest , shape_inference_empty_const_map) {
    const auto axes_node = std::make_shared<op::v0::Constant>(element::i32, ov::Shape{axes.size()}, axes);
    const auto op = make_op(arg, axes_node);

    unit_test::cpu_test_shape_infer(op.get(), input_shapes, output_shapes);
}

TEST_P(AdaptiveAvgPoolV8CpuShapeInferenceTest , shape_inference_with_const_map) {
    const auto axes_node = std::make_shared<op::v0::Parameter>(element::i32, PartialShape::dynamic());
    const auto op = make_op(arg, axes_node);

    const auto axes_tensor = ov::Tensor(element::i32, ov::Shape{axes.size()}, axes.data());
    const std::unordered_map<size_t, ov::Tensor> constant_data = {{1, axes_tensor}};

    unit_test::cpu_test_shape_infer(op.get(), input_shapes, output_shapes, constant_data);
}

INSTANTIATE_TEST_SUITE_P(
    CpuShapeInfer,
    AdaptiveAvgPoolV8CpuShapeInferenceTest ,
    Values(make_tuple(unit_test::ShapeVector{{1, 3, 1, 2}, {2}}, std::vector<int32_t>{10, 20}, StaticShape({1, 3, 10, 20})),
           make_tuple(unit_test::ShapeVector{{1, 2, 10}, {1}}, std::vector<int32_t>{17}, StaticShape({1, 2, 17}))),
    AdaptiveAvgPoolV8CpuShapeInferenceTest::getTestCaseName);
} // namespace cpu_shape_infer
} // namespace unit_test
} // namespace intel_cpu
} // namespace ov
