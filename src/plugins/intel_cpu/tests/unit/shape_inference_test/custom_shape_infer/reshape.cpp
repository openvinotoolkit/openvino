// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <ostream>

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

using ReshapeTestParams = std::tuple<unit_test::ShapeVector, // Input shapes
                                     std::vector<int64_t>,   // reshape axes
                                     StaticShape,            // Expected shape
                                     bool                    // specal zero
                                     >;

class ReshapeCpuShapeInferenceTest  : public unit_test::OpCpuShapeInferenceTest<op::v1::Reshape>,
                                      public WithParamInterface<ReshapeTestParams> {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ReshapeTestParams>& obj) {
        unit_test::ShapeVector tmp_input_shapes;
        std::vector<int64_t> tmp_axes;
        StaticShape tmp_exp_shape;
        bool tmp_specialZero;
        std::tie(tmp_input_shapes, tmp_axes, tmp_exp_shape, tmp_specialZero) = obj.param;
        std::ostringstream result;
        result << "IS" << ov::test::utils::vec2str(tmp_input_shapes) << "_";
        result << "axes" << ov::test::utils::vec2str(tmp_axes) << "_";
        result << "exp_shape(" << tmp_exp_shape << ")_";
        result << "specalZero(" << unit_test::boolToString(tmp_specialZero) << ")";
        return result.str();
    }

protected:
    void SetUp() override {
        std::tie(input_shapes, axes, exp_shape, specalZero) = GetParam();
        output_shapes = unit_test::ShapeVector(0);
        arg = std::make_shared<op::v0::Parameter>(element::f32, input_shapes.front().get_shape());
    }

    std::vector<int64_t> axes;
    std::shared_ptr<op::v0::Parameter> arg;
    bool specalZero;
};

TEST_P(ReshapeCpuShapeInferenceTest , shape_inference_empty_const_map) {
    const auto axes_node = std::make_shared<op::v0::Constant>(element::i64, ov::Shape{axes.size()}, axes);
    const auto op = make_op(arg, axes_node, specalZero);
    output_shapes.push_back(exp_shape);
    unit_test::cpu_test_shape_infer(op.get(), input_shapes, output_shapes);
}

TEST_P(ReshapeCpuShapeInferenceTest , shape_inference_with_const_map) {
    const auto axes_node = std::make_shared<op::v0::Parameter>(element::i64, PartialShape::dynamic());
    const auto op = make_op(arg, axes_node, specalZero);

    const auto axes_tensor = ov::Tensor(element::i64, ov::Shape{axes.size()}, axes.data());
    const std::unordered_map<size_t, ov::Tensor> constant_data = {{1, axes_tensor}};

    output_shapes.push_back(exp_shape);
    unit_test::cpu_test_shape_infer(op.get(), input_shapes, output_shapes, constant_data);
}

INSTANTIATE_TEST_SUITE_P(
    CpuShapeInfer,
    ReshapeCpuShapeInferenceTest ,
    Values(make_tuple(unit_test::ShapeVector{{1, 2, 3, 1}, {2}}, std::vector<int64_t>{0, -1}, StaticShape({1, 6}), true),
           make_tuple(unit_test::ShapeVector{{1, 2, 3, 8}, {4}}, std::vector<int64_t>{1, 2, 3, 8}, StaticShape({1, 2, 3, 8}), true),
           make_tuple(unit_test::ShapeVector{{1, 2, 3, 8}, {4}}, std::vector<int64_t>{0, 2, 0, 8}, StaticShape({1, 2, 3, 8}), true),
           make_tuple(unit_test::ShapeVector{{0, 2, 2}, {2}}, std::vector<int64_t>{0, -1}, StaticShape({0, 4}), true),
           make_tuple(unit_test::ShapeVector{{0, 2, 2}, {2}}, std::vector<int64_t>{0, 4}, StaticShape({0, 4}), true),
           make_tuple(unit_test::ShapeVector{{4, 0, 2}, {2}}, std::vector<int64_t>{-1, 0}, StaticShape({8, 0}), true),
           make_tuple(unit_test::ShapeVector{{1, 2, 3, 8}, {4}}, std::vector<int64_t>{1, 2, 3, 8}, StaticShape({1, 2, 3, 8}), false),
           make_tuple(unit_test::ShapeVector{{0, 2, 2}, {2}}, std::vector<int64_t>{3, 0}, StaticShape({3, 0}), false),
           make_tuple(unit_test::ShapeVector{{0, 2, 2}, {2}}, std::vector<int64_t>{4, 0}, StaticShape({4, 0}), false),
           make_tuple(unit_test::ShapeVector{{3, 6, 5, 5}, {2}}, std::vector<int64_t>{0, -1}, StaticShape({3, 150}), true)),
        ReshapeCpuShapeInferenceTest::getTestCaseName);

using ReshapeCpuShapeInferenceThrowExceptionTest = ReshapeCpuShapeInferenceTest;

TEST_P(ReshapeCpuShapeInferenceThrowExceptionTest, wrong_pattern) {
    const auto axes_node = std::make_shared<op::v0::Parameter>(element::i64, PartialShape::dynamic());
    const auto op = make_op(arg, axes_node, specalZero);

    const auto axes_tensor = ov::Tensor(element::i64, ov::Shape{axes.size()}, axes.data());
    const std::unordered_map<size_t, ov::Tensor> constant_data = {{1, axes_tensor}};
    std::ostringstream os;
    os << "[cpu]reshape: the shape of input data ";
    os << "(";
    for (size_t i = 0; i < input_shapes[0].size(); i++) {
        os << input_shapes[0][i];
        if (i < input_shapes[0].size() - 1) {
            os << ".";
        }
    }
    os << ")";
    os << " conflicts with the reshape pattern ";
    os << "(";
    for (size_t i = 0; i < axes.size(); i++) {
        os << axes[i];
        if (i < axes.size() - 1) {
            os << ".";
        }
    }
    os << ")";

    OV_EXPECT_THROW(unit_test::cpu_test_shape_infer(op.get(), input_shapes, output_shapes, constant_data),
                    ov::Exception,
                    HasSubstr(os.str()));
}

INSTANTIATE_TEST_SUITE_P(
    CpuShapeInfer,
    ReshapeCpuShapeInferenceThrowExceptionTest,
    Values(make_tuple(unit_test::ShapeVector{{1, 2, 3, 1}, {3}}, std::vector<int64_t>{0, -1, -1}, StaticShape({}), true),
           make_tuple(unit_test::ShapeVector{{1, 2, 3, 8}, {4}}, std::vector<int64_t>{1, 2, 3, 6}, StaticShape({}), true),
           make_tuple(unit_test::ShapeVector{{1, 2, 3, 8}, {4}}, std::vector<int64_t>{0, 3, 0, 8}, StaticShape({}), true),
           make_tuple(unit_test::ShapeVector{{1, 2, 3, 8}, {4}}, std::vector<int64_t>{1, 2, 3, 6}, StaticShape({}), false),
           make_tuple(unit_test::ShapeVector{{0, 2, 2}, {2}}, std::vector<int64_t>{3, 3}, StaticShape({}), false)),
        ReshapeCpuShapeInferenceThrowExceptionTest::getTestCaseName);

} // namespace cpu_shape_infer
} // namespace unit_test
} // namespace intel_cpu
} // namespace ov
