// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "common_test_utils/test_assertions.hpp"
#include <gtest/gtest.h>
#include "custom_shape_infer.hpp"
#include "openvino/op/ops.hpp"
namespace ov {
namespace intel_cpu {
namespace unit_test {
namespace cpu_shape_infer {

using namespace ov;
using namespace ov::intel_cpu;
using namespace testing;

using transpose_params = std::tuple<unit_test::ShapeVector, // Input shapes
                                    std::vector<int64_t>,   // transpose order
                                    StaticShape             // Expected shape
                                    >;

class TransposeCpuShapeInferenceTest  : public unit_test::OpCpuShapeInferenceTest<op::v1::Transpose>,
                                        public WithParamInterface<transpose_params> {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<transpose_params>& obj) {
        unit_test::ShapeVector tmp_input_shapes;
        std::vector<int64_t> tmp_transpose_order;
        StaticShape tmp_exp_shape;
        std::tie(tmp_input_shapes, tmp_transpose_order, tmp_exp_shape) = obj.param;
        std::ostringstream result;
        result << "input_shapes(" << ov::test::utils::vec2str(tmp_input_shapes) << ")_";
        result << "order(" << ov::test::utils::vec2str(tmp_transpose_order) << ")_";
        result << "exp_shape(" << tmp_exp_shape << ")";
        return result.str();
    }

protected:
    void SetUp() override {
        std::tie(input_shapes, transpose_order, exp_shape) = GetParam();
        output_shapes = unit_test::ShapeVector(0);
        output_shapes.push_back(exp_shape);
        ASSERT_EQ(input_shapes.size(), 2);
        arg = std::make_shared<op::v0::Parameter>(element::f32, input_shapes.front().get_shape());
    }

    std::vector<int64_t> transpose_order;
    std::shared_ptr<op::v0::Parameter> arg;
};

TEST_P(TransposeCpuShapeInferenceTest , shape_inference_empty_const_map) {
    const auto order =
        std::make_shared<op::v0::Constant>(element::i64, ov::Shape{transpose_order.size()}, transpose_order);
    auto op = make_op(arg, order);
    unit_test::cpu_test_shape_infer(op.get(), input_shapes, output_shapes);
}

/** \brief Use transpose order -> output shape dimensions shall be as transpose order. */
INSTANTIATE_TEST_SUITE_P(
    CpuShapeInfer,
    TransposeCpuShapeInferenceTest,
    Values(make_tuple(unit_test::ShapeVector{{3}, {1}}, std::vector<int64_t>{0}, StaticShape({3})),
           make_tuple(unit_test::ShapeVector{{5, 2}, {2}}, std::vector<int64_t>{0, 1}, StaticShape({5, 2})),
           make_tuple(unit_test::ShapeVector{{8, 3}, {2}}, std::vector<int64_t>{1, 0}, StaticShape({3, 8})),
           make_tuple(unit_test::ShapeVector{{1, 0, 2}, {3}}, std::vector<int64_t>{2, 0, 1}, StaticShape({2, 1, 0})),
           make_tuple(unit_test::ShapeVector{{10, 8, 9, 2}, {4}}, std::vector<int64_t>{2, 0, 3, 1}, StaticShape({9, 10, 2, 8})),
           make_tuple(unit_test::ShapeVector{{1, 2, 3, 4}, {4}}, std::vector<int64_t>{1, 3, 2, 0}, StaticShape({2, 4, 3, 1})),
           make_tuple(unit_test::ShapeVector{{1}, {0}}, std::vector<int64_t>{}, StaticShape({1})),
           make_tuple(unit_test::ShapeVector{{23}, {0}}, std::vector<int64_t>{}, StaticShape({23})),
           make_tuple(unit_test::ShapeVector{{3, 8}, {0}}, std::vector<int64_t>{}, StaticShape({8, 3})),
           make_tuple(unit_test::ShapeVector{{1, 0, 2}, {0}}, std::vector<int64_t>{}, StaticShape({2, 0, 1})),
           make_tuple(unit_test::ShapeVector{{21, 1, 5, 9}, {0}}, std::vector<int64_t>{}, StaticShape({9, 5, 1, 21})),
           make_tuple(unit_test::ShapeVector{{0, 0, 0}, {0}}, std::vector<int64_t>{}, StaticShape({0, 0, 0})),
           make_tuple(unit_test::ShapeVector{{0, 2, 0}, {0}}, std::vector<int64_t>{}, StaticShape({0, 2, 0})),
           make_tuple(unit_test::ShapeVector{{0, 2, 0, 0}, {0}}, std::vector<int64_t>{}, StaticShape({0, 0, 2, 0}))),
    TransposeCpuShapeInferenceTest::getTestCaseName);

using TransposeCpuShapeInferenceThrowExceptionTest = TransposeCpuShapeInferenceTest;
TEST_P(TransposeCpuShapeInferenceThrowExceptionTest, shape_inference_in_const_map) {
    const auto order = std::make_shared<op::v0::Parameter>(element::i64, PartialShape::dynamic());
    auto op = make_op(arg, order);

    const auto const_tensor = transpose_order.empty() ? ov::Tensor(element::i64, ov::Shape{transpose_order.size()})
                                                      : ov::Tensor(element::i64, ov::Shape{transpose_order.size()}, transpose_order.data());
    const std::unordered_map<size_t, ov::Tensor> const_map = {{1, const_tensor}};

    OV_EXPECT_THROW(unit_test::cpu_test_shape_infer(op.get(), input_shapes, output_shapes, const_map),
                    ov::Exception,
                    HasSubstr("TODO: Support parameterized Order input for dynamic shapes."));
}

INSTANTIATE_TEST_SUITE_P(
    CpuShapeInfer,
    TransposeCpuShapeInferenceThrowExceptionTest,
    Values(make_tuple(unit_test::ShapeVector{{3}, {1}}, std::vector<int64_t>{0}, StaticShape({3})),
           make_tuple(unit_test::ShapeVector{{1}, {0}}, std::vector<int64_t>{}, StaticShape({1}))),
    TransposeCpuShapeInferenceThrowExceptionTest::getTestCaseName);

} // namespace cpu_shape_infer
} // namespace unit_test
} // namespace intel_cpu
} // namespace ov
