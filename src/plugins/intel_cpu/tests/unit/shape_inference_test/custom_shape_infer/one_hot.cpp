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

using OneHotTestParams = std::tuple<unit_test::ShapeVector, // Input shapes
                                    int64_t,                // depth
                                    int32_t,                // on_value
                                    int32_t,                // off_value
                                    StaticShape             // Expected shape
                                    >;

class OneHotCpuShapeInferenceTest  : public unit_test::OpCpuShapeInferenceTest<op::v1::OneHot>,
                                      public WithParamInterface<OneHotTestParams> {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<OneHotTestParams>& obj) {
        unit_test::ShapeVector tmp_input_shapes;
        int64_t tmp_depth;
        int32_t tmp_on;
        int32_t tmp_off;
        StaticShape tmp_exp_shape;
        std::tie(tmp_input_shapes, tmp_depth, tmp_on, tmp_off, tmp_exp_shape) = obj.param;
        std::ostringstream result;
        result << "IS" << ov::test::utils::vec2str(tmp_input_shapes) << "_";
        result << "depth" << tmp_depth << "_";
        result << "on" << tmp_on << "_";
        result << "off" << tmp_off << "_";
        result << "exp_shape" << tmp_exp_shape;
        return result.str();
    }

protected:
    void SetUp() override {
        std::tie(input_shapes, m_depth, m_on, m_off, exp_shape) = GetParam();
        output_shapes = unit_test::ShapeVector(0);
        output_shapes.push_back(exp_shape);
        arg = std::make_shared<op::v0::Parameter>(element::i64, input_shapes.front().get_shape());
    }

    int64_t m_depth;
    int32_t m_on;
    int32_t m_off;
    std::shared_ptr<op::v0::Parameter> arg;
};

TEST_P(OneHotCpuShapeInferenceTest , shape_inference_empty_const_map) {
    const auto depth = op::v0::Constant::create(element::i64, ov::Shape{}, {m_depth});
    const auto on_value = op::v0::Constant::create(element::i32, ov::Shape{}, {m_on});
    const auto off_value = op::v0::Constant::create(element::i32, ov::Shape{}, {m_off});
    int64_t axis = -1;
    const auto op = make_op(arg, depth, on_value, off_value, axis);
    unit_test::cpu_test_shape_infer(op.get(), input_shapes, output_shapes);
}

TEST_P(OneHotCpuShapeInferenceTest , shape_inference_with_const_map) {
    const auto depth = std::make_shared<op::v0::Parameter>(element::i64, ov::Shape{});
    const auto on = std::make_shared<op::v0::Parameter>(element::i32, ov::Shape{});
    const auto off = std::make_shared<op::v0::Parameter>(element::i32, ov::Shape{});
    int64_t axis = -1;
    const auto op = make_op(arg, depth, on, off, axis);

    const auto depth_tensor = ov::Tensor(element::i64, ov::Shape{}, &m_depth);
    const auto on_tensor = ov::Tensor(element::i32, ov::Shape{}, &m_on);
    const auto off_tensor = ov::Tensor(element::i32, ov::Shape{}, &m_off);
    const std::unordered_map<size_t, ov::Tensor> constant_data = {{1, depth_tensor}, {2, on_tensor}, {3, off_tensor}};

    unit_test::cpu_test_shape_infer(op.get(), input_shapes, output_shapes, constant_data);
}

INSTANTIATE_TEST_SUITE_P(
    CpuShapeInfer,
    OneHotCpuShapeInferenceTest ,
    Values(make_tuple(unit_test::ShapeVector{{3}, {}, {}, {}}, 2, 5, 10, StaticShape({3, 2})),
           make_tuple(unit_test::ShapeVector{{3}, {}, {}, {}}, 2, 1, 0, StaticShape({3, 2}))),
    OneHotCpuShapeInferenceTest::getTestCaseName);

using OneHotCpuShapeInferenceThrowExceptionTest = OneHotCpuShapeInferenceTest;
TEST_P(OneHotCpuShapeInferenceThrowExceptionTest, wrong_pattern) {
    const auto depth = std::make_shared<op::v0::Parameter>(element::i64, ov::Shape{});
    const auto on = std::make_shared<op::v0::Parameter>(element::i32, ov::Shape{});
    const auto off = std::make_shared<op::v0::Parameter>(element::i32, ov::Shape{});
    int64_t axis = -1;
    const auto op = make_op(arg, depth, on, off, axis);

    const auto depth_tensor = ov::Tensor(element::i64, ov::Shape{}, &m_depth);
    const auto on_tensor = ov::Tensor(element::i32, ov::Shape{}, &m_on);
    const auto off_tensor = ov::Tensor(element::i32, ov::Shape{}, &m_off);
    const std::unordered_map<size_t, ov::Tensor> constant_data = {{1, depth_tensor}, {2, on_tensor}, {3, off_tensor}};

    OV_EXPECT_THROW(unit_test::cpu_test_shape_infer(op.get(), input_shapes, output_shapes, constant_data),
                    ov::Exception,
                    testing::HasSubstr("OneHot depth value can't be negative"));
}

INSTANTIATE_TEST_SUITE_P(
    CpuShapeInfer,
    OneHotCpuShapeInferenceThrowExceptionTest,
    Values(make_tuple(unit_test::ShapeVector{{3}, {}, {}, {}}, -2, 1, 0, StaticShape({}))),
    OneHotCpuShapeInferenceThrowExceptionTest::getTestCaseName);

} // namespace cpu_shape_infer
} // namespace unit_test
} // namespace intel_cpu
} // namespace ov
