// Copyright (C) 2022 Intel Corporation
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

using SqueezeTestParams = std::tuple<unit_test::ShapeVector, // Input shapes
                                     std::vector<int64_t>,   // Squeeze axes
                                     StaticShape             // Expected shape
                                     >;

class SqueezeCpuShapeInferenceTest  : public unit_test::OpCpuShapeInferenceTest<op::v0::Squeeze>,
                                      public WithParamInterface<SqueezeTestParams> {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<SqueezeTestParams>& obj) {
        unit_test::ShapeVector tmp_input_shapes;
        std::vector<int64_t> tmp_axes;
        StaticShape tmp_exp_shape;
        std::tie(tmp_input_shapes, tmp_axes, tmp_exp_shape) = obj.param;
        std::ostringstream result;
        result << "IS" << ov::test::utils::vec2str(tmp_input_shapes) << "_";
        result << "axes" << ov::test::utils::vec2str(tmp_axes) << "_";
        result << "exp_shape(" << tmp_exp_shape << ")";
        return result.str();
    }

protected:
    void SetUp() override {
        std::tie(input_shapes, axes, exp_shape) = GetParam();
        output_shapes = unit_test::ShapeVector(0);
        output_shapes.push_back(exp_shape);
        arg = std::make_shared<op::v0::Parameter>(element::f32, input_shapes.front().get_shape());
    }

    std::vector<int64_t> axes;
    std::shared_ptr<op::v0::Parameter> arg;
};

TEST_P(SqueezeCpuShapeInferenceTest , shape_inference_empty_const_map) {
    const auto axes_node = std::make_shared<op::v0::Constant>(element::i64, ov::Shape{axes.size()}, axes);
    const auto op = make_op(arg, axes_node);
    unit_test::cpu_test_shape_infer(op.get(), input_shapes, output_shapes);
}

TEST_P(SqueezeCpuShapeInferenceTest , shape_inference_with_const_map) {
    const auto axes_node = std::make_shared<op::v0::Parameter>(element::i64, PartialShape::dynamic());
    const auto op = make_op(arg, axes_node);

    const auto axes_tensor = axes.empty() ? ov::Tensor(element::i64, ov::Shape{axes.size()})
                                          : ov::Tensor(element::i64, ov::Shape{axes.size()}, axes.data());
    const std::unordered_map<size_t, ov::Tensor> constant_data = {{1, axes_tensor}};

    unit_test::cpu_test_shape_infer(op.get(), input_shapes, output_shapes, constant_data);
}

INSTANTIATE_TEST_SUITE_P(
    CpuShapeInfer,
    SqueezeCpuShapeInferenceTest ,
    Values(make_tuple(unit_test::ShapeVector{{1}, {1}}, std::vector<int64_t>{-1}, StaticShape({})),
           make_tuple(unit_test::ShapeVector{{1}, {1}}, std::vector<int64_t>{0}, StaticShape({})),
           make_tuple(unit_test::ShapeVector{{1, 2, 3, 1}, {2}}, std::vector<int64_t>{0, 3}, StaticShape({2, 3})),
           make_tuple(unit_test::ShapeVector{{2, 1, 1, 4}, {2}}, std::vector<int64_t>{2, 1}, StaticShape({2, 4})),
           make_tuple(unit_test::ShapeVector{{1, 3, 1, 2, 1}, {3}}, std::vector<int64_t>{0, 2, 4}, StaticShape({3, 2})),
           make_tuple(unit_test::ShapeVector{{1, 3, 1, 2, 1}, {3}}, std::vector<int64_t>{4, 2, 0}, StaticShape({3, 2})),
           make_tuple(unit_test::ShapeVector{{1, 3, 1, 2, 1}, {3}}, std::vector<int64_t>{2, 0, 4}, StaticShape({3, 2})),
           make_tuple(unit_test::ShapeVector{{10, 1, 0, 1, 3, 1, 1}, {4}},
                      std::vector<int64_t>{1, -1, 3, -2},
                      StaticShape({10, 0, 3})),
           make_tuple(unit_test::ShapeVector{{10, 1, 0, 1, 3, 1, 1}}, std::vector<int64_t>{}, StaticShape({10, 0, 3})),
           make_tuple(unit_test::ShapeVector{{10, 1, 0, 1, 3, 1, 1}, {}}, std::vector<int64_t>{}, StaticShape({10, 0, 3})),
           make_tuple(unit_test::ShapeVector{{2, 1, 7, 8, 3}, {1}}, std::vector<int64_t>{1}, StaticShape({2, 7, 8, 3})),
           make_tuple(unit_test::ShapeVector{{2, 1, 3}, {2}}, std::vector<int64_t>{1, 1}, StaticShape({2, 3})),
           make_tuple(unit_test::ShapeVector{{3, 1, 2, 1}, {3}}, std::vector<int64_t>{1, -1, 1}, StaticShape({3, 2})),
           make_tuple(unit_test::ShapeVector{{3, 1, 2, 1}, {4}}, std::vector<int64_t>{1, -1, 1, -1}, StaticShape({3, 2})),
           make_tuple(unit_test::ShapeVector{{1, 3, 1, 2, 1}, {5}}, std::vector<int64_t>{2, -1, 2, -1, 0}, StaticShape({3, 2})),
           make_tuple(unit_test::ShapeVector{{2, 6, 7, 8, 1}, {2}}, std::vector<int64_t>{-1, -1}, StaticShape({2, 6, 7, 8}))),
    SqueezeCpuShapeInferenceTest::getTestCaseName);

// Tests with non-squeezable dims pointed by axes (no throw, ignore)
class SqueezeCpuShapeInferenceTestNonSqueeezable : public SqueezeCpuShapeInferenceTest {};
TEST_P(SqueezeCpuShapeInferenceTestNonSqueeezable, shape_inference_non_squeezable_with_const_map) {
    const auto axes_node = std::make_shared<op::v0::Parameter>(element::i64, PartialShape::dynamic());
    const auto op = make_op(arg, axes_node);
    const auto axes_tensor = ov::Tensor(element::i64, ov::Shape{axes.size()}, axes.data());
    const std::unordered_map<size_t, ov::Tensor> constant_data = {{1, axes_tensor}};
    unit_test::cpu_test_shape_infer(op.get(), input_shapes, output_shapes, constant_data);
}

INSTANTIATE_TEST_SUITE_P(
    CpuShapeInfer_Squeeze_non_squeezable,
    SqueezeCpuShapeInferenceTestNonSqueeezable,
    Values(
        make_tuple(unit_test::ShapeVector{{1, 2, 3, 1}, {1}}, std::vector<int64_t>{1}, StaticShape({1, 2, 3, 1})),
        make_tuple(unit_test::ShapeVector{{1, 2, 3, 1}, {2}}, std::vector<int64_t>{0, 1}, StaticShape({2, 3, 1})),
        make_tuple(unit_test::ShapeVector{{1, 2, 3, 8}, {1}}, std::vector<int64_t>{2}, StaticShape({1, 2, 3, 8})),
        make_tuple(unit_test::ShapeVector{{1, 2, 3, 8}, {2}}, std::vector<int64_t>{1, 2}, StaticShape({1, 2, 3, 8})),
        make_tuple(unit_test::ShapeVector{{1, 2, 3, 8}, {1}}, std::vector<int64_t>{-1}, StaticShape({1, 2, 3, 8})),
        make_tuple(unit_test::ShapeVector{{1, 2, 3, 8}, {2}}, std::vector<int64_t>{-1, -1}, StaticShape({1, 2, 3, 8})),
        make_tuple(unit_test::ShapeVector{{1, 2, 3, 8}, {2}}, std::vector<int64_t>{-1, -2}, StaticShape({1, 2, 3, 8}))),
    SqueezeCpuShapeInferenceTest::getTestCaseName);

}  // namespace cpu_shape_infer
}  // namespace unit_test
}  // namespace intel_cpu
}  // namespace ov
