// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/test_assertions.hpp"
#include <gtest/gtest.h>
#include "custom_shape_infer.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/unsqueeze.hpp"
namespace ov {
namespace intel_cpu {
namespace unit_test {
namespace cpu_shape_infer {

using namespace ov;
using namespace ov::intel_cpu;
using namespace testing;
using UnsqueezeTestParams = std::tuple<unit_test::ShapeVector, // Input shapes
                                       std::vector<int64_t>,   // Unsqueeze axes
                                       StaticShape             // Expected shape
                                       >;

class UnsqueezeCpuShapeInferenceTest  : public unit_test::OpCpuShapeInferenceTest<op::v0::Unsqueeze>,
                                          public WithParamInterface<UnsqueezeTestParams> {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<UnsqueezeTestParams>& obj) {
        const auto& [tmp_input_shapes, tmp_axes, tmp_exp_shape] = obj.param;
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
        arg = std::make_shared<op::v0::Parameter>(element::f32, input_shapes.front().get_shape());
    }

    std::vector<int64_t> axes;
    std::shared_ptr<op::v0::Parameter> arg;
};

TEST_P(UnsqueezeCpuShapeInferenceTest , shape_inference_empty_const_map) {
    const auto axes_node = std::make_shared<op::v0::Constant>(element::i64, ov::Shape{axes.size()}, axes);
    op = std::make_shared<op::v0::Unsqueeze>(arg, axes_node);
    output_shapes.push_back(exp_shape);
    unit_test::cpu_test_shape_infer(op.get(), input_shapes, output_shapes);
}

TEST_P(UnsqueezeCpuShapeInferenceTest , shape_inference_with_const_map) {
    const auto axes_node = std::make_shared<op::v0::Parameter>(element::i64, PartialShape::dynamic());
    op = std::make_shared<op::v0::Unsqueeze>(arg, axes_node);

    const auto axes_tensor = ov::Tensor(element::i64, ov::Shape{axes.size()}, axes.data());
    const std::unordered_map<size_t, ov::Tensor> constant_data = {{1, axes_tensor}};
    output_shapes.push_back(exp_shape);
    unit_test::cpu_test_shape_infer(op.get(), input_shapes, output_shapes, constant_data);
}

INSTANTIATE_TEST_SUITE_P(
    CpuShapeInfer,
    UnsqueezeCpuShapeInferenceTest ,
    Values(make_tuple(unit_test::ShapeVector{{0}, {1}}, std::vector<int64_t>{-1}, StaticShape({0, 1})),
           make_tuple(unit_test::ShapeVector{{0}, {1}}, std::vector<int64_t>{0}, StaticShape({1, 0})),
           make_tuple(unit_test::ShapeVector{{1}, {1}}, std::vector<int64_t>{1}, StaticShape({1, 1})),
           make_tuple(unit_test::ShapeVector{{2}, {1}}, std::vector<int64_t>{0}, StaticShape({1, 2})),
           make_tuple(unit_test::ShapeVector{{2}, {1}}, std::vector<int64_t>{1}, StaticShape({2, 1})),
           make_tuple(unit_test::ShapeVector{{2}, {1}}, std::vector<int64_t>{-1}, StaticShape({2, 1})),
           make_tuple(unit_test::ShapeVector{{2}, {1}}, std::vector<int64_t>{-2}, StaticShape({1, 2})),
           make_tuple(unit_test::ShapeVector{{2, 3}, {2}}, std::vector<int64_t>{0, 3}, StaticShape({1, 2, 3, 1})),
           make_tuple(unit_test::ShapeVector{{2, 4}, {2}}, std::vector<int64_t>{2, 1}, StaticShape({2, 1, 1, 4})),
           make_tuple(unit_test::ShapeVector{{3, 2}, {3}}, std::vector<int64_t>{0, 2, 4}, StaticShape({1, 3, 1, 2, 1})),
           make_tuple(unit_test::ShapeVector{{3, 2}, {3}}, std::vector<int64_t>{4, 2, 0}, StaticShape({1, 3, 1, 2, 1})),
           make_tuple(unit_test::ShapeVector{{3, 2}, {3}}, std::vector<int64_t>{2, 0, 4}, StaticShape({1, 3, 1, 2, 1})),
           make_tuple(unit_test::ShapeVector{{10, 0, 3}, {4}},
                      std::vector<int64_t>{1, -1, 3, -2},
                      StaticShape({10, 1, 0, 1, 3, 1, 1})),
           make_tuple(unit_test::ShapeVector{{2, 6, 7, 8, 3}, {1}}, std::vector<int64_t>{0}, StaticShape({1, 2, 6, 7, 8, 3})),
           make_tuple(unit_test::ShapeVector{{2, 3}, {2}}, std::vector<int64_t>{1, 1}, StaticShape({2, 1, 3})),
           make_tuple(unit_test::ShapeVector{{3, 2}, {3}}, std::vector<int64_t>{1, -1, 1}, StaticShape({3, 1, 2, 1})),
           make_tuple(unit_test::ShapeVector{{3, 2}, {4}}, std::vector<int64_t>{1, -1, 1, -1}, StaticShape({3, 1, 2, 1})),
           make_tuple(unit_test::ShapeVector{{3, 2}, {5}}, std::vector<int64_t>{2, -1, 2, -1, 0}, StaticShape({1, 3, 1, 2, 1})),
           make_tuple(unit_test::ShapeVector{{2, 6, 7, 8, 3}, {2}}, std::vector<int64_t>{-1, -1}, StaticShape({2, 6, 7, 8, 3, 1}))),
    UnsqueezeCpuShapeInferenceTest::getTestCaseName);

using UnsqueezeCpuShapeInferenceThrowExceptionTest = UnsqueezeCpuShapeInferenceTest;
TEST_P(UnsqueezeCpuShapeInferenceThrowExceptionTest, wrong_pattern) {
    const auto axes_node = std::make_shared<op::v0::Parameter>(element::i64, PartialShape::dynamic());
    const auto op = make_op(arg, axes_node);

    const auto axes_tensor = ov::Tensor(element::i64, ov::Shape{axes.size()}, axes.data());
    const std::unordered_map<size_t, ov::Tensor> constant_data = {{1, axes_tensor}};
    std::ostringstream os;
    os << "[cpu]unsqueeze: the shape of input data ";
    os << "(";
    for (size_t i = 0; i < input_shapes[0].size(); i++) {
        os << input_shapes[0][i];
        if (i < input_shapes[0].size() - 1) {
            os << ".";
        }
    }
    os << ")";
    os << " conflicts with the unsqueeze pattern ";
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
    UnsqueezeCpuShapeInferenceThrowExceptionTest,
    Values(make_tuple(unit_test::ShapeVector{{1, 2}, {1}}, std::vector<int64_t>{3}, StaticShape({})),
           make_tuple(unit_test::ShapeVector{{1, 2}, {2}}, std::vector<int64_t>{3, -1}, StaticShape({})),
           make_tuple(unit_test::ShapeVector{{1, 2}, {1}}, std::vector<int64_t>{-4}, StaticShape({}))),
    UnsqueezeCpuShapeInferenceThrowExceptionTest::getTestCaseName);

} // namespace cpu_shape_infer
} // namespace unit_test
} // namespace intel_cpu
} // namespace ov
