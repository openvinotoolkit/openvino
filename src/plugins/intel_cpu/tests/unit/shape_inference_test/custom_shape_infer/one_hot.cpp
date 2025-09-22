// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/one_hot.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/test_assertions.hpp"
#include "custom_shape_infer.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
namespace ov {
namespace intel_cpu {
namespace unit_test {
namespace cpu_shape_infer {

using namespace ov;
using namespace ov::intel_cpu;
using namespace testing;

using OneHotTestParams = std::tuple<unit_test::ShapeVector,  // Input shapes
                                    int64_t,                 // depth
                                    int32_t,                 // on_value
                                    int32_t,                 // off_value
                                    StaticShape              // Expected shape
                                    >;
// Parameters for typed test used test case internal loop.
const auto OneHotTestData =
    std::vector<OneHotTestParams>{make_tuple(unit_test::ShapeVector{{3}, {}, {}, {}}, 2, 5, 10, StaticShape({3, 2})),
                                  make_tuple(unit_test::ShapeVector{{3}, {}, {}, {}}, 2, 1, 0, StaticShape({3, 2}))};

template <typename TOp>
class OneHotCpuShapeInferenceTest : public unit_test::OpCpuShapeInferenceTest<TOp> {
protected:
    void SetUp() override {
        this->output_shapes.resize(0);
    }

    template <class... Args>
    std::shared_ptr<TOp> make_one_hot(Args&&... args) {
        if constexpr (std::is_same_v<TOp, ov::op::v16::OneHot>) {
            return this->make_op(std::forward<Args>(args)...,
                                 ov::op::v16::OneHot::NegativeIndicesMode::IGNORE_NEGATIVE);
        }
        return this->make_op(std::forward<Args>(args)...);
    }
};

TYPED_TEST_SUITE_P(OneHotCpuShapeInferenceTest);

TYPED_TEST_P(OneHotCpuShapeInferenceTest, shape_inference_empty_const_map) {
    for (auto&& data : OneHotTestData) {
        int64_t depth;
        int32_t on;
        int32_t off;
        std::tie(this->input_shapes, depth, on, off, this->exp_shape) = data;
        this->output_shapes = {this->exp_shape};
        const auto depth_op = op::v0::Constant::create(element::i64, ov::Shape{}, {depth});
        const auto on_value_op = op::v0::Constant::create(element::i32, ov::Shape{}, {on});
        const auto off_value_op = op::v0::Constant::create(element::i32, ov::Shape{}, {off});
        const auto input = std::make_shared<op::v0::Parameter>(element::i64, this->input_shapes.front().get_shape());
        int64_t axis = -1;
        const auto op = this->make_one_hot(input, depth_op, on_value_op, off_value_op, axis);
        unit_test::cpu_test_shape_infer(op.get(), this->input_shapes, this->output_shapes);
    }
}

TYPED_TEST_P(OneHotCpuShapeInferenceTest, shape_inference_with_const_map) {
    for (auto&& data : OneHotTestData) {
        int64_t depth;
        int32_t on;
        int32_t off;
        std::tie(this->input_shapes, depth, on, off, this->exp_shape) = data;
        this->output_shapes = {this->exp_shape};
        const auto depth_op = std::make_shared<op::v0::Parameter>(element::i64, ov::Shape{});
        const auto on_value_op = std::make_shared<op::v0::Parameter>(element::i32, ov::Shape{});
        const auto off_value_op = std::make_shared<op::v0::Parameter>(element::i32, ov::Shape{});
        const auto input = std::make_shared<op::v0::Parameter>(element::i64, this->input_shapes.front().get_shape());
        int64_t axis = -1;
        const auto op = this->make_one_hot(input, depth_op, on_value_op, off_value_op, axis);

        const auto depth_tensor = ov::Tensor(element::i64, ov::Shape{}, &depth);
        const auto on_tensor = ov::Tensor(element::i32, ov::Shape{}, &on);
        const auto off_tensor = ov::Tensor(element::i32, ov::Shape{}, &off);
        const std::unordered_map<size_t, ov::Tensor> constant_data = {{1, depth_tensor},
                                                                      {2, on_tensor},
                                                                      {3, off_tensor}};

        unit_test::cpu_test_shape_infer(op.get(), this->input_shapes, this->output_shapes, constant_data);
    }
}

REGISTER_TYPED_TEST_SUITE_P(OneHotCpuShapeInferenceTest,
                            shape_inference_empty_const_map,
                            shape_inference_with_const_map);
using OneHotTypes = Types<op::v1::OneHot, op::v16::OneHot>;
INSTANTIATE_TYPED_TEST_SUITE_P(CpuShapeInfer, OneHotCpuShapeInferenceTest, OneHotTypes);

template <typename TOp>
using OneHotCpuShapeInferenceThrowExceptionTest = OneHotCpuShapeInferenceTest<TOp>;

TYPED_TEST_SUITE_P(OneHotCpuShapeInferenceThrowExceptionTest);
TYPED_TEST_P(OneHotCpuShapeInferenceThrowExceptionTest, wrong_pattern) {
    this->input_shapes = unit_test::ShapeVector{{3}, {}, {}, {}};
    this->exp_shape = StaticShape({});
    this->output_shapes = {this->exp_shape};
    const auto depth_op = std::make_shared<op::v0::Parameter>(element::i64, ov::Shape{});
    const auto on_op = std::make_shared<op::v0::Parameter>(element::i32, ov::Shape{});
    const auto off_op = std::make_shared<op::v0::Parameter>(element::i32, ov::Shape{});
    const auto input = std::make_shared<op::v0::Parameter>(element::i64, this->input_shapes.front().get_shape());
    int64_t axis = -1;

    const auto op = this->make_one_hot(input, depth_op, on_op, off_op, axis);

    const int64_t depth = -2;
    const int32_t on = 1;
    const int32_t off = 0;

    const auto depth_tensor = ov::Tensor(element::i64, ov::Shape{}, &depth);
    const auto on_tensor = ov::Tensor(element::i32, ov::Shape{}, &on);
    const auto off_tensor = ov::Tensor(element::i32, ov::Shape{}, &off);
    const std::unordered_map<size_t, ov::Tensor> constant_data = {{1, depth_tensor}, {2, on_tensor}, {3, off_tensor}};
    OV_EXPECT_THROW(unit_test::cpu_test_shape_infer(op.get(), this->input_shapes, this->output_shapes, constant_data),
                    ov::Exception,
                    testing::HasSubstr("OneHot depth value can't be negative"));
}

REGISTER_TYPED_TEST_SUITE_P(OneHotCpuShapeInferenceThrowExceptionTest, wrong_pattern);
INSTANTIATE_TYPED_TEST_SUITE_P(CpuShapeInfer, OneHotCpuShapeInferenceThrowExceptionTest, OneHotTypes);

}  // namespace cpu_shape_infer
}  // namespace unit_test
}  // namespace intel_cpu
}  // namespace ov
