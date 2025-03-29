// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include "openvino/op/ops.hpp"
#include "common_test_utils/test_assertions.hpp"
#include "custom_shape_infer.hpp"
namespace ov {
namespace intel_cpu {
namespace unit_test {
namespace cpu_shape_infer {

using namespace ov;
using namespace ov::intel_cpu;

template <class T>
class CpuShapeInferenceTest_BEA : public testing::Test {};

// CpuShapeInferenceTest for BinaryElementwiseArithmetis (BEA) operations
TYPED_TEST_SUITE_P(CpuShapeInferenceTest_BEA);

TYPED_TEST_P(CpuShapeInferenceTest_BEA, shape_inference_autob_numpy_equal_rank) {
    auto A = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1});
    auto B = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1});

    auto node = std::make_shared<TypeParam>(A, B);

    std::vector<StaticShape> static_input_shapes = {StaticShape{3, 1, 1, 5}, StaticShape{3, 1, 6, 1}},
            static_output_shapes = {StaticShape{3, 1, 6, 5}};

    unit_test::cpu_test_shape_infer(node.get(), static_input_shapes, static_output_shapes);
}

TYPED_TEST_P(CpuShapeInferenceTest_BEA, shape_inference_autob_numpy_a_rank_higher) {
    auto A = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1});
    auto B = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1});

    auto node = std::make_shared<TypeParam>(A, B);

    std::vector<StaticShape> static_input_shapes = {StaticShape{3, 4, 1, 5}, StaticShape{4, 6, 1}},
            static_output_shapes = {StaticShape{3, 4, 6, 5}};

    unit_test::cpu_test_shape_infer(node.get(), static_input_shapes, static_output_shapes);
}

TYPED_TEST_P(CpuShapeInferenceTest_BEA, shape_inference_autob_numpy_b_rank_higher) {
    auto A = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1});
    auto B = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1});

    auto node = std::make_shared<TypeParam>(A, B);

    std::vector<StaticShape> static_input_shapes = {StaticShape{4, 6, 1}, StaticShape{3, 4, 1, 5}},
            static_output_shapes = {StaticShape{3, 4, 6, 5}};

    unit_test::cpu_test_shape_infer(node.get(), static_input_shapes, static_output_shapes);
}

TYPED_TEST_P(CpuShapeInferenceTest_BEA, shape_inference_autob_numpy_incompatible_shapes) {
    auto A = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1});
    auto B = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1});

    auto node = std::make_shared<TypeParam>(A, B);

    std::vector<StaticShape> static_input_shapes = {StaticShape{3, 4, 6, 5}, StaticShape{2, 4, 6, 5}},
            static_output_shapes = {StaticShape{}};

    OV_EXPECT_THROW(unit_test::cpu_test_shape_infer(node.get(), static_input_shapes, static_output_shapes),
                    ov::Exception,
                    testing::HasSubstr("Eltwise shape infer input shapes dim index:"));
}

TYPED_TEST_P(CpuShapeInferenceTest_BEA, shape_inference_aubtob_none) {
    auto A = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1});
    auto B = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1});

    auto node = std::make_shared<TypeParam>(A, B, op::AutoBroadcastType::NONE);

    std::vector<StaticShape> static_input_shapes = {StaticShape{3, 4, 6, 5}, StaticShape{3, 4, 6, 5}},
            static_output_shapes = {StaticShape{3, 4, 6, 5}};

    unit_test::cpu_test_shape_infer(node.get(), static_input_shapes, static_output_shapes);
}

TYPED_TEST_P(CpuShapeInferenceTest_BEA, shape_inference_aubtob_none_incompatible_shapes) {
    GTEST_SKIP() << "CVS-122351 Skipping test, eltwiseShapeInfer only implemented numpy type boardcast";
    auto A = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1});
    auto B = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1});

    auto node = std::make_shared<TypeParam>(A, B, op::AutoBroadcastType::NONE);

    std::vector<StaticShape> static_input_shapes = {StaticShape{3, 4, 6, 5}, StaticShape{3, 1, 6, 1}},
            static_output_shapes = {StaticShape{}};

    OV_EXPECT_THROW(unit_test::cpu_test_shape_infer(node.get(), static_input_shapes, static_output_shapes),
                    ov::Exception,
                    testing::HasSubstr("Eltwise shape infer input shapes dim index:"));
}

REGISTER_TYPED_TEST_SUITE_P(CpuShapeInferenceTest_BEA,
                            shape_inference_autob_numpy_equal_rank,
                            shape_inference_autob_numpy_a_rank_higher,
                            shape_inference_autob_numpy_b_rank_higher,
                            shape_inference_autob_numpy_incompatible_shapes,
                            shape_inference_aubtob_none,
                            shape_inference_aubtob_none_incompatible_shapes);

INSTANTIATE_TYPED_TEST_SUITE_P(CpuShapeInfer_add, CpuShapeInferenceTest_BEA, ::testing::Types<op::v1::Add>);
INSTANTIATE_TYPED_TEST_SUITE_P(CpuShapeInfer_divide, CpuShapeInferenceTest_BEA, ::testing::Types<op::v1::Divide>);
INSTANTIATE_TYPED_TEST_SUITE_P(CpuShapeInfer_floor_mod, CpuShapeInferenceTest_BEA, ::testing::Types<op::v1::FloorMod>);
INSTANTIATE_TYPED_TEST_SUITE_P(CpuShapeInfer_maximum, CpuShapeInferenceTest_BEA, ::testing::Types<op::v1::Maximum>);
INSTANTIATE_TYPED_TEST_SUITE_P(CpuShapeInfer_minimum, CpuShapeInferenceTest_BEA, ::testing::Types<op::v1::Minimum>);
INSTANTIATE_TYPED_TEST_SUITE_P(CpuShapeInfer_mod, CpuShapeInferenceTest_BEA, ::testing::Types<op::v1::Mod>);
INSTANTIATE_TYPED_TEST_SUITE_P(CpuShapeInfer_multiply, CpuShapeInferenceTest_BEA, ::testing::Types<op::v1::Multiply>);
INSTANTIATE_TYPED_TEST_SUITE_P(CpuShapeInfer_power, CpuShapeInferenceTest_BEA, ::testing::Types<op::v1::Power>);
INSTANTIATE_TYPED_TEST_SUITE_P(CpuShapeInfer_squared_difference, CpuShapeInferenceTest_BEA, ::testing::Types<op::v0::SquaredDifference>);
INSTANTIATE_TYPED_TEST_SUITE_P(CpuShapeInfer_subtract, CpuShapeInferenceTest_BEA, ::testing::Types<op::v1::Subtract>);

} // namespace cpu_shape_infer
} // namespace unit_test
} // namespace intel_cpu
} // namespace ov

