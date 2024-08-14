// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>
#include <openvino/core/model.hpp>
#include <openvino/opsets/opset9.hpp>
#include <openvino/pass/manager.hpp>
#include <plugin/transformations/decompose_reduce_scalar_output.hpp>
#include <string>

#include "common_test_utils/ov_test_utils.hpp"
#include "intel_gpu/primitives/reduce.hpp"

using namespace testing;
using namespace ov::intel_gpu;
using namespace cldnn;
using ReduceType = cldnn::reduce_mode;

#define create_reduce(arg, reduction, keep_dims, reduce_type)                         \
    if (reduce_type == reduce_mode::sum)                                              \
        reduce = std::make_shared<ov::op::v1::ReduceSum>(arg, reduction, keep_dims);  \
    else if (reduce_type == reduce_mode::min)                                         \
        reduce = std::make_shared<ov::op::v1::ReduceMin>(arg, reduction, keep_dims);  \
    else if (reduce_type == reduce_mode::max)                                         \
        reduce = std::make_shared<ov::op::v1::ReduceMax>(arg, reduction, keep_dims);  \
    else if (reduce_type == reduce_mode::prod)                                        \
        reduce = std::make_shared<ov::op::v1::ReduceProd>(arg, reduction, keep_dims); \
    OPENVINO_ASSERT(reduce != nullptr, "cannot create reduce: ", static_cast<int>(reduce_type));

static std::shared_ptr<ov::Model> build_model(const ov::PartialShape& input_shape,
                                              const ov::element::Type& input_type,
                                              const std::vector<size_t>& reduction_axes,
                                              const bool keep_dim,
                                              const ReduceType reduce_type) {
    const auto in = std::make_shared<ov::op::v0::Parameter>(input_type, input_shape);
    std::shared_ptr<ov::op::util::ArithmeticReductionKeepDims> reduce = nullptr;
    create_reduce(in->get_default_output(),
                  ov::op::v0::Constant::create(ov::element::i64, ov::Shape{reduction_axes.size()}, reduction_axes),
                  keep_dim,
                  reduce_type);

    return std::make_shared<ov::Model>(ov::NodeVector{reduce}, ov::ParameterVector{in});
}

#define decompose_reduce_static_shape(reduce_type)                                                            \
    const ov::PartialShape in_shape = {1, 256, 1024, 10};                                                     \
    const ov::element::Type in_type = ov::element::Type_t::f16;                                               \
    const std::vector<size_t> reduction_axes = {0, 1, 2, 3};                                                  \
    disable_rt_info_check();                                                                                  \
    {                                                                                                         \
        model = build_model(in_shape, in_type, reduction_axes, false, reduce_type);                           \
        manager.register_pass<ov::intel_gpu::DecomposeReduceForScalarOutput>();                               \
    }                                                                                                         \
    {                                                                                                         \
        const auto in = std::make_shared<ov::op::v0::Parameter>(in_type, in_shape);                           \
        std::shared_ptr<ov::op::util::ArithmeticReductionKeepDims> reduce = nullptr;                          \
        create_reduce(in->get_default_output(),                                                               \
                      ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {2}),                      \
                      true,                                                                                   \
                      reduce_type);                                                                           \
        create_reduce(                                                                                        \
            reduce->get_default_output(),                                                                     \
            ov::op::v0::Constant::create(ov::element::i64, ov::Shape{reduction_axes.size()}, reduction_axes), \
            false,                                                                                            \
            reduce_mode::max);                                                                                \
        model_ref = std::make_shared<ov::Model>(ov::NodeVector{reduce}, ov::ParameterVector{in});             \
    }

// Static shape reduce to scalar output, decompose reduce.
TEST_F(TransformationTestsF, DecomposeReduceMaxTest_static_shape){decompose_reduce_static_shape(reduce_mode::max)}

TEST_F(TransformationTestsF, DecomposeReduceMinTest_static_shape){decompose_reduce_static_shape(reduce_mode::min)}

TEST_F(TransformationTestsF, DecomposeReduceSumTest_static_shape){decompose_reduce_static_shape(reduce_mode::sum)}

TEST_F(TransformationTestsF, DecomposeReduceProbTest_static_shape){decompose_reduce_static_shape(reduce_mode::prod)}

// Static shape reduce to non scalar output, don't decompose reduce.
TEST_F(TransformationTestsF, DecomposeReduceMaxTest_static_shape_skip) {
    const ov::PartialShape in_shape = {256, 1024, 10};
    const ov::element::Type in_type = ov::element::Type_t::f16;
    const std::vector<size_t> reduction_axes = {1};
    {
        model = build_model(in_shape, in_type, reduction_axes, true, reduce_mode::max);
        manager.register_pass<ov::intel_gpu::DecomposeReduceForScalarOutput>();
    }
    { model_ref = build_model(in_shape, in_type, reduction_axes, true, reduce_mode::max); }
}

// Dynamic shape reduce to scalar output, decompose reduce.
#define decompose_reduce_dynamic_shape(reduce_type)                                                           \
    const ov::PartialShape in_shape = {4, -1, -1, 10};                                                        \
    const ov::element::Type in_type = ov::element::Type_t::f16;                                               \
    const std::vector<size_t> reduction_axes = {0, 1, 2, 3};                                                  \
    disable_rt_info_check();                                                                                  \
    {                                                                                                         \
        model = build_model(in_shape, in_type, reduction_axes, false, reduce_mode::max);                      \
        manager.register_pass<ov::intel_gpu::DecomposeReduceForScalarOutput>();                               \
    }                                                                                                         \
    {                                                                                                         \
        const auto in = std::make_shared<ov::op::v0::Parameter>(in_type, in_shape);                           \
        std::shared_ptr<ov::op::util::ArithmeticReductionKeepDims> reduce = nullptr;                          \
        create_reduce(in->get_default_output(),                                                               \
                      ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {3}),                      \
                      true,                                                                                   \
                      reduce_type);                                                                           \
        create_reduce(reduce->get_default_output(),                                                           \
                      ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {2}),                      \
                      true,                                                                                   \
                      reduce_type);                                                                           \
        create_reduce(reduce->get_default_output(),                                                           \
                      ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {1}),                      \
                      true,                                                                                   \
                      reduce_type);                                                                           \
        create_reduce(                                                                                        \
            reduce->get_default_output(),                                                                     \
            ov::op::v0::Constant::create(ov::element::i64, ov::Shape{reduction_axes.size()}, reduction_axes), \
            false,                                                                                            \
            reduce_type);                                                                                     \
        model_ref = std::make_shared<ov::Model>(ov::NodeVector{reduce}, ov::ParameterVector{in});             \
    }

TEST_F(TransformationTestsF, DecomposeReduceMaxTest_dynamic_shape){decompose_reduce_dynamic_shape(reduce_mode::max)}

TEST_F(TransformationTestsF, DecomposeReduceMinTest_dynamic_shape){decompose_reduce_dynamic_shape(reduce_mode::min)}

TEST_F(TransformationTestsF, DecomposeReduceSumTest_dynamic_shape){decompose_reduce_dynamic_shape(reduce_mode::sum)}

TEST_F(TransformationTestsF, DecomposeReduceProbTest_dynamic_shape){decompose_reduce_dynamic_shape(reduce_mode::prod)}

// Dynamic shape reduce to non-scalar output, don't decompose reduce.
TEST_F(TransformationTestsF, DecomposeReduceMaxTest_dynamic_shape_skip) {
    const ov::PartialShape in_shape = {4, -1, -1, 10};
    const ov::element::Type in_type = ov::element::Type_t::f16;
    const std::vector<size_t> reduction_axes = {2};
    {
        model = build_model(in_shape, in_type, reduction_axes, false, reduce_mode::max);
        manager.register_pass<ov::intel_gpu::DecomposeReduceForScalarOutput>();
    }
    { model_ref = build_model(in_shape, in_type, reduction_axes, false, reduce_mode::max); }
}