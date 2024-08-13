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

using namespace testing;
using namespace ov::intel_gpu;

static std::shared_ptr<ov::Model> build_model(const ov::PartialShape& input_shape,
                                              const ov::element::Type& input_type,
                                              const std::vector<size_t>& reduction_axes,
                                              const bool keep_dim) {
    const auto in = std::make_shared<ov::op::v0::Parameter>(input_type, input_shape);
    const auto reduce = std::make_shared<ov::op::v1::ReduceMax>(
        in->get_default_output(),
        ov::op::v0::Constant::create(ov::element::i64, ov::Shape{reduction_axes.size()}, reduction_axes),
        keep_dim);

    return std::make_shared<ov::Model>(ov::NodeVector{reduce}, ov::ParameterVector{in});
}

// Static shape reduce to scalar output, decompose reduce.
TEST_F(TransformationTestsF, DecomposeReduceMaxTest_static_shape_1) {
    const ov::PartialShape in_shape = {1, 256, 1024, 10};
    const ov::element::Type in_type = ov::element::Type_t::f16;
    const std::vector<size_t> reduction_axes = {0, 1, 2, 3};
    disable_rt_info_check();
    {
        model = build_model(in_shape, in_type, reduction_axes, false);
        manager.register_pass<ov::intel_gpu::DecomposeReduceForScalarOutput>();
    }
    {
        const auto in = std::make_shared<ov::op::v0::Parameter>(in_type, in_shape);
        const auto reduce_1 =
            std::make_shared<ov::op::v1::ReduceMax>(in->get_default_output(),
                                                    ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {2}),
                                                    true);
        const auto reduce_2 = std::make_shared<ov::op::v1::ReduceMax>(
            reduce_1->get_default_output(),
            ov::op::v0::Constant::create(ov::element::i64, ov::Shape{reduction_axes.size()}, reduction_axes),
            false);

        model_ref = std::make_shared<ov::Model>(ov::NodeVector{reduce_2}, ov::ParameterVector{in});
    }
}

// Static shape reduce to non scalar output, don't decompose reduce.
TEST_F(TransformationTestsF, DecomposeReduceMaxTest_static_shape_2) {
    const ov::PartialShape in_shape = {256, 1024, 10};
    const ov::element::Type in_type = ov::element::Type_t::f16;
    const std::vector<size_t> reduction_axes = {1};
    {
        model = build_model(in_shape, in_type, reduction_axes, true);
        manager.register_pass<ov::intel_gpu::DecomposeReduceForScalarOutput>();
    }
    { model_ref = build_model(in_shape, in_type, reduction_axes, true); }
}

// Dynamic shape reduce to scalar output, decompose reduce.
TEST_F(TransformationTestsF, DecomposeReduceMaxTest_dynamic_shape_1) {
    const ov::PartialShape in_shape = {4, -1, -1, 10};
    const ov::element::Type in_type = ov::element::Type_t::f16;
    const std::vector<size_t> reduction_axes = {0, 1, 2, 3};
    disable_rt_info_check();
    {
        model = build_model(in_shape, in_type, reduction_axes, false);
        manager.register_pass<ov::intel_gpu::DecomposeReduceForScalarOutput>();
    }
    {
        const auto in = std::make_shared<ov::op::v0::Parameter>(in_type, in_shape);
        const auto reduce_1 =
            std::make_shared<ov::op::v1::ReduceMax>(in,
                                                    ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {3}),
                                                    true);
        const auto reduce_2 =
            std::make_shared<ov::op::v1::ReduceMax>(reduce_1->get_default_output(),
                                                    ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {2}),
                                                    true);
        const auto reduce_3 =
            std::make_shared<ov::op::v1::ReduceMax>(reduce_2->get_default_output(),
                                                    ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {1}),
                                                    true);
        const auto reduce = std::make_shared<ov::op::v1::ReduceMax>(
            reduce_3->get_default_output(),
            ov::op::v0::Constant::create(ov::element::i64, ov::Shape{reduction_axes.size()}, reduction_axes),
            false);
        model_ref = std::make_shared<ov::Model>(ov::NodeVector{reduce}, ov::ParameterVector{in});
    }
}

// Dynamic shape reduce to non-scalar output, don't decompose reduce.
TEST_F(TransformationTestsF, DecomposeReduceMaxTest_dynamic_shape_2) {
    const ov::PartialShape in_shape = {4, -1, -1, 10};
    const ov::element::Type in_type = ov::element::Type_t::f16;
    const std::vector<size_t> reduction_axes = {2};
    {
        model = build_model(in_shape, in_type, reduction_axes, false);
        manager.register_pass<ov::intel_gpu::DecomposeReduceForScalarOutput>();
    }
    { model_ref = build_model(in_shape, in_type, reduction_axes, false); }
}