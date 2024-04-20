// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>
#include <openvino/core/model.hpp>
#include <openvino/opsets/opset9.hpp>
#include <openvino/pass/manager.hpp>
#include <string>

#include "plugin/transformations/convert_reducemax_scalar_output.hpp"

using namespace testing;
using namespace ov::intel_gpu;

static std::shared_ptr<ov::Model> BuildFunction(const ov::PartialShape& input_shape,
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

TEST(TransformationTests, SplitReduceMaxTest1) {
    ov::pass::Manager manager;
    manager.set_per_pass_validation(false);
    manager.register_pass<ov::intel_gpu::ConvertReduceMaxScalarOutput>();

    const std::vector<size_t> reduction_axes = {0, 1, 2, 3};
    auto func = BuildFunction({1, 256, 1024, 10}, ov::element::Type_t::f16, reduction_axes, false);
    manager.run_passes(func);

    size_t reduce_count = 0;
    for (auto& ops : func->get_ops()) {
        std::string type_name(ops->get_type_name());
        if (type_name.find("ReduceMax") != std::string::npos) {
            reduce_count++;
        }
    }
    ASSERT_TRUE(reduce_count == 2);
}

TEST(TransformationTests, SplitReduceMaxTest2) {
    ov::pass::Manager manager;
    manager.set_per_pass_validation(false);
    manager.register_pass<ov::intel_gpu::ConvertReduceMaxScalarOutput>();

    const std::vector<size_t> reduction_axes = {0, 1, 2};
    auto func = BuildFunction({256, 1024, 10}, ov::element::Type_t::f16, reduction_axes, true);
    manager.run_passes(func);

    size_t reduce_count = 0;
    for (auto& ops : func->get_ops()) {
        std::string type_name(ops->get_type_name());
        if (type_name.find("ReduceMax") != std::string::npos) {
            reduce_count++;
        }
    }
    ASSERT_TRUE(reduce_count == 2);
}

TEST(TransformationTests, SplitReduceMaxTest3) {
    ov::pass::Manager manager;
    manager.set_per_pass_validation(false);
    manager.register_pass<ov::intel_gpu::ConvertReduceMaxScalarOutput>();

    const std::vector<size_t> reduction_axes = {1};
    auto func = BuildFunction({256, 1024, 10}, ov::element::Type_t::f16, reduction_axes, true);
    manager.run_passes(func);

    size_t reduce_count = 0;
    for (auto& ops : func->get_ops()) {
        std::string type_name(ops->get_type_name());
        if (type_name.find("ReduceMax") != std::string::npos) {
            reduce_count++;
        }
    }
    ASSERT_TRUE(reduce_count == reduction_axes.size());
}

TEST(TransformationTests, SplitReduceMaxTest4) {
    ov::pass::Manager manager;
    manager.set_per_pass_validation(false);
    manager.register_pass<ov::intel_gpu::ConvertReduceMaxScalarOutput>();

    const std::vector<size_t> reduction_axes = {0, 1, 2, 3};
    auto func = BuildFunction({4, -1, -1, 10}, ov::element::Type_t::f16, reduction_axes, false);
    manager.run_passes(func);

    size_t reduce_count = 0;
    for (auto& ops : func->get_ops()) {
        std::string type_name(ops->get_type_name());
        if (type_name.find("ReduceMax") != std::string::npos) {
            reduce_count++;
        }
    }
    ASSERT_TRUE(reduce_count == reduction_axes.size());
}

TEST(TransformationTests, SplitReduceMaxTest5) {
    ov::pass::Manager manager;
    manager.set_per_pass_validation(false);
    manager.register_pass<ov::intel_gpu::ConvertReduceMaxScalarOutput>();

    const std::vector<size_t> reduction_axes = {1};
    auto func = BuildFunction({256, -1, 10}, ov::element::Type_t::f16, reduction_axes, true);
    manager.run_passes(func);

    size_t reduce_count = 0;
    for (auto& ops : func->get_ops()) {
        std::string type_name(ops->get_type_name());
        if (type_name.find("ReduceMax") != std::string::npos) {
            reduce_count++;
        }
    }
    ASSERT_TRUE(reduce_count == reduction_axes.size());
}