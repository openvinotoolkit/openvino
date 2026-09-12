// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plugin.hpp"

#include <gtest/gtest.h>

#include <memory>

#include "intel_npu/common/filtered_config.hpp"
#include "intel_npu/config/options.hpp"
#include "intel_npu/utils/logger/logger.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/runtime/intel_npu/properties.hpp"

using namespace intel_npu;

namespace {

std::shared_ptr<ov::Model> make_relu_model(const ov::PartialShape& shape) {
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, shape);
    auto relu = std::make_shared<ov::op::v0::Relu>(param);
    auto result = std::make_shared<ov::op::v0::Result>(relu);
    return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param}, "relu_model");
}

std::shared_ptr<ov::Model> make_dynamic_input_static_output_model(const ov::PartialShape& shape) {
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, shape);
    auto shapeOf = std::make_shared<ov::op::v3::ShapeOf>(param);
    auto result = std::make_shared<ov::op::v0::Result>(shapeOf);
    return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param}, "shape_of_model");
}

std::shared_ptr<ov::Model> make_two_input_relu_model(const ov::PartialShape& shape0,
                                                     const ov::PartialShape& shape1) {
    auto param0 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, shape0);
    auto relu0 = std::make_shared<ov::op::v0::Relu>(param0);
    auto result0 = std::make_shared<ov::op::v0::Result>(relu0);

    auto param1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, shape1);
    auto relu1 = std::make_shared<ov::op::v0::Relu>(param1);
    auto result1 = std::make_shared<ov::op::v0::Result>(relu1);

    return std::make_shared<ov::Model>(ov::ResultVector{result0, result1},
                                       ov::ParameterVector{param0, param1},
                                       "two_input_relu_model");
}

class EnableHostCompileTest : public ::testing::Test {
protected:
    EnableHostCompileTest() {
        auto options = std::make_shared<OptionsDesc>();
        options->add<COMPILER_TYPE>();
        options->add<COMPILATION_MODE>();
        options->add<DYNAMIC_SHAPE_TO_STATIC>();
        config = std::make_unique<FilteredConfig>(options);
        config->update({{ov::intel_npu::compiler_type.name(), "PLUGIN"}});
    }

    bool run(const std::shared_ptr<const ov::Model>& model) {
        intel_npu::enable_host_compile_if_needed(model, *config, Logger("EnableHostCompileTest", ov::log::Level::NO));
        return config->has<COMPILATION_MODE>() &&
               config->get<COMPILATION_MODE>() == "HostCompile_Interpreter";
    }

    std::unique_ptr<FilteredConfig> config;
};

constexpr int64_t UPPER_BOUND = 224;

ov::Dimension bounded() {
    return ov::Dimension(1, UPPER_BOUND);
}

ov::Dimension unbounded() {
    return ov::Dimension::dynamic();
}

TEST_F(EnableHostCompileTest, BoundedDynamicFourDimensionalInputAndOutputEnableHostCompile) {
    EXPECT_TRUE(run(make_relu_model({1, bounded(), 16, 32})));
}

TEST_F(EnableHostCompileTest, NonPluginCompilerDoesNotEnableHostCompile) {
    config->update({{ov::intel_npu::compiler_type.name(), "DRIVER"}});

    EXPECT_FALSE(run(make_relu_model({1, bounded(), 16, 32})));
}

TEST_F(EnableHostCompileTest, ExplicitCompilationModeIsNotOverridden) {
    config->update({{ov::intel_npu::compilation_mode.name(), "ReferenceSW"}});

    EXPECT_FALSE(run(make_relu_model({1, bounded(), 16, 32})));
    EXPECT_EQ(config->get<COMPILATION_MODE>(), "ReferenceSW");
}

TEST_F(EnableHostCompileTest, DynamicShapeToStaticDoesNotEnableHostCompile) {
    config->update({{ov::intel_npu::dynamic_shape_to_static.name(), "YES"}});

    EXPECT_FALSE(run(make_relu_model({1, bounded(), 16, 32})));
}

TEST_F(EnableHostCompileTest, StaticModelDoesNotEnableHostCompile) {
    EXPECT_FALSE(run(make_relu_model({1, 3, 16, 32})));
}

TEST_F(EnableHostCompileTest, DynamicRankDoesNotEnableHostCompile) {
    EXPECT_FALSE(run(make_relu_model(ov::PartialShape::dynamic())));
}

TEST_F(EnableHostCompileTest, NonFourDimensionalModelDoesNotEnableHostCompile) {
    EXPECT_FALSE(run(make_relu_model({1, bounded(), 16})));
}

TEST_F(EnableHostCompileTest, DynamicBatchDoesNotEnableHostCompile) {
    EXPECT_FALSE(run(make_relu_model({bounded(), 3, 16, 32})));
}

TEST_F(EnableHostCompileTest, UnboundedDimensionDoesNotEnableHostCompile) {
    EXPECT_FALSE(run(make_relu_model({1, unbounded(), 16, 32})));
}

TEST_F(EnableHostCompileTest, StaticOutputDoesNotEnableHostCompile) {
    EXPECT_FALSE(run(make_dynamic_input_static_output_model({1, bounded(), 16, 32})));
}

TEST_F(EnableHostCompileTest, UnboundedAdditionalPortDoesNotEnableHostCompile) {
    const auto model = make_two_input_relu_model({1, bounded(), 16, 32}, {1, unbounded(), 16, 32});

    EXPECT_FALSE(run(model));
}

}  // namespace
