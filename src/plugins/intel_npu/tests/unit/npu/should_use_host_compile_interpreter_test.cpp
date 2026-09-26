// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>

#include "intel_npu/config/config.hpp"
#include "intel_npu/config/options.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/runtime/intel_npu/properties.hpp"
#include "transformations.hpp"

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

std::shared_ptr<ov::Model> make_two_input_relu_model(const ov::PartialShape& shape0, const ov::PartialShape& shape1) {
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

// Slice with a runtime (Parameter) stop keeps the static input while producing a bounded dynamic 4D output.
std::shared_ptr<ov::Model> make_static_input_dynamic_output_model(const ov::PartialShape& shape) {
    auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, shape);
    auto start = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{4}, {0, 0, 0, 0});
    auto step = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{4}, {1, 1, 1, 1});
    auto stop = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{4});
    auto slice = std::make_shared<ov::op::v8::Slice>(data, start, stop, step);
    auto result = std::make_shared<ov::op::v0::Result>(slice);
    return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{data, stop}, "slice_model");
}

std::shared_ptr<ov::Model> make_no_input_model() {
    auto constant = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1, 3, 16, 32}, {0.0f});
    auto result = std::make_shared<ov::op::v0::Result>(constant);
    return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{}, "no_input_model");
}

class ShouldUseHostCompileInterpreterTest : public ::testing::Test {
protected:
    ShouldUseHostCompileInterpreterTest() {
        auto desc = std::make_shared<OptionsDesc>();
        desc->add<COMPILER_TYPE>();
        desc->add<COMPILATION_MODE>();
        desc->add<DYNAMIC_SHAPE_TO_STATIC>();
        config = std::make_unique<Config>(desc);
        config->update({{ov::intel_npu::compiler_type.name(), "PLUGIN"}});
    }

    bool run(const std::shared_ptr<const ov::Model>& model) {
        return intel_npu::should_use_host_compile_interpreter(model,
                                                              config->get<COMPILER_TYPE>(),
                                                              config->has<COMPILATION_MODE>(),
                                                              config->get<DYNAMIC_SHAPE_TO_STATIC>());
    }

    std::unique_ptr<Config> config;
};

constexpr int64_t UPPER_BOUND = 224;

ov::Dimension bounded() {
    return ov::Dimension(1, UPPER_BOUND);
}

ov::Dimension unbounded() {
    return ov::Dimension::dynamic();
}

TEST_F(ShouldUseHostCompileInterpreterTest, BoundedDynamicFourDimensionalInputAndOutputEnableHostCompile) {
    EXPECT_TRUE(run(make_relu_model({1, bounded(), 16, 32})));
}

TEST_F(ShouldUseHostCompileInterpreterTest, DynamicSpatialDimensionsEnableHostCompile) {
    EXPECT_TRUE(run(make_relu_model({1, 3, bounded(), bounded()})));
}

TEST_F(ShouldUseHostCompileInterpreterTest, NonPluginCompilerDoesNotEnableHostCompile) {
    config->update({{ov::intel_npu::compiler_type.name(), "DRIVER"}});

    EXPECT_FALSE(run(make_relu_model({1, bounded(), 16, 32})));
}

TEST_F(ShouldUseHostCompileInterpreterTest, ExplicitCompilationModeIsNotOverridden) {
    config->update({{ov::intel_npu::compilation_mode.name(), "ReferenceSW"}});

    EXPECT_FALSE(run(make_relu_model({1, bounded(), 16, 32})));
    EXPECT_EQ(config->get<COMPILATION_MODE>(), "ReferenceSW");
}

TEST_F(ShouldUseHostCompileInterpreterTest, DynamicShapeToStaticDoesNotEnableHostCompile) {
    config->update({{ov::intel_npu::dynamic_shape_to_static.name(), "YES"}});

    EXPECT_FALSE(run(make_relu_model({1, bounded(), 16, 32})));
}

TEST_F(ShouldUseHostCompileInterpreterTest, StaticModelDoesNotEnableHostCompile) {
    EXPECT_FALSE(run(make_relu_model({1, 3, 16, 32})));
}

TEST_F(ShouldUseHostCompileInterpreterTest, DynamicRankDoesNotEnableHostCompile) {
    EXPECT_FALSE(run(make_relu_model(ov::PartialShape::dynamic())));
}

TEST_F(ShouldUseHostCompileInterpreterTest, NonFourDimensionalModelDoesNotEnableHostCompile) {
    EXPECT_FALSE(run(make_relu_model({1, bounded(), 16})));
}

TEST_F(ShouldUseHostCompileInterpreterTest, DynamicBatchDoesNotEnableHostCompile) {
    EXPECT_FALSE(run(make_relu_model({bounded(), 3, 16, 32})));
}

TEST_F(ShouldUseHostCompileInterpreterTest, DynamicBatchWithDynamicSpatialDoesNotEnableHostCompile) {
    EXPECT_FALSE(run(make_relu_model({bounded(), 3, bounded(), 32})));
}

TEST_F(ShouldUseHostCompileInterpreterTest, UnboundedDimensionDoesNotEnableHostCompile) {
    EXPECT_FALSE(run(make_relu_model({1, unbounded(), 16, 32})));
}

TEST_F(ShouldUseHostCompileInterpreterTest, StaticOutputDoesNotEnableHostCompile) {
    EXPECT_FALSE(run(make_dynamic_input_static_output_model({1, bounded(), 16, 32})));
}

TEST_F(ShouldUseHostCompileInterpreterTest, StaticInputDynamicOutputDoesNotEnableHostCompile) {
    EXPECT_FALSE(run(make_static_input_dynamic_output_model({1, 3, 16, 32})));
}

TEST_F(ShouldUseHostCompileInterpreterTest, UnboundedAdditionalPortDoesNotEnableHostCompile) {
    const auto model = make_two_input_relu_model({1, bounded(), 16, 32}, {1, unbounded(), 16, 32});

    EXPECT_FALSE(run(model));
}

TEST_F(ShouldUseHostCompileInterpreterTest, MultipleDynamicOutputsEnableHostCompile) {
    EXPECT_TRUE(run(make_two_input_relu_model({1, bounded(), 16, 32}, {1, 3, bounded(), bounded()})));
}

TEST_F(ShouldUseHostCompileInterpreterTest, NoInputModelDoesNotEnableHostCompile) {
    EXPECT_FALSE(run(make_no_input_model()));
}

}  // namespace
