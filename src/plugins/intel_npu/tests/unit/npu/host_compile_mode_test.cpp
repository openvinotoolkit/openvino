// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "host_compile_mode.hpp"

#include <gtest/gtest.h>

#include <memory>

#include "intel_npu/common/filtered_config.hpp"
#include "intel_npu/config/options.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/runtime/intel_npu/properties.hpp"

using namespace intel_npu;

namespace {

// Shape-preserving model: single 4D input feeds a Relu, so the output partial shape matches the input.
std::shared_ptr<ov::Model> make_relu_model(const ov::PartialShape& shape) {
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, shape);
    auto relu = std::make_shared<ov::op::v0::Relu>(param);
    auto result = std::make_shared<ov::op::v0::Result>(relu);
    return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param}, "relu_model");
}

// Dynamic 4D input but a static (1D, bounded) output produced by ShapeOf.
std::shared_ptr<ov::Model> make_dynamic_input_static_output_model(const ov::PartialShape& shape) {
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, shape);
    auto shape_of = std::make_shared<ov::op::v3::ShapeOf>(param);
    auto result = std::make_shared<ov::op::v0::Result>(shape_of);
    return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param}, "shape_of_model");
}

class EnableHostCompileTest : public ::testing::Test {
protected:
    EnableHostCompileTest() {
        auto desc = std::make_shared<OptionsDesc>();
        desc->add<COMPILER_TYPE>();
        desc->add<COMPILATION_MODE>();
        desc->add<DYNAMIC_SHAPE_TO_STATIC>();
        config = std::make_unique<FilteredConfig>(desc);
        config->enableAll();
        config->update({{ov::intel_npu::compiler_type.name(), "PLUGIN"}});
    }

    bool run(const std::shared_ptr<const ov::Model>& model) {
        return enable_host_compile_if_needed(model, *config);
    }

    std::unique_ptr<FilteredConfig> config;
};

constexpr int64_t UPPER = 224;

// Fully bounded dynamic dimension with a finite upper bound.
ov::Dimension bounded() {
    return ov::Dimension(1, UPPER);
}

// Fully dynamic dimension without an upper bound.
ov::Dimension unbounded() {
    return ov::Dimension::dynamic();
}

}  // namespace

// Dynamic spatial dimensions (H/W) with a static batch must enable HostCompile.
TEST_F(EnableHostCompileTest, DynamicSpatialEnablesHostCompile) {
    auto model = make_relu_model({1, 3, bounded(), bounded()});
    EXPECT_TRUE(run(model));
    EXPECT_TRUE(config->has<COMPILATION_MODE>());
    EXPECT_EQ(config->get<COMPILATION_MODE>(), "HostCompile_Interpreter");
}

// Dynamic batch is excluded from automatic HostCompile because the compiler's ConvertBatchedLayerTo1N and
// AdjustScaleShiftForDWConv passes do not support dynamic reshape; such models use the regular batch handling path.
TEST_F(EnableHostCompileTest, DynamicBatchDoesNotEnableHostCompile) {
    auto model = make_relu_model({bounded(), 3, UPPER, UPPER});
    EXPECT_FALSE(run(model));
    EXPECT_FALSE(config->has<COMPILATION_MODE>());
}

// A dynamic batch combined with dynamic spatial dimensions is still excluded due to the dynamic batch dimension.
TEST_F(EnableHostCompileTest, DynamicBatchAndSpatialDoesNotEnableHostCompile) {
    auto model = make_relu_model({bounded(), 3, bounded(), bounded()});
    EXPECT_FALSE(run(model));
    EXPECT_FALSE(config->has<COMPILATION_MODE>());
}

// A fully static model is not a HostCompile candidate.
TEST_F(EnableHostCompileTest, StaticModelDoesNotEnable) {
    auto model = make_relu_model({1, 3, UPPER, UPPER});
    EXPECT_FALSE(run(model));
    EXPECT_FALSE(config->has<COMPILATION_MODE>());
}

// A dynamic dimension without a finite upper bound blocks HostCompile buffer allocation.
TEST_F(EnableHostCompileTest, UnboundedDimensionDoesNotEnable) {
    auto model = make_relu_model({1, 3, unbounded(), UPPER});
    EXPECT_FALSE(run(model));
    EXPECT_FALSE(config->has<COMPILATION_MODE>());
}

// Only 4D static-rank ports qualify; a dynamic 2D model must not enable HostCompile.
TEST_F(EnableHostCompileTest, NonFourDimensionalDoesNotEnable) {
    auto model = make_relu_model({bounded(), bounded()});
    EXPECT_FALSE(run(model));
    EXPECT_FALSE(config->has<COMPILATION_MODE>());
}

// A dynamic rank is never a HostCompile candidate.
TEST_F(EnableHostCompileTest, DynamicRankDoesNotEnable) {
    auto model = make_relu_model(ov::PartialShape::dynamic());
    EXPECT_FALSE(run(model));
    EXPECT_FALSE(config->has<COMPILATION_MODE>());
}

// Both inputs and outputs must be dynamic; a dynamic input with a static output must not enable HostCompile.
TEST_F(EnableHostCompileTest, DynamicInputStaticOutputDoesNotEnable) {
    auto model = make_dynamic_input_static_output_model({1, 3, bounded(), bounded()});
    EXPECT_FALSE(run(model));
    EXPECT_FALSE(config->has<COMPILATION_MODE>());
}

// Automatic selection applies only to the Plugin compiler.
TEST_F(EnableHostCompileTest, NonPluginCompilerDoesNotEnable) {
    config->update({{ov::intel_npu::compiler_type.name(), "DRIVER"}});
    auto model = make_relu_model({1, 3, bounded(), bounded()});
    EXPECT_FALSE(run(model));
    EXPECT_FALSE(config->has<COMPILATION_MODE>());
}

// An explicit compilation mode is respected and never overridden.
TEST_F(EnableHostCompileTest, ExplicitCompilationModeIsRespected) {
    config->update({{ov::intel_npu::compilation_mode.name(), "ReferenceSW"}});
    auto model = make_relu_model({1, 3, bounded(), bounded()});
    EXPECT_FALSE(run(model));
    EXPECT_EQ(config->get<COMPILATION_MODE>(), "ReferenceSW");
}

// Dynamic-to-static conversion disables automatic HostCompile selection.
TEST_F(EnableHostCompileTest, DynamicShapeToStaticDisablesSelection) {
    config->update({{ov::intel_npu::dynamic_shape_to_static.name(), "YES"}});
    auto model = make_relu_model({1, 3, bounded(), bounded()});
    EXPECT_FALSE(run(model));
    EXPECT_FALSE(config->has<COMPILATION_MODE>());
}

// A null model must be handled gracefully.
TEST_F(EnableHostCompileTest, NullModelDoesNotEnable) {
    EXPECT_FALSE(run(nullptr));
    EXPECT_FALSE(config->has<COMPILATION_MODE>());
}

class UsesHostCompileDynamicGraphTest : public ::testing::Test {
protected:
    UsesHostCompileDynamicGraphTest() {
        auto desc = std::make_shared<OptionsDesc>();
        desc->add<COMPILER_TYPE>();
        desc->add<COMPILATION_MODE>();
        config = std::make_unique<FilteredConfig>(desc);
        config->enableAll();
        config->update({{ov::intel_npu::compiler_type.name(), "PLUGIN"}});
    }

    void setCompilationMode(const std::string& mode) {
        config->update({{ov::intel_npu::compilation_mode.name(), mode}});
    }

    bool run(const std::shared_ptr<const ov::Model>& model) {
        return uses_host_compile_dynamic_graph(model, *config);
    }

    std::unique_ptr<FilteredConfig> config;
};

// A dynamic model compiled by the Plugin compiler with an automatically selected HostCompile mode uses the dynamic
// graph path.
TEST_F(UsesHostCompileDynamicGraphTest, DynamicPluginHostCompileInterpreterUsesDynamicGraph) {
    setCompilationMode("HostCompile_Interpreter");
    EXPECT_TRUE(run(make_relu_model({1, 3, bounded(), bounded()})));
}

// Any compilation mode starting with "HostCompile" selects the dynamic graph path.
TEST_F(UsesHostCompileDynamicGraphTest, DynamicPluginHostCompilePrefixUsesDynamicGraph) {
    setCompilationMode("HostCompile");
    EXPECT_TRUE(run(make_relu_model({1, 3, bounded(), bounded()})));
}

// A static model never uses the dynamic graph path.
TEST_F(UsesHostCompileDynamicGraphTest, StaticModelDoesNotUseDynamicGraph) {
    setCompilationMode("HostCompile_Interpreter");
    EXPECT_FALSE(run(make_relu_model({1, 3, UPPER, UPPER})));
}

// Non-Plugin compilers never use the HostCompile dynamic graph path.
TEST_F(UsesHostCompileDynamicGraphTest, NonPluginCompilerDoesNotUseDynamicGraph) {
    config->update({{ov::intel_npu::compiler_type.name(), "DRIVER"}});
    setCompilationMode("HostCompile_Interpreter");
    EXPECT_FALSE(run(make_relu_model({1, 3, bounded(), bounded()})));
}

// A non-HostCompile mode does not select the dynamic graph path.
TEST_F(UsesHostCompileDynamicGraphTest, NonHostCompileModeDoesNotUseDynamicGraph) {
    setCompilationMode("ReferenceSW");
    EXPECT_FALSE(run(make_relu_model({1, 3, bounded(), bounded()})));
}

// An empty (unset) compilation mode does not start with "HostCompile".
TEST_F(UsesHostCompileDynamicGraphTest, EmptyModeDoesNotUseDynamicGraph) {
    EXPECT_FALSE(run(make_relu_model({1, 3, bounded(), bounded()})));
}

// A null model must be handled gracefully.
TEST_F(UsesHostCompileDynamicGraphTest, NullModelDoesNotUseDynamicGraph) {
    setCompilationMode("HostCompile_Interpreter");
    EXPECT_FALSE(run(nullptr));
}
