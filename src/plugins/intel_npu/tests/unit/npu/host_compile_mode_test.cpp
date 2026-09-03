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

// Two independent 4D input/output pairs; used to exercise the "every I/O port must be bounded" aggregate check.
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

// A dynamic batch alone, with H and W static, is the only N/H/W combination that does not enable HostCompile.
TEST_F(EnableHostCompileTest, OnlyBatchDynamicDoesNotEnableHostCompile) {
    auto model = make_relu_model({bounded(), 3, UPPER, UPPER});
    EXPECT_FALSE(run(model));
    EXPECT_FALSE(config->has<COMPILATION_MODE>());
}

// A static batch with only H dynamic (W static) is enough on its own - only H or W dynamic is required.
TEST_F(EnableHostCompileTest, OnlyHeightDynamicEnablesHostCompile) {
    auto model = make_relu_model({1, 3, bounded(), UPPER});
    EXPECT_TRUE(run(model));
    EXPECT_TRUE(config->has<COMPILATION_MODE>());
    EXPECT_EQ(config->get<COMPILATION_MODE>(), "HostCompile_Interpreter");
}

// Same as above, but only W is dynamic (H static).
TEST_F(EnableHostCompileTest, OnlyWidthDynamicEnablesHostCompile) {
    auto model = make_relu_model({1, 3, UPPER, bounded()});
    EXPECT_TRUE(run(model));
    EXPECT_TRUE(config->has<COMPILATION_MODE>());
    EXPECT_EQ(config->get<COMPILATION_MODE>(), "HostCompile_Interpreter");
}

// A dynamic batch is accepted as long as H and W are also dynamic (the "NHW dynamic" pattern).
TEST_F(EnableHostCompileTest, DynamicBatchAndSpatialEnablesHostCompile) {
    auto model = make_relu_model({bounded(), 3, bounded(), bounded()});
    EXPECT_TRUE(run(model));
    EXPECT_TRUE(config->has<COMPILATION_MODE>());
    EXPECT_EQ(config->get<COMPILATION_MODE>(), "HostCompile_Interpreter");
}

// Batch and height both dynamic (width static) is the "NH" pattern - accepted because H alone is dynamic.
TEST_F(EnableHostCompileTest, DynamicBatchAndHeightEnablesHostCompile) {
    auto model = make_relu_model({bounded(), 3, bounded(), UPPER});
    EXPECT_TRUE(run(model));
    EXPECT_TRUE(config->has<COMPILATION_MODE>());
    EXPECT_EQ(config->get<COMPILATION_MODE>(), "HostCompile_Interpreter");
}

// Batch and width both dynamic (height static) is the "NW" pattern - also accepted.
TEST_F(EnableHostCompileTest, DynamicBatchAndWidthEnablesHostCompile) {
    auto model = make_relu_model({bounded(), 3, UPPER, bounded()});
    EXPECT_TRUE(run(model));
    EXPECT_TRUE(config->has<COMPILATION_MODE>());
    EXPECT_EQ(config->get<COMPILATION_MODE>(), "HostCompile_Interpreter");
}

// H/W being static means neither the "HW" nor the "NHW" pattern applies, regardless of channel (C) dynamism.
TEST_F(EnableHostCompileTest, DynamicChannelWithStaticSpatialDoesNotEnableHostCompile) {
    auto model = make_relu_model({1, bounded(), UPPER, UPPER});
    EXPECT_FALSE(run(model));
    EXPECT_FALSE(config->has<COMPILATION_MODE>());
}

// Channel (C) dynamism is ignored: HostCompile is still selected when H and W are dynamic even if C is dynamic too.
TEST_F(EnableHostCompileTest, DynamicChannelWithDynamicSpatialEnablesHostCompile) {
    auto model = make_relu_model({1, bounded(), bounded(), bounded()});
    EXPECT_TRUE(run(model));
    EXPECT_TRUE(config->has<COMPILATION_MODE>());
    EXPECT_EQ(config->get<COMPILATION_MODE>(), "HostCompile_Interpreter");
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

// One port matching the HW pattern is not enough: every I/O port must independently be bounded, so an unrelated
// port with an unbounded dimension still blocks HostCompile.
TEST_F(EnableHostCompileTest, UnboundedUnrelatedPortDoesNotEnable) {
    auto model = make_two_input_relu_model({1, 3, bounded(), bounded()}, {1, 3, unbounded(), UPPER});
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
// Use an uncommon compilation mode to check for overrides.
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
