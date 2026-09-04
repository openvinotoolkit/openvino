// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <optional>
#include <string>
#include <string_view>

#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "plugin.hpp"

namespace {

using namespace intel_npu;

constexpr std::string_view HOST_COMPILE_MODE = "HostCompile_Interpreter";

FilteredConfig make_config(ov::intel_npu::CompilerType compilerType = ov::intel_npu::CompilerType::PLUGIN,
                           bool dynamicShapeToStatic = false,
                           const std::string& compilationMode = {}) {
    auto options = std::make_shared<OptionsDesc>();
    options->add<COMPILER_TYPE>();
    options->add<COMPILATION_MODE>();
    options->add<DYNAMIC_SHAPE_TO_STATIC>();

    FilteredConfig config(options);
    config.enableAll();
    config.update({{ov::intel_npu::compiler_type.name(), COMPILER_TYPE::toString(compilerType)},
                   {ov::intel_npu::dynamic_shape_to_static.name(), dynamicShapeToStatic ? "YES" : "NO"}});
    if (!compilationMode.empty()) {
        config.update({{ov::intel_npu::compilation_mode.name(), compilationMode}});
    }
    return config;
}

std::shared_ptr<ov::Model> make_model(const ov::PartialShape& inputShape,
                                      bool returnInput = true,
                                      const std::optional<ov::PartialShape>& additionalInputShape = std::nullopt) {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, inputShape);
    ov::ParameterVector inputs{input};
    if (additionalInputShape.has_value()) {
        inputs.push_back(std::make_shared<ov::op::v0::Parameter>(ov::element::f32, *additionalInputShape));
    }

    if (returnInput) {
        return std::make_shared<ov::Model>(ov::OutputVector{input}, inputs);
    }

    auto output = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1}, {0.0f});
    return std::make_shared<ov::Model>(ov::OutputVector{output}, inputs);
}

bool host_compile_enabled(const FilteredConfig& config) {
    return config.get<COMPILATION_MODE>() == HOST_COMPILE_MODE;
}

TEST(EnableHostCompileTest, EnablesForBoundedDynamicFourDimensionalInputAndOutputWithStaticBatch) {
    auto config = make_config();
    const auto model = make_model({1, ov::Dimension(1, 8), ov::Dimension(2, 16), 32});

    enable_host_compile_if_needed(model, config);

    EXPECT_TRUE(host_compile_enabled(config));
}

struct ConfigShortCircuitParams {
    ov::intel_npu::CompilerType compilerType;
    bool dynamicShapeToStatic;
    std::string compilationMode;
};

class EnableHostCompileConfigShortCircuitTest : public testing::TestWithParam<ConfigShortCircuitParams> {};

TEST_P(EnableHostCompileConfigShortCircuitTest, DoesNotOverrideConfiguration) {
    const auto& params = GetParam();
    auto config = make_config(params.compilerType, params.dynamicShapeToStatic, params.compilationMode);
    const auto model = make_model({1, ov::Dimension(1, 8), ov::Dimension(2, 16), 32});

    enable_host_compile_if_needed(model, config);

    EXPECT_EQ(config.get<COMPILATION_MODE>(), params.compilationMode);
}

INSTANTIATE_TEST_SUITE_P(
    AllConfigConditions,
    EnableHostCompileConfigShortCircuitTest,
    testing::Values(ConfigShortCircuitParams{ov::intel_npu::CompilerType::DRIVER, false, ""},
                    ConfigShortCircuitParams{ov::intel_npu::CompilerType::PLUGIN, true, ""},
                    ConfigShortCircuitParams{ov::intel_npu::CompilerType::PLUGIN, false, "ReferenceSW"}));

class EnableHostCompileInvalidShapeTest : public testing::TestWithParam<ov::PartialShape> {};

TEST_P(EnableHostCompileInvalidShapeTest, DoesNotEnableForUnsupportedPortShape) {
    auto config = make_config();
    const auto model = make_model(GetParam());

    enable_host_compile_if_needed(model, config);

    EXPECT_FALSE(host_compile_enabled(config));
}

INSTANTIATE_TEST_SUITE_P(AllShapeConditions,
                         EnableHostCompileInvalidShapeTest,
                         testing::Values(ov::PartialShape::dynamic(),
                                         ov::PartialShape{1, ov::Dimension(1, 8), 16},
                                         ov::PartialShape{1, 8, 16, 32},
                                         ov::PartialShape{ov::Dimension(1, 2), ov::Dimension(1, 8), 16, 32},
                                         ov::PartialShape{1, ov::Dimension::dynamic(), 16, 32}));

TEST(EnableHostCompileTest, DoesNotEnableWithoutDynamicOutput) {
    auto config = make_config();
    const auto model = make_model({1, ov::Dimension(1, 8), 16, 32}, false);

    enable_host_compile_if_needed(model, config);

    EXPECT_FALSE(host_compile_enabled(config));
}

TEST(EnableHostCompileTest, DoesNotEnableWhenAnotherPortHasUnboundedDimension) {
    auto config = make_config();
    const auto model =
        make_model({1, ov::Dimension(1, 8), 16, 32}, true, ov::PartialShape{1, ov::Dimension::dynamic(), 16, 32});

    enable_host_compile_if_needed(model, config);

    EXPECT_FALSE(host_compile_enabled(config));
}

}  // namespace