// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plugin_compiler_adapter.hpp"

#include <gtest/gtest.h>

#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "fake_vcl_compiler.hpp"
#include "intel_npu/config/options.hpp"
#include "intel_npu/npu_private_properties.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/rt_info/weightless_caching_attributes.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"

using ::fake_vcl::FakeVCLCompiler;
using ::intel_npu::AdapterDescriptor;
using ::intel_npu::PluginCompilerAdapter;

namespace {

std::shared_ptr<ov::Model> makeModel() {
    auto weights = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{5}, std::vector<float>{1.0f});
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{5});
    auto add = std::make_shared<ov::op::v1::Add>(input, weights);
    return std::make_shared<ov::Model>(ov::OutputVector{add}, ov::ParameterVector{input}, "adapter_test_model");
}

// The weights-separation flow requires every participating Constant to carry a
// WeightlessCacheAttribute; WeightlessGraph asserts on its absence.
std::shared_ptr<ov::Model> makeWeightlessModel() {
    auto weights = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{5}, std::vector<float>{1.0f});
    weights->get_rt_info()[ov::WeightlessCacheAttribute::get_type_info_static()] =
        ov::WeightlessCacheAttribute(weights->get_byte_size(), /* bin_offset = */ 0, ov::element::f32);

    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{5});
    auto add = std::make_shared<ov::op::v1::Add>(input, weights);
    return std::make_shared<ov::Model>(ov::OutputVector{add}, ov::ParameterVector{input}, "adapter_ws_test_model");
}

struct PluginCompilerAdapterTest : public ::testing::Test {
    std::shared_ptr<FakeVCLCompiler> compiler = std::make_shared<FakeVCLCompiler>();

    // The adapter is built with a null ZeroInitStructsHolder throughout: that is the no-driver path,
    // which is the only one reachable without an NPU.
    std::unique_ptr<PluginCompilerAdapter> makeAdapter() {
        return std::make_unique<PluginCompilerAdapter>(nullptr, ov::SoPtr<::intel_npu::IVCLCompiler>(compiler));
    }

    // The adapter receives only the compiler properties, values stored as strings.
    static std::map<std::string, std::string> makeCompilerProperties(
        const std::optional<std::string>& compilationMode = std::nullopt,
        const std::optional<std::string>& wsVersion = std::nullopt) {
        std::map<std::string, std::string> compilerProperties;
        if (compilationMode.has_value()) {
            compilerProperties[ov::intel_npu::compilation_mode.name()] = *compilationMode;
        }
        if (wsVersion.has_value()) {
            compilerProperties[ov::intel_npu::separate_weights_version.name()] = *wsVersion;
        }
        return compilerProperties;
    }
};

TEST_F(PluginCompilerAdapterTest, InjectedCompilerIsAdapted) {
    auto adapter = makeAdapter();
    ASSERT_NE(adapter, nullptr);
    EXPECT_EQ(adapter->get_version(), compiler->version);
}

TEST_F(PluginCompilerAdapterTest, NullCompilerIsRejected) {
    EXPECT_THROW(
        {
            auto adapter =
                std::make_unique<PluginCompilerAdapter>(nullptr, ov::SoPtr<::intel_npu::IVCLCompiler>(nullptr));
            (void)adapter;
        },
        ov::Exception);
}

TEST_F(PluginCompilerAdapterTest, CompileProducesAGraphEvenWithoutADriver) {
    auto adapter = makeAdapter();
    auto compilerProperties = makeCompilerProperties();

    const auto graph = adapter->compile(makeModel(), compilerProperties, AdapterDescriptor{});

    ASSERT_NE(graph, nullptr);
    EXPECT_EQ(compiler->compileCalls, 1);
    EXPECT_EQ(compiler->lastCompilerProperties, compilerProperties);
    // No driver means no Level Zero metadata; the graph is export-only but still constructed.
    EXPECT_TRUE(graph->get_metadata().name.empty());
}

TEST_F(PluginCompilerAdapterTest, CompilePropagatesCompilerFailures) {
    compiler->throwOnCompile = true;
    auto adapter = makeAdapter();
    auto compilerProperties = makeCompilerProperties();

    EXPECT_THROW(adapter->compile(makeModel(), compilerProperties, AdapterDescriptor{}), ov::Exception);
}

TEST_F(PluginCompilerAdapterTest, CompileDefaultsToTheElfBlobType) {
    // Without a HostCompile mode the ELF path is taken, which does not touch the VM runtime.
    auto adapter = makeAdapter();
    auto compilerProperties = makeCompilerProperties(std::string("DefaultHW"));

    const auto graph = adapter->compile(makeModel(), compilerProperties, AdapterDescriptor{});
    ASSERT_NE(graph, nullptr);
}

TEST_F(PluginCompilerAdapterTest, CompileWSDefaultsToOneShotWhenTheVersionIsUnset) {
    auto adapter = makeAdapter();
    // SEPARATE_WEIGHTS_VERSION is never set, so the adapter must default it.
    auto compilerProperties = makeCompilerProperties();
    ASSERT_EQ(compilerProperties.count(ov::intel_npu::separate_weights_version.name()), 0u);

    const auto graph = adapter->compileWS(makeWeightlessModel(), compilerProperties, AdapterDescriptor{});

    ASSERT_NE(graph, nullptr);
    EXPECT_EQ(compiler->compileWsOneShotCalls, 1);
    EXPECT_EQ(compiler->compileWsIterativeCalls, 0);
    // The default is forwarded to the compiler, without touching the caller's properties.
    const auto wsVersionIt = compiler->lastCompilerProperties.find(ov::intel_npu::separate_weights_version.name());
    ASSERT_NE(wsVersionIt, compiler->lastCompilerProperties.end());
    EXPECT_EQ(wsVersionIt->second,
              ::intel_npu::SEPARATE_WEIGHTS_VERSION::toString(ov::intel_npu::WSVersion::ONE_SHOT));
    EXPECT_EQ(compilerProperties.count(ov::intel_npu::separate_weights_version.name()), 0u);
}

TEST_F(PluginCompilerAdapterTest, CompileWSOneShotSplitsMainOffTheBack) {
    // Three tensors: two init schedules plus the main one, which is the last entry.
    compiler->wsOneShotResult = {ov::Tensor(ov::element::u8, ov::Shape{4096}),
                                 ov::Tensor(ov::element::u8, ov::Shape{4096}),
                                 ov::Tensor(ov::element::u8, ov::Shape{8192})};
    auto adapter = makeAdapter();
    auto compilerProperties = makeCompilerProperties(std::nullopt, std::string("ONE_SHOT"));

    const auto graph = adapter->compileWS(makeWeightlessModel(), compilerProperties, AdapterDescriptor{});

    ASSERT_NE(graph, nullptr);
    EXPECT_EQ(compiler->compileWsOneShotCalls, 1);
}

TEST_F(PluginCompilerAdapterTest, CompileWSOneShotToleratesASingleTensor) {
    // Only the main schedule came back: the adapter warns but must still produce a graph.
    compiler->wsOneShotResult = {ov::Tensor(ov::element::u8, ov::Shape{4096})};
    auto adapter = makeAdapter();
    auto compilerProperties = makeCompilerProperties(std::nullopt, std::string("ONE_SHOT"));

    const auto graph = adapter->compileWS(makeWeightlessModel(), compilerProperties, AdapterDescriptor{});

    ASSERT_NE(graph, nullptr);
    EXPECT_EQ(compiler->compileWsOneShotCalls, 1);
}

TEST_F(PluginCompilerAdapterTest, CompileWSIterativeRequiresAGraphHandle) {
    auto adapter = makeAdapter();
    auto compilerProperties = makeCompilerProperties(std::nullopt, std::string("ITERATIVE"));

    // The iterative flow cannot work without a Level Zero graph handle.
    try {
        adapter->compileWS(makeWeightlessModel(), compilerProperties, AdapterDescriptor{});
        FAIL() << "Expected compileWS(ITERATIVE) to throw without a graph handle";
    } catch (const ov::Exception& error) {
        EXPECT_NE(std::string(error.what()).find("weights separation"), std::string::npos) << error.what();
    }
    EXPECT_EQ(compiler->compileWsIterativeCalls, 0);
}

TEST_F(PluginCompilerAdapterTest, QueryIsDelegatedToTheCompiler) {
    compiler->queryResult = {{"Add_1", "NPU"}};
    auto adapter = makeAdapter();
    auto compilerProperties = makeCompilerProperties();

    const auto supported = adapter->query(makeModel(), compilerProperties);

    EXPECT_EQ(compiler->queryCalls, 1);
    EXPECT_EQ(compiler->lastCompilerProperties, compilerProperties);
    ASSERT_EQ(supported.size(), 1u);
    EXPECT_EQ(supported.at("Add_1"), "NPU");
}

TEST_F(PluginCompilerAdapterTest, GetVersionIsDelegatedToTheCompiler) {
    compiler->version = 0x000B0002;
    auto adapter = makeAdapter();
    EXPECT_EQ(adapter->get_version(), 0x000B0002u);
}

TEST_F(PluginCompilerAdapterTest, GetSupportedOptionsIsDelegatedToTheCompiler) {
    auto adapter = makeAdapter();

    EXPECT_EQ(adapter->get_supported_options(), std::vector<std::string>({"OPT_A", "OPT_B"}));
    EXPECT_EQ(compiler->getSupportedOptionsCalls, 1);
}

TEST_F(PluginCompilerAdapterTest, GetSupportedOptionsDelegatesOnEveryCall) {
    auto adapter = makeAdapter();

    adapter->get_supported_options();
    adapter->get_supported_options();

    // The adapter holds no state of its own: it must not answer the second call itself.
    EXPECT_EQ(compiler->getSupportedOptionsCalls, 2);
}

TEST_F(PluginCompilerAdapterTest, IsOptionSupportedForwardsTheValueUnchanged) {
    auto adapter = makeAdapter();

    adapter->is_option_supported("OPT_A", std::string("SOME_VALUE"));

    ASSERT_EQ(compiler->optionSupportQueries.size(), 1u);
    EXPECT_EQ(compiler->optionSupportQueries[0].first, "OPT_A");
    ASSERT_TRUE(compiler->optionSupportQueries[0].second.has_value());
    EXPECT_EQ(*compiler->optionSupportQueries[0].second, "SOME_VALUE");
}

TEST_F(PluginCompilerAdapterTest, IsOptionSupportedIsDelegatedOnEveryCall) {
    auto adapter = makeAdapter();

    EXPECT_TRUE(adapter->is_option_supported("OPT_A"));
    EXPECT_FALSE(adapter->is_option_supported("UNKNOWN_OPTION"));
    // The adapter never answers on its own; deduplication is the compiler's concern.
    EXPECT_EQ(compiler->optionSupportQueries.size(), 2u);
}

}  // namespace
