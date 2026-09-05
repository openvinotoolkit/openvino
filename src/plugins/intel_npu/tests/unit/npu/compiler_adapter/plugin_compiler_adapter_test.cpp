// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plugin_compiler_adapter.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "fake_vcl_compiler.hpp"
#include "intel_npu/common/filtered_config.hpp"
#include "intel_npu/common/option_support_cache.hpp"
#include "intel_npu/config/options.hpp"
#include "intel_npu/npu_private_properties.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/rt_info/weightless_caching_attributes.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"

using ::fake_vcl::FakeVCLCompiler;
using ::intel_npu::FilteredConfig;
using ::intel_npu::OptionsDesc;
using ::intel_npu::OptionSupportCache;
using ::intel_npu::PluginCompilerAdapter;

namespace {

constexpr OptionSupportCache::CacheKey kPluginKey =
    static_cast<OptionSupportCache::CacheKey>(ov::intel_npu::CompilerType::PLUGIN);

std::shared_ptr<ov::Model> makeModel() {
    auto weights = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{5}, std::vector<float>{1.0f});
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{5});
    auto add = std::make_shared<ov::op::v1::Add>(input, weights);
    return std::make_shared<ov::Model>(ov::OutputVector{add}, ov::ParameterVector{input}, "adapter_test_model");
}

/// The weights-separation flow requires every participating Constant to carry a
/// WeightlessCacheAttribute; WeightlessGraph asserts on its absence.
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
    std::shared_ptr<OptionSupportCache> cache;

    /// The adapter is built with a null ZeroInitStructsHolder throughout: that is the no-driver path,
    /// which is the only one reachable without an NPU.
    std::unique_ptr<PluginCompilerAdapter> makeAdapter() {
        return std::make_unique<PluginCompilerAdapter>(ov::SoPtr<::intel_npu::IVCLCompiler>(compiler), nullptr, cache);
    }

    static std::shared_ptr<OptionsDesc> makeOptionsDesc() {
        auto desc = std::make_shared<OptionsDesc>();
        desc->add<::intel_npu::LOG_LEVEL>();
        desc->add<::intel_npu::COMPILATION_MODE>();
        desc->add<::intel_npu::SEPARATE_WEIGHTS_VERSION>();
        desc->add<::intel_npu::MODEL_SERIALIZER_VERSION>();
        return desc;
    }

    static FilteredConfig makeConfig(const std::optional<std::string>& compilationMode = std::nullopt,
                                     const std::optional<std::string>& wsVersion = std::nullopt) {
        FilteredConfig config(makeOptionsDesc());
        config.enableAll();
        if (compilationMode.has_value()) {
            config.update({{ov::intel_npu::compilation_mode.name(), *compilationMode}});
        }
        if (wsVersion.has_value()) {
            config.update({{ov::intel_npu::separate_weights_version.name(), *wsVersion}});
        }
        return config;
    }
};

//
// --- construction ---
//

TEST_F(PluginCompilerAdapterTest, InjectedCompilerIsAdapted) {
    auto adapter = makeAdapter();
    ASSERT_NE(adapter, nullptr);
    EXPECT_EQ(adapter->get_version(), compiler->version);
}

TEST_F(PluginCompilerAdapterTest, NullCompilerIsRejected) {
    EXPECT_THROW(
        {
            auto adapter = std::make_unique<PluginCompilerAdapter>(ov::SoPtr<::intel_npu::IVCLCompiler>(nullptr),
                                                                   nullptr,
                                                                   nullptr);
            (void)adapter;
        },
        ov::Exception);
}

//
// --- compile: no-driver path ---
//

TEST_F(PluginCompilerAdapterTest, CompileProducesAGraphEvenWithoutADriver) {
    auto adapter = makeAdapter();
    auto config = makeConfig();

    const auto graph = adapter->compile(makeModel(), config);

    ASSERT_NE(graph, nullptr);
    EXPECT_EQ(compiler->compileCalls, 1);
    // No driver means no Level Zero metadata; the graph is export-only but still constructed.
    EXPECT_TRUE(graph->get_metadata().name.empty());
}

TEST_F(PluginCompilerAdapterTest, CompilePropagatesCompilerFailures) {
    compiler->throwOnCompile = true;
    auto adapter = makeAdapter();
    auto config = makeConfig();

    EXPECT_THROW(adapter->compile(makeModel(), config), ov::Exception);
}

TEST_F(PluginCompilerAdapterTest, CompileDefaultsToTheElfBlobType) {
    // Without a HostCompile mode the ELF path is taken, which does not touch the VM runtime.
    auto adapter = makeAdapter();
    auto config = makeConfig(std::string("DefaultHW"));

    const auto graph = adapter->compile(makeModel(), config);
    ASSERT_NE(graph, nullptr);
}

//
// --- compileWS ---
//

TEST_F(PluginCompilerAdapterTest, CompileWSDefaultsToOneShotWhenTheVersionIsUnset) {
    auto adapter = makeAdapter();
    // SEPARATE_WEIGHTS_VERSION is registered but never set, so the adapter must default it.
    auto config = makeConfig();
    ASSERT_FALSE(config.has<::intel_npu::SEPARATE_WEIGHTS_VERSION>());

    const auto graph = adapter->compileWS(makeWeightlessModel(), config);

    ASSERT_NE(graph, nullptr);
    EXPECT_EQ(compiler->compileWsOneShotCalls, 1);
    EXPECT_EQ(compiler->compileWsIterativeCalls, 0);
}

TEST_F(PluginCompilerAdapterTest, CompileWSOneShotSplitsMainOffTheBack) {
    // Three tensors: two init schedules plus the main one, which is the last entry.
    compiler->wsOneShotResult = {ov::Tensor(ov::element::u8, ov::Shape{4096}),
                                 ov::Tensor(ov::element::u8, ov::Shape{4096}),
                                 ov::Tensor(ov::element::u8, ov::Shape{8192})};
    auto adapter = makeAdapter();
    auto config = makeConfig(std::nullopt, std::string("ONE_SHOT"));

    const auto graph = adapter->compileWS(makeWeightlessModel(), config);

    ASSERT_NE(graph, nullptr);
    EXPECT_EQ(compiler->compileWsOneShotCalls, 1);
}

TEST_F(PluginCompilerAdapterTest, CompileWSOneShotToleratesASingleTensor) {
    // Only the main schedule came back: the adapter warns but must still produce a graph.
    compiler->wsOneShotResult = {ov::Tensor(ov::element::u8, ov::Shape{4096})};
    auto adapter = makeAdapter();
    auto config = makeConfig(std::nullopt, std::string("ONE_SHOT"));

    const auto graph = adapter->compileWS(makeWeightlessModel(), config);

    ASSERT_NE(graph, nullptr);
    EXPECT_EQ(compiler->compileWsOneShotCalls, 1);
}

TEST_F(PluginCompilerAdapterTest, CompileWSIterativeRequiresAGraphHandle) {
    auto adapter = makeAdapter();
    auto config = makeConfig(std::nullopt, std::string("ITERATIVE"));

    // The iterative flow cannot work without a Level Zero graph handle.
    try {
        adapter->compileWS(makeWeightlessModel(), config);
        FAIL() << "Expected compileWS(ITERATIVE) to throw without a graph handle";
    } catch (const ov::Exception& error) {
        EXPECT_NE(std::string(error.what()).find("weights separation"), std::string::npos) << error.what();
    }
    EXPECT_EQ(compiler->compileWsIterativeCalls, 0);
}

//
// --- query / get_version delegation ---
//

TEST_F(PluginCompilerAdapterTest, QueryIsDelegatedToTheCompiler) {
    compiler->queryResult = {{"Add_1", "NPU"}};
    auto adapter = makeAdapter();
    auto config = makeConfig();

    const auto supported = adapter->query(makeModel(), config);

    EXPECT_EQ(compiler->queryCalls, 1);
    ASSERT_EQ(supported.size(), 1u);
    EXPECT_EQ(supported.at("Add_1"), "NPU");
}

TEST_F(PluginCompilerAdapterTest, GetVersionIsDelegatedToTheCompiler) {
    compiler->version = 0x000B0002;
    auto adapter = makeAdapter();
    EXPECT_EQ(adapter->get_version(), 0x000B0002u);
}

//
// --- get_supported_options and the option-support cache ---
//

TEST_F(PluginCompilerAdapterTest, GetSupportedOptionsIsDelegatedWithoutACache) {
    cache = nullptr;
    auto adapter = makeAdapter();

    EXPECT_EQ(adapter->get_supported_options(), std::vector<std::string>({"OPT_A", "OPT_B"}));
    EXPECT_EQ(compiler->getSupportedOptionsCalls, 1);
}

TEST_F(PluginCompilerAdapterTest, GetSupportedOptionsWritesThroughToTheCache) {
    cache = std::make_shared<OptionSupportCache>();
    auto adapter = makeAdapter();

    const auto options = adapter->get_supported_options();
    EXPECT_EQ(options, std::vector<std::string>({"OPT_A", "OPT_B"}));

    // The bulk list is now cached, so per-option lookups resolve without the compiler.
    const auto cachedA = cache->isOptionSupported(kPluginKey, "OPT_A");
    ASSERT_TRUE(cachedA.has_value());
    EXPECT_TRUE(*cachedA);
}

TEST_F(PluginCompilerAdapterTest, IsOptionSupportedMissWritesThroughToTheCache) {
    cache = std::make_shared<OptionSupportCache>();
    auto adapter = makeAdapter();

    EXPECT_TRUE(adapter->is_option_supported("OPT_A"));
    EXPECT_EQ(compiler->optionSupportQueries.size(), 1u);

    const auto cached = cache->isOptionSupported(kPluginKey, "OPT_A");
    ASSERT_TRUE(cached.has_value());
    EXPECT_TRUE(*cached);
}

TEST_F(PluginCompilerAdapterTest, IsOptionSupportedHitShortCircuitsTheCompiler) {
    cache = std::make_shared<OptionSupportCache>();
    auto adapter = makeAdapter();

    EXPECT_TRUE(adapter->is_option_supported("OPT_A"));
    ASSERT_EQ(compiler->optionSupportQueries.size(), 1u);

    // Second call must be served from the cache.
    EXPECT_TRUE(adapter->is_option_supported("OPT_A"));
    EXPECT_EQ(compiler->optionSupportQueries.size(), 1u);
}

TEST_F(PluginCompilerAdapterTest, IsOptionSupportedCachesNegativeResultsToo) {
    cache = std::make_shared<OptionSupportCache>();
    auto adapter = makeAdapter();

    EXPECT_FALSE(adapter->is_option_supported("UNKNOWN_OPTION"));
    ASSERT_EQ(compiler->optionSupportQueries.size(), 1u);

    EXPECT_FALSE(adapter->is_option_supported("UNKNOWN_OPTION"));
    EXPECT_EQ(compiler->optionSupportQueries.size(), 1u);
}

TEST_F(PluginCompilerAdapterTest, IsOptionSupportedWithAValueBypassesTheCache) {
    cache = std::make_shared<OptionSupportCache>();
    auto adapter = makeAdapter();

    // A value-qualified query is not cacheable: it must reach the compiler every time.
    EXPECT_TRUE(adapter->is_option_supported("OPT_A", std::string("VALUE")));
    EXPECT_EQ(compiler->optionSupportQueries.size(), 1u);

    EXPECT_TRUE(adapter->is_option_supported("OPT_A", std::string("VALUE")));
    EXPECT_EQ(compiler->optionSupportQueries.size(), 2u);

    // And it must not have populated the cache.
    EXPECT_FALSE(cache->isOptionSupported(kPluginKey, "OPT_A").has_value());
}

TEST_F(PluginCompilerAdapterTest, IsOptionSupportedForwardsTheValueUnchanged) {
    auto adapter = makeAdapter();

    adapter->is_option_supported("OPT_A", std::string("SOME_VALUE"));

    ASSERT_EQ(compiler->optionSupportQueries.size(), 1u);
    EXPECT_EQ(compiler->optionSupportQueries[0].first, "OPT_A");
    ASSERT_TRUE(compiler->optionSupportQueries[0].second.has_value());
    EXPECT_EQ(*compiler->optionSupportQueries[0].second, "SOME_VALUE");
}

TEST_F(PluginCompilerAdapterTest, IsOptionSupportedWorksWithoutACache) {
    cache = nullptr;
    auto adapter = makeAdapter();

    EXPECT_TRUE(adapter->is_option_supported("OPT_A"));
    EXPECT_FALSE(adapter->is_option_supported("UNKNOWN_OPTION"));
    // Every call reaches the compiler when there is nothing to cache into.
    EXPECT_EQ(compiler->optionSupportQueries.size(), 2u);
}

}  // namespace
