// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/common/compiler_options_cache.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "intel_npu/common/filtered_config.hpp"
#include "intel_npu/common/icompiler_adapter.hpp"
#include "intel_npu/common/igraph.hpp"

namespace {

class StubCompilerAdapter final : public intel_npu::ICompilerAdapter {
public:
    std::optional<std::vector<std::string>> supportedOptions;
    uint32_t version = 0;
    bool isOptionSupportedResult = false;

    mutable int getSupportedOptionsCalls = 0;
    mutable int getVersionCalls = 0;
    mutable int isOptionSupportedCalls = 0;
    mutable std::vector<std::pair<std::string, std::optional<std::string>>> optionQueries;

    std::shared_ptr<intel_npu::IGraph> compile(const std::shared_ptr<const ov::Model>&,
                                               const intel_npu::FilteredConfig&) const override {
        return {};
    }

    std::shared_ptr<intel_npu::IGraph> compileWS(std::shared_ptr<ov::Model>&&,
                                                 const intel_npu::FilteredConfig&) const override {
        return {};
    }

    ov::SupportedOpsMap query(const std::shared_ptr<const ov::Model>&,
                              const intel_npu::FilteredConfig&) const override {
        return {};
    }

    uint32_t get_version() const override {
        ++getVersionCalls;
        return version;
    }

    std::optional<std::vector<std::string>> get_supported_options() const override {
        ++getSupportedOptionsCalls;
        return supportedOptions;
    }

    bool is_option_supported(const std::string& optName,
                             const std::optional<std::string>& optValue = std::nullopt) const override {
        ++isOptionSupportedCalls;
        optionQueries.emplace_back(optName, optValue);
        return isOptionSupportedResult;
    }
};

TEST(CompilerOptionsCacheTests, NonLegacySupportedListAndProbeCaching) {
    StubCompilerAdapter compiler;
    compiler.supportedOptions = std::vector<std::string>{"CACHE_TEST_OPT_A", "CACHE_TEST_OPT_B=1"};
    compiler.isOptionSupportedResult = true;

    const bool supportedA = intel_npu::CompilerOptionsCache::isOptionSupported(ov::intel_npu::CompilerType::DRIVER,
                                                                               "CACHE_TEST_OPT_A",
                                                                               std::nullopt,
                                                                               &compiler);
    const bool supportedB = intel_npu::CompilerOptionsCache::isOptionSupported(ov::intel_npu::CompilerType::DRIVER,
                                                                               "CACHE_TEST_OPT_B",
                                                                               std::string{"1"},
                                                                               &compiler);
    const bool firstProbe = intel_npu::CompilerOptionsCache::isOptionSupported(ov::intel_npu::CompilerType::DRIVER,
                                                                               "CACHE_TEST_OPT_PROBE",
                                                                               std::nullopt,
                                                                               &compiler);
    const bool secondProbe =
        intel_npu::CompilerOptionsCache::isOptionSupported(ov::intel_npu::CompilerType::DRIVER, "CACHE_TEST_OPT_PROBE");

    EXPECT_TRUE(supportedA);
    EXPECT_TRUE(supportedB);
    EXPECT_TRUE(firstProbe);
    EXPECT_TRUE(secondProbe);
    EXPECT_GE(compiler.getSupportedOptionsCalls, 1);
    EXPECT_EQ(compiler.isOptionSupportedCalls, 1);
}

TEST(CompilerOptionsCacheTests, LegacyCacheVersionSpecificAndReusableWithoutVersion) {
    const auto compilerType = ov::intel_npu::CompilerType::PLUGIN;
    StubCompilerAdapter compiler;
    compiler.supportedOptions = std::nullopt;
    compiler.version = 20;
    compiler.isOptionSupportedResult = true;

    const bool first = intel_npu::CompilerOptionsCache::isOptionSupported(compilerType,
                                                                          "CACHE_TEST_LEGACY_OPT",
                                                                          std::nullopt,
                                                                          &compiler,
                                                                          10U);
    const bool second = intel_npu::CompilerOptionsCache::isOptionSupported(compilerType,
                                                                           "CACHE_TEST_LEGACY_OPT",
                                                                           std::nullopt,
                                                                           nullptr,
                                                                           30U);
    const bool third = intel_npu::CompilerOptionsCache::isOptionSupported(compilerType, "CACHE_TEST_LEGACY_OPT");

    EXPECT_TRUE(first);
    EXPECT_FALSE(second);
    EXPECT_TRUE(third);
    EXPECT_GE(compiler.getSupportedOptionsCalls, 1);
    EXPECT_GE(compiler.getVersionCalls, 1);
    EXPECT_EQ(compiler.isOptionSupportedCalls, 0);
}

}  // namespace
