// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>
#include <string>

#include "intel_npu/common/filtered_config.hpp"
#include "intel_npu/config/config.hpp"
#include "intel_npu/config/options.hpp"
#include "model_serializer.hpp"

namespace {

using namespace intel_npu;

class SerializeConfigTests : public ::testing::Test {
protected:
    std::shared_ptr<OptionsDesc> options;
    std::unique_ptr<FilteredConfig> config;

    void initialize_config() {
        config = std::make_unique<FilteredConfig>(options);
        config->enableRuntimeOptions();
    }

    static ze_graph_compiler_version_info_t modernCompilerVersion() {
        ze_graph_compiler_version_info_t version{};
        version.major = 7;
        version.minor = 0;
        return version;
    }

    std::string serialize() const {
        const auto allSupported = [](const std::string&) {
            return true;
        };
        return compiler_utils::serializeConfig(*config, modernCompilerVersion(), allSupported);
    }
};

class CompileLogLevelSerializeConfigTests : public SerializeConfigTests {
protected:
    void SetUp() override {
        options = std::make_shared<OptionsDesc>();
        options->add<LOG_LEVEL>();
        options->add<COMPILE_LOG_LEVEL>();

        initialize_config();
        config->enable(ov::log::level.name(), true);
    }
};

TEST_F(CompileLogLevelSerializeConfigTests, BackwardCompatibleCompilerLogUnsetPluginLogSet) {
    config->update({{ov::log::level.name(), "LOG_DEBUG"}});

    const std::string flags = serialize();

    EXPECT_NE(flags.find(std::string(ov::log::level.name()) + "=\"LOG_DEBUG\""), std::string::npos) << flags;
    EXPECT_EQ(flags.find(ov::intel_npu::compile_log_level.name()), std::string::npos)
        << "NPU_COMPILE_LOG_LEVEL must never be serialized under its own key: " << flags;
}

TEST_F(CompileLogLevelSerializeConfigTests, CompileLogLevelSetPrioritizedOverUnchangedPluginLogLevel) {
    config->update({{ov::log::level.name(), "LOG_DEBUG"}, {ov::intel_npu::compile_log_level.name(), "LOG_ERROR"}});

    const std::string flags = serialize();

    EXPECT_NE(flags.find(std::string(ov::log::level.name()) + "=\"LOG_ERROR\""), std::string::npos) << flags;
    EXPECT_EQ(flags.find(std::string(ov::log::level.name()) + "=\"LOG_DEBUG\""), std::string::npos) << flags;
    EXPECT_EQ(flags.find(ov::intel_npu::compile_log_level.name()), std::string::npos)
        << "NPU_COMPILE_LOG_LEVEL must never be serialized under its own key: " << flags;
}

TEST_F(CompileLogLevelSerializeConfigTests, CompileLogLevelSetPrioritizedOverChangedPluginLogLevel) {
    config->update({{ov::intel_npu::compile_log_level.name(), "LOG_TRACE"}});

    const std::string flags = serialize();

    EXPECT_NE(flags.find(std::string(ov::log::level.name()) + "=\"LOG_TRACE\""), std::string::npos) << flags;
    EXPECT_EQ(flags.find(ov::intel_npu::compile_log_level.name()), std::string::npos)
        << "NPU_COMPILE_LOG_LEVEL must never be serialized under its own key: " << flags;
}

class PerfCountProfilingTypeSerializeConfigTests : public SerializeConfigTests {
protected:
    void SetUp() override {
        options = std::make_shared<OptionsDesc>();
        options->add<PERF_COUNT>();
        options->add<PROFILING_TYPE>();

        initialize_config();
        config->enable(ov::enable_profiling.name(), true);
    }

    static std::string perf_count_option(const char* value) {
        return std::string(ov::enable_profiling.name()) + "=\"" + value + "\"";
    }
};

TEST_F(PerfCountProfilingTypeSerializeConfigTests, ModelProfilingForwardsPerfCountToCompiler) {
    config->update({{ov::enable_profiling.name(), "YES"}, {ov::intel_npu::profiling_type.name(), "MODEL"}});

    const std::string flags = serialize();

    EXPECT_NE(flags.find(perf_count_option("YES")), std::string::npos) << flags;
}

TEST_F(PerfCountProfilingTypeSerializeConfigTests, InferProfilingDisablesPerfCountForCompiler) {
    config->update({{ov::enable_profiling.name(), "YES"}, {ov::intel_npu::profiling_type.name(), "INFER"}});

    const std::string flags = serialize();

    EXPECT_NE(flags.find(perf_count_option("NO")), std::string::npos) << flags;
    EXPECT_EQ(flags.find(perf_count_option("YES")), std::string::npos) << flags;
}

TEST_F(PerfCountProfilingTypeSerializeConfigTests, DefaultModelProfilingForwardsPerfCountToCompiler) {
    config->update({{ov::enable_profiling.name(), "YES"}});

    const std::string flags = serialize();

    EXPECT_NE(flags.find(perf_count_option("YES")), std::string::npos) << flags;
}

}  // namespace
