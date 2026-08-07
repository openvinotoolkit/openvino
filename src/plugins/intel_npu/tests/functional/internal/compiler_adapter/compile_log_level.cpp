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
#include "openvino/runtime/intel_npu/properties.hpp"

namespace {

class CompileLogLevelSerializeConfigTests : public ::testing::Test {
protected:
    std::shared_ptr<::intel_npu::OptionsDesc> options;
    std::unique_ptr<::intel_npu::FilteredConfig> config;

    void SetUp() override {
        using namespace ::intel_npu;

        options = std::make_shared<OptionsDesc>();
        options->add<LOG_LEVEL>();
        options->add<COMPILE_LOG_LEVEL>();

        config = std::make_unique<FilteredConfig>(options);

        config->enable(ov::log::level.name(), true);
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
        return ::intel_npu::compiler_utils::serializeConfig(*config, modernCompilerVersion(), allSupported);
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

}  // namespace
