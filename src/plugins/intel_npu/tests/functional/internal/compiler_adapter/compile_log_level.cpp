// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>

#include "model_serializer.hpp"
#include "openvino/runtime/intel_npu/properties.hpp"

namespace {

class CompileLogLevelSerializeConfigTests : public ::testing::Test {
protected:
    // The compiler properties hold the values as strings
    std::map<std::string, std::string> compilerProperties;

    static ze_graph_compiler_version_info_t modernCompilerVersion() {
        ze_graph_compiler_version_info_t version{};
        version.major = 7;
        version.minor = 0;
        return version;
    }

    std::string serialize() const {
        return ::intel_npu::compiler_utils::serializeConfig(compilerProperties, modernCompilerVersion());
    }
};

TEST_F(CompileLogLevelSerializeConfigTests, BackwardCompatibleCompilerLogUnsetPluginLogSet) {
    compilerProperties[ov::log::level.name()] = std::string("LOG_DEBUG");

    const std::string flags = serialize();

    EXPECT_NE(flags.find(std::string(ov::log::level.name()) + "=\"LOG_DEBUG\""), std::string::npos) << flags;
    EXPECT_EQ(flags.find(ov::intel_npu::compile_log_level.name()), std::string::npos)
        << "NPU_COMPILE_LOG_LEVEL must never be serialized under its own key: " << flags;
}

TEST_F(CompileLogLevelSerializeConfigTests, CompileLogLevelSetPrioritizedOverUnchangedPluginLogLevel) {
    compilerProperties[ov::log::level.name()] = std::string("LOG_DEBUG");
    compilerProperties[ov::intel_npu::compile_log_level.name()] = std::string("LOG_ERROR");

    const std::string flags = serialize();

    EXPECT_NE(flags.find(std::string(ov::log::level.name()) + "=\"LOG_ERROR\""), std::string::npos) << flags;
    EXPECT_EQ(flags.find(std::string(ov::log::level.name()) + "=\"LOG_DEBUG\""), std::string::npos) << flags;
    EXPECT_EQ(flags.find(ov::intel_npu::compile_log_level.name()), std::string::npos)
        << "NPU_COMPILE_LOG_LEVEL must never be serialized under its own key: " << flags;
}

TEST_F(CompileLogLevelSerializeConfigTests, CompileLogLevelSetPrioritizedOverChangedPluginLogLevel) {
    compilerProperties[ov::intel_npu::compile_log_level.name()] = std::string("LOG_TRACE");

    const std::string flags = serialize();

    EXPECT_NE(flags.find(std::string(ov::log::level.name()) + "=\"LOG_TRACE\""), std::string::npos) << flags;
    EXPECT_EQ(flags.find(ov::intel_npu::compile_log_level.name()), std::string::npos)
        << "NPU_COMPILE_LOG_LEVEL must never be serialized under its own key: " << flags;
}

}  // namespace
