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

class ProfilingSerializeConfigTests : public ::testing::Test {
protected:
    std::shared_ptr<::intel_npu::OptionsDesc> options;
    std::unique_ptr<::intel_npu::FilteredConfig> config;

    void SetUp() override {
        using namespace ::intel_npu;

        options = std::make_shared<OptionsDesc>();
        options->add<PERF_COUNT>();
        options->add<PROFILING>();

        config = std::make_unique<FilteredConfig>(options);

        config->enable(ov::enable_profiling.name(), true);
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

    static std::string perf_count_option(const char* value) {
        return std::string(ov::enable_profiling.name()) + "=\"" + value + "\"";
    }
};

TEST_F(ProfilingSerializeConfigTests, PerfCountIsForwardedAsSet) {
    config->update({{ov::enable_profiling.name(), "YES"}});

    const std::string flags = serialize();

    EXPECT_NE(flags.find(perf_count_option("YES")), std::string::npos) << flags;
}

TEST_F(ProfilingSerializeConfigTests, NpuProfilingIsNeverSerializedUnderItsOwnKey) {
    config->update({{ov::intel_npu::profiling.name(), "YES"}});

    const std::string flags = serialize();

    EXPECT_EQ(flags.find(ov::intel_npu::profiling.name()), std::string::npos)
        << "NPU_PROFILING is a runtime option and must never reach the compiler: " << flags;
}

}  // namespace
