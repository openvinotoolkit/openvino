// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <behavior/ov_plugin/caching_tests.hpp>

#include "intel_npu/utils/zero/zero_init.hpp"
#include "openvino/core/log_util.hpp"
#include "openvino/runtime/intel_npu/properties.hpp"

namespace ov {
namespace test {
namespace behavior {

using OVCompileModelLoadFromFileTestBaseNPU = CompileModelLoadFromFileTestBase;

TEST_P(OVCompileModelLoadFromFileTestBaseNPU, BlobWithOVHeaderAligmentCanBeImported) {
    core->set_property(ov::cache_dir(m_cacheFolderName));

    if (!::intel_npu::ZeroInitStructsHolder::getInstance()->isExternalMemoryStandardAllocationSupported()) {
        GTEST_SKIP() << "Standard allocation is not supported by the current configuration.";
    }

    std::stringstream custom_logger;
    ov::util::LogCallback custom_log_callback =
        [&](std::string_view s) {  // switch to query allocation info for import flag when possible
            custom_logger << s << std::endl;
        };
    ov::util::set_log_callback(custom_log_callback);
    struct ResetLogCallbackGuard {
        ~ResetLogCallbackGuard() {
            ov::util::reset_log_callback();
        }
    } reset_log_callback_guard;

    for (size_t i = 0; i < 2; ++i) {
        if (i != 0) {
            configuration.emplace(ov::log::level(ov::log::Level::DEBUG));
        }
        std::ignore = core->compile_model(m_modelName, targetDevice, configuration);
        configuration.erase(ov::log::level.name());
    }
    EXPECT_THAT(custom_logger.str(),
                ::testing::HasSubstr("getGraphDescriptor - set ZE_GRAPH_FLAG_INPUT_GRAPH_PERSISTENT"));
}

TEST_P(OVCompileModelLoadFromFileTestBaseNPU, EncryptionCallbacksSetSecureCompileFlag) {
    core->set_property(ov::cache_dir(m_cacheFolderName));

    if (::intel_npu::ZeroInitStructsHolder::getInstance()->getGraphDdiTable().version() < ZE_MAKE_VERSION(1, 17)) {
        GTEST_SKIP()
            << "Secure compilation when blob encryption is requested requires ze_graph_ext version 1.17 or higher.";
    }

    if (auto it = configuration.find(ov::intel_npu::compiler_type.name()); it != configuration.end()) {
        if (it->second.as<ov::intel_npu::CompilerType>() == ov::intel_npu::CompilerType::PLUGIN) {
            GTEST_SKIP() << "Compiler in Plugin doesn't need yet secure compilation when blob encryption is requested.";
        }
    }

    std::stringstream custom_logger;
    ov::util::LogCallback custom_log_callback = [&](std::string_view s) {
        custom_logger << s << std::endl;
    };
    ov::util::set_log_callback(custom_log_callback);
    struct ResetLogCallbackGuard {
        ~ResetLogCallbackGuard() {
            ov::util::reset_log_callback();
        }
    } reset_log_callback_guard;

    configuration.emplace(ov::log::level(ov::log::Level::DEBUG));
    std::ignore = core->compile_model(m_modelName, targetDevice, configuration);
    EXPECT_THAT(custom_logger.str(), ::testing::HasSubstr("getGraphDescriptor - set ZE_GRAPH_FLAG_SECURE_COMPILE"));
    configuration.erase(ov::log::level.name());
}

}  // namespace behavior
}  // namespace test
}  // namespace ov
