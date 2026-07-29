// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gmock/gmock-matchers.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <exception>
#include <memory>
#include <random>
#include <thread>

#include "common/npu_test_env_cfg.hpp"
#include "common/utils.hpp"
#include "common/zero_init_mock.hpp"
#include "functional_test_utils/ov_plugin_cache.hpp"
#include "intel_npu/utils/zero/zero_init.hpp"
#include "intel_npu/utils/zero/zero_wrappers.hpp"
#include "openvino/core/any.hpp"
#include "openvino/core/log.hpp"
#include "openvino/runtime/core.hpp"
#include "shared_test_classes/base/ov_behavior_test_utils.hpp"

namespace ov {
namespace test {
namespace behavior {
using CompilationParams = std::string;

using ::testing::AllOf;
using ::testing::HasSubstr;
class ZeroWrappersTests : public ov::test::behavior::OVPluginTestBase,
                          public testing::WithParamInterface<CompilationParams> {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<CompilationParams>& obj) {
        std::string target_device;
        ov::AnyMap configuration;
        target_device = obj.param;
        std::replace(target_device.begin(), target_device.end(), ':', '_');
        target_device = ov::test::utils::getTestsPlatformFromEnvironmentOr(ov::test::utils::DEVICE_NPU);

        std::ostringstream result;
        result << "targetDevice=" << target_device << "_";
        result << "targetPlatform=" << ov::test::utils::getTestsPlatformFromEnvironmentOr(target_device) << "_";
        if (!configuration.empty()) {
            for (auto& configItem : configuration) {
                result << "configItem=" << configItem.first << "_";
                configItem.second.print(result);
            }
        }

        return result.str();
    }

    void SetUp() override {
        target_device = this->GetParam();

        SKIP_IF_CURRENT_TEST_IS_DISABLED()
        OVPluginTestBase::SetUp();
    }

    void TearDown() override {
        APIBaseTest::TearDown();
    }
};

TEST_P(ZeroWrappersTests, DontDestroyZeroCommandListWhenZeroContextIsDestroyed) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    std::string logs;
    std::mutex logs_mutex;

    // Keep this std::function alive while logging is active.
    std::function<void(std::string_view)> log_cb = [&](std::string_view msg) {
        std::lock_guard<std::mutex> lock(logs_mutex);
        logs.append(msg);
        logs.push_back('\n');
    };

    auto zero_init_mock = std::make_shared<::intel_npu::ZeroInitStructsMock>();

    std::shared_ptr<::intel_npu::ZeroInitStructsHolder> zero_init_struct =
        std::reinterpret_pointer_cast<::intel_npu::ZeroInitStructsHolder>(zero_init_mock);

    {
        utils::LogCallbackGuard log_callback_guard(log_cb);
        utils::LoggerLevelGuard logger_level_guard(ov::log::Level::WARNING);

        auto command_list = std::make_shared<::intel_npu::CommandList>(zero_init_struct);
        ::intel_npu::ZeroInitStructsMock::destroyContextForInstance(zero_init_mock);

        try {
            command_list = {};
        } catch (const std::exception& ex) {
            ASSERT_FALSE(true) << ex.what();
        }
    }
    ASSERT_NE(logs.find("Context or CommandList handle is null during destruction."), std::string::npos);
    ASSERT_EQ(logs.find("zeCommandListDestroy failed"), std::string::npos);
}

TEST_P(ZeroWrappersTests, DontDestroyZeroCommandQueueWhenZeroContextIsDestroyed) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    std::string logs;
    std::mutex logs_mutex;

    std::function<void(std::string_view)> log_cb = [&](std::string_view msg) {
        std::lock_guard<std::mutex> lock(logs_mutex);
        logs.append(msg);
        logs.push_back('\n');
    };

    auto zero_init_mock = std::make_shared<::intel_npu::ZeroInitStructsMock>();

    std::shared_ptr<::intel_npu::ZeroInitStructsHolder> zero_init_struct =
        std::reinterpret_pointer_cast<::intel_npu::ZeroInitStructsHolder>(zero_init_mock);

    {
        utils::LogCallbackGuard log_callback_guard(log_cb);
        utils::LoggerLevelGuard logger_level_guard(ov::log::Level::WARNING);

        auto command_queue =
            std::make_shared<::intel_npu::CommandQueue>(zero_init_struct, ::intel_npu::CommandQueueDesc{});
        ::intel_npu::ZeroInitStructsMock::destroyContextForInstance(zero_init_mock);

        try {
            command_queue = {};
        } catch (const std::exception& ex) {
            ASSERT_FALSE(true) << ex.what();
        }
    }
    ASSERT_NE(logs.find("Context or CommandQueue handle is null during destruction."), std::string::npos);
    ASSERT_EQ(logs.find("zeCommandQueueDestroy failed"), std::string::npos);
}

TEST_P(ZeroWrappersTests, DontDestroyZeroFenceWhenZeroContextIsDestroyed) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    std::string logs;
    std::mutex logs_mutex;

    std::function<void(std::string_view)> log_cb = [&](std::string_view msg) {
        std::lock_guard<std::mutex> lock(logs_mutex);
        logs.append(msg);
        logs.push_back('\n');
    };

    auto zero_init_mock = std::make_shared<::intel_npu::ZeroInitStructsMock>();

    std::shared_ptr<::intel_npu::ZeroInitStructsHolder> zero_init_struct =
        std::reinterpret_pointer_cast<::intel_npu::ZeroInitStructsHolder>(zero_init_mock);

    {
        utils::LogCallbackGuard log_callback_guard(log_cb);
        utils::LoggerLevelGuard logger_level_guard(ov::log::Level::WARNING);

        auto command_queue =
            std::make_shared<::intel_npu::CommandQueue>(zero_init_struct, ::intel_npu::CommandQueueDesc{});
        auto fence = std::make_shared<::intel_npu::Fence>(command_queue);
        ::intel_npu::ZeroInitStructsMock::destroyContextForInstance(zero_init_mock);

        try {
            fence = {};
            command_queue = {};
        } catch (const std::exception& ex) {
            ASSERT_FALSE(true) << ex.what();
        }
    }
    ASSERT_NE(logs.find("Context or Fence handle is null during destruction."), std::string::npos);
    ASSERT_EQ(logs.find("zeFenceDestroy failed"), std::string::npos);
}

TEST_P(ZeroWrappersTests, DontDestroyZeroEventPoolWhenZeroContextIsDestroyed) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    std::string logs;
    std::mutex logs_mutex;

    std::function<void(std::string_view)> log_cb = [&](std::string_view msg) {
        std::lock_guard<std::mutex> lock(logs_mutex);
        logs.append(msg);
        logs.push_back('\n');
    };

    auto zero_init_mock = std::make_shared<::intel_npu::ZeroInitStructsMock>();

    std::shared_ptr<::intel_npu::ZeroInitStructsHolder> zero_init_struct =
        std::reinterpret_pointer_cast<::intel_npu::ZeroInitStructsHolder>(zero_init_mock);

    {
        utils::LogCallbackGuard log_callback_guard(log_cb);
        utils::LoggerLevelGuard logger_level_guard(ov::log::Level::WARNING);

        auto event_pool = std::make_shared<::intel_npu::EventPool>(zero_init_struct, 1);
        ::intel_npu::ZeroInitStructsMock::destroyContextForInstance(zero_init_mock);

        try {
            event_pool = {};
        } catch (const std::exception& ex) {
            ASSERT_FALSE(true) << ex.what();
        }
    }
    ASSERT_NE(logs.find("Context or EventPool handle is null during destruction."), std::string::npos);
    ASSERT_EQ(logs.find("zeEventPoolDestroy failed"), std::string::npos);
}

TEST_P(ZeroWrappersTests, DontDestroyZeroEventWhenZeroContextIsDestroyed) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    std::string logs;
    std::mutex logs_mutex;

    std::function<void(std::string_view)> log_cb = [&](std::string_view msg) {
        std::lock_guard<std::mutex> lock(logs_mutex);
        logs.append(msg);
        logs.push_back('\n');
    };

    auto zero_init_mock = std::make_shared<::intel_npu::ZeroInitStructsMock>();

    std::shared_ptr<::intel_npu::ZeroInitStructsHolder> zero_init_struct =
        std::reinterpret_pointer_cast<::intel_npu::ZeroInitStructsHolder>(zero_init_mock);

    {
        utils::LogCallbackGuard log_callback_guard(log_cb);
        utils::LoggerLevelGuard logger_level_guard(ov::log::Level::WARNING);

        auto event_pool = std::make_shared<::intel_npu::EventPool>(zero_init_struct, 1);
        auto event = std::make_shared<::intel_npu::Event>(event_pool, 0);
        ::intel_npu::ZeroInitStructsMock::destroyContextForInstance(zero_init_mock);

        try {
            event = {};
            event_pool = {};
        } catch (const std::exception& ex) {
            ASSERT_FALSE(true) << ex.what();
        }
    }
    ASSERT_NE(logs.find("Context or Event handle is null during destruction."), std::string::npos);
    ASSERT_EQ(logs.find("zeEventDestroy failed"), std::string::npos);
}

}  // namespace behavior
}  // namespace test
}  // namespace ov
