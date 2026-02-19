// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/subgraph_builders/multi_single_conv.hpp"
#include "common_test_utils/test_assertions.hpp"
#include "intel_npu/npu_private_properties.hpp"
#include "openvino/core/log.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/runtime/intel_npu/properties.hpp"
#include "openvino/runtime/properties.hpp"

using namespace ov::intel_npu;

using OfflineCompilationUnitTests = ::testing::Test;

TEST_F(OfflineCompilationUnitTests, CompilationPassesWithCiPWhenDriverNotInstalledSetProperty) {
    ov::Core core;
    ov::AnyMap config;
    std::string logs;
    std::mutex logs_mutex;

    std::function<void(std::string_view)> log_cb = [&](std::string_view msg) {
        std::lock_guard<std::mutex> lock(logs_mutex);
        logs.append(msg);
        logs.push_back('\n');
    };

    ov::util::set_log_callback(log_cb);

    config[ov::intel_npu::compiler_type.name()] = ov::intel_npu::CompilerType::PLUGIN;
    config[ov::intel_npu::platform.name()] = ov::intel_npu::Platform::NPU5010;
    config[ov::log::level.name()] = ov::log::Level::WARNING;
    core.set_property("NPU", config);

    std::shared_ptr<ov::Model> model = ov::test::utils::make_multi_single_conv();
    OV_ASSERT_NO_THROW(core.compile_model(model, "NPU"));

    ov::util::reset_log_callback();

    ASSERT_NE(logs.find("Only offline compilation can be done"), std::string::npos);
}

TEST_F(OfflineCompilationUnitTests, CompilationPassesWithCiPWhenDriverNotInstalled) {
    ov::Core core;
    ov::AnyMap config;
    std::string logs;
    std::mutex logs_mutex;

    std::function<void(std::string_view)> log_cb = [&](std::string_view msg) {
        std::lock_guard<std::mutex> lock(logs_mutex);
        logs.append(msg);
        logs.push_back('\n');
    };

    ov::util::set_log_callback(log_cb);

    config[ov::intel_npu::compiler_type.name()] = ov::intel_npu::CompilerType::PLUGIN;
    config[ov::intel_npu::platform.name()] = ov::intel_npu::Platform::NPU5010;
    config[ov::log::level.name()] = ov::log::Level::WARNING;
    core.set_property("NPU", config);

    std::shared_ptr<ov::Model> model = ov::test::utils::make_multi_single_conv();
    OV_ASSERT_NO_THROW(core.compile_model(model, "NPU", config));

    ov::util::reset_log_callback();

    ASSERT_NE(logs.find("Only offline compilation can be done"), std::string::npos);
}
