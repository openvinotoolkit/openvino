// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "base/ov_behavior_test_utils.hpp"
#include "openvino/runtime/threading/executor_manager.hpp"

namespace ov {
namespace test {
namespace behavior {

typedef std::tuple<size_t,       // Stream executor number
                   std::string,  // Device name
                   ov::AnyMap    // Config
                   >
    InferRequestPropertiesParams;

class InferRequestPropertiesTest : public testing::WithParamInterface<InferRequestPropertiesParams>,
                                   public ov::test::behavior::OVInferRequestTestBase {
public:
    void SetUp() override {
        std::tie(streamExecutorNumber, target_device, configuration) = this->GetParam();
        // Skip test according to plugin specific disabledTestPatterns() (if any)
        SKIP_IF_CURRENT_TEST_IS_DISABLED()
        APIBaseTest::SetUp();
        // Create model
        function = ov::test::behavior::getDefaultNGraphFunctionForTheDevice();
    }

    void TearDown() override {
        if (!configuration.empty()) {
            ov::test::utils::PluginCache::get().reset();
        }
        APIBaseTest::TearDown();
    }

    static std::string getTestCaseName(testing::TestParamInfo<InferRequestPropertiesParams> obj) {
        std::string target_device;
        size_t streamExecutorNumber;
        ov::AnyMap configuration;
        std::tie(streamExecutorNumber, target_device, configuration) = obj.param;
        std::ostringstream result;
        result << "target_device=" << target_device << "_";
        result << "streamExecutorNumber=" << target_device << "_";
        if (!configuration.empty()) {
            for (auto& configItem : configuration) {
                result << "configItem=" << configItem.first << "_";
                configItem.second.print(result);
            }
        }

        return result.str();
    }

protected:
    ov::CompiledModel execNet;
    std::shared_ptr<ov::Core> core = ov::test::utils::PluginCache::get().core();
    std::shared_ptr<ov::Model> function;
    ov::AnyMap configuration;
    size_t streamExecutorNumber;

    void set_api_entity() override {
        api_entity = ov::test::utils::ov_entity::ov_infer_request;
    }

    inline ov::InferRequest createInferRequestWithConfig() {
        // Load config
        configuration.insert({ov::internal::exclusive_async_requests(true)});
        if (target_device.find(ov::test::utils::DEVICE_AUTO) == std::string::npos &&
            target_device.find(ov::test::utils::DEVICE_MULTI) == std::string::npos &&
            target_device.find(ov::test::utils::DEVICE_HETERO) == std::string::npos &&
            target_device.find(ov::test::utils::DEVICE_BATCH) == std::string::npos) {
            core->set_property(target_device, configuration);
        }
        // Compile model to target plugins
        execNet = core->compile_model(function, target_device, configuration);
        auto req = execNet.create_infer_request();
        return req;
    }
};

TEST_P(InferRequestPropertiesTest, canSetExclusiveAsyncRequests) {
    ASSERT_EQ(0ul, ov::threading::executor_manager()->get_executors_number());
    OV_ASSERT_NO_THROW(createInferRequestWithConfig());
    if (target_device.find(ov::test::utils::DEVICE_AUTO) == std::string::npos &&
        target_device.find(ov::test::utils::DEVICE_MULTI) == std::string::npos &&
        target_device.find(ov::test::utils::DEVICE_HETERO) == std::string::npos &&
        target_device.find(ov::test::utils::DEVICE_BATCH) == std::string::npos) {
        ASSERT_EQ(streamExecutorNumber, ov::threading::executor_manager()->get_executors_number());
    }
}

TEST_P(InferRequestPropertiesTest, ReusableCPUStreamsExecutor) {
    ov::threading::executor_manager()->clear();
    ASSERT_EQ(0u, ov::threading::executor_manager()->get_executors_number());
    ASSERT_EQ(0u, ov::threading::executor_manager()->get_idle_cpu_streams_executors_number());

    {
        // Load config
        ov::AnyMap config = {{ov::internal::exclusive_async_requests(false)}};
        config.insert(configuration.begin(), configuration.end());
        if (target_device.find(ov::test::utils::DEVICE_AUTO) == std::string::npos &&
            target_device.find(ov::test::utils::DEVICE_MULTI) == std::string::npos &&
            target_device.find(ov::test::utils::DEVICE_HETERO) == std::string::npos &&
            target_device.find(ov::test::utils::DEVICE_BATCH) == std::string::npos) {
            OV_ASSERT_NO_THROW(core->set_property(target_device, config));
        }
        // Load CNNNetwork to target plugins
        execNet = core->compile_model(function, target_device, config);
        auto req = execNet.create_infer_request();
        if (target_device == ov::test::utils::DEVICE_NPU) {
            ASSERT_EQ(1u, ov::threading::executor_manager()->get_executors_number());
            ASSERT_EQ(0u, ov::threading::executor_manager()->get_idle_cpu_streams_executors_number());
        } else if ((target_device == ov::test::utils::DEVICE_AUTO) ||
                   (target_device == ov::test::utils::DEVICE_MULTI)) {
        } else {
            ASSERT_EQ(0u, ov::threading::executor_manager()->get_executors_number());
            ASSERT_GE(2u, ov::threading::executor_manager()->get_idle_cpu_streams_executors_number());
        }
    }
}

TEST_P(InferRequestPropertiesTest, ConfigHasUnsupportedPluginProperty) {
    configuration.insert({ov::enable_mmap(false)});
    if (target_device.find(ov::test::utils::DEVICE_AUTO) == std::string::npos &&
        target_device.find(ov::test::utils::DEVICE_MULTI) == std::string::npos &&
        target_device.find(ov::test::utils::DEVICE_HETERO) == std::string::npos &&
        target_device.find(ov::test::utils::DEVICE_BATCH) == std::string::npos) {
        OV_ASSERT_NO_THROW(core->set_property(target_device, configuration));
    }
    // Compile model to target plugins
    execNet = core->compile_model(function, target_device, configuration);
    OV_ASSERT_NO_THROW(execNet.create_infer_request());
}
}  // namespace behavior
}  // namespace test
}  // namespace ov
