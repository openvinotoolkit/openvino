// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <signal.h>
#include <setjmp.h>

#ifdef _WIN32
#include <process.h>
#endif

#include <gtest/gtest.h>


#include "common_test_utils/test_common.hpp"
#include "common_test_utils/test_constants.hpp"
#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/crash_handler.hpp"
#include "common_test_utils/file_utils.hpp"

#include "common_test_utils/ov_plugin_cache.hpp"
#include "functional_test_utils/skip_tests_config.hpp"
#include "functional_test_utils/summary/api_summary.hpp"
#include "openvino/util/file_util.hpp"
#include "common_test_utils/subgraph_builders/split_conv_concat.hpp"
#include "common_test_utils/subgraph_builders/kso_func.hpp"
#include "common_test_utils/subgraph_builders/single_concat_with_constant.hpp"
#include "common_test_utils/subgraph_builders/concat_with_params.hpp"
#include "common_test_utils/subgraph_builders/split_concat.hpp"

#define MARK_MANDATORY_PROPERTY_FOR_HW_DEVICE(GET_TEST_NAME)                                            \
    [](const testing::TestParamInfo<PropertiesParams>& info) {                                          \
        std::string name = GET_TEST_NAME(info);                                                         \
        return (sw_plugin_in_target_device(ov::test::utils::target_device) ? "" : "mandatory_") + name; \
    }

#define MARK_MANDATORY_API_FOR_HW_DEVICE_WITH_PARAM(GET_TEST_NAME)                                      \
    [](const testing::TestParamInfo<std::string>& info) {                                               \
        std::string name = GET_TEST_NAME(info);                                                         \
        return (sw_plugin_in_target_device(ov::test::utils::target_device) ? "" : "mandatory_") + name; \
    }

#define MARK_MANDATORY_API_FOR_HW_DEVICE_WITHOUT_PARAM()                                                \
    [](const testing::TestParamInfo<std::string>& info) {                                               \
        return sw_plugin_in_target_device(ov::test::utils::target_device) ? "" : "mandatory_";          \
    }

namespace ov {
namespace test {
namespace behavior {

inline std::shared_ptr<ov::Model> getDefaultNGraphFunctionForTheDevice(std::vector<size_t> inputShape = {1, 2, 32, 32},
                                                                              ov::element::Type_t ngPrc = ov::element::Type_t::f32) {
    return ov::test::utils::make_split_concat(inputShape, ngPrc);
}

inline bool sw_plugin_in_target_device(std::string targetDevice) {
    return (targetDevice.find("MULTI") != std::string::npos || targetDevice.find("BATCH") != std::string::npos ||
            targetDevice.find("HETERO") != std::string::npos || targetDevice.find("AUTO") != std::string::npos);
}

class APIBaseTest : public ov::test::TestsCommon {
private:
    // in case of crash jump will be made and work will be continued
    const std::unique_ptr<ov::test::utils::CrashHandler> crashHandler = std::unique_ptr<ov::test::utils::CrashHandler>(
                                                                        new ov::test::utils::CrashHandler(ov::test::utils::CONFORMANCE_TYPE::api));

protected:
    double k = 1.0;
    std::string target_device = "";
    ov::test::utils::ov_entity api_entity = ov::test::utils::ov_entity::undefined;
    ov::test::utils::ApiSummary& api_summary = ov::test::utils::ApiSummary::getInstance();

public:
    APIBaseTest() = default;

    virtual void set_api_entity() { api_entity = ov::test::utils::ov_entity::undefined; }

    void SetUp() override {
        set_api_entity();
        auto test_name = this->GetFullTestName();
        k = test_name.find("_mandatory") != std::string::npos || test_name.find("mandatory_") != std::string::npos ? 1.0 : 0.0;
        if (ov::test::utils::is_print_rel_influence_coef)
            std::cout << "[ CONFORMANCE ] Influence coefficient: " << k << std::endl;
        api_summary.updateStat(api_entity, target_device, ov::test::utils::PassRate::Statuses::CRASHED, k);
        crashHandler->StartTimer();
    }

    void TearDown() override {
        if (api_entity == ov::test::utils::ov_entity::undefined) {
            set_api_entity();
        }
        if (this->HasFailure()) {
            api_summary.updateStat(api_entity, target_device, ov::test::utils::PassRate::Statuses::FAILED, k);
        } else if (this->IsSkipped()) {
            api_summary.updateStat(api_entity, target_device, ov::test::utils::PassRate::Statuses::SKIPPED, k);
        } else {
            api_summary.updateStat(api_entity, target_device, ov::test::utils::PassRate::Statuses::PASSED, k);
        }
    }
};

class OVInferRequestTestBase :  public APIBaseTest {
private:
    void set_api_entity() override {
        api_entity = ov::test::utils::ov_entity::ov_infer_request;
    };
};

class OVCompiledNetworkTestBase :  public APIBaseTest {
private:
    void set_api_entity() override {
        api_entity = ov::test::utils::ov_entity::ov_compiled_model;
    };
};

class OVPluginTestBase :  public APIBaseTest {
private:
    void set_api_entity() override {
        api_entity = ov::test::utils::ov_entity::ov_plugin;
    };
};

typedef std::tuple<
        std::string, // Device name
        ov::AnyMap   // Config
> InferRequestParams;

class OVInferRequestTests : public testing::WithParamInterface<InferRequestParams>,
                            public OVInferRequestTestBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<InferRequestParams> obj) {
        std::string targetDevice;
        ov::AnyMap configuration;
        std::tie(targetDevice, configuration) = obj.param;
        std::replace(targetDevice.begin(), targetDevice.end(), ':', '.');
        std::ostringstream result;
        result << "targetDevice=" << targetDevice << "_";
        if (!configuration.empty()) {
            using namespace ov::test::utils;
            for (auto &configItem : configuration) {
                result << "configItem=" << configItem.first << "_";
                configItem.second.print(result);
            }
        }
        return result.str();
    }

    void SetUp() override {
        std::tie(target_device, configuration) = this->GetParam();
        // Skip test according to plugin specific disabledTestPatterns() (if any)
        SKIP_IF_CURRENT_TEST_IS_DISABLED()
        APIBaseTest::SetUp();
        function = ov::test::behavior::getDefaultNGraphFunctionForTheDevice();
        ov::AnyMap params;
        for (auto&& v : configuration) {
            params.emplace(v.first, v.second);
        }
        execNet = core->compile_model(function, target_device, params);
    }

    void TearDown() override {
        if (!configuration.empty()) {
            ov::test::utils::PluginCache::get().reset();
        }
        APIBaseTest::TearDown();
    }

protected:
    ov::CompiledModel execNet;
    std::shared_ptr<ov::Core> core = utils::PluginCache::get().core();
    std::string targetDevice;
    ov::AnyMap configuration;
    std::shared_ptr<ov::Model> function;
};

// DEPRECATED
// Replace the usage by `ov::test::utils::create_core()`
// in NVIDIA and NPU plugin
inline ov::Core createCoreWithTemplate() {
    ov::test::utils::PluginCache::get().reset();
    ov::Core core;
#ifndef OPENVINO_STATIC_LIBRARY
    std::string pluginName = "openvino_template_plugin";
    pluginName += OV_BUILD_POSTFIX;
    core.register_plugin(ov::util::make_plugin_library_name(ov::test::utils::getExecutableDirectory(), pluginName),
        ov::test::utils::DEVICE_TEMPLATE);
#endif // !OPENVINO_STATIC_LIBRARY
    return core;
}

class OVClassNetworkTest {
public:
    std::shared_ptr<ov::Model> actualNetwork, simpleNetwork, multinputNetwork, ksoNetwork;

    void SetUp() {
        SKIP_IF_CURRENT_TEST_IS_DISABLED();
        // Generic network
        actualNetwork = ov::test::utils::make_split_concat();
        // Quite simple network
        simpleNetwork = ov::test::utils::make_single_concat_with_constant();
        // Multinput to substruct network
        multinputNetwork = ov::test::utils::make_concat_with_params();
        // Network with KSO
        ksoNetwork = ov::test::utils::make_kso_function();
    }

    virtual void setHeteroNetworkAffinity(const std::string &targetDevice) {
        const std::map<std::string, std::string> deviceMapping = {{"Split_2",       targetDevice},
                                                                  {"Convolution_4", targetDevice},
                                                                  {"Convolution_7", ov::test::utils::DEVICE_CPU},
                                                                  {"Relu_5",        ov::test::utils::DEVICE_CPU},
                                                                  {"Relu_8",        targetDevice},
                                                                  {"Concat_9",      ov::test::utils::DEVICE_CPU}};

        for (const auto &op : actualNetwork->get_ops()) {
            auto it = deviceMapping.find(op->get_friendly_name());
            if (it != deviceMapping.end()) {
                std::string affinity = it->second;
                op->get_rt_info()["affinity"] = affinity;
            }
        }
    }
};

class OVClassBaseTestP : public OVClassNetworkTest,
                         public ::testing::WithParamInterface<std::string>,
                         public OVPluginTestBase {
public:
    void SetUp() override {
        target_device = GetParam();
        SKIP_IF_CURRENT_TEST_IS_DISABLED();
        APIBaseTest::SetUp();
        OVClassNetworkTest::SetUp();
    }
};
using OVClassModelTestP = OVClassBaseTestP;
using OVClassModelOptionalTestP = OVClassBaseTestP;

class OVCompiledModelClassBaseTestP : public OVClassNetworkTest,
                                      public ::testing::WithParamInterface<std::string>,
                                      public OVCompiledNetworkTestBase {
public:
    void SetUp() override {
        target_device = GetParam();
        SKIP_IF_CURRENT_TEST_IS_DISABLED();
        APIBaseTest::SetUp();
        OVClassNetworkTest::SetUp();
    }
};

class OVClassSetDevicePriorityConfigPropsTest : public OVPluginTestBase,
                                                public ::testing::WithParamInterface<std::tuple<std::string, AnyMap>> {
protected:
    std::string deviceName;
    ov::AnyMap configuration;
    std::shared_ptr<ov::Model> actualNetwork;

public:
    void SetUp() override {
        std::tie(target_device, configuration) = GetParam();
        SKIP_IF_CURRENT_TEST_IS_DISABLED();
        APIBaseTest::SetUp();
        actualNetwork = ov::test::utils::make_split_conv_concat();
    }
};

class OVClassSeveralDevicesTests : public OVPluginTestBase,
                                   public OVClassNetworkTest,
                                   public ::testing::WithParamInterface<std::vector<std::string>> {
public:
    std::vector<std::string> target_devices;

    void SetUp() override {
        target_device = ov::test::utils::DEVICE_MULTI;
        SKIP_IF_CURRENT_TEST_IS_DISABLED()
        APIBaseTest::SetUp();
        OVClassNetworkTest::SetUp();
        target_devices = GetParam();
    }
};

} // namespace behavior
} // namespace test
} // namespace ov
