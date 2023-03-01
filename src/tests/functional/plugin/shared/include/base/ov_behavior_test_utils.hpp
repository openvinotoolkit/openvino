// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <signal.h>
#include <setjmp.h>

#ifdef _WIN32
#include <process.h>
#endif

#include <gtest/gtest.h>

#include "ngraph_functions/subgraph_builders.hpp"

#include "common_test_utils/test_common.hpp"
#include "common_test_utils/test_constants.hpp"
#include "common_test_utils/common_utils.hpp"
#include "common_test_utils/crash_handler.hpp"
#include "common_test_utils/file_utils.hpp"

#include "functional_test_utils/plugin_cache.hpp"
#include "functional_test_utils/ov_plugin_cache.hpp"
#include "functional_test_utils/skip_tests_config.hpp"
#include "functional_test_utils/blob_utils.hpp"
#include "functional_test_utils/summary/api_summary.hpp"
#include "openvino/util/file_util.hpp"

namespace ov {
namespace test {
namespace behavior {

inline std::shared_ptr<ngraph::Function> getDefaultNGraphFunctionForTheDevice(std::string targetDevice,
                                                                              std::vector<size_t> inputShape = {1, 1, 32, 32},
                                                                              ngraph::element::Type_t ngPrc = ngraph::element::Type_t::f32) {
    // auto-batching (which now relies on the dim tracking) needs a ngraph function without reshapes in that
    if (targetDevice.find(CommonTestUtils::DEVICE_BATCH) != std::string::npos)
        return ngraph::builder::subgraph::makeConvPoolReluNoReshapes(inputShape, ngPrc);
    else  // for compatibility with the GNA that fails on any other ngraph function
        return ngraph::builder::subgraph::makeConvPoolRelu(inputShape, ngPrc);
}

class APIBaseTest : public CommonTestUtils::TestsCommon {
private:
    // place to jump in case of a crash
    int jmpRes = 0;
    // in case of crash jump will be made and work will be continued
    const std::unique_ptr<CommonTestUtils::CrashHandler> crashHandler = std::unique_ptr<CommonTestUtils::CrashHandler>(new CommonTestUtils::CrashHandler());

protected:
    size_t k = 1;
    std::string target_device = "";
    ov::test::utils::ov_entity api_entity = ov::test::utils::ov_entity::undefined;
    ov::test::utils::ApiSummary& api_summary = ov::test::utils::ApiSummary::getInstance();

public:
    APIBaseTest() = default;

    virtual void set_api_entity() { api_entity = ov::test::utils::ov_entity::undefined; }

    void SetUp() override {
        set_api_entity();
        auto test_name = this->GetTestName();
        k = test_name.find("_mandatory") != std::string::npos || test_name.find("mandatory_") != std::string::npos ? 1 : 0;
        std::cout << "[ CONFORMANCE ] Influence coefficient: " << k << std::endl;
        api_summary.updateStat(api_entity, target_device, ov::test::utils::PassRate::Statuses::CRASHED, k);
#ifdef _WIN32
        jmpRes = setjmp(CommonTestUtils::env);
#else
        jmpRes = sigsetjmp(CommonTestUtils::env, 0);
#endif
        if (jmpRes == CommonTestUtils::JMP_STATUS::ok) {
            crashHandler->StartTimer();
        } else if (jmpRes == CommonTestUtils::JMP_STATUS::alarmErr) {
            api_summary.updateStat(api_entity, target_device, ov::test::utils::PassRate::Statuses::HANGED, k);
            GTEST_FAIL();
        }
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
            using namespace CommonTestUtils;
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
        function = ov::test::behavior::getDefaultNGraphFunctionForTheDevice(target_device);
        ov::AnyMap params;
        for (auto&& v : configuration) {
            params.emplace(v.first, v.second);
        }
        execNet = core->compile_model(function, target_device, params);
    }

    void TearDown() override {
        if (!configuration.empty()) {
            PluginCache::get().reset();
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

inline ov::Core createCoreWithTemplate() {
    ov::test::utils::PluginCache::get().reset();
    ov::Core core;
#ifndef OPENVINO_STATIC_LIBRARY
    std::string pluginName = "openvino_template_plugin";
    pluginName += IE_BUILD_POSTFIX;
    core.register_plugin(ov::util::make_plugin_library_name(CommonTestUtils::getExecutableDirectory(), pluginName),
        CommonTestUtils::DEVICE_TEMPLATE);
#endif // !OPENVINO_STATIC_LIBRARY
    return core;
}

class OVClassNetworkTest {
public:
    std::shared_ptr<ngraph::Function> actualNetwork, simpleNetwork, multinputNetwork, ksoNetwork;

    void SetUp() {
        SKIP_IF_CURRENT_TEST_IS_DISABLED();
        // Generic network
        actualNetwork = ngraph::builder::subgraph::makeSplitConvConcat();
        // Quite simple network
        simpleNetwork = ngraph::builder::subgraph::makeSingleConv();
        // Multinput to substruct network
        multinputNetwork = ngraph::builder::subgraph::make2InputSubtract();
        // Network with KSO
        ksoNetwork = ngraph::builder::subgraph::makeKSOFunction();
    }

    virtual void setHeteroNetworkAffinity(const std::string &targetDevice) {
        const std::map<std::string, std::string> deviceMapping = {{"Split_2",       targetDevice},
                                                                  {"Convolution_4", targetDevice},
                                                                  {"Convolution_7", CommonTestUtils::DEVICE_CPU},
                                                                  {"Relu_5",        CommonTestUtils::DEVICE_CPU},
                                                                  {"Relu_8",        targetDevice},
                                                                  {"Concat_9",      CommonTestUtils::DEVICE_CPU}};

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

using PriorityParams = std::tuple<
        std::string,            // Device name
        ov::AnyMap              // device priority Configuration key
>;
class OVClassExecutableNetworkGetMetricTest_Priority : public ::testing::WithParamInterface<PriorityParams>,
                                                       public OVCompiledNetworkTestBase {
protected:
    ov::AnyMap configuration;
    std::shared_ptr<ngraph::Function> simpleNetwork;

public:
    static std::string getTestCaseName(testing::TestParamInfo<PriorityParams> obj);
    void SetUp() override {
        std::tie(target_device, configuration) = GetParam();
        SKIP_IF_CURRENT_TEST_IS_DISABLED();
        APIBaseTest::SetUp();
        simpleNetwork = ngraph::builder::subgraph::makeSingleConv();
    }
};
using OVClassExecutableNetworkGetMetricTest_DEVICE_PRIORITY = OVClassExecutableNetworkGetMetricTest_Priority;
using OVClassExecutableNetworkGetMetricTest_MODEL_PRIORITY = OVClassExecutableNetworkGetMetricTest_Priority;

#define SKIP_IF_NOT_IMPLEMENTED(...)                   \
{                                                      \
    try {                                              \
        __VA_ARGS__;                                   \
    } catch (const InferenceEngine::NotImplemented&) { \
        GTEST_SKIP();                                  \
    }                                                  \
}
} // namespace behavior
} // namespace test
} // namespace ov
