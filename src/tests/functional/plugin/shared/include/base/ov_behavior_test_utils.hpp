// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gtest/gtest.h>

#include "ngraph_functions/subgraph_builders.hpp"

#include "common_test_utils/test_common.hpp"
#include "common_test_utils/test_constants.hpp"
#include "common_test_utils/common_utils.hpp"

#include "functional_test_utils/plugin_cache.hpp"
#include "functional_test_utils/ov_plugin_cache.hpp"
#include "functional_test_utils/skip_tests_config.hpp"
#include "functional_test_utils/blob_utils.hpp"

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

typedef std::tuple<
        std::string,            // Device name
        ov::AnyMap   // Config
> InferRequestParams;

class OVInferRequestTests : public testing::WithParamInterface<InferRequestParams>,
                            public CommonTestUtils::TestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<InferRequestParams> obj) {
        std::string targetDevice;
        ov::AnyMap configuration;
        std::tie(targetDevice, configuration) = obj.param;
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
        // Skip test according to plugin specific disabledTestPatterns() (if any)
        SKIP_IF_CURRENT_TEST_IS_DISABLED()
        std::tie(targetDevice, configuration) = this->GetParam();
        function = ov::test::behavior::getDefaultNGraphFunctionForTheDevice(targetDevice);
        ov::AnyMap params;
        for (auto&& v : configuration) {
            params.emplace(v.first, v.second);
        }
        execNet = core->compile_model(function, targetDevice, params);
    }

    void TearDown() override {
        if (!configuration.empty()) {
            utils::PluginCache::get().reset();
        }
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
    core.register_plugin(pluginName, CommonTestUtils::DEVICE_TEMPLATE);
#endif // !OPENVINO_STATIC_LIBRARY
    return core;
}

class OVClassNetworkTest : public ::testing::Test {
public:
    std::shared_ptr<ngraph::Function> actualNetwork, simpleNetwork, multinputNetwork, ksoNetwork;

    void SetUp() override {
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

class OVClassBaseTestP : public OVClassNetworkTest, public ::testing::WithParamInterface<std::string> {
public:
    std::string deviceName;

    void SetUp() override {
        // TODO: Remove it after fixing issue 69529
        // w/a for myriad (cann't store 2 caches simultaneously)
        PluginCache::get().reset();

        SKIP_IF_CURRENT_TEST_IS_DISABLED();
        OVClassNetworkTest::SetUp();
        deviceName = GetParam();
    }
};

using PriorityParams = std::tuple<
        std::string,            // Device name
        ov::AnyMap              // device priority Configuration key
>;
class OVClassExecutableNetworkGetMetricTest_Priority : public ::testing::Test, public ::testing::WithParamInterface<PriorityParams> {
protected:
    std::string deviceName;
    ov::AnyMap configuration;
    std::shared_ptr<ngraph::Function> simpleNetwork;

public:
    void SetUp() override {
        SKIP_IF_CURRENT_TEST_IS_DISABLED();
        std::tie(deviceName, configuration) = GetParam();
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
