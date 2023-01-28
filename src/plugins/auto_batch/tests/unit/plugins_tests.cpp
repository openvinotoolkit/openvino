// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "mock_auto_batch_plugin.hpp"
#include "unit_test_utils/mocks/cpp_interfaces/impl/mock_inference_plugin_internal.hpp"
#include "unit_test_utils/mocks/cpp_interfaces/interface/mock_icore.hpp"
#include "unit_test_utils/mocks/cpp_interfaces/interface/mock_iinference_plugin.hpp"

using ::testing::_;
using ::testing::AnyNumber;
using ::testing::AtLeast;
using ::testing::Eq;
using ::testing::NiceMock;
using ::testing::Return;
using ::testing::ReturnRef;
using ::testing::StrEq;
using ::testing::StrNe;
using ::testing::Throw;
using namespace MockAutoBatchPlugin;
using namespace MockAutoBatchDevice;
using BatchDeviceConfigParams = std::tuple<std::string,  // Batch devices
                                           std::string,  // Expected device name
                                           int,          // Expected batch size
                                           bool          // Throw exception
                                           >;
using MetricConfigParams = std::tuple<std::string, std::string, bool>;
using MetaDeviceParams = std::tuple<std::string,                           // Device batch cfg
                                    std::map<std::string, std::string>,    // Config
                                    DeviceInformation,                     // Expected result
                                    bool>;                                 // Throw exception
using SetGetConfigParams = std::tuple<std::map<std::string, std::string>,  // Set Config
                                      std::string,                         // Get Config
                                      bool>;                               // Throw exception

const std::vector<std::string> cpu_supported_properties = {
    "CACHE_DIR",
};
const std::vector<std::string> gpu_supported_properties = {
    "CACHE_DIR",
    "OPTIMAL_BATCH_SIZE",
};

class SetGetConfigTest : public ::testing::TestWithParam<SetGetConfigParams> {
public:
    std::shared_ptr<NiceMock<MockICore>> core;
    std::shared_ptr<NiceMock<MockAutoBatchInferencePlugin>> plugin;

public:
    static std::string getTestCaseName(testing::TestParamInfo<SetGetConfigParams> obj) {
        std::map<std::string, std::string> set_config;
        std::string get_config;
        bool throw_exception;

        std::tie(set_config, get_config, throw_exception) = obj.param;
        std::string res = "";
        if (set_config.size() > 0) {
            res += "GetConfig_";
            for (auto& it : set_config) {
                res += it.first + "_" + it.second + "_";
            }
        }
        if (!get_config.empty()) {
            res += "GetConfig_" + get_config;
        }
        if (throw_exception)
            res += "_throw";
        return res;
    }

    void TearDown() override {
        core.reset();
        plugin.reset();
    }

    void SetUp() override {
        core = std::shared_ptr<NiceMock<MockICore>>(new NiceMock<MockICore>());
        plugin = std::shared_ptr<NiceMock<MockAutoBatchInferencePlugin>>(new NiceMock<MockAutoBatchInferencePlugin>());
        plugin->SetCore(core);

        ON_CALL(*plugin, ParseBatchDevice).WillByDefault([this](const std::string& batchDevice) {
            return plugin->AutoBatchInferencePlugin::ParseBatchDevice(batchDevice);
        });
    }
};

TEST_P(SetGetConfigTest, SetConfigTestCase) {
    std::map<std::string, std::string> set_config;
    std::string temp;
    bool throw_exception;
    std::tie(set_config, temp, throw_exception) = this->GetParam();

    if (set_config.size() == 0) {
        ASSERT_NO_THROW(plugin->SetConfig(set_config));
        return;
    }

    if (throw_exception) {
        ASSERT_ANY_THROW(plugin->SetConfig(set_config));
    } else {
        ASSERT_NO_THROW(plugin->SetConfig(set_config));
    }
}

TEST_P(SetGetConfigTest, GetConfigTestCase) {
    std::map<std::string, std::string> temp;
    std::string get_config;
    bool throw_exception;
    std::tie(temp, get_config, throw_exception) = this->GetParam();

    if (get_config.empty() || temp.size() > 0) {
        return;
    }

    std::map<std::string, InferenceEngine::Parameter> options = {};
    if (throw_exception) {
        ASSERT_ANY_THROW(plugin->GetConfig(get_config, options));
    } else {
        ASSERT_NO_THROW(plugin->GetConfig(get_config, options));
    }
}

TEST_P(SetGetConfigTest, SetGetConfigTestCase) {
    std::map<std::string, std::string> set_config;
    std::string get_config;
    bool throw_exception;
    std::tie(set_config, get_config, throw_exception) = this->GetParam();

    if (get_config.empty() || set_config.size() == 0) {
        return;
    }

    std::map<std::string, InferenceEngine::Parameter> options = {};
    ASSERT_NO_THROW(plugin->SetConfig(set_config));
    InferenceEngine::Parameter result;
    ASSERT_NO_THROW(result = plugin->GetConfig(get_config, options));
    EXPECT_EQ(result.as<std::string>(), set_config[get_config]);
}

class ParseMetaDeviceTest : public ::testing::TestWithParam<MetaDeviceParams> {
public:
    std::shared_ptr<NiceMock<MockICore>> core;
    std::shared_ptr<NiceMock<MockAutoBatchInferencePlugin>> plugin;

public:
    static std::string getTestCaseName(testing::TestParamInfo<MetaDeviceParams> obj) {
        std::string batch_cfg;
        std::map<std::string, std::string> config;
        DeviceInformation info;
        bool throw_exception;

        std::tie(batch_cfg, config, info, throw_exception) = obj.param;
        std::string res = batch_cfg;
        for (auto& c : config) {
            res += "_" + c.first + "_" + c.second;
        }
        if (throw_exception)
            res += "_throw";
        return res;
    }

    void TearDown() override {
        core.reset();
        plugin.reset();
    }

    void SetUp() override {
        core = std::shared_ptr<NiceMock<MockICore>>(new NiceMock<MockICore>());
        plugin = std::shared_ptr<NiceMock<MockAutoBatchInferencePlugin>>(new NiceMock<MockAutoBatchInferencePlugin>());
        plugin->SetCore(core);

        ON_CALL(*core, GetSupportedConfig)
            .WillByDefault([](const std::string& device, const std::map<std::string, std::string>& configs) {
                std::map<std::string, std::string> res_config;
                if (device == "CPU") {
                    for (auto& c : configs) {
                        if (std::find(begin(cpu_supported_properties), end(cpu_supported_properties), c.first) !=
                            cpu_supported_properties.end())
                            res_config[c.first] = c.second;
                    }
                } else if (device == "GPU") {
                    for (auto& c : configs) {
                        if (std::find(begin(gpu_supported_properties), end(gpu_supported_properties), c.first) !=
                            gpu_supported_properties.end())
                            res_config[c.first] = c.second;
                    }
                }
                return res_config;
            });

        ON_CALL(*plugin, ParseBatchDevice).WillByDefault([this](const std::string& batchDevice) {
            return plugin->AutoBatchInferencePlugin::ParseBatchDevice(batchDevice);
        });
    }

    bool compare(std::map<std::string, std::string> a, std::map<std::string, std::string> b) {
        if (a.size() != b.size())
            return false;

        for (auto& it : a) {
            auto item = b.find(it.first);
            if (item == b.end())
                return false;
            if (it.second != item->second)
                return false;
        }
        return true;
    }
};

TEST_P(ParseMetaDeviceTest, ParseMetaDeviceTestCase) {
    std::string batch_cfg;
    std::map<std::string, std::string> config;
    DeviceInformation expected;
    bool throw_exception;

    std::tie(batch_cfg, config, expected, throw_exception) = this->GetParam();

    if (throw_exception) {
        ASSERT_ANY_THROW(plugin->ParseMetaDevice(batch_cfg, config));
    } else {
        auto result = plugin->ParseMetaDevice(batch_cfg, config);
        EXPECT_EQ(result.deviceName, expected.deviceName);
        EXPECT_EQ(result.batchForDevice, expected.batchForDevice);
        EXPECT_TRUE(compare(result.config, expected.config));
    }
}

class ParseBatchDeviceTest : public ::testing::TestWithParam<BatchDeviceConfigParams> {
public:
    std::shared_ptr<NiceMock<MockICore>> core;
    std::shared_ptr<NiceMock<MockAutoBatchInferencePlugin>> plugin;

public:
    static std::string getTestCaseName(testing::TestParamInfo<BatchDeviceConfigParams> obj) {
        std::string batchDevice;
        std::string deviceName;
        int batchSize;
        bool throw_exception;
        std::tie(batchDevice, deviceName, batchSize, throw_exception) = obj.param;
        return batchDevice;
    }

    void TearDown() override {
        core.reset();
        plugin.reset();
    }

    void SetUp() override {
        core = std::shared_ptr<NiceMock<MockICore>>(new NiceMock<MockICore>());
        plugin = std::shared_ptr<NiceMock<MockAutoBatchInferencePlugin>>(new NiceMock<MockAutoBatchInferencePlugin>());
        plugin->SetCore(core);

        ON_CALL(*plugin, ParseBatchDevice).WillByDefault([this](const std::string& batchDevice) {
            return plugin->AutoBatchInferencePlugin::ParseBatchDevice(batchDevice);
        });
    }
};

TEST_P(ParseBatchDeviceTest, ParseBatchDeviceTestCase) {
    std::string batchDevice;
    std::string deviceName;
    int batchSize;
    bool throw_exception;
    std::tie(batchDevice, deviceName, batchSize, throw_exception) = this->GetParam();

    if (throw_exception) {
        ASSERT_ANY_THROW(plugin->ParseBatchDevice(batchDevice));
    } else {
        auto result = plugin->ParseBatchDevice(batchDevice);
        EXPECT_EQ(result.deviceName, deviceName);
        EXPECT_EQ(result.batchForDevice, batchSize);
    }
}

class PluginMetricTest : public ::testing::TestWithParam<MetricConfigParams> {
public:
    std::shared_ptr<NiceMock<MockICore>> core;
    std::shared_ptr<NiceMock<MockAutoBatchInferencePlugin>> plugin;

public:
    static std::string getTestCaseName(testing::TestParamInfo<MetricConfigParams> obj) {
        std::string metricName;
        std::string value;
        bool throw_exception;
        std::tie(metricName, value, throw_exception) = obj.param;
        return "Metric_" + metricName;
    }

    void TearDown() override {
        core.reset();
        plugin.reset();
    }

    void SetUp() override {
        core = std::shared_ptr<NiceMock<MockICore>>(new NiceMock<MockICore>());
        plugin = std::shared_ptr<NiceMock<MockAutoBatchInferencePlugin>>(new NiceMock<MockAutoBatchInferencePlugin>());
        plugin->SetCore(core);

        ON_CALL(*plugin, GetMetric)
            .WillByDefault(
                [this](const std::string& name, const std::map<std::string, InferenceEngine::Parameter>& options) {
                    return plugin->AutoBatchInferencePlugin::GetMetric(name, options);
                });
    }
};

TEST_P(PluginMetricTest, GetPluginMetricTest) {
    std::string metricName;
    std::string expected;
    bool throw_exception;
    std::tie(metricName, expected, throw_exception) = this->GetParam();

    if (throw_exception) {
        ASSERT_ANY_THROW(plugin->GetMetric(metricName, {}));
    } else {
        auto value = plugin->GetMetric(metricName, {});
        EXPECT_EQ(value.as<std::string>(), expected);
    }
}

const char supported_metric[] = "SUPPORTED_METRICS FULL_DEVICE_NAME SUPPORTED_CONFIG_KEYS";
const char supported_config_keys[] = "AUTO_BATCH_DEVICE_CONFIG AUTO_BATCH_TIMEOUT CACHE_DIR";

const std::vector<BatchDeviceConfigParams> batchDeviceTestConfigs = {
    BatchDeviceConfigParams{"CPU(4)", "CPU", 4, false},
    BatchDeviceConfigParams{"GPU(8)", "GPU", 8, false},
    BatchDeviceConfigParams{"CPU(0)", "CPU", 0, true},
    BatchDeviceConfigParams{"GPU(-1)", "GPU", 0, true},
};

const std::vector<MetricConfigParams> metricTestConfigs = {
    MetricConfigParams{METRIC_KEY(SUPPORTED_METRICS), supported_metric, false},
    MetricConfigParams{METRIC_KEY(FULL_DEVICE_NAME), "BATCH", false},
    MetricConfigParams{METRIC_KEY(SUPPORTED_CONFIG_KEYS), supported_config_keys, false},
    MetricConfigParams{"CPU_THREADS_NUM", "16", true},
    MetricConfigParams{"PERFORMANCE_HINT", "LATENCY", true},
};

const std::vector<MetaDeviceParams> testMetaDeviceConfigs = {
    MetaDeviceParams{"CPU(4)", {}, DeviceInformation{"CPU", {}, 4}, false},
    MetaDeviceParams{"CPU(4)", {{}}, DeviceInformation{"CPU", {{}}, 4}, true},
    MetaDeviceParams{"CPU(4)", {{"CACHE_DIR", "./"}}, DeviceInformation{"CPU", {{"CACHE_DIR", "./"}}, 4}, false},
    MetaDeviceParams{"GPU(4)", {{"CACHE_DIR", "./"}}, DeviceInformation{"GPU", {{"CACHE_DIR", "./"}}, 4}, false},
    MetaDeviceParams{"GPU(8)",
                     {{"CACHE_DIR", "./"}, {"OPTIMAL_BATCH_SIZE", "16"}},
                     DeviceInformation{"GPU", {{"CACHE_DIR", "./"}, {"OPTIMAL_BATCH_SIZE", "16"}}, 8},
                     false},
    MetaDeviceParams{"CPU(4)", {{"OPTIMAL_BATCH_SIZE", "16"}}, DeviceInformation{"CPU", {{}}, 4}, true},
    MetaDeviceParams{"CPU(4)",
                     {{"CACHE_DIR", "./"}, {"OPTIMAL_BATCH_SIZE", "16"}},
                     DeviceInformation{"CPU", {{"CACHE_DIR", "./"}}, 4},
                     true},
};

const std::vector<SetGetConfigParams> testSetGetConfigParams = {
    // Set Config
    SetGetConfigParams{{{"AUTO_BATCH_TIMEOUT", "200"}}, {}, false},
    SetGetConfigParams{{{"AUTO_BATCH_DEVICE_CONFIG", "CPU(4)"}}, {}, false},
    SetGetConfigParams{{{"CACHE_DIR", "./xyz"}}, {}, false},
    SetGetConfigParams{{{"AUTO_BATCH_TIMEOUT", "200"}, {"AUTO_BATCH_DEVICE_CONFIG", "CPU(4)"}}, {}, false},
    SetGetConfigParams{{{"AUTO_BATCH_TIMEOUT", "200"}, {"AUTO_BATCH_DEVICE_CONFIG", "CPU(4)"}, {"CACHE_DIR", "./xyz"}},
                       {},
                       false},
    SetGetConfigParams{{{"XYZ", "200"}}, {}, true},
    SetGetConfigParams{{{"XYZ", "200"}, {"AUTO_BATCH_DEVICE_CONFIG", "CPU(4)"}, {"CACHE_DIR", "./xyz"}}, {}, true},
    // Get Config
    SetGetConfigParams{{}, "AUTO_BATCH_TIMEOUT", false},
    SetGetConfigParams{{}, "AUTO_BATCH_DEVICE_CONFIG", true},
    SetGetConfigParams{{}, "CACHE_DIR", true},
    // Set and get Config
    SetGetConfigParams{{{"AUTO_BATCH_TIMEOUT", "200"}}, "AUTO_BATCH_TIMEOUT", false},
    SetGetConfigParams{{{"AUTO_BATCH_DEVICE_CONFIG", "CPU(4)"}}, "AUTO_BATCH_DEVICE_CONFIG", false},
    SetGetConfigParams{{{"CACHE_DIR", "./abc"}}, "CACHE_DIR", false},
};

INSTANTIATE_TEST_SUITE_P(smoke_AutoBatch_BehaviorTests,
                         SetGetConfigTest,
                         ::testing::ValuesIn(testSetGetConfigParams),
                         SetGetConfigTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_AutoBatch_BehaviorTests,
                         ParseBatchDeviceTest,
                         ::testing::ValuesIn(batchDeviceTestConfigs),
                         ParseBatchDeviceTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_AutoBatch_BehaviorTests,
                         PluginMetricTest,
                         ::testing::ValuesIn(metricTestConfigs),
                         PluginMetricTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_AutoBatch_BehaviorTests,
                         ParseMetaDeviceTest,
                         ::testing::ValuesIn(testMetaDeviceConfigs),
                         ParseMetaDeviceTest::getTestCaseName);