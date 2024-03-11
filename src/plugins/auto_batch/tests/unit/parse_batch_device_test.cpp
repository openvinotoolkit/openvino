// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mock_common.hpp"

using batch_device_config_params = std::tuple<std::string,  // Batch devices
                                              ov::AnyMap,   // Properties contains ov::device::properties
                                              int,          // Expected batch size
                                              bool          // Throw exception
                                              >;

class ParseBatchDeviceTest : public ::testing::TestWithParam<batch_device_config_params> {
public:
    std::string m_device_name;
    ov::AnyMap properties;
    int m_batch_size;
    bool m_throw_exception;
    std::shared_ptr<NiceMock<MockAutoBatchInferencePlugin>> m_plugin;

public:
    static std::string getTestCaseName(testing::TestParamInfo<batch_device_config_params> obj) {
        ov::AnyMap properties;
        std::string device_name;
        int batch_size;
        bool throw_exception;
        std::tie(device_name, properties, batch_size, throw_exception) = obj.param;
        std::string res = device_name;
        for (const auto& it : properties) {
            res += "_" + it.first + "_" + it.second.as<std::string>();
        }
        res += "_" + std::to_string(batch_size);
        if (throw_exception)
            res += "_throw";
        return res;
    }

    void TearDown() override {
        m_plugin.reset();
    }

    void SetUp() override {
        std::tie(m_device_name, properties, m_batch_size, m_throw_exception) = this->GetParam();
        m_plugin =
            std::shared_ptr<NiceMock<MockAutoBatchInferencePlugin>>(new NiceMock<MockAutoBatchInferencePlugin>());
    }
};

TEST_P(ParseBatchDeviceTest, ParseBatchDeviceTestCase) {
    if (m_throw_exception) {
        ASSERT_ANY_THROW(m_plugin->parse_batch_size(m_device_name, properties));
    } else {
        auto result = m_plugin->parse_batch_size(m_device_name, properties);
        EXPECT_EQ(result, m_batch_size);
    }
}

auto DeviceProperties = [](const std::string& device_name, const uint32_t batch_size) {
    auto prop = ov::AnyMap({{device_name, ov::AnyMap({ov::hint::num_requests(batch_size)})}});
    return ov::AnyMap({{ov::device::properties.name(), prop}});
};

const std::vector<batch_device_config_params>
    batch_device_test_configs = {
        batch_device_config_params{"CPU", DeviceProperties("CPU", 4), 4, false},
        batch_device_config_params{"CPU", DeviceProperties("CPU", -1), -1, true},
        batch_device_config_params{"CPU", DeviceProperties("CPU", 0), 0, true},
        batch_device_config_params{"CPU", {}, 0, false},
};

INSTANTIATE_TEST_SUITE_P(smoke_AutoBatch_BehaviorTests,
                         ParseBatchDeviceTest,
                         ::testing::ValuesIn(batch_device_test_configs),
                         ParseBatchDeviceTest::getTestCaseName);