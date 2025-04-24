// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mock_common.hpp"

using batch_device_config_params = std::tuple<std::string,  // Batch devices
                                              std::string,  // Expected device name
                                              int,          // Expected batch size
                                              bool          // Throw exception
                                              >;

class ParseBatchDeviceTest : public ::testing::TestWithParam<batch_device_config_params> {
public:
    std::string m_batch_device_config;
    std::string m_device_name;
    int m_batch_size;
    bool m_throw_exception;
    std::shared_ptr<NiceMock<MockAutoBatchInferencePlugin>> m_plugin;

public:
    static std::string getTestCaseName(testing::TestParamInfo<batch_device_config_params> obj) {
        std::string batch_device_config;
        std::string device_name;
        int batch_size;
        bool throw_exception;
        std::tie(batch_device_config, device_name, batch_size, throw_exception) = obj.param;
        std::string res = batch_device_config;
        if (throw_exception)
            res += "_throw";
        return res;
    }

    void TearDown() override {
        m_plugin.reset();
    }

    void SetUp() override {
        std::tie(m_batch_device_config, m_device_name, m_batch_size, m_throw_exception) = this->GetParam();
        m_plugin =
            std::shared_ptr<NiceMock<MockAutoBatchInferencePlugin>>(new NiceMock<MockAutoBatchInferencePlugin>());
    }
};

TEST_P(ParseBatchDeviceTest, ParseBatchDeviceTestCase) {
    if (m_throw_exception) {
        ASSERT_ANY_THROW(m_plugin->parse_batch_device(m_batch_device_config));
    } else {
        auto result = m_plugin->parse_batch_device(m_batch_device_config);
        EXPECT_EQ(result.device_name, m_device_name);
        EXPECT_EQ(result.device_batch_size, m_batch_size);
    }
}

const std::vector<batch_device_config_params> batch_device_test_configs = {
    batch_device_config_params{"CPU(4)", "CPU", 4, false},
    batch_device_config_params{"GPU(8)", "GPU", 8, false},
    batch_device_config_params{"CPU(0)", "CPU", 0, true},
    batch_device_config_params{"GPU(-1)", "GPU", 0, true},
};

INSTANTIATE_TEST_SUITE_P(smoke_AutoBatch_BehaviorTests,
                         ParseBatchDeviceTest,
                         ::testing::ValuesIn(batch_device_test_configs),
                         ParseBatchDeviceTest::getTestCaseName);