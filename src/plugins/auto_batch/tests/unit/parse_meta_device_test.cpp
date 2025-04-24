// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mock_common.hpp"
#include "unit_test_utils/mocks/openvino/runtime/mock_icore.hpp"

using meta_device_params = std::tuple<std::string,        // Device batch cfg
                                      ov::AnyMap,         // property map
                                      DeviceInformation,  // Expected result
                                      bool>;              // Throw exception

const std::vector<std::string> cpu_supported_properties = {
    ov::cache_dir.name(),
};

const std::vector<std::string> gpu_supported_properties = {
    ov::cache_dir.name(),
    ov::optimal_batch_size.name(),
};

class ParseMetaDeviceTest : public ::testing::TestWithParam<meta_device_params> {
public:
    std::shared_ptr<NiceMock<ov::MockICore>> m_core;
    std::shared_ptr<NiceMock<MockAutoBatchInferencePlugin>> m_plugin;

    std::string m_batch_cfg;
    ov::AnyMap m_config;
    DeviceInformation m_expected_device_info;
    bool m_throw_exception;

public:
    static std::string getTestCaseName(testing::TestParamInfo<meta_device_params> obj) {
        std::string batch_cfg;
        ov::AnyMap config;
        DeviceInformation info;
        bool throw_exception;

        std::tie(batch_cfg, config, info, throw_exception) = obj.param;
        std::string res = batch_cfg;
        for (auto& c : config) {
            res += "_" + c.first + "_" + c.second.as<std::string>();
        }
        if (throw_exception)
            res += "_throw";
        return res;
    }

    void TearDown() override {
        m_core.reset();
        m_plugin.reset();
    }

    void SetUp() override {
        m_core = std::shared_ptr<NiceMock<ov::MockICore>>(new NiceMock<ov::MockICore>());
        m_plugin =
            std::shared_ptr<NiceMock<MockAutoBatchInferencePlugin>>(new NiceMock<MockAutoBatchInferencePlugin>());
        m_plugin->set_core(m_core);

        std::tie(m_batch_cfg, m_config, m_expected_device_info, m_throw_exception) = this->GetParam();

        ON_CALL(*m_core, get_supported_property)
            .WillByDefault([](const std::string& device, const ov::AnyMap& configs, const bool keep_core_property) {
                ov::AnyMap res_config;
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
    }

    bool compare(ov::AnyMap a, ov::AnyMap b) {
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
    if (m_throw_exception) {
        ASSERT_ANY_THROW(m_plugin->parse_meta_device(m_batch_cfg, m_config));
    } else {
        auto result = m_plugin->parse_meta_device(m_batch_cfg, m_config);
        EXPECT_EQ(result.device_name, m_expected_device_info.device_name);
        EXPECT_EQ(result.device_batch_size, m_expected_device_info.device_batch_size);
        EXPECT_TRUE(compare(result.device_config, m_expected_device_info.device_config));
    }
}

const std::vector<meta_device_params> meta_device_test_configs = {
    meta_device_params{"CPU(4)", {}, DeviceInformation{"CPU", {}, 4}, false},
    meta_device_params{"CPU(4)", {{}}, DeviceInformation{"CPU", {{}}, 4}, true},
    meta_device_params{"CPU(4)", {{ov::cache_dir("./")}}, DeviceInformation{"CPU", {{ov::cache_dir("./")}}, 4}, false},
    meta_device_params{"GPU(4)", {{ov::cache_dir("./")}}, DeviceInformation{"GPU", {{ov::cache_dir("./")}}, 4}, false},
    meta_device_params{"GPU(8)",
                       {{ov::cache_dir("./")}, {ov::optimal_batch_size.name(), "16"}},
                       DeviceInformation{"GPU", {{ov::cache_dir("./")}, {ov::optimal_batch_size.name(), "16"}}, 8},
                       false},
    meta_device_params{"CPU(4)", {{ov::optimal_batch_size.name(), "16"}}, DeviceInformation{"CPU", {{}}, 4}, true},
    meta_device_params{"CPU(4)",
                       {{ov::cache_dir("./")}, {ov::optimal_batch_size.name(), "16"}},
                       DeviceInformation{"CPU", {{ov::cache_dir("./")}}, 4},
                       true},
};

INSTANTIATE_TEST_SUITE_P(smoke_AutoBatch_BehaviorTests,
                         ParseMetaDeviceTest,
                         ::testing::ValuesIn(meta_device_test_configs),
                         ParseMetaDeviceTest::getTestCaseName);
