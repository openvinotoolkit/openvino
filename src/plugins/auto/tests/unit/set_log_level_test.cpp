// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/auto_unit_test.hpp"
namespace {
void custom_unsetenv(const char* name) {
#ifdef _WIN32
    _putenv((std::string(name) + "=").c_str());
#else
    ::unsetenv(name);
#endif
}
}  // namespace

using ConfigParams = std::tuple<std::string, ov::AnyMap>;
using namespace ov::mock_auto_plugin;

class AutoSetLogLevel : public tests::AutoTest, public ::testing::TestWithParam<ConfigParams> {
public:
    static std::string getTestCaseName(testing::TestParamInfo<ConfigParams> obj) {
        std::string log_level;
        ov::AnyMap config;
        std::tie(log_level, config) = obj.param;
        std::ostringstream result;
        result << log_level;
        return result.str();
    }

    void SetUp() override {
        ON_CALL(*core,
                compile_model(::testing::Matcher<const std::shared_ptr<const ov::Model>&>(_),
                              ::testing::Matcher<const std::string&>(_),
                              ::testing::Matcher<const ov::AnyMap&>(_)))
            .WillByDefault(Return(mockExeNetwork));

        metaDevices = {{ov::test::utils::DEVICE_CPU, {}, -1}, {ov::test::utils::DEVICE_GPU, {}, -1}};
        // DeviceInformation devInfo;
        ON_CALL(*plugin, parse_meta_devices(_, _)).WillByDefault(Return(metaDevices));
        ON_CALL(*plugin, get_valid_device)
            .WillByDefault([](const std::vector<DeviceInformation>& metaDevices, const std::string& netPrecision) {
                std::list<DeviceInformation> devices(metaDevices.begin(), metaDevices.end());
                return devices;
            });
        ON_CALL(*plugin, select_device(_, _, _)).WillByDefault(Return(metaDevices[1]));
    }

    void TearDown() override {
        MockLog::release();
    }
};

TEST_P(AutoSetLogLevel, setLogLevelFromConfig) {
    custom_unsetenv("OPENVINO_LOG_LEVEL");
    std::string log_level;
    ov::AnyMap config;
    std::tie(log_level, config) = this->GetParam();
    plugin->set_device_name("AUTO");
    plugin->compile_model(model, config);
    int a = 0;
    DEBUG_RUN([&a]() {
        a++;
    });
    INFO_RUN([&a]() {
        a++;
    });
    if (log_level == "LOG_DEBUG" || log_level == "LOG_TRACE") {
        EXPECT_EQ(a, 2);
    } else if (log_level == "LOG_INFO") {
        EXPECT_EQ(a, 1);
    } else {
        EXPECT_EQ(a, 0);
    }
}

const std::vector<ConfigParams> testConfigs = {ConfigParams{"LOG_NONE", {{"LOG_LEVEL", "LOG_NONE"}}},
                                               ConfigParams{"LOG_ERROR", {{"LOG_LEVEL", "LOG_ERROR"}}},
                                               ConfigParams{"LOG_WARNING", {{"LOG_LEVEL", "LOG_WARNING"}}},
                                               ConfigParams{"LOG_INFO", {{"LOG_LEVEL", "LOG_INFO"}}},
                                               ConfigParams{"LOG_DEBUG", {{"LOG_LEVEL", "LOG_DEBUG"}}},
                                               ConfigParams{"LOG_TRACE", {{"LOG_LEVEL", "LOG_TRACE"}}}};

INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests,
                         AutoSetLogLevel,
                         ::testing::ValuesIn(testConfigs),
                         AutoSetLogLevel::getTestCaseName);
