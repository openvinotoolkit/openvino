// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/auto_unit_test.hpp"

using Config = std::map<std::string, std::string>;
using namespace ov::mock_auto_plugin;

using DeviceParams = std::tuple<std::string, bool>;

enum MODEL {
    GENERAL = 0,
    LATENCY = 1,
    THROUGHPUT = 2,
};

using ConfigParams = std::tuple<bool,                       // if can continue to run
                                bool,                       // if select throw exception
                                MODEL,                      // config model general, latency, throughput
                                std::vector<DeviceParams>,  // {device, loadSuccess}
                                unsigned int,               // select count
                                unsigned int,               // load count
                                unsigned int                // load device success count
                                >;

class AutoLoadFailedTest : public tests::AutoTest, public ::testing::TestWithParam<ConfigParams> {
public:
    static std::string getTestCaseName(testing::TestParamInfo<ConfigParams> obj) {
        unsigned int selectCount;
        unsigned int loadCount;
        unsigned int loadSuccessCount;
        std::vector<std::tuple<std::string, bool>> deviceConfigs;
        bool continueRun;
        bool thrExcWheSelect;
        MODEL configModel;
        std::tie(continueRun, thrExcWheSelect, configModel, deviceConfigs, selectCount, loadCount, loadSuccessCount) =
            obj.param;
        std::ostringstream result;
        for (auto& item : deviceConfigs) {
            if (std::get<1>(item)) {
                result << std::get<0>(item) << "_success_";
            } else {
                result << std::get<0>(item) << "_failed_";
            }
        }
        if (thrExcWheSelect) {
            result << "select_failed_";
        } else {
            result << "select_success_";
        }

        switch (configModel) {
        case GENERAL:
            result << "GENERAL";
            break;
        case LATENCY:
            result << "LATENCY";
            break;
        case THROUGHPUT:
            result << "THROUGHPUT";
            break;
        default:
            LOG_ERROR("should not come here");
            break;
        }

        result << "select_" << selectCount << "_loadCount_" << loadCount << "_loadSuccessCount_" << loadSuccessCount;
        return result.str();
    }
    void SetUp() override {
        unsigned int optimalNum = (uint32_t)2;
        ON_CALL(*mockIExeNet.get(), get_property(StrEq(ov::optimal_number_of_infer_requests.name())))
            .WillByDefault(Return(optimalNum));
    }
};

TEST_P(AutoLoadFailedTest, LoadCNNetWork) {
    // get Parameter
    unsigned int selectCount;
    unsigned int loadCount;
    unsigned int loadSuccessCount;
    std::vector<std::tuple<std::string, bool>> deviceConfigs;
    bool continueRun;
    bool thrExcWheSelect;
    MODEL configModel;
    std::tie(continueRun, thrExcWheSelect, configModel, deviceConfigs, selectCount, loadCount, loadSuccessCount) =
        this->GetParam();

    // test auto plugin
    plugin->set_device_name("AUTO");
    std::string devicesStr = "";
    auto selDevsSize = deviceConfigs.size();
    for (auto iter = deviceConfigs.begin(); iter != deviceConfigs.end(); selDevsSize--) {
        std::string deviceName = std::get<0>(*iter);
        bool loadSuccess = std::get<1>(*iter);
        // accoding to device loading config, set if the loading will successful or throw exception.
        if (loadSuccess) {
            ON_CALL(*core,
                    compile_model(::testing::Matcher<const std::shared_ptr<const ov::Model>&>(_),
                                  ::testing::Matcher<const std::string&>(StrEq(deviceName)),
                                  (_)))
                .WillByDefault(Return(mockExeNetwork));
        } else {
            ON_CALL(*core,
                    compile_model(::testing::Matcher<const std::shared_ptr<const ov::Model>&>(_),
                                  ::testing::Matcher<const std::string&>(StrEq(deviceName)),
                                  (_)))
                .WillByDefault(ov::Throw("compile error"));
        }
        DeviceInformation devInfo;
        switch (configModel) {
        case GENERAL:
            devInfo = {deviceName, {}, 2, ""};
            break;
        case LATENCY:
            devInfo = {deviceName,
                       {ov::hint::performance_mode("LATENCY"),
                        ov::hint::allow_auto_batching(true),
                        ov::auto_batch_timeout(1000)},
                       2,
                       ""};
            break;
        case THROUGHPUT:
            devInfo = {deviceName, {ov::hint::performance_mode("THROUGHPUT")}, 2, ""};
            break;
        default:
            LOG_ERROR("should not come here");
            break;
        }

        metaDevices.push_back(std::move(devInfo));
        // set the return value of SelectDevice
        // for example if there are three device, if will return GPU on the first call, and then NPU
        // at last CPU
        ON_CALL(*plugin, select_device(Property(&std::vector<DeviceInformation>::size, Eq(selDevsSize)), _, _))
            .WillByDefault(Return(metaDevices[deviceConfigs.size() - selDevsSize]));
        devicesStr += deviceName;
        devicesStr += ((++iter) == deviceConfigs.end()) ? "" : ",";
    }
    ON_CALL(*plugin, parse_meta_devices(_, _)).WillByDefault(Return(metaDevices));
    ON_CALL(*plugin, get_valid_device)
        .WillByDefault([](const std::vector<DeviceInformation>& metaDevices, const std::string& netPrecision) {
            std::list<DeviceInformation> devices(metaDevices.begin(), metaDevices.end());
            return devices;
        });
    config.insert(ov::device::priorities(devicesStr));
    // if set this parameter true, the second selecting call will thrown exception,
    // if there is only one device, it will thrown exception at the first call
    if (thrExcWheSelect) {
        selDevsSize = deviceConfigs.size();
        if (selDevsSize > 1) {
            ON_CALL(*plugin, select_device(Property(&std::vector<DeviceInformation>::size, Eq(selDevsSize - 1)), _, _))
                .WillByDefault(ov::Throw(""));
        } else {
            ON_CALL(*plugin, select_device(Property(&std::vector<DeviceInformation>::size, Eq(1)), _, _))
                .WillByDefault(ov::Throw(""));
        }
    }

    EXPECT_CALL(*plugin, parse_meta_devices(_, _)).Times(AtLeast(1));
    EXPECT_CALL(*plugin, select_device(_, _, _)).Times(selectCount);
    EXPECT_CALL(*core,
                compile_model(::testing::Matcher<const std::shared_ptr<const ov::Model>&>(_),
                              ::testing::Matcher<const std::string&>(_),
                              ::testing::Matcher<const ov::AnyMap&>(_)))
        .Times(loadCount);

    // if loadSuccess will get the optimalNum requset of per device, in this test is 2;
    EXPECT_CALL(*mockIExeNet.get(), get_property(StrEq(ov::optimal_number_of_infer_requests.name())))
        .Times(loadSuccessCount);
    EXPECT_CALL(*mockIExeNet.get(), create_infer_request()).Times(loadSuccessCount * 2);
    if (continueRun) {
        OV_ASSERT_NO_THROW(plugin->compile_model(model, config));
    } else {
        ASSERT_THROW(plugin->compile_model(model, config), ov::Exception);
    }
}

// the test configure, for example
// ConfigParams {true, false,  GENERAL, {DeviceParams {ov::test::utils::DEVICE_GPU, false},
//               DeviceParams {"OTHER", true},
//                DeviceParams {ov::test::utils::DEVICE_CPU, true}}, 2, 3, 2},
//
// every element for ConfigParams
// {continueRun, selectThrowException,  config model,  deviceLoadsuccessVector, selectCount, loadCount,
// loadSuccessCount} {       true,                false,       GENERAL,                 3 device,           2, 3, 2}
//
// there are three devices for loading
// CPU load for accelerator success, but GPU will load faild and then select NPU and load again
// LoadExeNetworkImpl will not throw exception and can continue to run,
// it will select twice, first select GPU, second select NPU
// it will load network three times(CPU, GPU, NPU)
// the inference request num is loadSuccessCount * optimalNum, in this test case optimalNum is 2
// so inference request num is 4 (CPU 2, NPU 2)
//
const std::vector<ConfigParams> testConfigs = {
    ConfigParams{true,
                 false,
                 GENERAL,
                 {DeviceParams{ov::test::utils::DEVICE_GPU, true},
                  DeviceParams{"OTHER", true},
                  DeviceParams{ov::test::utils::DEVICE_CPU, true}},
                 1,
                 2,
                 2},
    ConfigParams{true,
                 false,
                 GENERAL,
                 {DeviceParams{ov::test::utils::DEVICE_GPU, false},
                  DeviceParams{"OTHER", true},
                  DeviceParams{ov::test::utils::DEVICE_CPU, true}},
                 2,
                 3,
                 2},
    ConfigParams{true,
                 false,
                 GENERAL,
                 {DeviceParams{ov::test::utils::DEVICE_GPU, true},
                  DeviceParams{"OTHER", false},
                  DeviceParams{ov::test::utils::DEVICE_CPU, true}},
                 1,
                 2,
                 2},
    ConfigParams{true,
                 false,
                 GENERAL,
                 {DeviceParams{ov::test::utils::DEVICE_GPU, true},
                  DeviceParams{"OTHER", true},
                  DeviceParams{ov::test::utils::DEVICE_CPU, false}},
                 1,
                 2,
                 1},
    ConfigParams{true,
                 false,
                 GENERAL,
                 {DeviceParams{ov::test::utils::DEVICE_GPU, true},
                  DeviceParams{"OTHER", false},
                  DeviceParams{ov::test::utils::DEVICE_CPU, false}},
                 1,
                 2,
                 1},
    ConfigParams{true,
                 false,
                 GENERAL,
                 {DeviceParams{ov::test::utils::DEVICE_GPU, false},
                  DeviceParams{"OTHER", true},
                  DeviceParams{ov::test::utils::DEVICE_CPU, false}},
                 2,
                 3,
                 1},
    ConfigParams{true,
                 false,
                 GENERAL,
                 {DeviceParams{ov::test::utils::DEVICE_GPU, false},
                  DeviceParams{"OTHER", false},
                  DeviceParams{ov::test::utils::DEVICE_CPU, true}},
                 3,
                 4,
                 2},
    ConfigParams{false,
                 false,
                 GENERAL,
                 {DeviceParams{ov::test::utils::DEVICE_GPU, false},
                  DeviceParams{"OTHER", false},
                  DeviceParams{ov::test::utils::DEVICE_CPU, false}},
                 3,
                 4,
                 0},
    ConfigParams{true,
                 false,
                 GENERAL,
                 {DeviceParams{ov::test::utils::DEVICE_GPU, true}, DeviceParams{ov::test::utils::DEVICE_CPU, true}},
                 1,
                 2,
                 2},
    ConfigParams{true,
                 false,
                 GENERAL,
                 {DeviceParams{ov::test::utils::DEVICE_GPU, false}, DeviceParams{ov::test::utils::DEVICE_CPU, true}},
                 2,
                 3,
                 2},
    ConfigParams{true,
                 false,
                 GENERAL,
                 {DeviceParams{ov::test::utils::DEVICE_GPU, true}, DeviceParams{ov::test::utils::DEVICE_CPU, false}},
                 1,
                 2,
                 1},
    ConfigParams{false,
                 false,
                 GENERAL,
                 {DeviceParams{ov::test::utils::DEVICE_GPU, false}, DeviceParams{ov::test::utils::DEVICE_CPU, false}},
                 2,
                 3,
                 0},
    ConfigParams{false, false, GENERAL, {DeviceParams{ov::test::utils::DEVICE_GPU, false}}, 1, 1, 0},
    ConfigParams{false, false, GENERAL, {DeviceParams{ov::test::utils::DEVICE_CPU, false}}, 1, 1, 0},
    ConfigParams{true, false, GENERAL, {DeviceParams{ov::test::utils::DEVICE_GPU, true}}, 1, 1, 1},
    ConfigParams{true, false, GENERAL, {DeviceParams{ov::test::utils::DEVICE_CPU, true}}, 1, 1, 1},
    ConfigParams{false, true, GENERAL, {DeviceParams{ov::test::utils::DEVICE_GPU, true}}, 1, 0, 0},
    ConfigParams{false, true, GENERAL, {DeviceParams{ov::test::utils::DEVICE_CPU, true}}, 1, 0, 0},
    ConfigParams{true,
                 true,
                 GENERAL,
                 {DeviceParams{ov::test::utils::DEVICE_GPU, false},
                  DeviceParams{"OTHER", true},
                  DeviceParams{ov::test::utils::DEVICE_CPU, true}},
                 2,
                 2,
                 1},
    ConfigParams{false,
                 true,
                 GENERAL,
                 {DeviceParams{ov::test::utils::DEVICE_GPU, false},
                  DeviceParams{"OTHER", true},
                  DeviceParams{ov::test::utils::DEVICE_CPU, false}},
                 2,
                 2,
                 0},
    ConfigParams{true,
                 true,
                 GENERAL,
                 {DeviceParams{ov::test::utils::DEVICE_GPU, false}, DeviceParams{ov::test::utils::DEVICE_CPU, true}},
                 2,
                 2,
                 1},
    ConfigParams{true,
                 false,
                 LATENCY,
                 {DeviceParams{ov::test::utils::DEVICE_GPU, false},
                  DeviceParams{"OTHER", false},
                  DeviceParams{ov::test::utils::DEVICE_CPU, true}},
                 3,
                 3,
                 1},
    ConfigParams{true,
                 false,
                 LATENCY,
                 {DeviceParams{ov::test::utils::DEVICE_GPU, false}, DeviceParams{ov::test::utils::DEVICE_CPU, true}},
                 2,
                 2,
                 1},
    ConfigParams{true,
                 false,
                 THROUGHPUT,
                 {DeviceParams{ov::test::utils::DEVICE_GPU, false},
                  DeviceParams{"OTHER", false},
                  DeviceParams{ov::test::utils::DEVICE_CPU, true}},
                 3,
                 4,
                 2},
    ConfigParams{true,
                 false,
                 THROUGHPUT,
                 {DeviceParams{ov::test::utils::DEVICE_GPU, false}, DeviceParams{ov::test::utils::DEVICE_CPU, true}},
                 2,
                 3,
                 2}};

INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests,
                         AutoLoadFailedTest,
                         ::testing::ValuesIn(testConfigs),
                         AutoLoadFailedTest::getTestCaseName);
