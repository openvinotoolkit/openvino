// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <thread>

#include "include/auto_unit_test.hpp"
#include "openvino/runtime/properties.hpp"

using Config = std::map<std::string, std::string>;
using namespace ov::mock_auto_plugin;

using ConfigParams = std::tuple<bool,          // if THROUGHPUT
                                unsigned int,  // cpu OPTIMAL_NUMBER_OF_INFER_REQUESTS
                                int,           // cpu infer requet num of customer want
                                bool,          // if cpu sleep, cpu device will load slow
                                unsigned int,  // Actual device OPTIMAL_NUMBER_OF_INFER_REQUESTS
                                int,           // Actual device infer requet num of customer want
                                bool,          // if Actual device sleep, cpu device will load slow
                                std::string,   // Actual Device Name
                                unsigned int,  // expect OPTIMAL_NUMBER_OF_INFER_REQUESTS
                                int,           // Actual PERFORMANCE_HINT_NUM_REQUESTS
                                bool,          // if actual device supported OptimalNum
                                bool           // if actual device supported ov::hint::requests
                                >;
class ExecNetworkget_propertyOptimalNumInferReq : public tests::AutoTest,
                                                  public ::testing::TestWithParam<ConfigParams> {
public:
    static std::string getTestCaseName(testing::TestParamInfo<ConfigParams> obj) {
        unsigned int cpuOptimalNum;
        int cpuCustomerNum;
        unsigned int actualOptimalNum;
        int actualCustomerNum;
        unsigned int expectOptimalNum;
        bool cpuSleep;
        bool actualSleep;
        bool isThroughput;
        int gpuPerfHintNum;
        bool isSupportOptimalNum;
        bool isSupportNumRequests;
        std::string actualDeviceName;
        std::tie(isThroughput,
                 cpuOptimalNum,
                 cpuCustomerNum,
                 cpuSleep,
                 actualOptimalNum,
                 actualCustomerNum,
                 actualSleep,
                 actualDeviceName,
                 expectOptimalNum,
                 gpuPerfHintNum,
                 isSupportOptimalNum,
                 isSupportNumRequests) = obj.param;
        std::ostringstream result;
        result << "cpuOptimalNum_" << cpuOptimalNum << "cpuCustomerNum_" << cpuCustomerNum;
        result << "actualOptimalNum_" << actualOptimalNum << "actualCustomerNum_" << actualCustomerNum;
        result << "expectOptimalNum_" << expectOptimalNum;
        if (isThroughput) {
            result << "_isThroughput"
                   << "true";
        } else {
            result << "__isThroughput"
                   << "false";
        }
        if (cpuSleep) {
            result << "_cpuSleep_"
                   << "true";
        } else {
            result << "_cpuSleep_"
                   << "false";
        }

        if (actualSleep) {
            result << "_actualSleep_"
                   << "true";
        } else {
            result << "_actualSleep_"
                   << "false";
        }
        result << "_actualDeviceName_" << actualDeviceName;
        result << "_gpuPerfHintNum_" << gpuPerfHintNum;
        if (isSupportOptimalNum) {
            result << "_supportOptimalNum_"
                   << "true";
        } else {
            result << "_supportOptimalNum_"
                   << "false";
        }
        if (isSupportNumRequests) {
            result << "_supportNumRequests_"
                   << "true";
        } else {
            result << "_supportNumRequests_"
                   << "false";
        }
        return result.str();
    }
};

using modelPrioPerfHintTestParams = std::tuple<bool,         // is New API
                                               bool,         // if Actual device sleep, cpu device will load slow
                                               std::string,  // Actual Device Name
                                               ov::Any,      // performance mode
                                               ov::Any       // model Priority
                                               >;

class ExecNetworkget_propertyOtherTest : public tests::AutoTest,
                                         public ::testing::TestWithParam<modelPrioPerfHintTestParams> {
public:
    static std::string getTestCaseName(testing::TestParamInfo<modelPrioPerfHintTestParams> obj) {
        bool isNewAPI;
        bool actualSleep;
        std::string actualDeviceName;
        ov::Any performanceMode;
        ov::Any modelPriority;
        std::tie(isNewAPI, actualSleep, actualDeviceName, performanceMode, modelPriority) = obj.param;
        std::ostringstream result;
        if (isNewAPI) {
            result << "_isNewAPI_"
                   << "true";
        } else {
            result << "_isNewAPI_"
                   << "false";
        }
        if (actualSleep) {
            result << "_actualSleep_"
                   << "true";
        } else {
            result << "_actualSleep_"
                   << "false";
        }
        result << "_actualDeviceName_" << actualDeviceName;
        result << "_performanceMode_" << performanceMode.as<std::string>();
        result << "_modelPriority" << modelPriority.as<std::string>();
        return result.str();
    }
};

TEST_P(ExecNetworkget_propertyOptimalNumInferReq, OPTIMAL_NUMBER_OF_INFER_REQUESTS) {
    unsigned int cpuOptimalNum;
    int cpuCustomerNum;
    unsigned int actualOptimalNum;
    int actualCustomerNum;
    unsigned int expectOptimalNum;
    bool cpuSleep;
    bool actualSleep;
    bool isThroughput;
    unsigned int gpuPerfHintNum;
    bool isSupportedOptimalNum;
    bool isSupportNumRequests;
    std::string actualDeviceName;
    std::tie(isThroughput,
             cpuOptimalNum,
             cpuCustomerNum,
             cpuSleep,
             actualOptimalNum,
             actualCustomerNum,
             actualSleep,
             actualDeviceName,
             expectOptimalNum,
             gpuPerfHintNum,
             isSupportedOptimalNum,
             isSupportNumRequests) = this->GetParam();
    config.insert(ov::device::priorities(ov::test::utils::DEVICE_CPU + std::string(",") + actualDeviceName));
    std::vector<ov::PropertyName> supported_props = {ov::range_for_streams,
                                                     ov::optimal_batch_size,
                                                     ov::hint::performance_mode};
    if (isSupportedOptimalNum) {
        supported_props.push_back(ov::optimal_number_of_infer_requests);
        ON_CALL(*core,
                get_property(StrEq(ov::test::utils::DEVICE_GPU), StrEq(ov::optimal_number_of_infer_requests.name()), _))
            .WillByDefault(RETURN_MOCK_VALUE(actualOptimalNum));
        ON_CALL(
            *core,
            get_property(StrEq(ov::test::utils::DEVICE_NPU), StrEq(ov::optimal_number_of_infer_requests.name()), _))
            .WillByDefault(RETURN_MOCK_VALUE(actualOptimalNum));
    }
    if (isSupportNumRequests)
        supported_props.push_back(ov::hint::num_requests);

    ON_CALL(*core, get_property(_, StrEq(ov::supported_properties.name()), _))
        .WillByDefault(RETURN_MOCK_VALUE(supported_props));

    ov::AnyMap metaConfig = {};
    if (gpuPerfHintNum >= 0) {
        metaConfig.insert(ov::hint::num_requests(gpuPerfHintNum));
    }
    if (isThroughput) {
        metaConfig.insert(ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT));
        metaDevices.push_back({ov::test::utils::DEVICE_CPU, metaConfig, cpuCustomerNum, ""});
        metaDevices.push_back({actualDeviceName, metaConfig, actualCustomerNum, ""});
        // enable autoBatch
        unsigned int gpuOptimalBatchNum = 8;
        unsigned int npuOptimalBatchNum = 1;
        ov::hint::PerformanceMode mode = ov::hint::PerformanceMode::THROUGHPUT;
        std::tuple<unsigned int, unsigned int> rangeOfStreams = std::make_tuple<unsigned int, unsigned int>(1, 3);
        ON_CALL(*core, get_property(StrEq(ov::test::utils::DEVICE_GPU), StrEq(ov::optimal_batch_size.name()), _))
            .WillByDefault(RETURN_MOCK_VALUE(gpuOptimalBatchNum));
        ON_CALL(*core, get_property(StrEq(ov::test::utils::DEVICE_NPU), StrEq(ov::optimal_batch_size.name()), _))
            .WillByDefault(RETURN_MOCK_VALUE(npuOptimalBatchNum));
        ON_CALL(*core, get_property(_, StrEq(ov::range_for_streams.name()), _))
            .WillByDefault(RETURN_MOCK_VALUE(rangeOfStreams));
        ON_CALL(*core, get_property(_, StrEq(ov::hint::performance_mode.name()), _))
            .WillByDefault(RETURN_MOCK_VALUE(mode));
        if (isSupportNumRequests) {
            ON_CALL(*core, get_property(_, StrEq(ov::hint::num_requests.name()), _))
                .WillByDefault(RETURN_MOCK_VALUE(gpuPerfHintNum));
        }
        ON_CALL(*core, get_property(_, StrEq(ov::compilation_num_threads.name()), _)).WillByDefault(Return(8));
    } else {
        metaDevices.push_back({ov::test::utils::DEVICE_CPU, metaConfig, cpuCustomerNum, ""});
        metaDevices.push_back({actualDeviceName, metaConfig, actualCustomerNum, ""});
        ON_CALL(*core, get_property(_, StrEq(ov::compilation_num_threads.name()), _)).WillByDefault(Return(8));
    }
    ON_CALL(*plugin, select_device(_, _, _)).WillByDefault(Return(metaDevices[1]));
    ON_CALL(*plugin, parse_meta_devices(_, _)).WillByDefault(Return(metaDevices));
    ON_CALL(*plugin, get_valid_device)
        .WillByDefault([](const std::vector<DeviceInformation>& metaDevices, const std::string& netPrecision) {
            std::list<DeviceInformation> devices(metaDevices.begin(), metaDevices.end());
            return devices;
        });
    EXPECT_CALL(*plugin, parse_meta_devices(_, _)).Times(1);
    EXPECT_CALL(*plugin, select_device(_, _, _)).Times(1);

    if (cpuSleep) {
        ON_CALL(*core,
                compile_model(::testing::Matcher<const std::shared_ptr<const ov::Model>&>(_),
                              ::testing::Matcher<const std::string&>(StrEq(ov::test::utils::DEVICE_CPU)),
                              _))
            .WillByDefault(InvokeWithoutArgs([this]() {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                return mockExeNetwork;
            }));
    } else {
        ON_CALL(*core,
                compile_model(::testing::Matcher<const std::shared_ptr<const ov::Model>&>(_),
                              ::testing::Matcher<const std::string&>(StrEq(ov::test::utils::DEVICE_CPU)),
                              _))
            .WillByDefault(Return(mockExeNetwork));
    }

    if (actualSleep) {
        ON_CALL(*core,
                compile_model(::testing::Matcher<const std::shared_ptr<const ov::Model>&>(_),
                              ::testing::Matcher<const std::string&>(StrEq(actualDeviceName)),
                              _))
            .WillByDefault(InvokeWithoutArgs([this]() {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                return mockExeNetworkActual;
            }));
    } else {
        ON_CALL(*core,
                compile_model(::testing::Matcher<const std::shared_ptr<const ov::Model>&>(_),
                              ::testing::Matcher<const std::string&>(StrEq(actualDeviceName)),
                              _))
            .WillByDefault(Return(mockExeNetworkActual));
    }

    ON_CALL(*mockIExeNet.get(), get_property(StrEq(ov::optimal_number_of_infer_requests.name())))
        .WillByDefault(RETURN_MOCK_VALUE(cpuOptimalNum));
    ON_CALL(*mockIExeNetActual.get(), get_property(StrEq(ov::optimal_number_of_infer_requests.name())))
        .WillByDefault(RETURN_MOCK_VALUE(actualOptimalNum));

    EXPECT_CALL(*mockIExeNet.get(), get_property(StrEq(ov::optimal_number_of_infer_requests.name()))).Times(AtLeast(1));

    EXPECT_CALL(*mockIExeNetActual.get(), get_property(StrEq(ov::optimal_number_of_infer_requests.name())))
        .Times(AtLeast(1));

    EXPECT_CALL(*core,
                compile_model(::testing::Matcher<const std::shared_ptr<const ov::Model>&>(_),
                              ::testing::Matcher<const std::string&>(StrEq(ov::test::utils::DEVICE_CPU)),
                              _))
        .Times(1);

    EXPECT_CALL(*core,
                compile_model(::testing::Matcher<const std::shared_ptr<const ov::Model>&>(_),
                              ::testing::Matcher<const std::string&>(StrEq(actualDeviceName)),
                              _))
        .Times(1);

    if (cpuCustomerNum == -1) {
        EXPECT_CALL(*mockIExeNet.get(), create_sync_infer_request()).Times(cpuOptimalNum);
    } else {
        EXPECT_CALL(*mockIExeNet.get(), create_sync_infer_request()).Times(cpuCustomerNum);
    }

    if (actualCustomerNum == -1) {
        EXPECT_CALL(*mockIExeNetActual.get(), create_sync_infer_request()).Times(actualOptimalNum);
    } else {
        EXPECT_CALL(*mockIExeNetActual.get(), create_sync_infer_request()).Times(actualCustomerNum);
    }

    auto AutoExecNetwork = plugin->compile_model(model, config);
    auto result = AutoExecNetwork->get_property(ov::optimal_number_of_infer_requests.name()).as<unsigned int>();
    EXPECT_EQ(result, expectOptimalNum);
}

// ConfigParams {bool, unsigned int, int, bool,
//               unsigned int, int, bool, std::string, unsigned int}
//
// every element for ConfigParams
// {is throughput mode, cpuOptimalNum, customer hope for cpu infer requset num, if cpu sleep when load,
//  actualOptimalNum, customer hope for actual infer requset num, if actual sleep when load, actual device Name
//  expectOptimalNum of Auto ExecNetwork, gpu Number of requests, if actual supported OptimalNum, default Value of
//  OptimalNum}
//
const std::vector<ConfigParams> testConfigs = {
    ConfigParams{false, 3, -1, false, 2, -1, true, ov::test::utils::DEVICE_GPU, 1, 0, false, true},
    ConfigParams{true, 3, -1, false, 2, -1, true, ov::test::utils::DEVICE_GPU, 48, 0, false, true},
    ConfigParams{false, 3, -1, true, 2, -1, false, ov::test::utils::DEVICE_GPU, 2, 0, false, true},
    ConfigParams{true, 3, -1, true, 2, -1, false, ov::test::utils::DEVICE_GPU, 2, 0, false, true},
    ConfigParams{false, 3, 5, false, 2, 5, true, ov::test::utils::DEVICE_GPU, 1, 0, false, true},
    ConfigParams{true, 3, 5, false, 2, 5, true, ov::test::utils::DEVICE_GPU, 48, 0, false, true},
    ConfigParams{false, 3, 5, true, 2, 5, false, ov::test::utils::DEVICE_GPU, 2, 0, false, true},
    ConfigParams{true, 3, 5, true, 2, 5, false, ov::test::utils::DEVICE_GPU, 2, 0, false, true},
    ConfigParams{true, 3, 5, false, 2, 5, true, ov::test::utils::DEVICE_GPU, 48, 48, false, true},
    ConfigParams{true, 3, 5, false, 2, 5, true, ov::test::utils::DEVICE_GPU, 6, 6, false, true},
    ConfigParams{true, 3, 5, false, 0, 5, true, ov::test::utils::DEVICE_GPU, 6, 6, false, true},
    ConfigParams{true, 3, 5, false, 2, 5, true, ov::test::utils::DEVICE_GPU, 2, 0, true, true},
    ConfigParams{true, 3, 5, false, 0, 5, true, ov::test::utils::DEVICE_GPU, 48, 0, false, true},
    ConfigParams{true, 3, 5, false, 0, 5, true, ov::test::utils::DEVICE_GPU, 48, 0, true, true},
    ConfigParams{true, 3, 5, false, 2, 5, true, ov::test::utils::DEVICE_GPU, 6, 6, false, false},
    ConfigParams{true, 3, 5, false, 2, 5, true, ov::test::utils::DEVICE_GPU, 8, 10, false, false},
    ConfigParams{true, 3, 5, false, 0, 5, true, ov::test::utils::DEVICE_GPU, 6, 6, true, false},
    ConfigParams{true, 3, 5, false, 2, 5, true, ov::test::utils::DEVICE_GPU, 2, 6, true, false},
    ConfigParams{true, 3, 5, false, 2, 5, true, ov::test::utils::DEVICE_GPU, 8, 0, false, false},
    ConfigParams{true, 3, 5, false, 0, 5, true, ov::test::utils::DEVICE_GPU, 8, 0, true, false},
    ConfigParams{true, 3, 5, false, 2, 5, true, ov::test::utils::DEVICE_GPU, 2, 0, true, false},
    ConfigParams{true, 3, 5, false, 0, 5, true, ov::test::utils::DEVICE_GPU, 6, 6, true, true},
    ConfigParams{false, 3, 5, false, 0, 5, true, ov::test::utils::DEVICE_GPU, 1, 6, true, true},
    ConfigParams{true, 3, 5, false, 6, 5, true, ov::test::utils::DEVICE_GPU, 6, 6, true, true},
    ConfigParams{false, 3, 5, false, 6, 5, true, ov::test::utils::DEVICE_GPU, 6, 6, true, true},
    ConfigParams{false, 3, -1, false, 2, -1, true, ov::test::utils::DEVICE_NPU, 1, 0, false, true},
    ConfigParams{true, 3, -1, false, 2, -1, true, ov::test::utils::DEVICE_NPU, 8, 0, false, true},
    ConfigParams{true, 3, -1, false, 2, -1, true, ov::test::utils::DEVICE_NPU, 8, 0, false, false},
    ConfigParams{true, 3, -1, false, 0, -1, true, ov::test::utils::DEVICE_NPU, 8, 0, true, false},
    ConfigParams{true, 3, -1, false, 2, -1, true, ov::test::utils::DEVICE_NPU, 2, 0, true, false},
    ConfigParams{true, 3, -1, false, 2, -1, true, ov::test::utils::DEVICE_NPU, 2, 1, true, false},
    ConfigParams{true, 3, -1, false, 2, -1, true, ov::test::utils::DEVICE_NPU, 6, 6, false, false},
    ConfigParams{true, 3, -1, false, 2, -1, true, ov::test::utils::DEVICE_NPU, 8, 10, false, false},
    ConfigParams{true, 3, -1, false, 4, -1, true, ov::test::utils::DEVICE_NPU, 4, 6, true, true},
    ConfigParams{true, 3, -1, false, 4, -1, true, ov::test::utils::DEVICE_NPU, 2, 2, true, true},
    ConfigParams{true, 3, -1, false, 0, -1, true, ov::test::utils::DEVICE_NPU, 8, 10, true, true},
    ConfigParams{true, 3, -1, false, 0, -1, true, ov::test::utils::DEVICE_NPU, 6, 6, true, true},
    ConfigParams{false, 3, -1, false, 0, -1, true, ov::test::utils::DEVICE_NPU, 1, 0, true, true},
    ConfigParams{true, 3, -1, false, 0, -1, true, ov::test::utils::DEVICE_NPU, 8, 0, true, true},
    ConfigParams{false, 3, -1, false, 2, -1, true, ov::test::utils::DEVICE_NPU, 2, 0, true, true},
    ConfigParams{true, 3, -1, false, 2, -1, true, ov::test::utils::DEVICE_NPU, 2, 0, true, true},
    ConfigParams{false, 3, -1, true, 2, -1, false, ov::test::utils::DEVICE_NPU, 2, 0, false, true},
    ConfigParams{true, 3, -1, true, 2, -1, false, ov::test::utils::DEVICE_NPU, 2, 0, false, true},
    ConfigParams{false, 3, 5, false, 2, 5, true, ov::test::utils::DEVICE_NPU, 1, 0, false, true},
    ConfigParams{true, 3, 5, false, 2, 5, true, ov::test::utils::DEVICE_NPU, 8, 0, false, true},
    ConfigParams{false, 3, 5, true, 2, 5, false, ov::test::utils::DEVICE_NPU, 2, 0, false, true},
    ConfigParams{true, 3, 5, true, 2, 5, false, ov::test::utils::DEVICE_NPU, 2, 0, false, true},
};

INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests,
                         ExecNetworkget_propertyOptimalNumInferReq,
                         ::testing::ValuesIn(testConfigs),
                         ExecNetworkget_propertyOptimalNumInferReq::getTestCaseName);
class ExecNetworkGetMetricOtherTest : public tests::AutoTest,
                                      public ::testing::TestWithParam<modelPrioPerfHintTestParams> {
public:
    static std::string getTestCaseName(testing::TestParamInfo<modelPrioPerfHintTestParams> obj) {
        bool isNewAPI;
        bool actualSleep;
        std::string actualDeviceName;
        ov::Any performanceMode;
        ov::Any modelPriority;
        std::tie(isNewAPI, actualSleep, actualDeviceName, performanceMode, modelPriority) = obj.param;
        std::ostringstream result;
        if (isNewAPI) {
            result << "_isNewAPI_"
                   << "true";
        } else {
            result << "_isNewAPI_"
                   << "false";
        }
        if (actualSleep) {
            result << "_actualSleep_"
                   << "true";
        } else {
            result << "_actualSleep_"
                   << "false";
        }
        result << "_actualDeviceName_" << actualDeviceName;
        result << "_performanceMode_" << performanceMode.as<std::string>();
        result << "_modelPriority" << modelPriority.as<std::string>();
        return result.str();
    }
};

TEST_P(ExecNetworkGetMetricOtherTest, modelPriority_perfHint_exclusiveAsyncReq_test) {
    unsigned int cpuOptimalNum = 3;
    unsigned int actualOptimalNum = 2;
    bool isNewAPI;
    bool actualSleep;
    std::string actualDeviceName;
    ov::Any performanceHint;
    ov::Any modelPriority;
    std::tie(isNewAPI, actualSleep, actualDeviceName, performanceHint, modelPriority) = this->GetParam();
    config.insert(ov::device::priorities(ov::test::utils::DEVICE_CPU + std::string(",") + actualDeviceName));
    config.insert(ov::hint::performance_mode(performanceHint.as<ov::hint::PerformanceMode>()));
    config.insert({ov::hint::model_priority.name(), modelPriority.as<std::string>()});

    metaDevices.push_back({ov::test::utils::DEVICE_CPU,
                           {ov::hint::performance_mode(performanceHint.as<ov::hint::PerformanceMode>())},
                           3,
                           ""});
    metaDevices.push_back(
        {actualDeviceName, {ov::hint::performance_mode(performanceHint.as<ov::hint::PerformanceMode>())}, 2, ""});

    ON_CALL(*plugin, select_device(_, _, _)).WillByDefault(Return(metaDevices[1]));
    ON_CALL(*plugin, parse_meta_devices(_, _)).WillByDefault(Return(metaDevices));
    ON_CALL(*plugin, get_valid_device)
        .WillByDefault([](const std::vector<DeviceInformation>& metaDevices, const std::string& netPrecision) {
            std::list<DeviceInformation> devices(metaDevices.begin(), metaDevices.end());
            return devices;
        });
    EXPECT_CALL(*plugin, parse_meta_devices(_, _)).Times(1);
    EXPECT_CALL(*plugin, select_device(_, _, _)).Times(1);

    ON_CALL(*core, get_property(_, StrEq(ov::compilation_num_threads.name()), _)).WillByDefault(Return(8));
    ON_CALL(*core,
            compile_model(::testing::Matcher<const std::shared_ptr<const ov::Model>&>(_),
                          ::testing::Matcher<const std::string&>(StrEq(ov::test::utils::DEVICE_CPU)),
                          _))
        .WillByDefault(Return(mockExeNetwork));

    if (actualSleep) {
        ON_CALL(*core,
                compile_model(::testing::Matcher<const std::shared_ptr<const ov::Model>&>(_),
                              ::testing::Matcher<const std::string&>(StrEq(actualDeviceName)),
                              _))
            .WillByDefault(InvokeWithoutArgs([this]() {
                std::this_thread::sleep_for(std::chrono::milliseconds(5000));
                return mockExeNetworkActual;
            }));
    } else {
        ON_CALL(*core,
                compile_model(::testing::Matcher<const std::shared_ptr<const ov::Model>&>(_),
                              ::testing::Matcher<const std::string&>(StrEq(actualDeviceName)),
                              _))
            .WillByDefault(Return(mockExeNetworkActual));
    }

    ON_CALL(*mockIExeNet.get(), get_property(StrEq(ov::optimal_number_of_infer_requests.name())))
        .WillByDefault(RETURN_MOCK_VALUE(cpuOptimalNum));
    ON_CALL(*mockIExeNetActual.get(), get_property(StrEq(ov::optimal_number_of_infer_requests.name())))
        .WillByDefault(RETURN_MOCK_VALUE(actualOptimalNum));

    auto AutoExecNetwork = plugin->compile_model(model, config);
    auto result = AutoExecNetwork->get_property(ov::hint::performance_mode.name()).as<std::string>();
    EXPECT_EQ(result, performanceHint.as<std::string>());
    auto resPriority = AutoExecNetwork->get_property(ov::hint::model_priority.name()).as<std::string>();
    EXPECT_EQ(resPriority, modelPriority.as<std::string>());
}

const std::vector<modelPrioPerfHintTestParams> modelPrioPerfHintConfig = {
    modelPrioPerfHintTestParams{false,
                                true,
                                ov::test::utils::DEVICE_GPU,
                                ov::hint::PerformanceMode::THROUGHPUT,
                                ov::hint::Priority::LOW},
    modelPrioPerfHintTestParams{false,
                                true,
                                ov::test::utils::DEVICE_GPU,
                                ov::hint::PerformanceMode::LATENCY,
                                ov::hint::Priority::LOW},
    modelPrioPerfHintTestParams{false,
                                true,
                                ov::test::utils::DEVICE_GPU,
                                ov::hint::PerformanceMode::THROUGHPUT,
                                ov::hint::Priority::MEDIUM},
    modelPrioPerfHintTestParams{false,
                                true,
                                ov::test::utils::DEVICE_GPU,
                                ov::hint::PerformanceMode::LATENCY,
                                ov::hint::Priority::MEDIUM},
    modelPrioPerfHintTestParams{false,
                                true,
                                ov::test::utils::DEVICE_GPU,
                                ov::hint::PerformanceMode::THROUGHPUT,
                                ov::hint::Priority::HIGH},
    modelPrioPerfHintTestParams{false,
                                true,
                                ov::test::utils::DEVICE_GPU,
                                ov::hint::PerformanceMode::LATENCY,
                                ov::hint::Priority::HIGH},
    modelPrioPerfHintTestParams{true,
                                true,
                                ov::test::utils::DEVICE_GPU,
                                ov::hint::PerformanceMode::THROUGHPUT,
                                ov::hint::Priority::LOW},
    modelPrioPerfHintTestParams{true,
                                true,
                                ov::test::utils::DEVICE_GPU,
                                ov::hint::PerformanceMode::LATENCY,
                                ov::hint::Priority::LOW},
    modelPrioPerfHintTestParams{true,
                                true,
                                ov::test::utils::DEVICE_GPU,
                                ov::hint::PerformanceMode::THROUGHPUT,
                                ov::hint::Priority::MEDIUM},
    modelPrioPerfHintTestParams{true,
                                true,
                                ov::test::utils::DEVICE_GPU,
                                ov::hint::PerformanceMode::LATENCY,
                                ov::hint::Priority::MEDIUM},
    modelPrioPerfHintTestParams{true,
                                true,
                                ov::test::utils::DEVICE_GPU,
                                ov::hint::PerformanceMode::THROUGHPUT,
                                ov::hint::Priority::HIGH},
    modelPrioPerfHintTestParams{true,
                                true,
                                ov::test::utils::DEVICE_GPU,
                                ov::hint::PerformanceMode::LATENCY,
                                ov::hint::Priority::HIGH}};

INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests,
                         ExecNetworkGetMetricOtherTest,
                         ::testing::ValuesIn(modelPrioPerfHintConfig),
                         ExecNetworkGetMetricOtherTest::getTestCaseName);
