// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <vector>

#include "common_test_utils/test_assertions.hpp"
#include "async_infer_request.hpp"
#include "common.hpp"
#include "cumulative_schedule.hpp"
#include "openvino/runtime/auto/properties.hpp"
#include "plugin.hpp"
using ConfigParams = std::tuple<std::vector<ov::auto_plugin::DeviceInformation>,  // device candidate list
                                ov::intel_auto::SchedulePolicy,                   // schedule policy
                                std::map<std::string, int>,  // number of infer request for each device
                                std::vector<std::string>  // the expected device where each of infer request comes from
                                >;
class MockCumuSchedule : public ov::auto_plugin::CumuSchedule, public ::testing::TestWithParam<ConfigParams> {
protected:
    std::vector<ov::auto_plugin::DeviceInformation> devicesInfo;
    ov::intel_auto::SchedulePolicy schedulePolicy;
    std::map<std::string, int> numOfInferRequests;
    std::vector<std::string> expectedScheDevs;

public:
    static std::string getTestCaseName(testing::TestParamInfo<ConfigParams> obj) {
        std::vector<ov::auto_plugin::DeviceInformation> devicesInfo;
        ov::intel_auto::SchedulePolicy schedulePolicy;
        std::map<std::string, int> numOfInferRequests;
        std::vector<std::string> expectedScheDevs;
        std::tie(devicesInfo, schedulePolicy, numOfInferRequests, expectedScheDevs) = obj.param;
        std::ostringstream result;
        std::string candidateDevList;
        result << "candaidateDeviceList_";
        for (auto dev : devicesInfo)
            result << dev.device_name << "_";
        result << "schedulePolicy_" << schedulePolicy << "_";
        result << "inferRequestNumberOnEachDevice_";
        for (auto ninfer : numOfInferRequests)
            result << ninfer.first << "_" << ninfer.second << "_";
        result << "expectedDeviceSelection_";
        for (auto dev : expectedScheDevs)
            result << dev << "_";
        return result.str();
    }

    void TearDown() override {
        devicesInfo.clear();
        numOfInferRequests.clear();
        expectedScheDevs.clear();
        m_context.reset();
    }

    void SetUp() override {
        std::tie(devicesInfo, schedulePolicy, numOfInferRequests, expectedScheDevs) = GetParam();
        m_context = std::make_shared<ov::auto_plugin::ScheduleContext>();
        m_context->m_schedule_policy = schedulePolicy;
    }
};

TEST_P(MockCumuSchedule, scheduleInferRequestBasedOnSchedulePolicy) {
    std::size_t deviceIndexWithInferReq = 0;
    int expectedDevIndex = 0;
    while (true) {
        std::string actualSelectedDev;
        OV_ASSERT_NO_THROW(actualSelectedDev = schedule_to_next_device(devicesInfo, deviceIndexWithInferReq));
        if (numOfInferRequests[actualSelectedDev] > 0) {
            EXPECT_EQ(actualSelectedDev, expectedScheDevs[expectedDevIndex++]);
            // consume an available infer request on selected device
            numOfInferRequests[actualSelectedDev]--;
        } else {
            // schecdule to next priority device
            deviceIndexWithInferReq++;
            if (deviceIndexWithInferReq >= devicesInfo.size()) {
                // no available infer request on all of the devices
                break;
            }
        }
    }
}

const std::vector<ov::auto_plugin::DeviceInformation> metaDevicesWithSingleDev = {
    {"DEVICE_0", {}, -1, "01", "DEVICE_0_01", 0}};
const std::vector<ov::auto_plugin::DeviceInformation> metaDevicesWithTwoDevs = {
    {"DEVICE_0", {}, -1, "01", "DEVICE_0_01", 0},
    {"DEVICE_1", {}, -1, "01", "DEVICE_1_01", 1}};
const std::vector<ov::auto_plugin::DeviceInformation> metaDevices = {{"DEVICE_0", {}, -1, "01", "DEVICE_0_01", 0},
                                                                     {"DEVICE_1", {}, -1, "01", "DEVICE_1_01", 1},
                                                                     {"DEVICE_2", {}, -1, "01", "DEVICE_2_01", 2}};
const std::vector<ConfigParams> configs = {
    ConfigParams{
        metaDevicesWithSingleDev,                     // param[in]: device candidate list for AUTO plugin
        ov::intel_auto::SchedulePolicy::ROUND_ROBIN,  // param[in]: specified schedule policy
        {{"DEVICE_0", 6}},  // param[in]: a map recorded the count of infer request on each hw device
        {"DEVICE_0",
         "DEVICE_0",
         "DEVICE_0",
         "DEVICE_0",
         "DEVICE_0",
         "DEVICE_0"}},  // param[output]: the expected device list where the next available infer request comes from
    ConfigParams{metaDevicesWithSingleDev,
                 ov::intel_auto::SchedulePolicy::DEVICE_PRIORITY,
                 {{"DEVICE_0", 6}},
                 {"DEVICE_0", "DEVICE_0", "DEVICE_0", "DEVICE_0", "DEVICE_0", "DEVICE_0"}},
    ConfigParams{metaDevicesWithTwoDevs,
                 ov::intel_auto::SchedulePolicy::ROUND_ROBIN,
                 {{"DEVICE_0", 3}, {"DEVICE_1", 2}},
                 {"DEVICE_0", "DEVICE_1", "DEVICE_0", "DEVICE_1", "DEVICE_0"}},
    ConfigParams{metaDevicesWithTwoDevs,
                 ov::intel_auto::SchedulePolicy::DEVICE_PRIORITY,
                 {{"DEVICE_0", 3}, {"DEVICE_1", 2}},
                 {"DEVICE_0", "DEVICE_0", "DEVICE_0", "DEVICE_1", "DEVICE_1"}},
    ConfigParams{metaDevicesWithTwoDevs,
                 ov::intel_auto::SchedulePolicy::DEVICE_PRIORITY,
                 {{"DEVICE_0", 2}, {"DEVICE_1", 3}},
                 {"DEVICE_0", "DEVICE_0", "DEVICE_1", "DEVICE_1", "DEVICE_1"}},
    ConfigParams{metaDevices,
                 ov::intel_auto::SchedulePolicy::ROUND_ROBIN,
                 {{"DEVICE_0", 3}, {"DEVICE_1", 2}, {"DEVICE_2", 1}},
                 {"DEVICE_0", "DEVICE_1", "DEVICE_2", "DEVICE_0", "DEVICE_1", "DEVICE_0"}},
    ConfigParams{metaDevices,
                 ov::intel_auto::SchedulePolicy::ROUND_ROBIN,
                 {{"DEVICE_0", 1}, {"DEVICE_1", 2}, {"DEVICE_2", 3}},
                 {"DEVICE_0", "DEVICE_1", "DEVICE_2", "DEVICE_1", "DEVICE_2", "DEVICE_2"}},
    ConfigParams{metaDevices,
                 ov::intel_auto::SchedulePolicy::ROUND_ROBIN,
                 {{"DEVICE_0", 1}, {"DEVICE_1", 3}, {"DEVICE_2", 2}},
                 {"DEVICE_0", "DEVICE_1", "DEVICE_2", "DEVICE_1", "DEVICE_2", "DEVICE_1"}},
    ConfigParams{metaDevices,
                 ov::intel_auto::SchedulePolicy::DEVICE_PRIORITY,
                 {{"DEVICE_0", 1}, {"DEVICE_1", 3}, {"DEVICE_2", 2}},
                 {"DEVICE_0", "DEVICE_1", "DEVICE_1", "DEVICE_1", "DEVICE_2", "DEVICE_2"}},
    ConfigParams{metaDevices,
                 ov::intel_auto::SchedulePolicy::DEVICE_PRIORITY,
                 {{"DEVICE_0", 3}, {"DEVICE_1", 2}, {"DEVICE_2", 1}},
                 {"DEVICE_0", "DEVICE_0", "DEVICE_0", "DEVICE_1", "DEVICE_1", "DEVICE_2"}}};

INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests,
                         MockCumuSchedule,
                         ::testing::ValuesIn(configs),
                         MockCumuSchedule::getTestCaseName);