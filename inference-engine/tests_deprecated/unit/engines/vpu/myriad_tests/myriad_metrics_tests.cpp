// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "myriad_test_case.h"
#include <memory>

using namespace InferenceEngine;
using namespace InferenceEngine::PluginConfigParams;
using namespace InferenceEngine::VPUConfigParams;

using str_vector = std::vector<std::string>;

TEST_P(MyriadGetMetricsTestCaseWithParam, CheckNames) {
    Parameter act_result;
    str_vector act_available_devices;

    EXPECT_CALL(*mvnc_stub_, AvailableDevicesNames()).Times(1)
        .WillOnce(Return(exp_devices_names_));

    ASSERT_NO_THROW(act_result = myriad_engine_->GetMetric(METRIC_KEY(AVAILABLE_DEVICES), options_););
    ASSERT_NO_THROW(act_available_devices = act_result.as<str_vector>(););

    ASSERT_TRUE(exp_devices_names_.size() ==
                     act_available_devices.size());

    for (int i = 0; i < act_available_devices.size(); ++i) {
        ASSERT_TRUE(act_available_devices[i] == exp_devices_names_[i]);
    }
}

TEST_P(MyriadGetMetricsTestCaseWithParam, ListOfDevicesShouldBeChanged) {
    Parameter act_name_result;
    std::string act_name;

    auto list_devices_1 = std::vector<std::string>{"0.0-maxxxx"};
    EXPECT_CALL(*mvnc_stub_, AvailableDevicesNames()).Times(2)
        .WillOnce(Return(list_devices_1))
        .WillOnce(Return(exp_devices_names_));

    auto act_available_devices = myriad_engine_->GetMetric(METRIC_KEY(AVAILABLE_DEVICES), options_)
        .as<str_vector>();

    //List devices should be changed by expected
    act_available_devices = myriad_engine_->GetMetric(METRIC_KEY(AVAILABLE_DEVICES), options_)
        .as<str_vector>();

    ASSERT_TRUE(exp_devices_names_.size() ==
                act_available_devices.size());

    for (int i = 0; i < act_available_devices.size(); ++i) {
        ASSERT_TRUE(act_available_devices[i] == exp_devices_names_[i]);
    }
}

TEST_P(MyriadGetMetricsTestCaseWithParam, CheckFullNames) {
    Parameter act_name_result;
    std::string act_name;

    // Will be called only once when the GetMetric with KEY_AVAILABLE_DEVICES is called.
    EXPECT_CALL(*mvnc_stub_, AvailableDevicesNames()).Times(1)
        .WillRepeatedly(Return(exp_devices_names_));

    auto act_available_devices = myriad_engine_->GetMetric(METRIC_KEY(AVAILABLE_DEVICES), options_)
                                                    .as<str_vector>();

    ASSERT_TRUE(exp_devices_names_.size() ==
                act_available_devices.size());

    for (int i = 0; i < act_available_devices.size(); ++i) {
        options_[KEY_DEVICE_ID] = act_available_devices[i];

        ASSERT_NO_THROW(act_name_result = myriad_engine_->GetMetric(METRIC_KEY(FULL_DEVICE_NAME), options_));
        ASSERT_NO_THROW(act_name = act_name_result.as<std::string>());

        ASSERT_TRUE(act_name.size());
        ASSERT_TRUE(act_name != act_available_devices[i]);
    }
}

TEST_F(MyriadGetMetricsTestCase, DevicesOrderShouldBeSaved) {
    Parameter act_name_result;
    std::string act_name;
    std::vector<std::string> device_names_1;
    std::vector<std::string> device_names_2;

    EXPECT_CALL(*mvnc_stub_, AvailableDevicesNames()).Times(2)
        .WillOnce(Return(std::vector<std::string>{"2.1-ma2480", "3.1-ma2085", "4.3-ma2085"}))
        .WillOnce(Return(std::vector<std::string>{"4.3-ma2085","2.1-ma2480",  "3.1-ma2085"}));

    auto act_available_devices_1 = myriad_engine_->GetMetric(METRIC_KEY(AVAILABLE_DEVICES), options_)
        .as<str_vector>();

    auto act_available_devices_2 = myriad_engine_->GetMetric(METRIC_KEY(AVAILABLE_DEVICES), options_)
        .as<str_vector>();

    ASSERT_TRUE(act_available_devices_1.size() ==
                    act_available_devices_2.size());

    for (int i = 0; i < act_available_devices_1.size(); ++i) {
        ASSERT_TRUE(act_available_devices_1[i] == act_available_devices_2[i]);
    }
}

TEST_F(MyriadGetMetricsTestCase, ShouldThrowExceptionWhenThereIsNoDevices) {
    Parameter act_name_result;
    std::string act_name;

    EXPECT_CALL(*mvnc_stub_, AvailableDevicesNames()).Times(1)
        .WillOnce(Return(std::vector<std::string>{}));

    ASSERT_ANY_THROW(act_name_result = myriad_engine_->GetMetric(METRIC_KEY(FULL_DEVICE_NAME), options_));
}

TEST_F(MyriadGetMetricsTestCase, ShouldThrowExceptionWhenThereIsMoreThanOneDeviceAndOptionsIsEmpty) {
    Parameter act_name_result;
    std::string act_name;

    exp_devices_names_ = std::vector<std::string>{"2.1-ma2480", "3.1-ma2085"};
    EXPECT_CALL(*mvnc_stub_, AvailableDevicesNames()).Times(2)
        .WillRepeatedly(Return(exp_devices_names_));

    auto act_available_devices = myriad_engine_->GetMetric(METRIC_KEY(AVAILABLE_DEVICES), options_)
        .as<str_vector>();

    ASSERT_TRUE(exp_devices_names_.size() ==
                act_available_devices.size());

    ASSERT_ANY_THROW(act_name_result = myriad_engine_->GetMetric(METRIC_KEY(FULL_DEVICE_NAME), options_));
}

TEST_F(MyriadGetMetricsTestCase, ShouldReturnFullNameWhenThereIsOneDeviceAndOptionsIsEmpty) {
    Parameter act_name_result;
    std::string act_name;

    SetupOneDevice();

    ASSERT_NO_THROW(act_name_result = myriad_engine_->GetMetric(METRIC_KEY(FULL_DEVICE_NAME), options_));
    ASSERT_NO_THROW(act_name = act_name_result.as<std::string>());

    ASSERT_TRUE(act_name.size());
    ASSERT_TRUE(act_name != exp_devices_names_[0]);
}

TEST_P(MyriadDeviceMetricsTestWithParam, CheckNames) {
    Parameter act_name_result;
    str_vector act_names;

    EXPECT_CALL(*mvnc_stub_, AvailableDevicesNames()).Times(1)
        .WillRepeatedly(Return(exp_unbooted_devices_names_));

    ASSERT_NO_THROW(act_names = metrics_container_->AvailableDevicesNames(mvnc_stub_, devices_));

    ASSERT_TRUE(act_names.size() ==
        (exp_unbooted_devices_names_.size() + exp_booted_devices_names_.size()));

    for(const auto& name : exp_unbooted_devices_names_) {
        ASSERT_TRUE(std::find(act_names.begin(), act_names.end(), name) != act_names.end());
    }

    for(const auto& name : exp_booted_devices_names_) {
        ASSERT_TRUE(std::find(act_names.begin(), act_names.end(), name) != act_names.end());
    }
}

TEST_F(MyriadMetricsTest, ShouldThrowExceptionWhenOption_MYRIAD_THROUGHPUT_STREAMS_isInvalid) {
    range_type act_res;
    std::map<std::string, std::string> config {
        {InferenceEngine::MYRIAD_THROUGHPUT_STREAMS, std::string("bad param")}
    };

    ASSERT_ANY_THROW(act_res = metrics_container_->RangeForAsyncInferRequests(config));
}

TEST_F(MyriadMetricsTest, ShouldReturnDefaultValueWhenOption_MYRIAD_THROUGHPUT_STREAMS_isEmpty) {
    range_type act_res;
    std::map<std::string, std::string> config {};
    auto exp_value = range_type(3,6,1);

    ASSERT_NO_THROW(act_res = metrics_container_->RangeForAsyncInferRequests(config));
    ASSERT_TRUE(act_res == exp_value);
}

TEST_P(MyriadRangeInferMetricsTestWithParam, CheckValues) {
    range_type act_res;
    std::map<std::string, std::string> config {
        {InferenceEngine::MYRIAD_THROUGHPUT_STREAMS, confir_param_}
    };

    ASSERT_NO_THROW(act_res = metrics_container_->RangeForAsyncInferRequests(config));
    ASSERT_TRUE(act_res == exp_range_);
}

INSTANTIATE_TEST_SUITE_P(
    MetricsTest,
    MyriadGetMetricsTestCaseWithParam,
    ::testing::Values(std::vector<std::string> {},
                      std::vector<std::string> {"2.1-ma2480"},
                      std::vector<std::string> {"2.1-ma2480", "3.1-ma2085"}));

INSTANTIATE_TEST_SUITE_P(
    MetricsTest,
    MyriadDeviceMetricsTestWithParam,
    Combine(::testing::Values(std::vector<std::string> {},
                      std::vector<std::string> {"2.1-ma2480"},
                      std::vector<std::string> {"2.2-ma2450", "3.1-ma2085"}),
            ::testing::Values(std::vector<std::string> {},
                                std::vector<std::string> {"1.4-ma2455"},
                                std::vector<std::string> {"1.1-ma2080", "3.3-ma2085"})));

INSTANTIATE_TEST_SUITE_P(
    MetricsTest,
    MyriadRangeInferMetricsTestWithParam,
    ::testing::Values(std::tuple<range_type, std::string>(range_type(3,6,1), "-1"),
                      std::tuple<range_type, std::string>(range_type(3,6,1), "0"),
                      std::tuple<range_type, std::string>(range_type(5,12,1), "4")));
