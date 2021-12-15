// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "myriad_test_case.h"
#include "ie_plugin_config.hpp"

//------------------------------------------------------------------------------
// Implementation of methods of class MyriadEngineTest
//------------------------------------------------------------------------------

MyriadEngineTest::MyriadEngineTest() {
    mvnc_stub_ = std::make_shared<MvncStub>();
    myriad_engine_ = std::make_shared<Engine>(mvnc_stub_);
}

//------------------------------------------------------------------------------
// Implementation of methods of class MyriadEngineSetConfigTest
//------------------------------------------------------------------------------

void MyriadEngineSetConfigTest::SetUp() {
    config_ = GetParam();
}

//------------------------------------------------------------------------------
// Implementation of methods of class MyriadGetMetricsTestCase
//------------------------------------------------------------------------------

void MyriadGetMetricsTestCase::SetupOneDevice() {
    exp_devices_names_ = std::vector<std::string>{"2.1-ma2480"};
    EXPECT_CALL(*mvnc_stub_, AvailableDevicesNames()).Times(exp_devices_names_.size() + 1)
        .WillRepeatedly(Return(exp_devices_names_));

    auto act_available_devices_ids = myriad_engine_->GetMetric(METRIC_KEY(AVAILABLE_DEVICES), options_)
        .as<std::vector<std::string>>();

    ASSERT_TRUE(exp_devices_names_.size() ==
                act_available_devices_ids.size());
}

//------------------------------------------------------------------------------
// Implementation of methods of class MyriadGetMetricsTestCaseWithParam
//------------------------------------------------------------------------------

void MyriadGetMetricsTestCaseWithParam::SetUp() {
    exp_devices_names_ = GetParam();
}

//------------------------------------------------------------------------------
// Implementation of methods of class MyriadMetricsTest
//------------------------------------------------------------------------------

MyriadMetricsTest::MyriadMetricsTest() {
    metrics_container_ = std::make_shared<MyriadMetrics>();
    mvnc_stub_ = std::make_shared<MvncStub>();
}

void MyriadMetricsTest::SetDevices(std::vector<std::string> deviceNames) {
    for(const auto& name : deviceNames) {
        auto deviceDesc = std::make_shared<DeviceDesc>();
        deviceDesc->_name = name;

        devices_.push_back(deviceDesc);
    }
}

//------------------------------------------------------------------------------
// Implementation of methods of class MyriadDeviceMetricsTestWithParam
//------------------------------------------------------------------------------

void MyriadDeviceMetricsTestWithParam::SetUp() {
    exp_unbooted_devices_names_ = std::get<0>(GetParam());
    exp_booted_devices_names_ = std::get<1>(GetParam());

    SetDevices(exp_booted_devices_names_);
}

//------------------------------------------------------------------------------
// Implementation of methods of class MyriadDeviceMetricsTestWithParam
//------------------------------------------------------------------------------

void MyriadRangeInferMetricsTestWithParam::SetUp() {
    exp_range_ = std::get<0>(GetParam());
    confir_param_ = std::get<1>(GetParam());
}