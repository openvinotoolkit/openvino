// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gtest/gtest.h>
#include <memory>
#include <tuple>
#include "myriad_plugin.h"
#include "myriad_mvnc_stub.h"

using namespace vpu::MyriadPlugin;
using namespace ::testing;

using AvailableDevices = std::vector<std::string>;
using UnbootedDevices = std::vector<std::string>;
using BootedDevices = std::vector<std::string>;
using range_type = std::tuple<unsigned int, unsigned int, unsigned int>;

using MyriadMetricsTestParam = WithParamInterface<AvailableDevices>;
using MyriadMetricsContainerTestParam = WithParamInterface<std::tuple<UnbootedDevices, BootedDevices>>;
using MyriadRangeMetricsContainerTestParam = WithParamInterface<std::tuple<range_type, std::string>>;

//------------------------------------------------------------------------------
// class MyriadEngineTest
//------------------------------------------------------------------------------

class MyriadEngineTest : public Test {
public:
    MyriadEngineTest();
protected:

    //Data section
    std::shared_ptr<MvncStub> mvnc_stub_;
    std::shared_ptr<Engine> myriad_engine_;
};

//------------------------------------------------------------------------------
// class MyriadGetMetricsTestCase
//------------------------------------------------------------------------------

class MyriadGetMetricsTestCase : public MyriadEngineTest {
public:
    MyriadGetMetricsTestCase() = default;

    //Operations
    void SetupOneDevice();

protected:

    //Data section
    std::vector<std::string> exp_devices_names_;
    std::map<std::string, InferenceEngine::Parameter> options_;
};

//------------------------------------------------------------------------------
// class MyriadGetMetricsTestCaseWithParam
//------------------------------------------------------------------------------

class MyriadGetMetricsTestCaseWithParam : public MyriadGetMetricsTestCase,
                                       public MyriadMetricsTestParam {
protected:
    //Operations
    void SetUp() override;
};

//------------------------------------------------------------------------------
// class MyriadMetricsTest
//------------------------------------------------------------------------------

class MyriadMetricsTest : public Test {
public:
    MyriadMetricsTest();

    //Operations
    void SetDevices(std::vector<std::string> deviceNames);
protected:
    //Data section
    std::shared_ptr<MyriadMetrics> metrics_container_;
    std::shared_ptr<MvncStub> mvnc_stub_;
    std::vector<DevicePtr> devices_;
};

//------------------------------------------------------------------------------
// class MyriadDeviceMetricsTestWithParam
//------------------------------------------------------------------------------

class MyriadDeviceMetricsTestWithParam : public MyriadMetricsTest,
                                         public MyriadMetricsContainerTestParam{
protected:
    //Operations
    void SetUp() override;

    //Data section
    std::vector<std::string> exp_unbooted_devices_names_;
    std::vector<std::string> exp_booted_devices_names_;
};

//------------------------------------------------------------------------------
// class MyriadDeviceMetricsTestWithParam
//------------------------------------------------------------------------------

class MyriadRangeInferMetricsTestWithParam : public MyriadMetricsTest,
                                             public MyriadRangeMetricsContainerTestParam{
protected:
    //Operations
    void SetUp() override;

    //Data section
    range_type exp_range_;
    std::string confir_param_;
};
