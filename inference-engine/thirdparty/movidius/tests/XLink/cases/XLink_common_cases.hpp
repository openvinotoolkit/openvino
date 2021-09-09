// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "XLink.h"
#include "XLink_tests_helpers.hpp"

#include "gtest/gtest.h"
#include <memory>


typedef enum {
    DEVICE_GET_THERMAL_STATS        = 0,
    DEVICE_GET_CAPABILITIES         = 1,
    DEVICE_GET_USED_MEMORY          = 2,
    DEVICE_GET_DEVICE_ID            = 3,
    DEVICE_WATCHDOG_PING            = 4,
    DEVICE_SET_STDIO_REDIRECT_XLINK = 5,
    DEVICE_SET_POWER_CONFIG         = 6,
    DEVICE_RESET_POWER_CONFIG       = 7,
    DEVICE_ENABLE_ASYNC_DMA         = 8,
    DEVICE_COMMAND_LAST             = 9
} deviceCommandType_t;

typedef struct {
    deviceCommandType_t type;
    uint32_t arg;
} deviceCommand_t;

using namespace ::testing;
using XLinkDeviceTestsCommonParam = WithParamInterface<std::tuple<XLinkProtocol_t, XLinkPlatform_t>>;

//------------------------------------------------------------------------------
//      class XLinkTests
//------------------------------------------------------------------------------
class XLinkTests : public ::testing::Test,
                   protected XLinkTestsHelper {
public:
    static void SetUpTestCase();
};

//------------------------------------------------------------------------------
//      class XLinkNullPtrTests
//------------------------------------------------------------------------------
class XLinkNullPtrTests: public XLinkTests {};

//------------------------------------------------------------------------------
//      class XLinkFindAllSuitableDevicesTests
//------------------------------------------------------------------------------

class XLinkFindAllSuitableDevicesTests : public XLinkTests {};


class XLinkWriteDataWithTimeoutTests : public XLinkTests {};


//------------------------------------------------------------------------------
//      class XLinkCommonTests
//------------------------------------------------------------------------------
class XLinkDeviceTestsCommon : public XLinkTests,
                               public XLinkDeviceTestsCommonParam {
public:
    //Operations
    static std::string getTestCaseName(
        const TestParamInfo<XLinkDeviceTestsCommonParam::ParamType>&  param);

    void SetUp() override;

protected:
    XLinkProtocol_t _protocol;
    XLinkPlatform_t _platform;
};

//------------------------------------------------------------------------------
//      class XLinkBootUSBTests
//------------------------------------------------------------------------------

class XLinkBootTests : public XLinkDeviceTestsCommon {};

//------------------------------------------------------------------------------
//      class XLinkConnectTests
//------------------------------------------------------------------------------
class XLinkConnectTests : public XLinkDeviceTestsCommon {};

//------------------------------------------------------------------------------
//      class XLinkFindFirstSuitableDeviceTests
//------------------------------------------------------------------------------

class XLinkFindFirstSuitableDeviceTests : public XLinkDeviceTestsCommon {};

//------------------------------------------------------------------------------
//      class XLinkFindFirstSuitableBootedDeviceTests
//------------------------------------------------------------------------------

class XLinkFindFirstSuitableDevicePlatformTests : public XLinkDeviceTestsCommon {};

//------------------------------------------------------------------------------
//      class XLinkResetRemoteTests
//------------------------------------------------------------------------------
class XLinkResetRemoteTests : public XLinkDeviceTestsCommon {};

//------------------------------------------------------------------------------
//      class XLinkResetAllTests
//------------------------------------------------------------------------------
class XLinkResetAllTests : public XLinkDeviceTestsCommon {};



//------------------------------------------------------------------------------
//      class XLinkOpenStreamTests
//------------------------------------------------------------------------------

class XLinkOpenStreamTests : public XLinkDeviceTestsCommon {
public:
    //Operations
    static std::string getTestCaseName(
        const TestParamInfo<XLinkDeviceTestsCommonParam::ParamType>& param);

protected:
    XLinkOpenStreamTests();

    void SetUp() override;
    void TearDown() override;

protected:
    std::unique_ptr<XLinkHandler_t> _handlerPtr;
    deviceDesc_t _deviceDesc = {};
    deviceDesc_t _bootedDesc = {};
};
