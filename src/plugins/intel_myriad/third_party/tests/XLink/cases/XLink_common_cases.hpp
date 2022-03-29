// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "XLink.h"
#include "XLink_tests_helpers.hpp"

#include "gtest/gtest.h"
#include <memory>

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

class XLinkFindAllSuitableDevicesTests : public XLinkTests,
                                         public XLinkDeviceTestsCommonParam {};


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
