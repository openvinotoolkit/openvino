// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//


#pragma once
#include "gtest/gtest.h"

#include "XLink.h"
#include "XLink_tests_helpers.hpp"

//------------------------------------------------------------------------------
//      class XLinkTests
//------------------------------------------------------------------------------
class XLinkTests : public ::testing::Test {
public:
    static void SetUpTestCase();
};

//------------------------------------------------------------------------------
//      class XLinkBootUSBTests
//------------------------------------------------------------------------------

class XLinkBootUSBTests :   public XLinkTestsHelpersBoot,
                            public XLinkTests {
protected:
    void SetUp() override;
};

//------------------------------------------------------------------------------
//      class XLinkOneDeviceUSBTests
//------------------------------------------------------------------------------

class XLinkOneDeviceUSBTests :  public XLinkTestsHelpersOneUSBDevice,
                                public XLinkTests {};

//------------------------------------------------------------------------------
//      class XLinkOpenStreamUSBTests
//------------------------------------------------------------------------------

class XLinkOpenStreamUSBTests : public XLinkOneDeviceUSBTests {
    void SetUp() override;
    void TearDown() override ;
};

//------------------------------------------------------------------------------
//      class XLinkFindFirstSuitableDeviceUSBTests
//------------------------------------------------------------------------------

class XLinkFindFirstSuitableDeviceUSBTests : public XLinkBootUSBTests {};

//------------------------------------------------------------------------------
//      class XLinkFindAllSuitableDevicesTests
//------------------------------------------------------------------------------

class XLinkFindAllSuitableDevicesTests : public XLinkBootUSBTests {};

//------------------------------------------------------------------------------
//      class XLinkResetAllUSBTests
//------------------------------------------------------------------------------
class XLinkResetAllUSBTests : public XLinkBootUSBTests {};

//------------------------------------------------------------------------------
//      class XLinkConnectUSBTests
//------------------------------------------------------------------------------
class XLinkConnectUSBTests : public XLinkBootUSBTests {};

//------------------------------------------------------------------------------
//      class XLinkNullPtrTests
//------------------------------------------------------------------------------
class XLinkNullPtrTests: public XLinkTests {};

//------------------------------------------------------------------------------
//      class XLinkFindPCIEDeviceTests
//------------------------------------------------------------------------------
class XLinkFindPCIEDeviceTests: public XLinkBootUSBTests {
public:
    static deviceDesc_t getPCIeDeviceRequirements();
    int available_devices = 0;
protected:
    void SetUp() override;
};

//------------------------------------------------------------------------------
//      class XLinkResetRemoteUSBTests
//------------------------------------------------------------------------------

class XLinkResetRemoteUSBTests : public XLinkTestsHelpersBoot,
                              public XLinkTests {};
