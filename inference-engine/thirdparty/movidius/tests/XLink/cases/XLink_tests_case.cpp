// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//


#include "XLink_tests_case.hpp"

#include <thread>

static XLinkGlobalHandler_t globalHandler;

//------------------------------------------------------------------------------
// Implementation of methods of class XLinkTests
//------------------------------------------------------------------------------

void XLinkTests::SetUpTestCase() {
    ASSERT_EQ(X_LINK_SUCCESS, XLinkInitialize(&globalHandler));

    // Deprecated field usage. Begin.
    globalHandler.protocol = USB_VSC;
    // Deprecated field usage. End.

    // Waiting for initialization
    std::this_thread::sleep_for(std::chrono::seconds(1));
}

//------------------------------------------------------------------------------
// Implementation of methods of class XLinkOpenStreamUSBTests
//------------------------------------------------------------------------------
void XLinkOpenStreamUSBTests::SetUp() {
    if (getAmountOfDevices(X_LINK_UNBOOTED, X_LINK_USB_VSC) != 0)
        bootUSBDevice();
}

void XLinkOpenStreamUSBTests::TearDown() {
    if (getAmountOfDevices(X_LINK_BOOTED, X_LINK_USB_VSC) != 0)
        closeUSBDevice();
}

//------------------------------------------------------------------------------
// Implementation of methods of class XLinkFindPCIEDeviceTests
//------------------------------------------------------------------------------

deviceDesc_t XLinkFindPCIEDeviceTests::getPCIeDeviceRequirements() {
    deviceDesc_t deviceDesc = {};
    deviceDesc.protocol = X_LINK_PCIE;
    deviceDesc.platform = X_LINK_ANY_PLATFORM;
    return deviceDesc;
}

void XLinkFindPCIEDeviceTests::SetUp() {
    available_devices = getAmountOfDevices(X_LINK_ANY_STATE, X_LINK_PCIE);
}

//------------------------------------------------------------------------------
// Implementation of methods of class XLinkBootUSBTests
//------------------------------------------------------------------------------

void XLinkBootUSBTests::SetUp() {
    ASSERT_GE(getAmountOfDevices(X_LINK_UNBOOTED, X_LINK_USB_VSC), 1);
}
