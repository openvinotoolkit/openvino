// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//


#include "XLink_common_cases.hpp"

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
// Implementation of methods of class XLinkDeviceTestsCommon
//------------------------------------------------------------------------------

static std::string protocolToString(XLinkProtocol_t protocol) {
    switch (protocol) {
        case X_LINK_USB_VSC:
            return std::string("USB");
        case X_LINK_PCIE:
            return std::string("PCIE");
        default:
            return std::string("ANY");
    }
}

static std::string platformToString(XLinkPlatform_t platform) {
    switch (platform) {
        case X_LINK_MYRIAD_2:
            return std::string("Myriad2");
        case X_LINK_MYRIAD_X:
            return std::string("MyriadX");
        default:
            return std::string("ANY");
    }
}

std::string XLinkDeviceTestsCommon::getTestCaseName(
    const TestParamInfo<XLinkDeviceTestsCommonParam::ParamType>&  param) {
    XLinkProtocol_t protocol = get<0>(param.param);
    XLinkPlatform_t platform = get<1>(param.param);

    return "protocol=" + protocolToString(protocol) +
            "_platform=" + platformToString(platform);
}

void XLinkDeviceTestsCommon::SetUp() {
    _protocol = get<0>(XLinkDeviceTestsCommonParam::GetParam());
    _platform = get<1>(XLinkDeviceTestsCommonParam::GetParam());
}

//------------------------------------------------------------------------------
// Implementation of methods of class XLinkOpenStreamUSBTests
//------------------------------------------------------------------------------

std::string XLinkOpenStreamTests::getTestCaseName(
    const TestParamInfo<XLinkDeviceTestsCommonParam::ParamType>& param) {
    const auto name = XLinkDeviceTestsCommon::getTestCaseName(param);

    XLinkProtocol_t protocol = get<0>(param.param);
    if (getCountSpecificDevices(X_LINK_UNBOOTED, protocol) == 0) {
        return "DISABLED_" + name;
    }

    return name;
}

XLinkOpenStreamTests::XLinkOpenStreamTests() : _handlerPtr(new XLinkHandler_t()) {
}

void XLinkOpenStreamTests::SetUp() {
    XLinkDeviceTestsCommon::SetUp();

    _deviceDesc.protocol = _protocol;
    _deviceDesc.platform = _platform;

    XLinkTestsHelper::bootDevice(_deviceDesc, _bootedDesc);
    XLinkTestsHelper::connectToDevice(_bootedDesc, _handlerPtr.get());
}

void XLinkOpenStreamTests::TearDown() {
    XLinkTestsHelper::closeDevice(_handlerPtr.get());
}
