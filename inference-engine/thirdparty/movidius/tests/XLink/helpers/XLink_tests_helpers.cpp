// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "XLink_tests_helpers.hpp"
#include <thread>

//------------------------------------------------------------------------------
// Implementation of methods of class XLinkTestsHelpersBoot
//------------------------------------------------------------------------------

void XLinkTestsHelper::bootDevice(const deviceDesc_t& in_deviceDesc, deviceDesc_t& out_bootedDeviceDesc) {
    deviceDesc_t tmp_deviceDesc = {};
    ASSERT_EQ(X_LINK_SUCCESS,
              XLinkFindFirstSuitableDevice(X_LINK_UNBOOTED, in_deviceDesc, &tmp_deviceDesc));

    std::string firmwarePath;
    ASSERT_NO_THROW(firmwarePath = getMyriadFirmwarePath(tmp_deviceDesc));
    printf("Would boot (%s) device with firmware (%s) \n", tmp_deviceDesc.name, firmwarePath.c_str());

    ASSERT_EQ(X_LINK_SUCCESS, XLinkBoot(&tmp_deviceDesc, firmwarePath.c_str()));
    // FIXME: need to find a way to avoid this sleep
    std::this_thread::sleep_for(kBootTimeoutSec);

    // Check, that device booted
    tmp_deviceDesc.platform = X_LINK_ANY_PLATFORM;
    memset(tmp_deviceDesc.name, 0, XLINK_MAX_NAME_SIZE);
    ASSERT_EQ(X_LINK_SUCCESS,
              XLinkFindFirstSuitableDevice(X_LINK_BOOTED, tmp_deviceDesc, &out_bootedDeviceDesc));
}

void XLinkTestsHelper::connectToDevice(deviceDesc_t& in_bootedDeviceDesc, XLinkHandler_t* out_handler) {
    if (!out_handler){
        GTEST_FAIL();
    }

    memset(out_handler, 0, sizeof(XLinkHandler_t));
    out_handler->protocol = in_bootedDeviceDesc.protocol;
    out_handler->devicePath = in_bootedDeviceDesc.name;

    ASSERT_EQ(X_LINK_SUCCESS, XLinkConnect(out_handler));
}

void XLinkTestsHelper::closeDevice(XLinkHandler_t* handler) {
    ASSERT_EQ(X_LINK_SUCCESS, XLinkResetRemote(handler->linkId));
    std::this_thread::sleep_for(kResetTimeoutSec);

    // Make sure that device is closed
    deviceDesc_t deviceDesc = {};
    deviceDesc.protocol = handler->protocol;
    deviceDesc.platform = X_LINK_ANY_PLATFORM;
    strcpy(deviceDesc.name, handler->devicePath);

    deviceDesc_t foundDeviceDesc = {};
    ASSERT_EQ(X_LINK_DEVICE_NOT_FOUND,
              XLinkFindFirstSuitableDevice(X_LINK_BOOTED, deviceDesc, &foundDeviceDesc));
}

void XLinkTestsHelper::connectAndCloseDevice(deviceDesc_t& in_bootedDeviceDesc) {
    XLinkHandler_t handler = {0};

    connectToDevice(in_bootedDeviceDesc, &handler);
    closeDevice(&handler);
}

std::string XLinkTestsHelper::getMyriadUSBFirmwarePath(const std::string& deviceName) {
    if (deviceName.find('-') == std::string::npos) {
        throw std::invalid_argument("Invalid device address");
    }

    std::string firmwareName = "usb-ma2450.mvcmd";
    if (deviceName.find("ma2480") != std::string::npos) {
        firmwareName = "usb-ma2x8x.mvcmd";
    }

    return FIRMWARE_SUBFOLDER + firmwareName;
}

std::string XLinkTestsHelper::getMyriadFirmwarePath(const deviceDesc_t& in_deviceDesc) {
    if(in_deviceDesc.protocol != X_LINK_USB_VSC &&
        in_deviceDesc.protocol != X_LINK_PCIE) {
        throw std::invalid_argument("Device protocol must be specified");
    }

    if(in_deviceDesc.protocol == X_LINK_PCIE) {
#if defined(_WIN32)
        const std::string extension = "elf";
#else
        const std::string extension = "mvcmd";
#endif
        return FIRMWARE_SUBFOLDER + std::string("pcie-ma2x8x.") + extension;
    }

    return getMyriadUSBFirmwarePath(in_deviceDesc.name);
}

XLinkError_t XLinkTestsHelper::findDeviceOnIndex(
    const int index,
    const XLinkDeviceState_t deviceState,
    const deviceDesc_t in_deviceRequirements,
    deviceDesc_t *out_foundDevicesPtr) {

    deviceDesc_t deviceDescArray[XLINK_MAX_DEVICES] = {};
    unsigned int foundDevices = 0;
    XLinkError_t rc = XLinkFindAllSuitableDevices(
            deviceState, in_deviceRequirements, deviceDescArray, XLINK_MAX_DEVICES, &foundDevices);

    if (rc != X_LINK_SUCCESS) {
        return rc;
    }

    if (foundDevices <= index) {
        return X_LINK_DEVICE_NOT_FOUND;
    }

    out_foundDevicesPtr->platform = deviceDescArray[index].platform;
    out_foundDevicesPtr->protocol = deviceDescArray[index].protocol;
    strncpy(out_foundDevicesPtr->name, deviceDescArray[index].name, XLINK_MAX_NAME_SIZE);
    return X_LINK_SUCCESS;
}

int XLinkTestsHelper::getCountSpecificDevices(const XLinkDeviceState_t state,
                                              const XLinkProtocol_t deviceProtocol,
                                              const XLinkPlatform_t devicePlatform) {
    deviceDesc_t req_deviceDesc = {};
    req_deviceDesc.protocol = deviceProtocol;
    req_deviceDesc.platform = devicePlatform;

    deviceDesc_t deviceDescArray[XLINK_MAX_DEVICES] = {};
    unsigned int foundDevices = 0;
    XLinkFindAllSuitableDevices(
            state, req_deviceDesc, deviceDescArray, XLINK_MAX_DEVICES, &foundDevices);

    return foundDevices;
}
