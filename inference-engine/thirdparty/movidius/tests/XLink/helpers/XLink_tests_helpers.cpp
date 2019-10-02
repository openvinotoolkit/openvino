// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "XLink_tests_helpers.hpp"
#include <thread>

//------------------------------------------------------------------------------
// Implementation of methods of class XLinkTestsHelpersBoot
//------------------------------------------------------------------------------

std::string XLinkTestsHelpersBoot::getMyriadStickFirmwarePath(const std::string& devAddr) {
    if (devAddr.find('-') == std::string::npos) {
        throw std::invalid_argument("Invalid device address");
    }

    if (devAddr.find("ma2480") != std::string::npos) {      // MX
        return FIRMWARE_SUBFOLDER + std::string("MvNCAPI-ma2x8x.mvcmd");
    } else {                                                // M2
        return FIRMWARE_SUBFOLDER + std::string("MvNCAPI-ma2450.mvcmd");
    }
}

std::string XLinkTestsHelpersBoot::getMyriadPCIeFirmware() {
    return FIRMWARE_SUBFOLDER + std::string("MvNCAPI-mv0262.mvcmd");
}

void XLinkTestsHelpersBoot::bootUSBDevice(deviceDesc_t& deviceDesc, deviceDesc_t& bootedDeviceDesc) {
    deviceDesc_t in_deviceDesc = {};
    in_deviceDesc.protocol = X_LINK_USB_VSC;
    in_deviceDesc.platform = X_LINK_ANY_PLATFORM;

    // Get device name
    ASSERT_EQ(X_LINK_SUCCESS,
            XLinkFindFirstSuitableDevice(X_LINK_UNBOOTED, in_deviceDesc, &deviceDesc));

    std::string firmwarePath;
    // Get firmware
    ASSERT_NO_THROW(firmwarePath = getMyriadStickFirmwarePath(deviceDesc.name));


    printf("Would boot (%s) device with firmware (%s) \n", deviceDesc.name, firmwarePath.c_str());

    // Boot device
    ASSERT_EQ(X_LINK_SUCCESS, XLinkBoot(&deviceDesc, firmwarePath.c_str()));
    // FIXME: need to find a way to avoid this sleep
    std::this_thread::sleep_for(std::chrono::seconds(2));

    // Check, that device booted
    ASSERT_EQ(X_LINK_SUCCESS,
            XLinkFindFirstSuitableDevice(X_LINK_BOOTED, in_deviceDesc, &bootedDeviceDesc));
}

void XLinkTestsHelpersBoot::connectToBootedUSB(deviceDesc_t& bootedDeviceDesc, XLinkHandler_t* handler) {
    // Handler should be preallocated
    if (!handler){
        GTEST_FAIL();
    }
    // Connect to device
    memset(handler, 0, sizeof(XLinkHandler_t));

    handler->protocol = bootedDeviceDesc.protocol;
    handler->devicePath = bootedDeviceDesc.name;
    ASSERT_EQ(X_LINK_SUCCESS, XLinkConnect(handler));
    std::this_thread::sleep_for(std::chrono::seconds(1));
}

void XLinkTestsHelpersBoot::connectUSBAndClose(deviceDesc_t& bootedDeviceDesc) {
    XLinkHandler_t *handler = (XLinkHandler_t *)malloc(sizeof(XLinkHandler_t));

    connectToBootedUSB(bootedDeviceDesc, handler);
    closeUSBDevice(handler);
}

void XLinkTestsHelpersBoot::closeUSBDevice(XLinkHandler_t* handler) {
    std::string deviceName(handler->devicePath);

    ASSERT_EQ(X_LINK_SUCCESS, XLinkResetRemote(handler->linkId));
    free(handler);

    std::this_thread::sleep_for(std::chrono::seconds(2));

    // Make sure that device is closed
    deviceDesc_t deviceDesc = {};
    deviceDesc.protocol = X_LINK_USB_VSC;
    deviceDesc.platform = X_LINK_ANY_PLATFORM;
    strcpy(deviceDesc.name, deviceName.c_str());

    deviceDesc_t foundDeviceDesc = {};
    ASSERT_EQ(X_LINK_DEVICE_NOT_FOUND,
            XLinkFindFirstSuitableDevice(X_LINK_BOOTED, deviceDesc, &foundDeviceDesc));

    deviceDesc_t findUnbootedDevice = {};
    findUnbootedDevice.protocol = X_LINK_USB_VSC;
    findUnbootedDevice.platform = X_LINK_ANY_PLATFORM;

    ASSERT_EQ(X_LINK_SUCCESS,
            XLinkFindFirstSuitableDevice(X_LINK_UNBOOTED, findUnbootedDevice, &foundDeviceDesc));
}

void XLinkTestsHelpersBoot::copyDeviceDescr(deviceDesc_t *destDeviceDescr,
                     const deviceDesc_t sourceDeviceDescr) {
    destDeviceDescr->platform = sourceDeviceDescr.platform;
    destDeviceDescr->protocol = sourceDeviceDescr.protocol;
    strncpy(destDeviceDescr->name, sourceDeviceDescr.name, XLINK_MAX_NAME_SIZE);
}


XLinkError_t XLinkTestsHelpersBoot::findDeviceOnIndex(
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
    if (foundDevices > index) {
        copyDeviceDescr(out_foundDevicesPtr, deviceDescArray[index]);
        return X_LINK_SUCCESS;
    }
    return X_LINK_DEVICE_NOT_FOUND;
}

int XLinkTestsHelpersBoot::getAmountOfDevices(const XLinkDeviceState_t state,
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

//------------------------------------------------------------------------------
// Implementation of methods of class XLinkTestsHelpersOneUSBDevice
//------------------------------------------------------------------------------

void XLinkTestsHelpersOneUSBDevice::bootUSBDevice() {
    XLinkTestsHelpersBoot::bootUSBDevice(_deviceDesc, _bootedDesc);
    handler = (XLinkHandler_t *)malloc(sizeof(XLinkHandler_t));
    XLinkTestsHelpersBoot::connectToBootedUSB(_bootedDesc, handler);
}

void XLinkTestsHelpersOneUSBDevice::closeUSBDevice() {
    XLinkTestsHelpersBoot::closeUSBDevice(handler);
}
