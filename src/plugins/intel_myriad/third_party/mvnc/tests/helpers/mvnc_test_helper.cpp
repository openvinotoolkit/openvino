// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <iostream>
#include "mvnc_data.h"
#include "mvnc_test_helper.h"

//------------------------------------------------------------------------------
//      Implementations of helpers - counters
//------------------------------------------------------------------------------
int getAmountOfDevices(const ncDeviceProtocol_t deviceProtocol,
                                        const XLinkDeviceState_t state) {
    deviceDesc_t req_deviceDesc = {};
    req_deviceDesc.protocol = convertProtocolToXlink(deviceProtocol);

    deviceDesc_t deviceDescArray[NC_MAX_DEVICES] = {};
    unsigned int foundDevices = 0;
    XLinkFindAllSuitableDevices(
            state, req_deviceDesc, deviceDescArray, NC_MAX_DEVICES, &foundDevices);

    return foundDevices;
}

long getAmountOfPCIeDevices() {
    return getAmountOfDevices(NC_PCIE);
}

long getAmountOfUSBDevices() {
    return getAmountOfDevices(NC_USB);
}

//------------------------------------------------------------------------------
//      Implementations of helpers - get devices
//------------------------------------------------------------------------------
std::vector<std::string> getDevicesList(const ncDeviceProtocol_t deviceProtocol,
                                                         const XLinkDeviceState_t state) {

    deviceDesc_t req_deviceDesc = {};
    req_deviceDesc.protocol = convertProtocolToXlink(deviceProtocol);

    deviceDesc_t deviceDescArray[NC_MAX_DEVICES] = {};
    unsigned int foundDevices = 0;
    XLinkFindAllSuitableDevices(
            state, req_deviceDesc, deviceDescArray, NC_MAX_DEVICES, &foundDevices);

    std::vector < std::string > devNames;
    for (int i = 0; i < foundDevices; ++i) {
        devNames.emplace_back(deviceDescArray[i].name);
    }

    return devNames;
}

//------------------------------------------------------------------------------
//      Implementation of helpers - comparators
//------------------------------------------------------------------------------
bool isMyriadXUSBDevice(const std::string &deviceName) {
    return (deviceName.find(MYRIAD_X_NAME_STR) != std::string::npos);
}

bool isMyriad2USBDevice(const std::string &deviceName) {
    return (deviceName.find(MYRIAD_2_NAME_STR) != std::string::npos);
}

bool isMyriadPCIeDevice(const std::string &deviceName) {
    return deviceName.find(std::string(PCIE_NAME_STR)) != std::string::npos;
}

bool isMyriadUSBDevice(const std::string &deviceName) {
    return (isMyriad2USBDevice(deviceName)
            || isMyriadXUSBDevice(deviceName)
            || isMyriadBootedUSBDevice(deviceName));
}

bool isMyriadBootedUSBDevice(const std::string &deviceName) {
    return (!isMyriad2USBDevice(deviceName) &&
            !isMyriadXUSBDevice(deviceName) &&
            !isMyriadPCIeDevice(deviceName));
}

bool isSameProtocolDevice(const std::string &deviceName, const ncDeviceProtocol_t expectedProtocol) {
    switch (expectedProtocol) {
        case NC_USB:            return isMyriadUSBDevice(deviceName);
        case NC_PCIE:           return isMyriadPCIeDevice(deviceName);
        case NC_ANY_PROTOCOL:
            return isMyriadPCIeDevice(deviceName) || isMyriadUSBDevice(deviceName);
        default:
            std::cout << "Unknown device protocol" << std::endl;
            return false;
    }
}

bool
isSamePlatformUSBDevice(const std::string &deviceName) {
    return isMyriadXUSBDevice(deviceName);
}

//------------------------------------------------------------------------------
//      Implementation of helpers - file loader
//------------------------------------------------------------------------------
bool readBINFile(const std::string &fileName, std::vector<char> &buf) {
    std::ifstream file(fileName, std::ios_base::binary | std::ios_base::ate);
    if (file.fail()) {
        std::cout << "Can't open file!" << std::endl;
        return false;
    }
    buf.resize(static_cast<unsigned int>(file.tellg()));
    file.seekg(0);
    file.read(buf.data(), buf.size());
    return true;
}
