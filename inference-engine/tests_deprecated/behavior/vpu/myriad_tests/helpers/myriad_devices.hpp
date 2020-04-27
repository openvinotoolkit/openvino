// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "string"
#include "vector"
#include <algorithm>
#include <mvnc.h>
#include <XLink.h>

//------------------------------------------------------------------------------
// class MyriadDevicesInfo
//------------------------------------------------------------------------------

class MyriadDevicesInfo {
public:
    // Constants
    static constexpr char kMyriadXName[] = "ma2480";
    static constexpr char kMyriad2Name[] = "ma2450";

    //Constructor
    MyriadDevicesInfo();

    //Accessors
    inline const std::string& firmwareDir();

    std::vector<std::string> getDevicesList(
            const ncDeviceProtocol_t deviceProtocol = NC_ANY_PROTOCOL,
            const ncDevicePlatform_t devicePlatform = NC_ANY_PLATFORM,
            const XLinkDeviceState_t state = X_LINK_ANY_STATE
            );

    inline bool isMyriadXDevice(const std::string &device_name);
    inline bool isMyriad2Device(const std::string &device_name);

    inline bool isMyriadBootedDevice(const std::string &device_name);
    inline bool isMyriadUnbootedDevice(const std::string &device_name);

    int getAmountOfDevices(const ncDeviceProtocol_t deviceProtocol = NC_ANY_PROTOCOL,
                           const ncDevicePlatform_t devicePlatform = NC_ANY_PLATFORM,
                           const XLinkDeviceState_t state = X_LINK_ANY_STATE);

    inline long getAmountOfBootedDevices(const ncDeviceProtocol_t deviceProtocol);
    inline long getAmountOfUnbootedDevices(const ncDeviceProtocol_t deviceProtocol);

private:
    std::string firmware_dir_;
};

const std::string& MyriadDevicesInfo::firmwareDir() {
    return firmware_dir_;
}

bool MyriadDevicesInfo::isMyriadXDevice(const std::string &device_name) {
    return (device_name.find(kMyriadXName) != std::string::npos);
}

bool MyriadDevicesInfo::isMyriad2Device(const std::string &device_name) {
    return (device_name.find(kMyriad2Name) != std::string::npos);
}

bool MyriadDevicesInfo::isMyriadBootedDevice(const std::string &device_name) {
    return (!isMyriad2Device(device_name) && !isMyriadXDevice(device_name));
}

bool MyriadDevicesInfo::isMyriadUnbootedDevice(const std::string &device_name) {
    return (isMyriad2Device(device_name) || isMyriadXDevice(device_name));
}

long MyriadDevicesInfo::getAmountOfUnbootedDevices(const ncDeviceProtocol_t deviceProtocol) {
    return getAmountOfDevices(deviceProtocol, NC_ANY_PLATFORM, X_LINK_UNBOOTED);
}

long MyriadDevicesInfo::getAmountOfBootedDevices(const ncDeviceProtocol_t deviceProtocol) {
    return getAmountOfDevices(deviceProtocol, NC_ANY_PLATFORM, X_LINK_BOOTED);
}