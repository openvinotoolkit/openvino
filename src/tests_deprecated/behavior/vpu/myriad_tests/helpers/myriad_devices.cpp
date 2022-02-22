// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "myriad_devices.hpp"
#include <usb_boot.h>
#include <mvnc_ext.h>
#include "mvnc_data.h"

//------------------------------------------------------------------------------
// Implementation of methods of class MyriadDevicesInfo
//------------------------------------------------------------------------------

constexpr char MyriadDevicesInfo::kMyriadXName[];
constexpr char MyriadDevicesInfo::kMyriad2Name[];
constexpr char MyriadDevicesInfo::kMyriadXPCIeName[];

MyriadDevicesInfo::MyriadDevicesInfo() {
#if (defined(_WIN32) || defined(_WIN64))
    initialize_usb_boot();
#endif


#if !(defined(_WIN32) || defined(_WIN64))
    firmware_dir_ = "./lib/";
#endif
}

std::vector<std::string> MyriadDevicesInfo::getDevicesList(
                    const ncDeviceProtocol_t deviceProtocol,
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

int MyriadDevicesInfo::getAmountOfDevices(
                            const ncDeviceProtocol_t deviceProtocol,
                            const XLinkDeviceState_t state) {
    deviceDesc_t req_deviceDesc = {};
    req_deviceDesc.protocol = convertProtocolToXlink(deviceProtocol);

    deviceDesc_t deviceDescArray[NC_MAX_DEVICES] = {};
    unsigned int foundDevices = 0;
    XLinkFindAllSuitableDevices(
            state, req_deviceDesc, deviceDescArray, NC_MAX_DEVICES, &foundDevices);

    return foundDevices;
}
