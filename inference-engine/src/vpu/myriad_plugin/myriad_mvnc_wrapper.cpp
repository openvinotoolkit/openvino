// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "myriad_mvnc_wrapper.h"

#include <ie_common.h>

using namespace vpu::MyriadPlugin;

//------------------------------------------------------------------------------
// Implementation of methods of class Mvnc
//------------------------------------------------------------------------------

Mvnc::Mvnc() {
    WatchdogHndl_t* watchdogHndl = nullptr;
    if (watchdog_create(&watchdogHndl) != WD_ERRNO) {
        IE_THROW() << "Cannot create watchdog.";
    }

    m_watcdogPtr = WatchdogUniquePtr(watchdogHndl, [](WatchdogHndl_t* watchdogHndl) {
        watchdog_destroy(watchdogHndl);
    });
}

std::vector<ncDeviceDescr_t> Mvnc::AvailableDevicesDesc() const {
    int deviceCount = 0;
    std::vector<ncDeviceDescr_t> availableDevices(NC_MAX_DEVICES);
    if (ncAvailableDevices(&availableDevices[0], NC_MAX_DEVICES, &deviceCount) != NC_OK) {
        IE_THROW() << "Cannot receive available devices.";
    }
    availableDevices.resize(deviceCount);

    return availableDevices;
}

std::vector<std::string> Mvnc::AvailableDevicesNames() const {
    auto _availableDevicesDesc = AvailableDevicesDesc();

    std::vector<std::string> availableDevices;
    for (size_t i = 0; i < _availableDevicesDesc.size(); ++i) {
        availableDevices.emplace_back(std::string(_availableDevicesDesc[i].name));
    }

    return availableDevices;
}
