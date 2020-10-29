// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "myriad_mvnc_wraper.h"
#include "details/ie_exception.hpp"

using namespace vpu::MyriadPlugin;

//------------------------------------------------------------------------------
// Implementation of methods of class Mvnc
//------------------------------------------------------------------------------

Mvnc::Mvnc() :
    _devicesPtr(new struct ncDeviceDescr_t[NC_MAX_DEVICES]) {
}

std::vector<ncDeviceDescr_t> Mvnc::AvailableDevicesDesc() const {
    int deviceCount = 0;
    if (ncAvailableDevices(_devicesPtr.get(), NC_MAX_DEVICES, &deviceCount) != NC_OK) {
        THROW_IE_EXCEPTION << "Cannot receive available devices.";
    }

    std::vector<ncDeviceDescr_t> availableDevices;
    for (int i = 0; i < deviceCount; ++i) {
        availableDevices.push_back(_devicesPtr[i]);
    }

    return availableDevices;
}

std::vector<std::string> Mvnc::AvailableDevicesNames() const {
    auto _availableDevicesDesc = AvailableDevicesDesc();

    std::vector<std::string> availableDevices;
    for (size_t i = 0; i < _availableDevicesDesc.size(); ++i) {
        availableDevices.emplace_back(std::string(_devicesPtr[i].name));
    }

    return availableDevices;
}
