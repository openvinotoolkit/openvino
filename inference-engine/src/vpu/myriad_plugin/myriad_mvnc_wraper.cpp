// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "myriad_mvnc_wraper.h"
#include "details/ie_exception.hpp"

using namespace vpu::MyriadPlugin;

//------------------------------------------------------------------------------
// Implementation of methods of class Mvnc
//------------------------------------------------------------------------------

Mvnc::Mvnc() {
    WatchdogHndl_t* watchdogHndl = nullptr;
    if (watchdog_create(&watchdogHndl) != WD_ERRNO) {
        THROW_IE_EXCEPTION << "Cannot create watchdog.";
    }

    m_watcdogPtr = WatchdogUniquePtr(watchdogHndl, [](WatchdogHndl_t* watchdogHndl) {
        watchdog_destroy(watchdogHndl);
    });
}

std::vector<ncDeviceDescr_t> Mvnc::AvailableDevicesDesc() const {
    int deviceCount = 0;
    std::vector<ncDeviceDescr_t> availableDevices(NC_MAX_DEVICES);
    if (ncAvailableDevices(&availableDevices[0], NC_MAX_DEVICES, &deviceCount) != NC_OK) {
        THROW_IE_EXCEPTION << "Cannot receive available devices.";
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

std::string Mvnc::ncStatusToStr(ncGraphHandle_t *graphHandle, ncStatus_t status) {
#define MVNC_STATUS_TO_STR(E) case E: return #E;
    switch (status) {
        MVNC_STATUS_TO_STR(NC_OK)
        MVNC_STATUS_TO_STR(NC_BUSY)
        MVNC_STATUS_TO_STR(NC_ERROR)
        MVNC_STATUS_TO_STR(NC_OUT_OF_MEMORY)
        MVNC_STATUS_TO_STR(NC_DEVICE_NOT_FOUND)
        MVNC_STATUS_TO_STR(NC_INVALID_PARAMETERS)
        MVNC_STATUS_TO_STR(NC_TIMEOUT)
        MVNC_STATUS_TO_STR(NC_MVCMD_NOT_FOUND)
        MVNC_STATUS_TO_STR(NC_NOT_ALLOCATED)
        MVNC_STATUS_TO_STR(NC_UNAUTHORIZED)
        MVNC_STATUS_TO_STR(NC_UNSUPPORTED_FEATURE)
        MVNC_STATUS_TO_STR(NC_UNSUPPORTED_GRAPH_FILE)
        MVNC_STATUS_TO_STR(NC_UNSUPPORTED_CONFIGURATION_FILE)
        case NC_MYRIAD_ERROR: {
            if (graphHandle == nullptr) {
                return "NC_MYRIAD_ERROR";
            } else {
                auto debugInfo = getGraphInfo<char>(graphHandle, NC_RO_GRAPH_DEBUG_INFO, NC_DEBUG_BUFFER_SIZE);
                if (debugInfo.empty()) {
                    return "NC_MYRIAD_ERROR";
                } else {
                    return std::string(debugInfo.begin(), debugInfo.end());
                }
            }
        }
        default:
            return "UNKNOWN MVNC STATUS";
    }
#undef MVNC_STATUS_TO_STR
}
