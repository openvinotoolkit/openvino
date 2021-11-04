// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <gmock/gmock.h>
#include "ie_icore.hpp"
#include "multi_device_plugin.hpp"
#include <iostream>

using namespace MultiDevicePlugin;
namespace MockMultiDevice {

class MockMultiDeviceInferencePlugin : public MultiDeviceInferencePlugin {
public:
    // MOCK_METHOD2(FilterDevice, std::vector<DeviceInformation>(const std::vector<DeviceInformation>&,
    //                                            const std::map<std::string, std::string>&));
    // MOCK_CONST_METHOD2(ParseMetaDevices, std::vector<DeviceInformation>(const std::string&,
    //                                            const std::map<std::string, std::string>&));
    //
    // virtual std::vector<MultiDevicePlugin::DeviceInformation> ParseMetaDevices(const std::string & devicesRequestsCfg,
    //                                                                    const std::map<std::string, std::string> & config) const;
    //
    MOCK_METHOD(DeviceInformation, SelectDevice, ((const std::vector<DeviceInformation>&),
                const std::string&), (override));
    MOCK_METHOD((std::vector<DeviceInformation>), ParseMetaDevices,
            (const std::string&, (const std::map<std::string, std::string>&)), (const, override));
};
}// namespace MockMultiDevice
