// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <gmock/gmock.h>
#include "ie_icore.hpp"
#include "plugin.hpp"
#include <iostream>

using namespace MockMultiDevicePlugin;
namespace MockMultiDevice {

class MockMultiDeviceInferencePlugin : public MultiDeviceInferencePlugin {
public:
    MOCK_METHOD((std::string), GetDeviceList, ((const std::map<std::string, std::string>&)), (const, override));
    MOCK_METHOD(DeviceInformation, SelectDevice, ((const std::vector<DeviceInformation>&),
                const std::string&, unsigned int), (override));
    MOCK_METHOD((std::vector<DeviceInformation>), ParseMetaDevices,
                (const std::string&, (const std::map<std::string, std::string>&)), (const, override));
};
}// namespace MockMultiDevice
