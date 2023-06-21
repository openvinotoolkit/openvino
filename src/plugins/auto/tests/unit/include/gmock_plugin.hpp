// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <gmock/gmock.h>
#include "openvino/runtime/core.hpp"
#include "plugin.hpp"
#include <iostream>

using namespace ov::mock_auto_plugin;
namespace ov {
namespace mock_auto_plugin {

class MockAutoPlugin : public Plugin {
public:
    MOCK_METHOD((std::string), get_device_list, ((const ov::AnyMap&)), (const, override));
    MOCK_METHOD((std::list<DeviceInformation>),
                get_valid_device,
                ((const std::vector<DeviceInformation>&), const std::string&),
                (const, override));
    MOCK_METHOD(DeviceInformation, select_device, ((const std::vector<DeviceInformation>&),
                const std::string&, unsigned int), (override));
    MOCK_METHOD((std::vector<DeviceInformation>), parse_meta_devices,
                (const std::string&, const ov::AnyMap&), (const, override));
};
} // namespace mock_auto_plugin
} // namespace ov
