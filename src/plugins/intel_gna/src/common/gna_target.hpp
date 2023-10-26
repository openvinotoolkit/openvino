// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>

#include "gna2-common-api.h"
#include "openvino/runtime/intel_gna/properties.hpp"

namespace ov {
namespace intel_gna {
namespace target {

enum class DeviceVersion {
    NotSet = -1,
    SoftwareEmulation = 0,
    GNA1_0 = 0x10,
    GNAEmbedded1_0 = 0x10e,
    GNA2_0 = 0x20,
    GNA3_0 = 0x30,
    GNA3_1 = 0x31e,
    GNA3_5 = 0x35,
    GNAEmbedded3_5 = 0x35e,
    GNA3_6 = 0x36e,
    GNA4_0 = 0x40e,
    Default = GNA3_5
};

class Target {
public:
    DeviceVersion get_detected_device_version() const;
    DeviceVersion get_user_set_execution_target() const;
    DeviceVersion get_user_set_compile_target() const;
    void set_detected_device_version(DeviceVersion detected_device);
    void set_user_set_execution_target(DeviceVersion execution_target);
    void set_user_set_compile_target(DeviceVersion compile_target);
    DeviceVersion get_effective_execution_target() const;
    DeviceVersion get_effective_compile_target() const;

private:
    DeviceVersion detected_device_version = DeviceVersion::SoftwareEmulation;
    DeviceVersion user_set_execution_target = DeviceVersion::NotSet;
    DeviceVersion user_set_compile_target = DeviceVersion::NotSet;
};

DeviceVersion HwGenerationToDevice(const HWGeneration& target);
HWGeneration DeviceToHwGeneration(const DeviceVersion& target);
DeviceVersion GnaToDevice(const Gna2DeviceVersion& target);
Gna2DeviceVersion DeviceToGna(const DeviceVersion& target);
DeviceVersion StringToDevice(const std::string& target);
std::string DeviceToString(const DeviceVersion& target);
bool IsEmbeddedDevice(const DeviceVersion& target);

}  // namespace target
}  // namespace intel_gna
}  // namespace ov
