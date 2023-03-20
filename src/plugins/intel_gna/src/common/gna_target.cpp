// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gna_target.hpp"

#include <map>
#include <set>

#include "log/debug.hpp"
#include "misc_utils.hpp"

namespace ov {
namespace intel_gna {
using namespace common;
namespace target {

static constexpr const char* kGnaTargetUnspecified = "";
static constexpr const char* kGnaTargetSoftwareEmulation = "GNA_SW_EMULATION";
static constexpr const char* kGnaTarget1_0 = "GNA_TARGET_1_0";
static constexpr const char* kGnaTarget2_0 = "GNA_TARGET_2_0";
static constexpr const char* kGnaTarget3_0 = "GNA_TARGET_3_0";
static constexpr const char* kGnaTarget3_1 = "GNA_TARGET_3_1";
static constexpr const char* kGnaTarget3_5 = "GNA_TARGET_3_5";
static constexpr const char* kGnaTarget3_5_e = "GNA_TARGET_3_5_E";
static constexpr const char* kGnaTarget3_6 = "GNA_TARGET_3_6";
static constexpr const char* kGnaTarget4_0 = "GNA_TARGET_4_0";

static const std::unordered_map<HWGeneration, DeviceVersion> HWGenerationDeviceMap{
    {HWGeneration::GNA_1_0, DeviceVersion::GNA1_0},
    {HWGeneration::GNA_2_0, DeviceVersion::GNA2_0},
    {HWGeneration::GNA_3_0, DeviceVersion::GNA3_0},
    {HWGeneration::GNA_3_1, DeviceVersion::GNA3_1},
    {HWGeneration::GNA_3_5, DeviceVersion::GNA3_5},
    {HWGeneration::GNA_3_5_E, DeviceVersion::GNAEmbedded3_5},
    {HWGeneration::GNA_3_6, DeviceVersion::GNA3_6},
    {HWGeneration::GNA_4_0, DeviceVersion::GNA4_0},
    {HWGeneration::UNDEFINED, DeviceVersion::NotSet}};

static const std::unordered_map<Gna2DeviceVersion, DeviceVersion> GnaDeviceMap{
    {Gna2DeviceVersionEmbedded1_0, DeviceVersion::GNA1_0},
    {Gna2DeviceVersion2_0, DeviceVersion::GNA2_0},
    {Gna2DeviceVersion3_0, DeviceVersion::GNA3_0},
    {Gna2DeviceVersionEmbedded3_1, DeviceVersion::GNA3_1},
    {Gna2DeviceVersion3_5, DeviceVersion::GNA3_5},
    {Gna2DeviceVersionEmbedded3_5, DeviceVersion::GNAEmbedded3_5},
    {Gna2DeviceVersionEmbedded3_5, DeviceVersion::GNA3_6},
    {Gna2DeviceVersionEmbedded3_5, DeviceVersion::GNA4_0},
    {Gna2DeviceVersionSoftwareEmulation, DeviceVersion::SoftwareEmulation}};

static const std::unordered_map<std::string, DeviceVersion> StringDeviceMap{
    {kGnaTarget1_0, DeviceVersion::GNA1_0},
    {kGnaTarget2_0, DeviceVersion::GNA2_0},
    {kGnaTarget3_0, DeviceVersion::GNA3_0},
    {kGnaTarget3_1, DeviceVersion::GNA3_1},
    {kGnaTarget3_5, DeviceVersion::GNA3_5},
    {kGnaTarget3_5_e, DeviceVersion::GNAEmbedded3_5},
    {kGnaTarget3_6, DeviceVersion::GNA3_6},
    {kGnaTarget4_0, DeviceVersion::GNA4_0},
    {kGnaTargetSoftwareEmulation, DeviceVersion::SoftwareEmulation},
    {kGnaTargetUnspecified, DeviceVersion::NotSet}};

static const std::vector<DeviceVersion> EmbeddedDevices{DeviceVersion::GNA1_0,
                                                        DeviceVersion::GNA3_1,
                                                        DeviceVersion::GNAEmbedded3_5,
                                                        DeviceVersion::GNA3_6,
                                                        DeviceVersion::GNA4_0};

DeviceVersion HwGenerationToDevice(const HWGeneration& target) {
    return GetValueForKey<HWGeneration, DeviceVersion>(target, HWGenerationDeviceMap);
}

HWGeneration DeviceToHwGeneration(const DeviceVersion& target) {
    return GetKeyForValue<HWGeneration, DeviceVersion>(target, HWGenerationDeviceMap);
}

DeviceVersion GnaToDevice(const Gna2DeviceVersion& target) {
    return GetValueForKey<Gna2DeviceVersion, DeviceVersion>(target, GnaDeviceMap);
}

Gna2DeviceVersion DeviceToGna(const DeviceVersion& target) {
    return GetKeyForValue<Gna2DeviceVersion, DeviceVersion>(target, GnaDeviceMap);
}

DeviceVersion StringToDevice(const std::string& target) {
    return GetValueForKey<std::string, DeviceVersion>(target, StringDeviceMap);
}

std::string DeviceToString(const DeviceVersion& target) {
    return GetKeyForValue<std::string, DeviceVersion>(target, StringDeviceMap);
}

bool IsEmbeddedDevice(const DeviceVersion& target) {
    return std::find(EmbeddedDevices.begin(), EmbeddedDevices.end(), target) != EmbeddedDevices.end();
}

DeviceVersion Target::get_detected_device_version() const {
    return detected_device_version;
}

DeviceVersion Target::get_user_set_execution_target() const {
    return user_set_execution_target;
}

DeviceVersion Target::get_user_set_compile_target() const {
    return user_set_compile_target;
}

void Target::set_detected_device_version(DeviceVersion detected_device) {
    detected_device_version = detected_device;
}

void Target::set_user_set_execution_target(DeviceVersion execution_target) {
    user_set_execution_target = execution_target;
}

void Target::set_user_set_compile_target(DeviceVersion compile_target) {
    user_set_compile_target = compile_target;
}

DeviceVersion Target::get_effective_execution_target() const {
    if (user_set_execution_target != DeviceVersion::NotSet) {
        return user_set_execution_target;
    } else if (detected_device_version == DeviceVersion::SoftwareEmulation) {
        return DeviceVersion::Default;
    } else {
        return detected_device_version;
    }
}

DeviceVersion Target::get_effective_compile_target() const {
    if (user_set_compile_target != DeviceVersion::NotSet) {
        return user_set_compile_target;
    } else if (detected_device_version == DeviceVersion::SoftwareEmulation) {
        return DeviceVersion::Default;
    } else {
        return get_effective_execution_target();
    }
}

}  // namespace target
}  // namespace intel_gna
}  // namespace ov
