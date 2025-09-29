//
// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "behavior/ov_plugin/properties_tests.hpp"
#include "common_test_utils/file_utils.hpp"
#include "common_test_utils/test_assertions.hpp"
#include "common_test_utils/unicode_utils.hpp"
#include "openvino/runtime/properties.hpp"
#include "openvino/util/common_util.hpp"
#include "shared_test_classes/base/ov_behavior_test_utils.hpp"

namespace ov::test::behavior {

using PropertiesParamsNPU = std::tuple<std::string, AnyMap>;

class OVPropertiesTestsNPU : public testing::WithParamInterface<PropertiesParamsNPU>, public OVPropertiesBase {
public:
    static inline std::string getTestCaseName(testing::TestParamInfo<PropertiesParamsNPU> obj);

    inline void SetUp() override;

    inline void TearDown() override;
};

using OVPropertiesIncorrectTestsNPU = OVPropertiesTestsNPU;

using CompileModelPropertiesParamsNPU = std::tuple<std::string, AnyMap>;

class OVPropertiesTestsWithCompileModelPropsNPU : public testing::WithParamInterface<PropertiesParamsNPU>,
                                                  public OVPropertiesBase {
public:
    static inline std::string getTestCaseName(testing::TestParamInfo<PropertiesParamsNPU> obj);

    inline void SetUp() override;

    inline void TearDown() override;

    AnyMap compileModelProperties;
};

std::string OVPropertiesTestsNPU::getTestCaseName(testing::TestParamInfo<PropertiesParamsNPU> obj) {
    std::string target_device;
    AnyMap properties;
    std::tie(target_device, properties) = obj.param;
    std::replace(target_device.begin(), target_device.end(), ':', '.');
    std::ostringstream result;
    result << "target_device=" << target_device << "_";
    if (!properties.empty()) {
        result << "properties=" << util::join(util::split(util::to_string(properties), ' '), "_");
    }
    return result.str();
}

void OVPropertiesTestsNPU::SetUp() {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    std::tie(target_device, properties) = this->GetParam();
    APIBaseTest::SetUp();
    model = ov::test::utils::make_split_concat();
}

void OVPropertiesTestsNPU::TearDown() {
    if (!properties.empty()) {
        utils::PluginCache::get().reset();
    }
    APIBaseTest::TearDown();
}

std::string OVPropertiesTestsWithCompileModelPropsNPU::getTestCaseName(
    testing::TestParamInfo<PropertiesParamsNPU> obj) {
    std::string target_device;
    AnyMap properties;
    std::tie(target_device, properties) = obj.param;
    std::replace(target_device.begin(), target_device.end(), ':', '.');
    std::ostringstream result;
    result << "target_device=" << target_device << "_";
    if (!properties.empty()) {
        result << "properties=" << util::join(util::split(util::to_string(properties), ' '), "_");
    }
    return result.str();
}

void OVPropertiesTestsWithCompileModelPropsNPU::SetUp() {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    std::string temp_device;
    std::tie(temp_device, properties) = this->GetParam();
    std::string::size_type pos = temp_device.find(":", 0);
    std::string hw_device;

    if (pos == std::string::npos) {
        target_device = temp_device;
        hw_device = temp_device;
    } else {
        target_device = temp_device.substr(0, pos);
        hw_device = temp_device.substr(++pos, std::string::npos);
    }

    if (target_device == std::string(ov::test::utils::DEVICE_MULTI) ||
        target_device == std::string(ov::test::utils::DEVICE_AUTO) ||
        target_device == std::string(ov::test::utils::DEVICE_HETERO) ||
        target_device == std::string(ov::test::utils::DEVICE_BATCH)) {
        compileModelProperties = {ov::device::priorities(hw_device)};
    }

    model = ov::test::utils::make_split_concat();

    APIBaseTest::SetUp();
}

void OVPropertiesTestsWithCompileModelPropsNPU::TearDown() {
    if (!properties.empty()) {
        utils::PluginCache::get().reset();
    }
    APIBaseTest::TearDown();
}

using OVCheckSetSupportedRWMetricsPropsTestsNPU = OVPropertiesTestsWithCompileModelPropsNPU;
}  // namespace ov::test::behavior
