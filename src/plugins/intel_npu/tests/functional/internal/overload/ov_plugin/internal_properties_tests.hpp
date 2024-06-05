//
// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "base/ov_behavior_test_utils.hpp"
#include "behavior/ov_plugin/properties_tests.hpp"
#include "common_test_utils/file_utils.hpp"
#include "common_test_utils/test_assertions.hpp"
#include "common_test_utils/unicode_utils.hpp"
#include "openvino/runtime/properties.hpp"
#include "openvino/util/common_util.hpp"

namespace ov {
namespace test {
namespace behavior {

#define OV_ASSERT_PROPERTY_SUPPORTED(property_key)                                  \
    {                                                                               \
        auto properties = ie.get_property(target_device, ov::supported_properties); \
        auto it = std::find(properties.begin(), properties.end(), property_key);    \
        ASSERT_NE(properties.end(), it);                                            \
    }

using PropertiesParamsNPU = std::tuple<std::string, AnyMap>;

class OVPropertiesTestsNPU : public testing::WithParamInterface<PropertiesParamsNPU>, public OVPropertiesBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<PropertiesParamsNPU> obj);

    void SetUp() override;

    void TearDown() override;
};

using OVPropertiesIncorrectTestsNPU = OVPropertiesTestsNPU;

using CompileModelPropertiesParamsNPU = std::tuple<std::string, AnyMap>;

class OVPropertiesTestsWithCompileModelPropsNPU :
        public testing::WithParamInterface<PropertiesParamsNPU>,
        public OVPropertiesBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<PropertiesParamsNPU> obj);

    void SetUp() override;

    void TearDown() override;

    AnyMap compileModelProperties;

    static std::vector<ov::AnyMap> getROMandatoryProperties(bool is_sw_device = false);
    static std::vector<ov::AnyMap> getROOptionalProperties(bool is_sw_device = false);
    static std::vector<ov::AnyMap> configureProperties(std::vector<std::string> props);

    static std::vector<ov::AnyMap> getRWMandatoryPropertiesValues(const std::vector<std::string>& props = {},
                                                                  bool is_sw_device = false);
    static std::vector<ov::AnyMap> getWrongRWMandatoryPropertiesValues(const std::vector<std::string>& props = {},
                                                                       bool is_sw_device = false);
    static std::vector<ov::AnyMap> getRWOptionalPropertiesValues(const std::vector<std::string>& props = {},
                                                                 bool is_sw_device = false);
    static std::vector<ov::AnyMap> getWrongRWOptionalPropertiesValues(const std::vector<std::string>& props = {},
                                                                      bool is_sw_device = false);

    static std::vector<ov::AnyMap> getModelDependcePropertiesValues();
};

using OVCheckSetSupportedRWMetricsPropsTestsNPU = OVPropertiesTestsWithCompileModelPropsNPU;

}  // namespace behavior
}  // namespace test
}  // namespace ov