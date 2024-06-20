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

namespace ov::test::behavior {

using PropertiesParamsNPU = std::tuple<std::string, AnyMap>;

class OVPropertiesTestsNPU : public testing::WithParamInterface<PropertiesParamsNPU>, public OVPropertiesBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<PropertiesParamsNPU> obj);

    void SetUp() override;

    void TearDown() override;
};

using OVPropertiesIncorrectTestsNPU = OVPropertiesTestsNPU;

using CompileModelPropertiesParamsNPU = std::tuple<std::string, AnyMap>;

class OVPropertiesTestsWithCompileModelPropsNPU : public testing::WithParamInterface<PropertiesParamsNPU>,
                                                  public OVPropertiesBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<PropertiesParamsNPU> obj);

    void SetUp() override;

    void TearDown() override;

    AnyMap compileModelProperties;
};

using OVCheckSetSupportedRWMetricsPropsTestsNPU = OVPropertiesTestsWithCompileModelPropsNPU;
}  // namespace ov::test::behavior
