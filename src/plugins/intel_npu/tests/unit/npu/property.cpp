// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/test_assertions.hpp"
#include "common_test_utils/test_constants.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/runtime/intel_npu/properties.hpp"
#include "openvino/runtime/properties.hpp"

using namespace ov::intel_npu;
using namespace ov::test::utils;

using PropertyUnitTests = ::testing::Test;

TEST_F(PropertyUnitTests, SetUnknownPropertyShouldFailWhenDriverNotInstalled) {
    ov::Core core;
    // triggers Plugin constructor
    OV_ASSERT_NO_THROW(core.get_property(DEVICE_NPU, ov::hint::enable_cpu_pinning.name()));

    OV_ASSERT_NO_THROW(
        core.set_property(DEVICE_NPU, {{ov::intel_npu::compiler_type.name(), ov::intel_npu::CompilerType::DRIVER}}));

    OV_EXPECT_THROW_HAS_SUBSTRING(core.set_property(DEVICE_NPU, {{"WRONG_PROPERTY", 0xDEADBEEF}}),
                                  ov::Exception,
                                  "cannot be created to validate the property");
}

TEST_F(PropertyUnitTests, SetPropertyShouldFailWhenDriverNotInstalled) {
    ov::Core core;
    // triggers Plugin constructor
    OV_ASSERT_NO_THROW(core.get_property(DEVICE_NPU, ov::hint::enable_cpu_pinning.name()));

    OV_ASSERT_NO_THROW(
        core.set_property(DEVICE_NPU, {{ov::intel_npu::compiler_type.name(), ov::intel_npu::CompilerType::DRIVER}}));

    OV_EXPECT_THROW_HAS_SUBSTRING(core.set_property(DEVICE_NPU, {{ov::intel_npu::platform.name(), "3720"}}),
                                  ov::Exception,
                                  "cannot be created to validate the property");
}
