// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/subgraph_builders/multi_single_conv.hpp"
#include "common_test_utils/test_assertions.hpp"
#include "common_test_utils/test_constants.hpp"
#include "intel_npu/npu_private_properties.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/runtime/intel_npu/properties.hpp"
#include "openvino/runtime/properties.hpp"
#include "zero_backend.hpp"

using namespace ov::intel_npu;
using namespace ov::test::utils;

class OfflineCompilationUnitTests : public ::testing::TestWithParam<ov::AnyMap> {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ParamType>& info) {
        std::string result;
        for (const auto& [key, value] : info.param) {
            result += value.as<std::string>();
        }
        return result;
    }

protected:
    void SetUp() override {
        config = GetParam();
        std::vector<std::string> availableDevices = core.get_available_devices();
        auto it = std::find(availableDevices.begin(), availableDevices.end(), DEVICE_NPU);
        ASSERT_TRUE(it == availableDevices.end());
    }

    ov::Core core;
    ov::AnyMap config;
};

TEST_P(OfflineCompilationUnitTests, CompileWithCiPWhenDriverNotInstalledSetProperty) {
    core.set_property(DEVICE_NPU, config);
    std::shared_ptr<ov::Model> model = ov::test::utils::make_multi_single_conv();
    OV_ASSERT_NO_THROW(core.compile_model(model, DEVICE_NPU));
}

TEST_P(OfflineCompilationUnitTests, CompileWithCiPWhenDriverNotInstalled) {
    std::shared_ptr<ov::Model> model = ov::test::utils::make_multi_single_conv();
    OV_ASSERT_NO_THROW(core.compile_model(model, DEVICE_NPU, config));
}

TEST_P(OfflineCompilationUnitTests, ExpectThrowWhenCreateInferRequestWhenDriverNotInstalled) {
    std::shared_ptr<ov::Model> model = ov::test::utils::make_multi_single_conv();
    ov::CompiledModel compiledModel;
    OV_ASSERT_NO_THROW(compiledModel = core.compile_model(model, DEVICE_NPU, config));
    OV_EXPECT_THROW_HAS_SUBSTRING(compiledModel.create_infer_request(),
                                  ov::Exception,
                                  "No available devices. Failed to create infer request!");
}

INSTANTIATE_TEST_SUITE_P(
    OfflineCompilationPlatforms,
    OfflineCompilationUnitTests,
    ::testing::Values(ov::AnyMap{{ov::intel_npu::platform.name(), ov::intel_npu::Platform::NPU5010}},
                      ov::AnyMap{{ov::intel_npu::platform.name(), ov::intel_npu::Platform::NPU5020}}),
    OfflineCompilationUnitTests::getTestCaseName);

using UnavailableDeviceTests = ::testing::Test;

TEST_F(UnavailableDeviceTests, GetDeviceNotAvailable) {
    ov::Core core;
    OV_ASSERT_NO_THROW(
        core.set_property("NPU", {{ov::intel_npu::compiler_type.name(), ov::intel_npu::CompilerType::DRIVER}}));

    std::shared_ptr<intel_npu::ZeroEngineBackend> backend;
    ASSERT_ANY_THROW(backend = std::make_shared<intel_npu::ZeroEngineBackend>());
}
