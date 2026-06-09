// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gtest/gtest.h>

#include <iostream>
#include <string>
#include <vector>

#include "common/utils.hpp"
#include "common_test_utils/subgraph_builders/multi_single_conv.hpp"
#include "common_test_utils/test_assertions.hpp"
#include "compiler_adapter_utils.hpp"
#include "driver_compiler_adapter.hpp"
#include "functional_test_utils/skip_tests_config.hpp"
#include "intel_npu/common/igraph.hpp"
#include "intel_npu/utils/zero/zero_api.hpp"
#include "openvino/runtime/intel_npu/properties.hpp"
#include "ze_graph_ext_wrappers.hpp"

namespace ov::test::behavior {

using namespace ::intel_npu;

using CompatStringParams = std::tuple<std::string, ov::AnyMap>;

class DriverCompatStringTest : public ::testing::TestWithParam<CompatStringParams> {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<CompatStringParams>& obj) {
        std::string targetDevice;
        ov::AnyMap configuration;
        std::tie(targetDevice, configuration) = obj.param;
        std::replace(targetDevice.begin(), targetDevice.end(), ':', '_');

        std::ostringstream result;
        result << "targetDevice=" << targetDevice << "_";
        result << "targetPlatform=" << ov::test::utils::getTestsPlatformFromEnvironmentOr(targetDevice);
        if (!configuration.empty()) {
            for (auto& configItem : configuration) {
                result << "_configItem=" << configItem.first << "_";
                configItem.second.print(result);
            }
        }
        return result.str();
    }

protected:
    void SetUp() override {
        SKIP_IF_CURRENT_TEST_IS_DISABLED();

        std::tie(targetDevice, configuration) = this->GetParam();

        zeroInitStruct = ZeroInitStructsHolder::getInstance();
        ASSERT_NE(zeroInitStruct, nullptr);
        ASSERT_NE(zeroInitStruct->getDevice(), nullptr);

        zel_version_t loaderVer = {};
        zel_component_version_t cv;
        if (zelGetLoaderVersion(&cv) == ZE_RESULT_SUCCESS) {
            loaderVer = cv.component_lib_version;
        }
        if (loaderVer.major == 1 && loaderVer.minor < 29) {
            GTEST_SKIP() << "ze_loader version " << loaderVer.major << "." << loaderVer.minor
                         << " < 1.29: runtime requirements extension entry-points not forwarded";
        }
        adapter = std::make_unique<DriverCompilerAdapter>(zeroInitStruct);
    }

    void TearDown() override {
        compiledGraph.reset();
    }

    void compileModel() {
        auto model = ov::test::utils::make_multi_single_conv();
        compiledGraph = adapter->compile(model, makeTestCompileConfig());
    }

    GraphDescriptor graphHandle() const {
        EXPECT_NE(compiledGraph, nullptr);
        return GraphDescriptor{compiledGraph->get_handle()};
    }

    std::string targetDevice;
    ov::AnyMap configuration;
    std::shared_ptr<ZeroInitStructsHolder> zeroInitStruct;
    std::unique_ptr<DriverCompilerAdapter> adapter;
    std::shared_ptr<IGraph> compiledGraph;
};

TEST_P(DriverCompatStringTest, CompileThenGetString) {
    OV_ASSERT_NO_THROW(compileModel());
    ASSERT_NE(compiledGraph->get_handle(), nullptr);

    std::optional<std::string> result;
    OV_ASSERT_NO_THROW(result = adapter->get_runtime_requirements(graphHandle()));

    if (result.has_value()) {
        ASSERT_FALSE(result->empty()) << "get_runtime_requirements returned an empty string";
    }
}

TEST_P(DriverCompatStringTest, GetStringThenValidate) {
    OV_ASSERT_NO_THROW(compileModel());
    ASSERT_NE(compiledGraph->get_handle(), nullptr);

    auto compatStr = adapter->get_runtime_requirements(graphHandle());

    if (compatStr.has_value()) {
        bool isCompatible = false;
        OV_ASSERT_NO_THROW(isCompatible = adapter->validate_compatibility_descriptor(*compatStr));
        EXPECT_TRUE(isCompatible) << "adapter rejected a compat string it just generated on the same device";
    }
}

TEST_P(DriverCompatStringTest, ValidateRejectsGarbageString) {
    bool isCompatible = true;
    OV_ASSERT_NO_THROW(isCompatible = adapter->validate_compatibility_descriptor("not_a_valid_compat_string"));
    EXPECT_FALSE(isCompatible);
}

TEST_P(DriverCompatStringTest, ValidateRejectsEmptyString) {
    bool isCompatible = true;
    OV_ASSERT_NO_THROW(isCompatible = adapter->validate_compatibility_descriptor(""));
    EXPECT_FALSE(isCompatible);
}

TEST_P(DriverCompatStringTest, IsOptionSupportedCompatibilityCheck) {
    bool supported = adapter->is_option_supported(COMPATIBILITY_CHECK::key().data());
    EXPECT_EQ(supported, zeDeviceValidateRuntimeRequirements != nullptr);
}

TEST_P(DriverCompatStringTest, IsOptionSupportedRuntimeRequirements) {
    bool supported = adapter->is_option_supported(RUNTIME_REQUIREMENTS::key().data());
    EXPECT_EQ(supported, zeDeviceGetRuntimeRequirements != nullptr);
}

TEST_P(DriverCompatStringTest, zeDeviceGetRuntimeRequirementsKey) {
    const char* key = nullptr;
    const ze_result_t result = zeDeviceGetRuntimeRequirementsKey(zeroInitStruct->getDevice(), &key);
    ASSERT_EQ(result, ZE_RESULT_SUCCESS)
        << "zeDeviceGetRuntimeRequirementsKey returned 0x" << std::hex << static_cast<uint32_t>(result);
    ASSERT_NE(key, nullptr) << "zeDeviceGetRuntimeRequirementsKey returned null key pointer";
    ASSERT_GT(std::strlen(key), 0) << "zeDeviceGetRuntimeRequirementsKey returned empty key string";
}

TEST_P(DriverCompatStringTest, zeDeviceValidateRuntimeRequirementsGarbage) {
    ze_validate_runtime_requirements_output_t output = {};
    output.stype = ZE_STRUCTURE_TYPE_RUNTIME_REQUIREMENTS_OUTPUT;
    const ze_result_t result =
        zeDeviceValidateRuntimeRequirements(zeroInitStruct->getDevice(), "garbage_string", &output);
    ASSERT_EQ(result, ZE_RESULT_SUCCESS)
        << "zeDeviceValidateRuntimeRequirements returned 0x" << std::hex << static_cast<uint32_t>(result);
    EXPECT_NE(output.result, ZE_VALIDATE_RUNTIME_REQUIREMENTS_RESULT_REQUIREMENTS_MET)
        << "Driver unexpectedly accepted a garbage requirements string";
    EXPECT_NE(output.result, ZE_VALIDATE_RUNTIME_REQUIREMENTS_RESULT_REQUIREMENTS_MET_RECOMPILATION_ADVISABLE)
        << "Driver unexpectedly accepted a garbage requirements string";
}

}  // namespace ov::test::behavior
