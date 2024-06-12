// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <filesystem>
#include "base/ov_behavior_test_utils.hpp"
#include "behavior/ov_plugin/properties_tests.hpp"
#include "common/utils.hpp"
#include "common/npu_test_env_cfg.hpp"
#include "common_test_utils/subgraph_builders/concat_with_params.hpp"
#include "common_test_utils/subgraph_builders/kso_func.hpp"
#include "common_test_utils/subgraph_builders/single_concat_with_constant.hpp"
#include "common_test_utils/subgraph_builders/split_conv_concat.hpp"
#include "intel_npu/al/config/common.hpp"

using CompilationParams = std::tuple<std::string,  // Device name
                                     ov::AnyMap    // Config
                                     >;

#ifdef OPENVINO_ENABLE_UNICODE_PATH_SUPPORT
#include <iostream>
#define GTEST_COUT std::cerr << "[          ] [ INFO ] "
#include <codecvt>
#include "openvino/pass/manager.hpp"
#endif

namespace ov {
namespace test {
namespace behavior {

class OVClassBaseTestPNPU :
        public OVClassNetworkTest,
        public testing::WithParamInterface<CompilationParams>,
        public OVPluginTestBase {
protected:
    ov::AnyMap configuration;
    std::string deathTestStyle;
    std::shared_ptr<ov::Model> function;

public:
    static std::string getTestCaseName(testing::TestParamInfo<CompilationParams> obj) {
        std::string target_device;
        ov::AnyMap configuration;
        std::tie(target_device, configuration) = obj.param;
        std::replace(target_device.begin(), target_device.end(), ':', '.');

        std::ostringstream result;
        result << "OVClassNetworkTestName_" << target_device;
        result << "_targetDevice=" << ov::test::utils::getTestsPlatformFromEnvironmentOr(ov::test::utils::DEVICE_NPU)
               << "_";
        if (!configuration.empty()) {
            for (auto& configItem : configuration) {
                result << "configItem=" << configItem.first << "_";
                configItem.second.print(result);
            }
        }
        return result.str();
    }

    void SetUp() override {
        std::tie(target_device, configuration) = this->GetParam();
        SKIP_IF_CURRENT_TEST_IS_DISABLED();
        APIBaseTest::SetUp();
        SKIP_IF_CURRENT_TEST_IS_DISABLED();
        // Generic network
        actualNetwork = ov::test::utils::make_split_conv_concat();
        // Quite simple network
        simpleNetwork = ov::test::utils::make_single_concat_with_constant();
        // Multinput to substruct network
        multinputNetwork = ov::test::utils::make_concat_with_params();
        // Network with KSO
        ksoNetwork = ov::test::utils::make_kso_function();

        ov::AnyMap params;
        for (auto&& v : configuration) {
            params.emplace(v.first, v.second);
        }
    }

    void TearDown() override {
        ::testing::GTEST_FLAG(death_test_style) = deathTestStyle;
        if (!configuration.empty()) {
            utils::PluginCache::get().reset();
        }
        APIBaseTest::TearDown();
    }
};

class OVClassBasicTestPNPU : public OVBasicPropertiesTestsP {

public:
    void TearDown() override {
        for (std::size_t testIndex = 0; testIndex < ov::test::utils::test_unicode_postfix_vector.size(); testIndex++) {
            std::wstring postfix = ov::test::utils::test_unicode_postfix_vector[testIndex];
            std::wstring unicode_path = ov::test::utils::stringToWString(ov::util::get_ov_lib_path() + "/") + postfix;
#ifndef _WIN32
            removeDirFilesRecursive(ov::util::wstring_to_string(unicode_path));
#else
            removeDirFilesRecursive(unicode_path);
#endif
        }
    }
};

using OVClassNetworkTestPNPU = OVClassBaseTestPNPU;
using OVClassLoadNetworkTestNPU = OVClassBaseTestPNPU;

TEST_P(OVClassNetworkTestPNPU, LoadNetworkActualNoThrow) {
    ov::Core ie = createCoreWithTemplate();
    OV_ASSERT_NO_THROW(ie.compile_model(actualNetwork, target_device, configuration));
}

TEST_P(OVClassNetworkTestPNPU, LoadNetworkActualHeteroDeviceNoThrow) {
    ov::Core ie = createCoreWithTemplate();
    OV_ASSERT_NO_THROW(ie.compile_model(
            actualNetwork, ov::test::utils::DEVICE_HETERO + std::string(":") + target_device, configuration));
}

TEST_P(OVClassNetworkTestPNPU, LoadNetworkActualHeteroDevice2NoThrow) {
    ov::Core ie = createCoreWithTemplate();

    OV_ASSERT_NO_THROW(ie.compile_model(actualNetwork, ov::test::utils::DEVICE_HETERO,
                                        ov::device::priorities(target_device),
                                        ov::device::properties(target_device, configuration)));
}

TEST_P(OVClassNetworkTestPNPU, LoadNetworkActualHeteroDeviceUsingDevicePropertiesNoThrow) {
    ov::Core ie = createCoreWithTemplate();
    configuration.emplace(ov::enable_profiling(true));

    OV_ASSERT_NO_THROW(ie.compile_model(actualNetwork, ov::test::utils::DEVICE_HETERO,
                                        ov::device::priorities(target_device),
                                        ov::device::properties(target_device, configuration)));
}

TEST_P(OVClassLoadNetworkTestNPU, LoadNetworkHETEROWithDeviceIDNoThrow) {
    ov::Core ie = createCoreWithTemplate();

    auto supported_properties = ie.get_property(target_device, ov::supported_properties);

    if (supported_properties.end() !=
        std::find(std::begin(supported_properties), std::end(supported_properties), ov::device::id)) {
        auto deviceIDs = ie.get_property(target_device, ov::available_devices);
        if (deviceIDs.empty())
            GTEST_FAIL();
        std::string heteroDevice = ov::test::utils::DEVICE_HETERO + std::string(":") + target_device + "." +
                                   deviceIDs[0] + "," + target_device;
        OV_ASSERT_NO_THROW(ie.compile_model(actualNetwork, heteroDevice, configuration));
    } else {
        GTEST_FAIL() << "Device does not support DeviceID property" << std::endl;
    }
}

TEST(OVClassBasicPropsTestNPU, smoke_SetConfigDevicePropertiesThrows) {
    ov::Core core;
    ASSERT_THROW(core.set_property("", ov::device::properties(ov::test::utils::DEVICE_NPU, ov::enable_profiling(true))),
                 ov::Exception);
    ASSERT_THROW(core.set_property(ov::test::utils::DEVICE_NPU,
                                   ov::device::properties(ov::test::utils::DEVICE_NPU, ov::enable_profiling(true))),
                 ov::Exception);
    ASSERT_THROW(core.set_property(ov::test::utils::DEVICE_AUTO,
                                   ov::device::properties(ov::test::utils::DEVICE_NPU, ov::enable_profiling(true))),
                 ov::Exception);
    ASSERT_THROW(core.set_property(ov::test::utils::DEVICE_AUTO,
                                   ov::device::properties(ov::test::utils::DEVICE_NPU, ov::num_streams(4))),
                 ov::Exception);
}

#ifdef OPENVINO_ENABLE_UNICODE_PATH_SUPPORT

TEST_P(OVClassBasicTestPNPU, smoke_registerPluginsLibrariesUnicodePath) {
    ov::Core core = createCoreWithTemplate();

    const std::vector<std::string> libs = {pluginName};

    for (std::size_t testIndex = 0; testIndex < ov::test::utils::test_unicode_postfix_vector.size(); testIndex++) {
        GTEST_COUT << testIndex;
        std::string unicode_target_device = target_device + "_UNICODE_" + std::to_string(testIndex);
        std::wstring postfix = ov::test::utils::test_unicode_postfix_vector[testIndex];
        std::wstring unicode_path =
                ov::test::utils::stringToWString(ov::test::utils::getOpenvinoLibDirectory() + "/") + postfix;
        try {
#ifndef _WIN32
            std::filesystem::create_directory(ov::util::wstring_to_string(unicode_path));
#else
            std::filesystem::create_directory(unicode_path);
#endif
            std::string pluginNamePath =
                    ov::util::make_plugin_library_name(ov::util::wstring_to_string(unicode_path), pluginName);

            for (auto&& lib : libs) {
                auto&& libPath = ov::test::utils::stringToWString(
                        ov::util::make_plugin_library_name(ov::test::utils::getOpenvinoLibDirectory(), lib));
                auto&& libPathNew = ov::test::utils::stringToWString(
                        ov::util::make_plugin_library_name(::ov::util::wstring_to_string(unicode_path), lib));
                bool is_copy_successfully = ov::test::utils::copyFile(libPath, libPathNew);
                if (!is_copy_successfully) {
                    FAIL() << "Unable to copy from '" << libPath << "' to '" << libPathNew << "'";
                }
            }

            OV_ASSERT_NO_THROW(core.register_plugin(pluginNamePath, unicode_target_device));
            OV_ASSERT_NO_THROW(core.get_versions(unicode_target_device));
            auto devices = core.get_available_devices();
            if (std::find_if(devices.begin(), devices.end(), [&unicode_target_device](std::string device) {
                    return device.find(unicode_target_device) != std::string::npos;
                }) == devices.end()) {
                FAIL() << unicode_target_device << " was not found within registered plugins.";
            }
            core.unload_plugin(unicode_target_device);
        } catch (const ov::Exception& e_next) {
            FAIL() << e_next.what();
        }
    }
}
#endif

}  // namespace behavior
}  // namespace test
}  // namespace ov
