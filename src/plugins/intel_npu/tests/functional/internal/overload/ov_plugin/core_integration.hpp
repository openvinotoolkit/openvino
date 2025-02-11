// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <filesystem>

#include "base/ov_behavior_test_utils.hpp"
#include "behavior/ov_plugin/properties_tests.hpp"
#include "common/functions.h"
#include "common/npu_test_env_cfg.hpp"
#include "common/utils.hpp"
#include "common_test_utils/data_utils.hpp"
#include "common_test_utils/subgraph_builders/concat_with_params.hpp"
#include "common_test_utils/subgraph_builders/conv_pool_relu.hpp"
#include "common_test_utils/subgraph_builders/kso_func.hpp"
#include "common_test_utils/subgraph_builders/single_concat_with_constant.hpp"
#include "common_test_utils/subgraph_builders/split_conv_concat.hpp"
#include "intel_npu/config/common.hpp"

using CompilationParams = std::tuple<std::string,  // Device name
                                     ov::AnyMap    // Config
                                     >;

#ifdef OPENVINO_ENABLE_UNICODE_PATH_SUPPORT
#    include <iostream>
#    define GTEST_COUT std::cerr << "[          ] [ INFO ] "
#    include <codecvt>
#    include <functional_test_utils/skip_tests_config.hpp>

#    include "openvino/pass/manager.hpp"

#endif

namespace ov {
namespace test {
namespace behavior {

class OVClassBaseTestPNPU : public OVClassNetworkTest,
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
    OV_ASSERT_NO_THROW(ie.compile_model(actualNetwork,
                                        ov::test::utils::DEVICE_HETERO + std::string(":") + target_device,
                                        configuration));
}

TEST_P(OVClassNetworkTestPNPU, LoadNetworkActualHeteroDevice2NoThrow) {
    ov::Core ie = createCoreWithTemplate();

    OV_ASSERT_NO_THROW(ie.compile_model(actualNetwork,
                                        ov::test::utils::DEVICE_HETERO,
                                        ov::device::priorities(target_device),
                                        ov::device::properties(target_device, configuration)));
}

TEST_P(OVClassNetworkTestPNPU, LoadNetworkActualHeteroDeviceUsingDevicePropertiesNoThrow) {
    ov::Core ie = createCoreWithTemplate();
    configuration.emplace(ov::enable_profiling(true));

    OV_ASSERT_NO_THROW(ie.compile_model(actualNetwork,
                                        ov::test::utils::DEVICE_HETERO,
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

TEST(compatibility_OVClassBasicPropsTestNPU, smoke_SetConfigDevicePropertiesThrows) {
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

//
// NPU specific metrics
//

using OVClassGetMetricAndPrintNoThrow = OVClassBaseTestP;
TEST_P(OVClassGetMetricAndPrintNoThrow, DeviceAllocMemSizeLesserThanTotalMemSizeNPU) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    ov::Core ie;
    ov::Any p;

    OV_ASSERT_NO_THROW(p = ie.get_property(target_device, ov::intel_npu::device_total_mem_size.name()));
    uint64_t t = p.as<uint64_t>();
    ASSERT_NE(t, 0);

    OV_ASSERT_NO_THROW(p = ie.get_property(target_device, ov::intel_npu::device_alloc_mem_size.name()));
    uint64_t a = p.as<uint64_t>();

    ASSERT_LT(a, t);

    std::cout << "OV NPU device alloc/total memory size: " << a << "/" << t << std::endl;
}

TEST_P(OVClassGetMetricAndPrintNoThrow, DeviceAllocMemSizeLesserAfterModelIsLoadedNPU) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    ov::Core ie;
    ov::Any p;

    OV_ASSERT_NO_THROW(p = ie.get_property(target_device, ov::intel_npu::device_alloc_mem_size.name()));
    uint64_t a1 = p.as<uint64_t>();

    SKIP_IF_CURRENT_TEST_IS_DISABLED() {
        auto model = ov::test::utils::make_conv_pool_relu();
        OV_ASSERT_NO_THROW(ie.compile_model(model, target_device, {}));
    }

    OV_ASSERT_NO_THROW(p = ie.get_property(target_device, ov::intel_npu::device_alloc_mem_size.name()));
    uint64_t a2 = p.as<uint64_t>();

    std::cout << "OV NPU device {alloc before load network/alloc after load network} memory size: {" << a1 << "/" << a2
              << "}" << std::endl;

    // after the network is loaded onto device, allocated memory value should increase
    ASSERT_LE(a1, a2);
}

TEST_P(OVClassGetMetricAndPrintNoThrow, VpuDeviceAllocMemSizeLesserAfterModelIsLoaded) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    ov::Core ie;
    ov::Any p;

    OV_ASSERT_NO_THROW(p = ie.get_property(target_device, ov::intel_npu::device_alloc_mem_size.name()));
    uint64_t a1 = p.as<uint64_t>();

    SKIP_IF_CURRENT_TEST_IS_DISABLED() {
        auto model = ov::test::utils::make_conv_pool_relu();
        OV_ASSERT_NO_THROW(ie.compile_model(model, target_device, ov::AnyMap{ov::log::level(ov::log::Level::ERR)}));
    }

    OV_ASSERT_NO_THROW(p = ie.get_property(target_device, ov::intel_npu::device_alloc_mem_size.name()));
    uint64_t a2 = p.as<uint64_t>();

    std::cout << "OV NPU device {alloc before load network/alloc after load network} memory size: {" << a1 << "/" << a2
              << "}" << std::endl;

    // after the network is loaded onto device, allocated memory value should increase
    ASSERT_LE(a1, a2);
}

TEST_P(OVClassGetMetricAndPrintNoThrow, VpuDeviceAllocMemSizeSameAfterDestroyCompiledModel) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    ov::Core core;
    ov::Any deviceAllocMemSizeAny;

    auto model = createModelWithLargeSize();

    OV_ASSERT_NO_THROW(deviceAllocMemSizeAny =
                           core.get_property(target_device, ov::intel_npu::device_alloc_mem_size.name()));
    uint64_t deviceAllocMemSize = deviceAllocMemSizeAny.as<uint64_t>();
    uint64_t deviceAllocMemSizeFinal;

    for (size_t i = 0; i < 1000; ++i) {
        ov::CompiledModel compiledModel;
        OV_ASSERT_NO_THROW(compiledModel = core.compile_model(model, target_device));

        compiledModel = {};

        OV_ASSERT_NO_THROW(deviceAllocMemSizeAny =
                               core.get_property(target_device, ov::intel_npu::device_alloc_mem_size.name()));
        deviceAllocMemSizeFinal = deviceAllocMemSizeAny.as<uint64_t>();
        ASSERT_EQ(deviceAllocMemSize, deviceAllocMemSizeFinal) << " at iteration " << i;
    }
}

TEST_P(OVClassGetMetricAndPrintNoThrow, VpuDeviceAllocMemSizeSameAfterDestroyInferRequest) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    ov::Core core;
    ov::Any deviceAllocMemSizeAny;

    ov::CompiledModel compiledModel;
    auto model = createModelWithLargeSize();

    OV_ASSERT_NO_THROW(compiledModel = core.compile_model(model, target_device));

    // After memory consumption updates, need to run first inference before measurements
    auto inferRequest = compiledModel.create_infer_request();
    inferRequest.infer();
    inferRequest = {};

    OV_ASSERT_NO_THROW(deviceAllocMemSizeAny =
                           core.get_property(target_device, ov::intel_npu::device_alloc_mem_size.name()));
    uint64_t deviceAllocMemSize = deviceAllocMemSizeAny.as<uint64_t>();
    uint64_t deviceAllocMemSizeFinal;

    for (size_t i = 0; i < 1000; ++i) {
        inferRequest = compiledModel.create_infer_request();
        inferRequest.infer();

        inferRequest = {};

        OV_ASSERT_NO_THROW(deviceAllocMemSizeAny =
                               core.get_property(target_device, ov::intel_npu::device_alloc_mem_size.name()));
        deviceAllocMemSizeFinal = deviceAllocMemSizeAny.as<uint64_t>();
        ASSERT_EQ(deviceAllocMemSize, deviceAllocMemSizeFinal) << " at iteration " << i;
    }
}

TEST_P(OVClassGetMetricAndPrintNoThrow, VpuDeviceAllocMemSizeSameAfterDestroyInferRequestSetTensor) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    ov::Core core;
    ov::Any deviceAllocMemSizeAny;

    ov::CompiledModel compiledModel;
    auto model = createModelWithLargeSize();

    OV_ASSERT_NO_THROW(compiledModel = core.compile_model(model, target_device));

    // After memory consumption updates, need to run first inference before measurements
    auto inferRequest = compiledModel.create_infer_request();
    inferRequest.infer();
    inferRequest = {};

    OV_ASSERT_NO_THROW(deviceAllocMemSizeAny =
                           core.get_property(target_device, ov::intel_npu::device_alloc_mem_size.name()));
    uint64_t deviceAllocMemSize = deviceAllocMemSizeAny.as<uint64_t>();
    uint64_t deviceAllocMemSizeFinal;

    for (size_t i = 0; i < 1000; ++i) {
        auto inferRequest = compiledModel.create_infer_request();
        auto tensor = ov::Tensor(model->input(0).get_element_type(), model->input(0).get_shape());
        ov::test::utils::fill_data_random(static_cast<ov::float16*>(tensor.data()), tensor.get_size());
        inferRequest.set_input_tensor(tensor);
        inferRequest.infer();

        inferRequest = {};

        OV_ASSERT_NO_THROW(deviceAllocMemSizeAny =
                               core.get_property(target_device, ov::intel_npu::device_alloc_mem_size.name()));
        deviceAllocMemSizeFinal = deviceAllocMemSizeAny.as<uint64_t>();
        ASSERT_EQ(deviceAllocMemSize, deviceAllocMemSizeFinal) << " at iteration " << i;
    }
}

TEST_P(OVClassGetMetricAndPrintNoThrow, VpuDeviceAllocMemSizeSameAfterDestroyInferRequestGetTensor) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    ov::Core core;
    ov::Any deviceAllocMemSizeAny;

    ov::CompiledModel compiledModel;
    auto model = createModelWithLargeSize();

    OV_ASSERT_NO_THROW(deviceAllocMemSizeAny =
                           core.get_property(target_device, ov::intel_npu::device_alloc_mem_size.name()));
    uint64_t deviceAllocMemSize = deviceAllocMemSizeAny.as<uint64_t>();
    uint64_t deviceAllocMemSizeFinal;

    for (size_t i = 0; i < 1000; ++i) {
        OV_ASSERT_NO_THROW(compiledModel = core.compile_model(model, target_device));

        auto inferRequest = compiledModel.create_infer_request();
        auto tensor = inferRequest.get_output_tensor();
        inferRequest.infer();

        inferRequest = {};
        compiledModel = {};

        OV_ASSERT_NO_THROW(deviceAllocMemSizeAny =
                               core.get_property(target_device, ov::intel_npu::device_alloc_mem_size.name()));
        deviceAllocMemSizeFinal = deviceAllocMemSizeAny.as<uint64_t>();
        ASSERT_LT(deviceAllocMemSize, deviceAllocMemSizeFinal) << " at iteration " << i;

        tensor = {};

        OV_ASSERT_NO_THROW(deviceAllocMemSizeAny =
                               core.get_property(target_device, ov::intel_npu::device_alloc_mem_size.name()));
        deviceAllocMemSizeFinal = deviceAllocMemSizeAny.as<uint64_t>();
        ASSERT_EQ(deviceAllocMemSize, deviceAllocMemSizeFinal) << " at iteration " << i;
    }
}

TEST_P(OVClassGetMetricAndPrintNoThrow, DriverVersionNPU) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    ov::Core ie;
    ov::Any p;

    OV_ASSERT_NO_THROW(p = ie.get_property(target_device, ov::intel_npu::driver_version.name()));
    uint32_t t = p.as<uint32_t>();

    std::cout << "NPU driver version is " << t << std::endl;

    OV_ASSERT_PROPERTY_SUPPORTED(ov::intel_npu::driver_version.name());
}

using OVClassCompileModel = OVClassBaseTestP;
TEST_P(OVClassCompileModel, CompileModelWithDifferentThreadNumbers) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    ov::Core ie;
    ov::Any p;

    auto model = ov::test::utils::make_conv_pool_relu();
    OV_ASSERT_NO_THROW(ie.compile_model(model, target_device, {{ov::compilation_num_threads.name(), ov::Any(1)}}));

    OV_ASSERT_NO_THROW(ie.compile_model(model, target_device, {{ov::compilation_num_threads.name(), ov::Any(2)}}));

    OV_ASSERT_NO_THROW(ie.compile_model(model, target_device, {{ov::compilation_num_threads.name(), ov::Any(4)}}));

    EXPECT_ANY_THROW(ie.compile_model(model, target_device, {{ov::compilation_num_threads.name(), ov::Any(-1)}}));
    OV_EXPECT_THROW(
        std::ignore = ie.compile_model(model, target_device, {{ov::compilation_num_threads.name(), ov::Any(-1)}}),
        ::ov::Exception,
        testing::HasSubstr("ov::compilation_num_threads must be positive int32 value"));
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
#    ifndef _WIN32
            std::filesystem::create_directory(ov::util::wstring_to_string(unicode_path));
#    else
            std::filesystem::create_directory(unicode_path);
#    endif
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
