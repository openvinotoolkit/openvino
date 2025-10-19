// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <fstream>
#include <thread>

#include "common_test_utils/common_utils.hpp"
#include "common_test_utils/file_utils.hpp"
#include "common_test_utils/test_assertions.hpp"
#include "functional_test_utils/test_model/test_model.hpp"
#include "openvino/core/so_extension.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/util/file_util.hpp"

#ifdef __GLIBC__
#    include <gnu/libc-version.h>
#    if __GLIBC_MINOR__ < 34
#        define OV_TEST_GLIBC_VERSION_LESS_2_34
#    endif
#endif

class CoreThreadingTests : public ::testing::Test {
protected:
    std::string modelName = "CoreThreadingTests.xml", weightsName = "CoreThreadingTests.bin";

public:
    void SetUp() override {
        auto prefix = ov::test::utils::generateTestFilePrefix();
        modelName = prefix + modelName;
        weightsName = prefix + weightsName;
        ov::test::utils::generate_test_model(modelName, weightsName);
    }

    void TearDown() override {
        ov::test::utils::removeIRFiles(modelName, weightsName);
    }

    void runParallel(std::function<void(void)> func,
                     const unsigned int iterations = 100,
                     const unsigned int threadsNum = 8) {
        std::vector<std::thread> threads(threadsNum);

        for (auto& thread : threads) {
            thread = std::thread([&]() {
                for (unsigned int i = 0; i < iterations; ++i) {
                    func();
                }
            });
        }

        for (auto& thread : threads) {
            if (thread.joinable())
                thread.join();
        }
    }

    void safeAddExtension(ov::Core& core) {
        try {
            auto extension = ov::detail::load_extensions(
                ov::util::make_plugin_library_name(ov::test::utils::getExecutableDirectory(),
                                                   std::string("openvino_template_extension") + OV_BUILD_POSTFIX));
            core.add_extension(extension);
        } catch (const ov::Exception& ex) {
            ASSERT_STR_CONTAINS(ex.what(), "name: custom_opset. Opset");
        }
    }
};

// tested function: SetConfig
TEST_F(CoreThreadingTests, SetConfigPluginDoesNotExist) {
    ov::Core core;

    runParallel(
        [&]() {
            core.set_property(ov::enable_profiling(true));
        },
        10000);
}

// TODO: CVS-68982
#ifndef OPENVINO_STATIC_LIBRARY

// tested function: RegisterPlugin
TEST_F(CoreThreadingTests, RegisterPlugin) {
    ov::Core core;
    std::atomic<int> index{0};
    auto plugin_path = ov::util::make_plugin_library_name(ov::test::utils::getExecutableDirectory(),
                                                          std::string("mock_engine") + OV_BUILD_POSTFIX);
    runParallel(
        [&]() {
            const std::string deviceName = std::to_string(index++);
            core.register_plugin(plugin_path, deviceName);
            core.get_versions(deviceName);
            core.unload_plugin(deviceName);
        },
        4000);
}

// tested function: RegisterPlugins
TEST_F(CoreThreadingTests, RegisterPlugins) {
#    ifdef _WIN32
    // TODO: CVS-133087
    GTEST_SKIP() << "This test sporadically crashes on Windows";
#    endif
    ov::Core core;
    std::atomic<unsigned int> index{0};
    auto file_prefix = ov::test::utils::generateTestFilePrefix();
    auto plugin_path = ov::util::make_plugin_library_name(ov::test::utils::getExecutableDirectory(),
                                                          std::string("mock_engine") + OV_BUILD_POSTFIX);

    auto getPluginXml = [&]() -> std::tuple<std::filesystem::path, std::string> {
        std::string indexStr = std::to_string(index++);
        std::filesystem::path pluginsXML = file_prefix + indexStr + ".xml";
        std::ofstream file(pluginsXML);

        file << "<ie><plugins><plugin location=\"";
        file << plugin_path;
        file << "\" name=\"";
        file << indexStr;
        file << "\"></plugin></plugins></ie>";
        file.flush();
        file.close();

        return std::tie(pluginsXML, indexStr);
    };

    runParallel(
        [&]() {
            const auto& [fileName, deviceName] = getPluginXml();
            core.register_plugins(fileName.string());
            core.get_versions(deviceName);
            core.unload_plugin(deviceName);
            std::filesystem::remove(fileName);
        },
        1000);
}

#endif  // !OPENVINO_STATIC_LIBRARY

// tested function: get_available_devices, unload_plugin
// TODO: some initialization (e.g. thread/dlopen) sporadically fails during such stress-test scenario
TEST_F(CoreThreadingTests, GetAvailableDevices) {
#ifdef OV_TEST_GLIBC_VERSION_LESS_2_34
    GTEST_SKIP();
#endif
    ov::Core core;
    runParallel(
        [&]() {
            std::vector<std::string> devices = core.get_available_devices();

            // unregister all the devices
            for (auto&& deviceName : devices) {
                try {
                    core.unload_plugin(deviceName);
                } catch (const ov::Exception& ex) {
                    // if several threads unload plugin at once, the first thread does this
                    // while all others will throw an exception that plugin is not registered
                    ASSERT_STR_CONTAINS(ex.what(), "name is not registered in the");
                }
            }
        },
        30);
}

#if defined(ENABLE_OV_IR_FRONTEND)
// tested function: read_model and add_extension
TEST_F(CoreThreadingTests, ReadModel) {
    ov::Core core;
    auto model = core.read_model(modelName, weightsName);
    constexpr size_t threads_num = 12;

    runParallel(
        [&]() {
            safeAddExtension(core);
            std::ignore = core.read_model(modelName, weightsName);
        },
        100,
        threads_num);
}
#endif  // defined(ENABLE_OV_IR_FRONTEND)
