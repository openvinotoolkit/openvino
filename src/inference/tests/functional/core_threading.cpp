// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <ie_extension.h>

#include <atomic>
#include <chrono>
#include <common_test_utils/file_utils.hpp>
#include <common_test_utils/test_assertions.hpp>
#include <fstream>
#include <functional_test_utils/test_model/test_model.hpp>
#include <ie_core.hpp>
#include <ie_plugin_config.hpp>
#include <mutex>
#include <thread>

#include "openvino/util/file_util.hpp"
#ifdef __GLIBC__
#    include <gnu/libc-version.h>
#    if __GLIBC_MINOR__ < 34
#        define OV_TEST_GLIBC_VERSION_LESS_2_34
#    endif
#endif

class IECoreThreadingTests : public ::testing::Test {
protected:
    std::string modelName = "IECoreThreadingTests.xml", weightsName = "IECoreThreadingTests.bin";

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

    void safeAddExtension(InferenceEngine::Core& ie) {
        try {
            auto extension = std::make_shared<InferenceEngine::Extension>(
                ov::util::make_plugin_library_name(ov::test::utils::getExecutableDirectory(),
                                                   std::string("template_extension") + OV_BUILD_POSTFIX));
            ie.AddExtension(extension);
        } catch (const InferenceEngine::Exception& ex) {
            ASSERT_STR_CONTAINS(ex.what(), "name: custom_opset. Opset");
        }
    }
};

// tested function: SetConfig
TEST_F(IECoreThreadingTests, SetConfigPluginDoesNotExist) {
    InferenceEngine::Core ie;
    std::map<std::string, std::string> localConfig = {
        {CONFIG_KEY(PERF_COUNT), InferenceEngine::PluginConfigParams::YES}};

    runParallel(
        [&]() {
            ie.SetConfig(localConfig);
        },
        10000);
}

// TODO: CVS-68982
#ifndef OPENVINO_STATIC_LIBRARY

// tested function: RegisterPlugin
TEST_F(IECoreThreadingTests, RegisterPlugin) {
    InferenceEngine::Core ie;
    std::atomic<int> index{0};
    runParallel(
        [&]() {
            const std::string deviceName = std::to_string(index++);
            ie.RegisterPlugin(ov::util::make_plugin_library_name(ov::test::utils::getExecutableDirectory(),
                                                                 std::string("mock_engine") + OV_BUILD_POSTFIX),
                              deviceName);
            ie.GetVersions(deviceName);
            ie.UnregisterPlugin(deviceName);
        },
        4000);
}

// tested function: RegisterPlugins
TEST_F(IECoreThreadingTests, RegisterPlugins) {
    InferenceEngine::Core ie;
    std::atomic<unsigned int> index{0};

    auto getPluginXml = [&]() -> std::tuple<std::string, std::string> {
        std::string indexStr = std::to_string(index++);
        std::string pluginsXML = "test_plugins" + indexStr + ".xml";
        std::ofstream file(pluginsXML);

        file << "<ie><plugins><plugin location=\"";
        file << ov::test::utils::getExecutableDirectory();
        file << ov::util::FileTraits<char>::file_separator;
        file << ov::util::FileTraits<char>::library_prefix();
        file << "mock_engine";
        file << OV_BUILD_POSTFIX;
        file << ov::util::FileTraits<char>::dot_symbol;
        file << ov::util::FileTraits<char>::library_ext();
        file << "\" name=\"";
        file << indexStr;
        file << "\"></plugin></plugins></ie>";
        file.flush();
        file.close();

        return std::tie(pluginsXML, indexStr);
    };

    runParallel(
        [&]() {
            std::string fileName, deviceName;
            std::tie(fileName, deviceName) = getPluginXml();
            ie.RegisterPlugins(fileName);
            ie.GetVersions(deviceName);
            ASSERT_EQ(0, std::remove(fileName.c_str()));
        },
        1000);
}

#endif  // !OPENVINO_STATIC_LIBRARY

// tested function: GetAvailableDevices, UnregisterPlugin
// TODO: some initialization (e.g. thread/dlopen) sporadically fails during such stress-test scenario
TEST_F(IECoreThreadingTests, GetAvailableDevices) {
#ifdef OV_TEST_GLIBC_VERSION_LESS_2_34
    GTEST_SKIP();
#endif
    InferenceEngine::Core ie;
    runParallel(
        [&]() {
            std::vector<std::string> devices = ie.GetAvailableDevices();

            // unregister all the devices
            for (auto&& deviceName : devices) {
                try {
                    ie.UnregisterPlugin(deviceName);
                } catch (const InferenceEngine::Exception& ex) {
                    // if several threads unload plugin at once, the first thread does this
                    // while all others will throw an exception that plugin is not registered
                    ASSERT_STR_CONTAINS(ex.what(), "name is not registered in the");
                }
            }
        },
        30);
}

#if defined(ENABLE_OV_IR_FRONTEND)
// tested function: ReadNetwork, AddExtension
TEST_F(IECoreThreadingTests, ReadNetwork) {
    InferenceEngine::Core ie;
    auto network = ie.ReadNetwork(modelName, weightsName);

    runParallel(
        [&]() {
            safeAddExtension(ie);
            (void)ie.ReadNetwork(modelName, weightsName);
        },
        100,
        12);
}
#endif  // defined(ENABLE_OV_IR_FRONTEND)
