// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ie_core.hpp>
#include <details/ie_exception.hpp>
#include <ie_plugin_config.hpp>
#include <ie_extension.h>

#include <file_utils.h>
#include <ngraph_functions/subgraph_builders.hpp>
#include <functional_test_utils/test_model/test_model.hpp>
#include <functional_test_utils/threading.hpp>
#include <common_test_utils/file_utils.hpp>
#include <common_test_utils/test_assertions.hpp>

#include <gtest/gtest.h>
#include <thread>
#include <atomic>
#include <mutex>
#include <chrono>
#include <fstream>

// tested function: SetConfig
TEST(CoreThreadingTests, SetConfigPluginDoesNotExist) {
    InferenceEngine::Core ie;
    std::map<std::string, std::string> localConfig = {
        { CONFIG_KEY(PERF_COUNT), InferenceEngine::PluginConfigParams::YES }
    };

    runParallel([&] () {
        ie.SetConfig(localConfig);
    }, 10000);
}

// tested function: RegisterPlugin
TEST(CoreThreadingTests, RegisterPlugin) {
    InferenceEngine::Core ie;
    std::atomic<int> index{0};
    runParallel([&] () {
        const std::string deviceName = std::to_string(index++);
        ie.RegisterPlugin(std::string("mock_engine") + IE_BUILD_POSTFIX, deviceName);
        ie.GetVersions(deviceName);
        ie.UnregisterPlugin(deviceName);
    }, 4000);
}

// tested function: RegisterPlugins
TEST(CoreThreadingTests, RegisterPlugins) {
    InferenceEngine::Core ie;
    std::atomic<unsigned int> index{0};

    auto getPluginXml = [&] () -> std::tuple<std::string, std::string> {
        std::string indexStr = std::to_string(index++);
        std::string pluginsXML = InferenceEngine::getIELibraryPath() +
            FileUtils::FileSeparator +
            "test_plugins" + indexStr + ".xml";
        std::ofstream file(pluginsXML);

        file << "<ie><plugins><plugin location=\"";
        file << FileUtils::FileTraits<char>::SharedLibraryPrefix();
        file << "mock_engine";
        file << IE_BUILD_POSTFIX;
        file << FileUtils::DotSymbol<char>::value;
        file << FileUtils::FileTraits<char>::SharedLibraryExt();
        file << "\" name=\"";
        file << indexStr;
        file << "\"></plugin></plugins></ie>";
        file.flush();
        file.close();

        return std::tie(pluginsXML, indexStr);
    };

    runParallel([&] () {
        std::string fileName, deviceName;
        std:tie(fileName, deviceName) = getPluginXml();
        ie.RegisterPlugins(fileName);
        ie.GetVersions(deviceName);
        ASSERT_EQ(0, std::remove(fileName.c_str()));
    }, 1000);
}

// tested function: GetAvailableDevices, UnregisterPlugin
// TODO: some plugins initialization (e.g. GNA) failed during such stress-test scenario
TEST(CoreThreadingTests, DISABLED_GetAvailableDevices) {
    InferenceEngine::Core ie;
    runParallel([&] () {
        std::vector<std::string> devices = ie.GetAvailableDevices();

        // unregister all the devices
        for (auto && deviceName : devices) {
            try {
                ie.UnregisterPlugin(deviceName);
            } catch (const InferenceEngine::details::InferenceEngineException & ex) {
                // if several threads unload plugin at once, the first thread does this
                // while all others will throw an exception that plugin is not registered
                ASSERT_STR_CONTAINS(ex.what(), "name is not registered in the");
            }
        }
    }, 30);
}

