// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ie_core.hpp>
#include <ie_plugin_config.hpp>
#include <ie_extension.h>

#include <file_utils.h>
#include <ngraph_functions/subgraph_builders.hpp>
#include <functional_test_utils/test_model/test_model.hpp>
#include <common_test_utils/file_utils.hpp>
#include <common_test_utils/test_assertions.hpp>

#include <gtest/gtest.h>
#include <thread>
#include <atomic>
#include <mutex>
#include <chrono>
#include <fstream>

class CoreThreadingTests : public ::testing::Test {
protected:
    std::string modelName = "CoreThreadingTests.xml", weightsName = "CoreThreadingTests.bin";

public:
    void SetUp() override {
        FuncTestUtils::TestModel::generateTestModel(modelName, weightsName);
    }

    void TearDown() override {
        CommonTestUtils::removeIRFiles(modelName, weightsName);
    }

    void runParallel(std::function<void(void)> func,
                     const unsigned int iterations = 100,
                     const unsigned int threadsNum = 8) {
        std::vector<std::thread> threads(threadsNum);

        for (auto & thread : threads) {
            thread = std::thread([&](){
                for (unsigned int i = 0; i < iterations; ++i) {
                    func();
                }
            });
        }

        for (auto & thread : threads) {
            if (thread.joinable())
                thread.join();
        }
    }

    void safeAddExtension(InferenceEngine::Core & ie) {
        try {
            auto extension = std::make_shared<InferenceEngine::Extension>(
                FileUtils::makePluginLibraryName<char>({},
                    std::string("template_extension") + IE_BUILD_POSTFIX));
            ie.AddExtension(extension);
        } catch (const InferenceEngine::Exception & ex) {
            ASSERT_STR_CONTAINS(ex.what(), "name: custom_opset. Opset");
        }
    }
};

// tested function: SetConfig
TEST_F(CoreThreadingTests, SetConfigPluginDoesNotExist) {
    InferenceEngine::Core ie;
    std::map<std::string, std::string> localConfig = {
        { CONFIG_KEY(PERF_COUNT), InferenceEngine::PluginConfigParams::YES }
    };

    runParallel([&] () {
        ie.SetConfig(localConfig);
    }, 10000);
}

// tested function: RegisterPlugin
TEST_F(CoreThreadingTests, RegisterPlugin) {
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
TEST_F(CoreThreadingTests, RegisterPlugins) {
    InferenceEngine::Core ie;
    std::atomic<unsigned int> index{0};

    auto getPluginXml = [&] () -> std::tuple<std::string, std::string> {
        std::string indexStr = std::to_string(index++);
        std::string pluginsXML = InferenceEngine::getIELibraryPath() +
            FileUtils::FileSeparator +
            "test_plugins" + indexStr + ".xml";
        std::ofstream file(pluginsXML);

        file << "<ie><plugins><plugin location=\"";
        file << FileUtils::FileTraits<char>::PluginLibraryPrefix();
        file << "mock_engine";
        file << IE_BUILD_POSTFIX;
        file << FileUtils::DotSymbol<char>::value;
        file << FileUtils::FileTraits<char>::PluginLibraryExt();
        file << "\" name=\"";
        file << indexStr;
        file << "\"></plugin></plugins></ie>";
        file.flush();
        file.close();

        return std::tie(pluginsXML, indexStr);
    };

    runParallel([&] () {
        std::string fileName, deviceName;
        std::tie(fileName, deviceName) = getPluginXml();
        ie.RegisterPlugins(fileName);
        ie.GetVersions(deviceName);
        ASSERT_EQ(0, std::remove(fileName.c_str()));
    }, 1000);
}

// tested function: GetAvailableDevices, UnregisterPlugin
// TODO: some initialization (e.g. thread/dlopen) sporadically fails during such stress-test scenario
TEST_F(CoreThreadingTests, DISABLED_GetAvailableDevices) {
    InferenceEngine::Core ie;
    runParallel([&] () {
        std::vector<std::string> devices = ie.GetAvailableDevices();

        // unregister all the devices
        for (auto && deviceName : devices) {
            try {
                ie.UnregisterPlugin(deviceName);
            } catch (const InferenceEngine::Exception & ex) {
                // if several threads unload plugin at once, the first thread does this
                // while all others will throw an exception that plugin is not registered
                ASSERT_STR_CONTAINS(ex.what(), "name is not registered in the");
            }
        }
    }, 30);
}

// tested function: ReadNetwork, AddExtension
TEST_F(CoreThreadingTests, ReadNetwork) {
    InferenceEngine::Core ie;
    auto network = ie.ReadNetwork(modelName, weightsName);

    runParallel([&] () {
        safeAddExtension(ie);
        (void)ie.ReadNetwork(modelName, weightsName);
    }, 100, 12);
}
