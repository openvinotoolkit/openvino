// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ie_core.hpp>
#include <details/ie_exception.hpp>
#include <ie_plugin_config.hpp>
#include <ie_extension.h>
#include <multi-device/multi_device_config.hpp>

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

using Device = std::string;
using Config = std::map<std::string, std::string>;
using Params = std::tuple<Device, Config>;

class CoreThreadingTestsBase {
public:
    static void runParallel(std::function<void(void)> func,
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

    void safePluginUnregister(InferenceEngine::Core & ie) {
        try {
            ie.UnregisterPlugin(deviceName);
        } catch (const InferenceEngine::details::InferenceEngineException & ex) {
            // if several threads unload plugin at once, the first thread does this
            // while all others will throw an exception that plugin is not registered
            ASSERT_STR_CONTAINS(ex.what(), "name is not registered in the");
        }
    }

    void safeAddExtension(InferenceEngine::Core & ie) {
        try {
            auto extension = InferenceEngine::make_so_pointer<InferenceEngine::IExtension>(
                FileUtils::makeSharedLibraryName<char>({}, "extension_tests"));
            ie.AddExtension(extension);
        } catch (const InferenceEngine::details::InferenceEngineException & ex) {
            ASSERT_STR_CONTAINS(ex.what(), "name: experimental");
        }
    }

    Device deviceName;
    Config config;
};

//
//  Common threading plugin tests
//

class CoreThreadingTests : public CoreThreadingTestsBase,
                           public ::testing::TestWithParam<Params> {
public:
    void SetUp() override {
        std::tie(deviceName, config) = GetParam();
    }
};

// tested function: GetVersions, UnregisterPlugin
TEST_P(CoreThreadingTests, smoke_GetVersions) {
    InferenceEngine::Core ie;

    runParallel([&] () {
        auto versions = ie.GetVersions(deviceName);
        ASSERT_LE(1u, versions.size());
        safePluginUnregister(ie);
    });
}

// tested function: SetConfig for already created plugins
TEST_P(CoreThreadingTests, smoke_SetConfigPluginExists) {
    InferenceEngine::Core ie;

    ie.SetConfig(config);
    auto versions = ie.GetVersions(deviceName);

    runParallel([&] () {
        ie.SetConfig(config);
    }, 10000);
}

// tested function: GetConfig, UnregisterPlugin
TEST_P(CoreThreadingTests, smoke_GetConfig) {
    InferenceEngine::Core ie;
    std::string configKey = config.begin()->first;

    ie.SetConfig(config);
    runParallel([&] () {
        ie.GetConfig(deviceName, configKey);
        safePluginUnregister(ie);
    });
}

// tested function: GetMetric, UnregisterPlugin
TEST_P(CoreThreadingTests, smoke_GetMetric) {
    InferenceEngine::Core ie;
    runParallel([&] () {
        ie.GetMetric(deviceName, METRIC_KEY(SUPPORTED_CONFIG_KEYS));
        safePluginUnregister(ie);
    });
}

// tested function: QueryNetwork
TEST_P(CoreThreadingTests, smoke_QueryNetwork) {
    InferenceEngine::Core ie;
    auto model = FuncTestUtils::TestModel::convReluNormPoolFcModelFP32;
    auto network = ie.ReadNetwork(model.model_xml_str, model.weights_blob);

    ie.SetConfig(config, deviceName);
    InferenceEngine::QueryNetworkResult refResult = ie.QueryNetwork(network, deviceName);

    runParallel([&] () {
        const auto result = ie.QueryNetwork(network, deviceName);
        safePluginUnregister(ie);

        // compare QueryNetworkResult with reference
        for (auto && r : refResult.supportedLayersMap) {
            ASSERT_NE(result.supportedLayersMap.end(), result.supportedLayersMap.find(r.first));
        }
        for (auto && r : result.supportedLayersMap) {
            ASSERT_NE(refResult.supportedLayersMap.end(), refResult.supportedLayersMap.find(r.first));
        }
    }, 3000);
}

//
//  Parametrized tests with numfer of parallel threads, iterations
//

using Threads = unsigned int;
using Iterations = unsigned int;

class CoreThreadingTestsWithIterations : public ::testing::TestWithParam<std::tuple<Params, Threads, Iterations> >,
                                         public CoreThreadingTestsBase {
public:
    void SetUp() override {
        std::tie(deviceName, config) = std::get<0>(GetParam());
        numThreads =  std::get<1>(GetParam());
        numIterations =  std::get<2>(GetParam());
    }

    unsigned int numIterations;
    unsigned int numThreads;
};

// tested function: LoadNetwork, AddExtension
TEST_P(CoreThreadingTestsWithIterations, smoke_LoadNetwork) {
    InferenceEngine::Core ie;
    std::atomic<unsigned int> counter{0u};

    const FuncTestUtils::TestModel::TestModel models[] = {
        FuncTestUtils::TestModel::convReluNormPoolFcModelFP32,
        FuncTestUtils::TestModel::convReluNormPoolFcModelFP16
    };
    std::vector<InferenceEngine::CNNNetwork> networks;
    for (auto & model : models) {
        networks.emplace_back(ie.ReadNetwork(model.model_xml_str, model.weights_blob));
    }

    // TODO: uncomment after fixing *-31414
    // networks.emplace_back(InferenceEngine::CNNNetwork(ngraph::builder::subgraph::make2InputSubtract()));
    // networks.emplace_back(InferenceEngine::CNNNetwork(ngraph::builder::subgraph::makeMultiSingleConv()));
    // networks.emplace_back(InferenceEngine::CNNNetwork(ngraph::builder::subgraph::makeSingleConv()));
    // networks.emplace_back(InferenceEngine::CNNNetwork(ngraph::builder::subgraph::makeSplitConvConcat()));
    // networks.emplace_back(InferenceEngine::CNNNetwork(ngraph::builder::subgraph::makeSplitMultiConvConcat()));

    ie.SetConfig(config, deviceName);
    runParallel([&] () {
        auto value = counter++;
        (void)ie.LoadNetwork(networks[(counter++) % networks.size()], deviceName);
    }, numIterations, numThreads);
}

// tested function: ReadNetwork, SetConfig, LoadNetwork, AddExtension
TEST_P(CoreThreadingTestsWithIterations, smoke_LoadNetwork_MultipleIECores) {
    std::atomic<unsigned int> counter{0u};

    // TODO: replace with subgraph builders after fixing *-31414
    const std::vector<FuncTestUtils::TestModel::TestModel> models = {
        FuncTestUtils::TestModel::convReluNormPoolFcModelFP32,
        FuncTestUtils::TestModel::convReluNormPoolFcModelFP16
    };

    runParallel([&] () {
        auto value = counter++;
        InferenceEngine::Core ie;
        ie.SetConfig(config, deviceName);
        auto model = models[(counter++) % models.size()];
        auto network = ie.ReadNetwork(model.model_xml_str, model.weights_blob);
        (void)ie.LoadNetwork(network, deviceName);
    }, numIterations, numThreads);
}
