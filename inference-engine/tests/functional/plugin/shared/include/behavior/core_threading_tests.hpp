// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ie_core.hpp>
#include <ie_plugin_config.hpp>
#include <ie_extension.h>
#include <cpp/ie_cnn_network.h>
#include <cpp/ie_executable_network.hpp>
#include <cpp/ie_infer_request.hpp>

#include <file_utils.h>
#include <ngraph_functions/subgraph_builders.hpp>
#include <functional_test_utils/blob_utils.hpp>
#include <common_test_utils/file_utils.hpp>
#include <common_test_utils/test_assertions.hpp>
#include <common_test_utils/test_constants.hpp>

#include <gtest/gtest.h>
#include <thread>
#include <atomic>
#include <mutex>
#include <chrono>
#include <fstream>
#include <functional_test_utils/skip_tests_config.hpp>

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
        } catch (const InferenceEngine::Exception & ex) {
            // if several threads unload plugin at once, the first thread does this
            // while all others will throw an exception that plugin is not registered
            ASSERT_STR_CONTAINS(ex.what(), "name is not registered in the");
        }
    }

    void safeAddExtension(InferenceEngine::Core & ie) {
        try {
            auto extension = std::make_shared<InferenceEngine::Extension>(
                FileUtils::makePluginLibraryName<char>({}, "template_extension"));
            ie.AddExtension(extension);
        } catch (const InferenceEngine::Exception & ex) {
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

    static std::string getTestCaseName(testing::TestParamInfo<Params> obj) {
        std::string deviceName;
        Config config;
        std::tie(deviceName, config) = obj.param;
        char separator('_');
        std::ostringstream result;
        result << "targetDevice=" << deviceName << separator;
        result << "config=";
        for (auto& confItem : config) {
            result << confItem.first << ":" << confItem.second << separator;
        }
        return result.str();
    }
};

// tested function: GetVersions, UnregisterPlugin
TEST_P(CoreThreadingTests, smoke_GetVersions) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    InferenceEngine::Core ie;
    runParallel([&] () {
        auto versions = ie.GetVersions(deviceName);
        ASSERT_LE(1u, versions.size());
        safePluginUnregister(ie);
    });
}

// tested function: SetConfig for already created plugins
TEST_P(CoreThreadingTests, smoke_SetConfigPluginExists) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    InferenceEngine::Core ie;

    ie.SetConfig(config);
    auto versions = ie.GetVersions(deviceName);

    runParallel([&] () {
        ie.SetConfig(config);
    }, 10000);
}

// tested function: GetConfig, UnregisterPlugin
TEST_P(CoreThreadingTests, smoke_GetConfig) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

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
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    InferenceEngine::Core ie;
    runParallel([&] () {
        ie.GetMetric(deviceName, METRIC_KEY(SUPPORTED_CONFIG_KEYS));
        safePluginUnregister(ie);
    });
}

// tested function: QueryNetwork
TEST_P(CoreThreadingTests, smoke_QueryNetwork) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    InferenceEngine::Core ie;
    InferenceEngine::CNNNetwork network(ngraph::builder::subgraph::make2InputSubtract());

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
//  Parameterized tests with number of parallel threads, iterations
//

using Threads = unsigned int;
using Iterations = unsigned int;

enum struct ModelClass : unsigned {
    Default,
    ConvPoolRelu
};

using CoreThreadingParams = std::tuple<Params, Threads, Iterations, ModelClass>;

class CoreThreadingTestsWithIterations : public ::testing::TestWithParam<CoreThreadingParams>,
    public CoreThreadingTestsBase {
public:
    void SetUp() override {
        std::tie(deviceName, config) = std::get<0>(GetParam());
        numThreads = std::get<1>(GetParam());
        numIterations = std::get<2>(GetParam());
        modelClass = std::get<3>(GetParam());
    }

    static std::string getTestCaseName(testing::TestParamInfo<CoreThreadingParams > obj) {
        unsigned int numThreads, numIterations;
        std::string deviceName;
        Config config;
        std::tie(deviceName, config) = std::get<0>(obj.param);
        numThreads = std::get<1>(obj.param);
        numIterations = std::get<2>(obj.param);
        char separator('_');
        std::ostringstream result;
        result << "targetDevice=" << deviceName << separator;
        result << "config=";
        for (auto& confItem : config) {
            result << confItem.first << ":" << confItem.second << separator;
        }
        result << "numThreads=" << numThreads << separator;
        result << "numIter=" << numIterations;
        return result.str();
    }

    ModelClass modelClass;
    unsigned int numIterations;
    unsigned int numThreads;

    std::vector<InferenceEngine::CNNNetwork> networks;
    void SetupNetworks() {
        if (modelClass == ModelClass::ConvPoolRelu) {
            for (unsigned i = 0; i < numThreads; i++) {
                networks.emplace_back(InferenceEngine::CNNNetwork(ngraph::builder::subgraph::makeConvPoolRelu()));
            }
        } else {
            networks.emplace_back(InferenceEngine::CNNNetwork(ngraph::builder::subgraph::make2InputSubtract()));
            networks.emplace_back(InferenceEngine::CNNNetwork(ngraph::builder::subgraph::makeMultiSingleConv()));
            networks.emplace_back(InferenceEngine::CNNNetwork(ngraph::builder::subgraph::makeSingleConv()));
            networks.emplace_back(InferenceEngine::CNNNetwork(ngraph::builder::subgraph::makeSplitConvConcat()));
            networks.emplace_back(InferenceEngine::CNNNetwork(ngraph::builder::subgraph::makeSplitMultiConvConcat()));
        }
    }
};

// tested function: LoadNetwork, AddExtension
TEST_P(CoreThreadingTestsWithIterations, smoke_LoadNetwork) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    InferenceEngine::Core ie;
    std::atomic<unsigned int> counter{0u};

    SetupNetworks();

    ie.SetConfig(config, deviceName);
    runParallel([&] () {
        auto value = counter++;
        (void)ie.LoadNetwork(networks[value % networks.size()], deviceName);
    }, numIterations, numThreads);
}

// tested function: LoadNetwork accuracy
TEST_P(CoreThreadingTestsWithIterations, smoke_LoadNetworkAccuracy) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    InferenceEngine::Core ie;
    std::atomic<unsigned int> counter{0u};

    SetupNetworks();

    ie.SetConfig(config, deviceName);
    runParallel([&] () {
        auto value = counter++;
        auto network = networks[value % networks.size()];

        InferenceEngine::BlobMap blobs;
        for (const auto & info : network.getInputsInfo()) {
            auto input = FuncTestUtils::createAndFillBlobFloatNormalDistribution(
                info.second->getTensorDesc(), 0.0f, 0.2f, 7235346);
            blobs[info.first] = input;
        }

        auto getOutputBlob = [&](InferenceEngine::Core & core) {
            auto exec = core.LoadNetwork(network, deviceName);
            auto req = exec.CreateInferRequest();
            req.SetInput(blobs);

            auto info = network.getOutputsInfo();
            auto outputInfo = info.begin();
            auto blob = make_blob_with_precision(outputInfo->second->getTensorDesc());
            blob->allocate();
            req.SetBlob(outputInfo->first, blob);

            req.Infer();
            return blob;
        };

        auto outputActual = getOutputBlob(ie);

        // compare actual value using the second Core
        {
            InferenceEngine::Core ie2;
            ie2.SetConfig(config, deviceName);
            auto outputRef = getOutputBlob(ie2);

            FuncTestUtils::compareBlobs(outputActual, outputRef);
        }
    }, numIterations, numThreads);
}

// tested function: ReadNetwork, SetConfig, LoadNetwork, AddExtension
TEST_P(CoreThreadingTestsWithIterations, smoke_LoadNetwork_MultipleIECores) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    std::atomic<unsigned int> counter{0u};

    SetupNetworks();

    runParallel([&] () {
        auto value = counter++;
        InferenceEngine::Core ie;
        ie.SetConfig(config, deviceName);
        (void)ie.LoadNetwork(networks[value % networks.size()], deviceName);
    }, numIterations, numThreads);
}
