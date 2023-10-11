// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ie_core.hpp>
#include <ie_plugin_config.hpp>
#include <ie_extension.h>
#include <cpp/ie_cnn_network.h>
#include <cpp/ie_executable_network.hpp>
#include <cpp/ie_infer_request.hpp>

#include <file_utils.h>
#include <ov_models/subgraph_builders.hpp>
#include <functional_test_utils/blob_utils.hpp>
#include <common_test_utils/file_utils.hpp>
#include <common_test_utils/test_assertions.hpp>
#include <common_test_utils/test_constants.hpp>
#include "base/behavior_test_utils.hpp"

#include <gtest/gtest.h>
#include <thread>
#include <atomic>
#include <mutex>
#include <chrono>
#include <fstream>
#include <functional_test_utils/skip_tests_config.hpp>
#include "base/ov_behavior_test_utils.hpp"

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

    void safePluginUnregister(InferenceEngine::Core & ie, const std::string& deviceName) {
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
                FileUtils::makePluginLibraryName<char>(ov::test::utils::getExecutableDirectory(), "template_extension"));
            ie.AddExtension(extension);
        } catch (const InferenceEngine::Exception & ex) {
            ASSERT_STR_CONTAINS(ex.what(), "name: experimental");
        }
    }

    Config config;
};

//
//  Common threading plugin tests
//

class CoreThreadingTests : public testing::WithParamInterface<Params>,
                           public BehaviorTestsUtils::IEPluginTestBase,
                           public CoreThreadingTestsBase {
public:
    void SetUp() override {
        std::tie(target_device, config) = GetParam();
        APIBaseTest::SetUp();
        SKIP_IF_CURRENT_TEST_IS_DISABLED();
    }

    static std::string getTestCaseName(testing::TestParamInfo<Params> obj) {
        std::string deviceName;
        Config config;
        std::tie(deviceName, config) = obj.param;
        std::replace(deviceName.begin(), deviceName.end(), ':', '.');
        char separator('_');
        std::ostringstream result;
        result << "targetDevice=" << deviceName << separator;
        result << "config=";
        for (auto& confItem : config) {
            result << confItem.first << "=" << confItem.second << separator;
        }
        return result.str();
    }
};

// tested function: GetVersions, UnregisterPlugin
TEST_P(CoreThreadingTests, smoke_GetVersions) {
    InferenceEngine::Core ie;
    runParallel([&] () {
        auto versions = ie.GetVersions(target_device);
        ASSERT_LE(1u, versions.size());
        safePluginUnregister(ie, target_device);
    });
}

// tested function: SetConfig for already created plugins
TEST_P(CoreThreadingTests, smoke_SetConfigPluginExists) {
    InferenceEngine::Core ie;

    ie.SetConfig(config);
    auto versions = ie.GetVersions(target_device);

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
        ie.GetConfig(target_device, configKey);
        safePluginUnregister(ie, target_device);
    });
}

// tested function: GetMetric, UnregisterPlugin
TEST_P(CoreThreadingTests, smoke_GetMetric) {
    InferenceEngine::Core ie;
    runParallel([&] () {
        ie.GetMetric(target_device, METRIC_KEY(SUPPORTED_CONFIG_KEYS));
        safePluginUnregister(ie, target_device);
    });
}

// tested function: QueryNetwork
TEST_P(CoreThreadingTests, smoke_QueryNetwork) {
    InferenceEngine::Core ie;
    InferenceEngine::CNNNetwork network(ngraph::builder::subgraph::make2InputSubtract());

    ie.SetConfig(config, target_device);
    InferenceEngine::QueryNetworkResult refResult = ie.QueryNetwork(network, target_device);

    runParallel([&] () {
        const auto result = ie.QueryNetwork(network, target_device);
        safePluginUnregister(ie, target_device);

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

class CoreThreadingTestsWithIterations : public testing::WithParamInterface<CoreThreadingParams>,
                                         public BehaviorTestsUtils::IEPluginTestBase,
                                         public CoreThreadingTestsBase {
public:
    void SetUp() override {
        SKIP_IF_CURRENT_TEST_IS_DISABLED();
        std::tie(target_device, config) = std::get<0>(GetParam());
        numThreads = std::get<1>(GetParam());
        numIterations = std::get<2>(GetParam());
        modelClass = std::get<3>(GetParam());
    }

    static std::string getTestCaseName(testing::TestParamInfo<CoreThreadingParams > obj) {
        unsigned int numThreads, numIterations;
        std::string deviceName;
        Config config;
        std::tie(deviceName, config) = std::get<0>(obj.param);
        std::replace(deviceName.begin(), deviceName.end(), ':', '.');
        numThreads = std::get<1>(obj.param);
        numIterations = std::get<2>(obj.param);
        char separator('_');
        std::ostringstream result;
        result << "targetDevice=" << deviceName << separator;
        result << "config=";
        for (auto& confItem : config) {
            result << confItem.first << "=" << confItem.second << separator;
        }
        result << "numThreads=" << numThreads << separator;
        result << "numIter=" << numIterations;
        return result.str();
    }


protected:
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
    InferenceEngine::Core ie;
    std::atomic<unsigned int> counter{0u};

    SetupNetworks();

    ie.SetConfig(config, target_device);
    runParallel([&] () {
        auto value = counter++;
        (void)ie.LoadNetwork(networks[value % networks.size()], target_device);
    }, numIterations, numThreads);
}

// tested function: single IECore LoadNetwork accuracy
TEST_P(CoreThreadingTestsWithIterations, smoke_LoadNetworkAccuracy_SingleIECore) {
    InferenceEngine::Core ie;
    std::atomic<unsigned int> counter{0u};

    SetupNetworks();

    ie.SetConfig(config, target_device);

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
            auto exec = core.LoadNetwork(network, target_device);
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

        // compare actual value using the same Core
        auto outputRef = getOutputBlob(ie);
        FuncTestUtils::compareBlobs(outputActual, outputRef);
    }, numIterations, numThreads);
}

// tested function: LoadNetwork accuracy
TEST_P(CoreThreadingTestsWithIterations, smoke_LoadNetworkAccuracy) {
    InferenceEngine::Core ie;
    std::atomic<unsigned int> counter{0u};

    SetupNetworks();

    ie.SetConfig(config, target_device);
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
            auto exec = core.LoadNetwork(network, target_device);
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
            ie2.SetConfig(config, target_device);
            auto outputRef = getOutputBlob(ie2);

            FuncTestUtils::compareBlobs(outputActual, outputRef);
        }
    }, numIterations, numThreads);
}

// tested function: ReadNetwork, SetConfig, LoadNetwork, AddExtension
TEST_P(CoreThreadingTestsWithIterations, smoke_LoadNetwork_MultipleIECores) {
    std::atomic<unsigned int> counter{0u};

    SetupNetworks();

    runParallel([&] () {
        auto value = counter++;
        InferenceEngine::Core ie;
        ie.SetConfig(config, target_device);
        (void)ie.LoadNetwork(networks[value % networks.size()], target_device);
    }, numIterations, numThreads);
}
