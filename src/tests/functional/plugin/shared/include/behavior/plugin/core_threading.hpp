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
#include <ngraph_functions/subgraph_builders.hpp>
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
#include "openvino/frontend/manager.hpp"

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
        auto hash = std::hash<std::string>()(::testing::UnitTest::GetInstance()->current_test_info()->name());
        std::stringstream ss;
        ss << std::this_thread::get_id();
        cache_path = "threading_test" + std::to_string(hash) + "_"
                + ss.str() + "_" + GetTimestamp() + "_cache";
    }
    void TearDown() override {
        std::remove(cache_path.c_str());
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
    std::string cache_path;

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

using CoreThreadingTestsWithCacheEnabled = CoreThreadingTestsWithIterations;
// tested function: SetConfig, LoadNetwork
TEST_P(CoreThreadingTestsWithCacheEnabled, smoke_LoadNetwork_cache_enabled) {
    InferenceEngine::Core ie;

    std::string ir_with_meta = R"V0G0N(
    <net name="Network" version="11">
        <layers>
            <layer name="in1" type="Parameter" id="0" version="opset1">
                <data element_type="f32" shape="1,3,22,22"/>
                <output>
                    <port id="0" precision="FP32">
                        <dim>1</dim>
                        <dim>3</dim>
                        <dim>22</dim>
                        <dim>22</dim>
                    </port>
                </output>
            </layer>
            <layer name="activation" id="1" type="ReLU" version="opset1">
                <input>
                    <port id="1" precision="FP32">
                        <dim>1</dim>
                        <dim>3</dim>
                        <dim>22</dim>
                        <dim>22</dim>
                    </port>
                </input>
                <output>
                    <port id="2" precision="FP32">
                        <dim>1</dim>
                        <dim>3</dim>
                        <dim>22</dim>
                        <dim>22</dim>
                    </port>
                </output>
            </layer>
            <layer name="output" type="Result" id="2" version="opset1">
                <input>
                    <port id="0" precision="FP32">
                        <dim>1</dim>
                        <dim>3</dim>
                        <dim>22</dim>
                        <dim>22</dim>
                    </port>
                </input>
            </layer>
        </layers>
        <edges>
            <edge from-layer="1" from-port="2" to-layer="2" to-port="0"/>
            <edge from-layer="0" from-port="0" to-layer="1" to-port="1"/>
        </edges>
        <meta_data>
            <MO_version value="TestVersion"/>
            <Runtime_version value="TestVersion"/>
            <cli_parameters>
                <input_shape value="[1, 3, 22, 22]"/>
                <transform value=""/>
                <use_new_frontend value="False"/>
            </cli_parameters>
        </meta_data>
        <framework_meta>
            <batch value="1"/>
            <chunk_size value="16"/>
        </framework_meta>
        <quantization_parameters>
            <config>{
            'compression': {
                'algorithms': [
                    {
                        'name': 'DefaultQuantization',
                        'params': {
                            'num_samples_for_tuning': 2000,
                            'preset': 'performance',
                            'stat_subset_size': 300,
                            'use_layerwise_tuning': false
                        }
                    }
                ],
                'dump_intermediate_model': true,
                'target_device': 'ANY'
            },
            'engine': {
                'models': [
                    {
                        'name': 'bert-small-uncased-whole-word-masking-squad-0001',
                        'launchers': [
                            {
                                'framework': 'openvino',
                                'adapter': {
                                    'type': 'bert_question_answering',
                                    'start_token_logits_output': 'output_s',
                                    'end_token_logits_output': 'output_e'
                                },
                                'inputs': [
                                    {
                                        'name': 'input_ids',
                                        'type': 'INPUT',
                                        'value': 'input_ids'
                                    },
                                    {
                                        'name': 'attention_mask',
                                        'type': 'INPUT',
                                        'value': 'input_mask'
                                    },
                                    {
                                        'name': 'token_type_ids',
                                        'type': 'INPUT',
                                        'value': 'segment_ids'
                                    }
                                ],
                                'device': 'cpu'
                            }
                        ],
                        'datasets': [
                            {
                                'name': 'squad_v1_1_msl384_mql64_ds128_lowercase',
                                'annotation_conversion': {
                                    'converter': 'squad',
                                    'testing_file': 'PATH',
                                    'max_seq_length': 384,
                                    'max_query_length': 64,
                                    'doc_stride': 128,
                                    'lower_case': true,
                                    'vocab_file': 'PATH'
                                },
                                'reader': {
                                    'type': 'annotation_features_extractor',
                                    'features': [
                                        'input_ids',
                                        'input_mask',
                                        'segment_ids'
                                    ]
                                },
                                'postprocessing': [
                                    {
                                        'type': 'extract_answers_tokens',
                                        'max_answer': 30,
                                        'n_best_size': 20
                                    }
                                ],
                                'metrics': [
                                    {
                                        'name': 'F1',
                                        'type': 'f1',
                                        'reference': 0.9157
                                    },
                                    {
                                        'name': 'EM',
                                        'type': 'exact_match',
                                        'reference': 0.8504
                                    }
                                ],
                                '_command_line_mapping': {
                                    'testing_file': 'PATH',
                                    'vocab_file': [
                                        'PATH'
                                    ]
                                }
                            }
                        ]
                    }
                ],
                'stat_requests_number': null,
                'eval_requests_number': null,
                'type': 'accuracy_checker'
            }
        }</config>
            <version value="invalid version"/>
            <cli_params value="{'quantize': None, 'preset': None, 'model': None, 'weights': None, 'name': None, 'engine': None, 'ac_config': None, 'max_drop': None, 'evaluate': False, 'output_dir': 'PATH', 'direct_dump': True, 'log_level': 'INFO', 'pbar': False, 'stream_output': False, 'keep_uncompressed_weights': False, 'data_source': None}"/>
        </quantization_parameters>
    </net>
    )V0G0N";
    InferenceEngine::Blob::Ptr weights = nullptr;
    auto model = ie.ReadNetwork(ir_with_meta, weights);

    ie.SetConfig(config, target_device);
    ie.SetConfig({{CONFIG_KEY(CACHE_DIR), cache_path}}, "");
    runParallel([&] () {
        (void)ie.LoadNetwork(model, target_device);
    }, numIterations, numThreads);
    ie.SetConfig({{CONFIG_KEY(CACHE_DIR), ""}}, "");;
}