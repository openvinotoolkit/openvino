// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <thread>

#include "base/ov_behavior_test_utils.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "common_test_utils/subgraph_builders/2_input_subtract.hpp"
#include "common_test_utils/subgraph_builders/multi_single_conv.hpp"
#include "common_test_utils/subgraph_builders/single_conv.hpp"
#include "common_test_utils/subgraph_builders/split_conv_concat.hpp"
#include "common_test_utils/subgraph_builders/split_multi_conv_concat.hpp"
#include "functional_test_utils/skip_tests_config.hpp"
#include "functional_test_utils/ov_plugin_cache.hpp"
#include "openvino/runtime/properties.hpp"

using Device = std::string;
using Config = ov::AnyMap;
using Params = std::tuple<Device, Config>;

class CoreThreadingTestsBase {
public:
    static void runParallel(std::function<void(void)> func,
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

    void safePluginUnload(ov::Core& core, const std::string& deviceName) {
        try {
            core.unload_plugin(deviceName);
        } catch (const ov::Exception& ex) {
            // if several threads unload plugin at once, the first thread does this
            // while all others will throw an exception that plugin is not registered
            ASSERT_STR_CONTAINS(ex.what(), "name is not registered in the");
        }
    }

    Config config;
};

//
//  Parameterized tests with number of parallel threads, iterations
//

using Threads = unsigned int;
using Iterations = unsigned int;

using CoreThreadingParams = std::tuple<Params, Threads, Iterations>;

class CoreThreadingTestsWithCacheEnabled : public testing::WithParamInterface<CoreThreadingParams>,
                                           public ov::test::behavior::OVPluginTestBase,
                                           public CoreThreadingTestsBase {
public:
    void SetUp() override {
        SKIP_IF_CURRENT_TEST_IS_DISABLED();
        std::tie(target_device, config) = std::get<0>(GetParam());
        numThreads = std::get<1>(GetParam());
        numIterations = std::get<2>(GetParam());
        auto hash = std::hash<std::string>()(::testing::UnitTest::GetInstance()->current_test_info()->name());
        std::stringstream ss;
        ss << std::this_thread::get_id();
        cache_path = "threading_test" + std::to_string(hash) + "_" + ss.str() + "_" + GetTimestamp() + "_cache";
        APIBaseTest::SetUp();
    }

    void TearDown() override {
        ov::test::utils::removeFilesWithExt(cache_path, "blob");
        std::remove(cache_path.c_str());
        APIBaseTest::TearDown();
    }

    static std::string getTestCaseName(testing::TestParamInfo<CoreThreadingParams> obj) {
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
            result << confItem.first << "=" << confItem.second.as<std::string>() << separator;
        }
        result << "numThreads=" << numThreads << separator;
        result << "numIter=" << numIterations;
        return result.str();
    }

protected:
    unsigned int numIterations;
    unsigned int numThreads;
    std::string cache_path;
    std::vector<std::shared_ptr<ov::Model>> models;

    void SetupModels() {
        ov::Core core;
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
        ov::Tensor weights = {};
        auto model = core.read_model(ir_with_meta, weights);
        OPENVINO_ASSERT(model);
        models.emplace_back(model); // model with cli_parameter
        // test model with runtime attributes -- layout
        model = ov::test::utils::make_split_multi_conv_concat();
        for (auto& iter : model->get_parameters())
            iter->set_layout("NCHW");
        for (auto& iter : model->get_results())
            iter->set_layout("NHCW");
        models.emplace_back(model);
    }
};

// tested function: set_property, compile_model
TEST_P(CoreThreadingTestsWithCacheEnabled, smoke_compiled_model_cache_enabled) {
    auto core = ov::test::utils::create_core();
    SetupModels();
    core.set_property(target_device, config);
    core.set_property(ov::cache_dir(cache_path));
    for (auto& model : models) {
        runParallel(
            [&]() {
                (void)core.compile_model(model, target_device);
            },
            numIterations,
            numThreads);
    }
    core.set_property(ov::cache_dir(""));
}

class CoreThreadingTest : public testing::WithParamInterface<Params>,
                            public ov::test::behavior::OVPluginTestBase,
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
            result << confItem.first << "=" << confItem.second.as<std::string>() << separator;
        }
        return result.str();
    }
};

// tested function: get_versions, unload_plugin
TEST_P(CoreThreadingTest, smoke_GetVersions) {
    auto core = ov::test::utils::create_core();
    runParallel([&]() {
        auto versions = core.get_versions(target_device);
        ASSERT_LE(1u, versions.size());
        safePluginUnload(core, target_device);
    });
}

// tested function: get_property, UnregisterPlugin
TEST_P(CoreThreadingTest, smoke_GetMetric) {
    auto core = ov::test::utils::create_core();

    runParallel([&] () {
        core.get_property(target_device, ov::internal::supported_properties);
        safePluginUnload(core, target_device);
    });
}

// tested function: set_property for already created plugins
TEST_P(CoreThreadingTest, smoke_SetProperty_PluginExists) {
    auto core = ov::test::utils::create_core();
    core.set_property(target_device, config);
    auto versions = core.get_versions(target_device);
    runParallel(
        [&]() {
            core.set_property(target_device, config);
        },
        10000);
}

// tested function: get_property, unload_plugin
TEST_P(CoreThreadingTest, smoke_GetProperty_PluginExists) {
    auto core = ov::test::utils::create_core();
    std::string configKey = config.begin()->first;
    core.set_property(target_device, config);
    runParallel([&]() {
        core.get_property(target_device, configKey);
        safePluginUnload(core, target_device);
    });
}

// tested function: query_model
TEST_P(CoreThreadingTest, smoke_QueryModel) {
    auto core = ov::test::utils::create_core();
    core.set_property(target_device, config);
    auto model = ov::test::utils::make_2_input_subtract();
    auto refResult = core.query_model(model, target_device);

    runParallel(
        [&]() {
            const auto result = core.query_model(model, target_device);
            safePluginUnload(core, target_device);

            // compare QueryNetworkResult with reference
            for (auto&& r : refResult) {
                ASSERT_NE(result.end(), result.find(r.first));
            }
            for (auto&& r : result) {
                ASSERT_NE(refResult.end(), refResult.find(r.first));
            }
        },
        3000);
}

class CoreThreadingTestsWithIter : public testing::WithParamInterface<CoreThreadingParams>,
                                          public ov::test::behavior::OVPluginTestBase,
                                          public CoreThreadingTestsBase {
public:
    void SetUp() override {
        SKIP_IF_CURRENT_TEST_IS_DISABLED();
        std::tie(target_device, config) = std::get<0>(GetParam());
        numThreads = std::get<1>(GetParam());
        numIterations = std::get<2>(GetParam());
        APIBaseTest::SetUp();
    }

    static std::string getTestCaseName(testing::TestParamInfo<CoreThreadingParams> obj) {
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
            result << confItem.first << "=" << confItem.second.as<std::string>() << separator;
        }
        result << "numThreads=" << numThreads << separator;
        result << "numIter=" << numIterations;
        return result.str();
    }

protected:
    unsigned int numIterations;
    unsigned int numThreads;

    std::vector<std::shared_ptr<ov::Model>> models;
    void SetupNetworks() {
        models.emplace_back(ov::test::utils::make_2_input_subtract());
        models.emplace_back(ov::test::utils::make_multi_single_conv());
        models.emplace_back(ov::test::utils::make_single_conv());
        models.emplace_back(ov::test::utils::make_split_conv_concat());
        models.emplace_back(ov::test::utils::make_split_multi_conv_concat());
    }
};

// tested function: compile_model
TEST_P(CoreThreadingTestsWithIter, smoke_CompileModel) {
    auto core = ov::test::utils::create_core();
    std::atomic<unsigned int> counter{0u};
    SetupNetworks();
    core.set_property(target_device, config);
    runParallel(
        [&]() {
            auto value = counter++;
            (void)core.compile_model(models[value % models.size()], target_device);
        },
        numIterations,
        numThreads);
}

// tested function: single Core compile_model accuracy
TEST_P(CoreThreadingTestsWithIter, smoke_CompileModel_Accuracy_SingleCore) {
    auto core = ov::test::utils::create_core();
    std::atomic<unsigned int> counter{0u};
    SetupNetworks();
    core.set_property(target_device, config);

    runParallel(
        [&]() {
            auto value = counter++;
            auto model = models[value % models.size()];

            std::map<ov::Output<ov::Node>, ov::Tensor> inputs;
            for (const auto& input : model->inputs()) {
                auto tensor = ov::test::utils::create_and_fill_tensor_normal_distribution(input.get_element_type(),
                                                                                          input.get_shape(),
                                                                                          0.0f,
                                                                                          0.2f,
                                                                                          7235346);
                inputs.insert({input, tensor});
            }

            auto getOutputBlob = [&](ov::Core& core) {
                ov::AnyMap f32_precision_property = {{ov::hint::inference_precision.name(), ov::element::f32.to_string()}};
                auto compiled_model = core.compile_model(model, target_device, f32_precision_property);
                auto req = compiled_model.create_infer_request();
                for (const auto& input : inputs) {
                    req.set_tensor(input.first, input.second);
                }
                auto output_tensor = ov::Tensor(model->output().get_element_type(), model->output().get_shape());
                req.set_output_tensor(output_tensor);
                req.infer();
                return output_tensor;
            };

            auto outputActual = getOutputBlob(core);

            // compare actual value using the same Core
            auto outputRef = getOutputBlob(core);
            ov::test::utils::compare(outputActual, outputRef);
        },
        numIterations,
        numThreads);
}

// tested function: multi Core compile_model accuracy
TEST_P(CoreThreadingTestsWithIter, smoke_CompileModel_Accuracy_MultipleCores) {
    auto core = ov::test::utils::create_core();
    std::atomic<unsigned int> counter{0u};
    SetupNetworks();
    core.set_property(target_device, config);
    runParallel(
        [&]() {
            auto value = counter++;
            auto model = models[value % models.size()];

            std::map<ov::Output<ov::Node>, ov::Tensor> inputs;
            for (const auto& input : model->inputs()) {
                auto tensor = ov::test::utils::create_and_fill_tensor_normal_distribution(input.get_element_type(),
                                                                                          input.get_shape(),
                                                                                          0.0f,
                                                                                          0.2f,
                                                                                          7235346);
                inputs.insert({input, tensor});
            }

            auto getOutputBlob = [&](ov::Core& core) {
                ov::AnyMap f32_precision_property = {{ov::hint::inference_precision.name(), ov::element::f32.to_string()}};
                auto compiled_model = core.compile_model(model, target_device, f32_precision_property);
                auto req = compiled_model.create_infer_request();
                for (const auto& input : inputs) {
                    req.set_tensor(input.first, input.second);
                }
                auto output_tensor = ov::Tensor(model->output().get_element_type(), model->output().get_shape());
                req.set_output_tensor(output_tensor);
                req.infer();
                return output_tensor;
            };

            auto outputActual = getOutputBlob(core);

            // compare actual value using the second Core
            {
                auto core_2 = ov::test::utils::create_core();
                core_2.set_property(target_device, config);
                auto outputRef = getOutputBlob(core_2);
                ov::test::utils::compare(outputActual, outputRef);
            }
        },
        numIterations,
        numThreads);
}

// tested function: set_property, compile_model
TEST_P(CoreThreadingTestsWithIter, smoke_CompileModel_MultipleCores) {
    std::atomic<unsigned int> counter{0u};

    SetupNetworks();

    runParallel(
        [&]() {
            auto value = counter++;
            auto core = ov::test::utils::create_core();;
            core.set_property(target_device, config);
            (void)core.compile_model(models[value % models.size()], target_device);
        },
        numIterations,
        numThreads);
}

TEST_P(CoreThreadingTestsWithIter, nightly_AsyncInfer_ShareInput) {
    SetupNetworks();
    auto model = models[0];
    ov::Tensor output_ref;
    std::map<ov::Output<ov::Node>, ov::Tensor> inputs;
    for (const auto& input : model->inputs()) {
        auto tensor = ov::test::utils::create_and_fill_tensor_normal_distribution(input.get_element_type(),
                                                                                  input.get_shape(),
                                                                                  0.0f,
                                                                                  0.2f,
                                                                                  7235346);
        inputs.insert({input, tensor});
    }

    runParallel(
        [&]() {
            auto core = ov::test::utils::create_core();
            core.set_property(target_device, config);
            auto compiled_model = core.compile_model(model, target_device);
            auto nireq = compiled_model.get_property(ov::optimal_number_of_infer_requests);
            std::vector<ov::InferRequest> inferReqsQueue;
            int count = nireq;
            while (count--) {
                ov::InferRequest req = compiled_model.create_infer_request();
                for (const auto& input : inputs) {
                    req.set_tensor(input.first, input.second);
                }
                inferReqsQueue.push_back(req);
            }
            for (auto& req : inferReqsQueue) {
                req.start_async();
            }
            for (auto& req : inferReqsQueue) {
                req.wait();
            }
        },
        numIterations,
        numThreads);
}
