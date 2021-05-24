// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <tuple>
#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <algorithm>

#include "ie_core.hpp"

#include "ie_transformations.hpp"
#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/blob_utils.hpp"
#include "functional_test_utils/precision_utils.hpp"
#include "functional_test_utils/plugin_cache.hpp"
#include "functional_test_utils/skip_tests_config.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"
#include "ngraph_functions/builders.hpp"

#include <transformations/op_conversions/lstm_cell_decomposition.hpp>
#include "transformations/control_flow/unroll_tensor_iterator.hpp"
#include "configuration_tests/dynamic_batch.hpp"

#include "ngraph_functions/subgraph_builders.hpp"

namespace ConfigurationTestsDefinitions {

    std::string DynamicBatchTest::getTestCaseName(const testing::TestParamInfo<dynamicBatchTestParams> &obj) {
        std::string targetDevice;
        InferenceEngine::Precision netPrecision;
        std::vector<size_t> batchSizes;
        bool runAsync;
        std::map<std::string, std::string> config;
        std::tie(targetDevice, netPrecision, batchSizes, runAsync, config) = obj.param;
        std::ostringstream result;

        result << "netPrecision=" << netPrecision.name() << "_";
        result << "BS=" << CommonTestUtils::vec2str(batchSizes) << "_";
        result << std::string(runAsync ? "Async" : "Sync") << "_";
        result << "targetDevice=" << targetDevice;
        return result.str();
    }

    size_t hiddenSize;


    void DynamicBatchTest::SetUp() {
        InferenceEngine::Precision netPrecision;
        std::map<std::string, std::string> config;
        std::tie(targetDevice, netPrecision, batch_sizes, run_async, config) = this->GetParam();
        configuration.insert(config.begin(), config.end());
        configuration[InferenceEngine::PluginConfigParams::KEY_DYN_BATCH_ENABLED] = InferenceEngine::PluginConfigParams::YES;

        max_batch_size = *std::max_element(batch_sizes.begin(), batch_sizes.end());

        function = ngraph::builder::subgraph::makeSingleConv();
    }

    void DynamicBatchTest::LoadNetwork() {
        cnnNetwork = InferenceEngine::CNNNetwork{function};
        ConfigureNetwork();
        cnnNetwork.setBatchSize(max_batch_size);
        executableNetwork = core->LoadNetwork(cnnNetwork, targetDevice, configuration);
    }

    void DynamicBatchTest::Infer() {
        inferRequest = executableNetwork.CreateInferRequest();
        inputs.clear();

        for (int i = 0; i < batch_sizes.size(); i++) {
            auto batch_size = batch_sizes[i];

            cnnNetwork.setBatchSize(batch_size);
            inputs.clear();
            for (const auto &input : cnnNetwork.getInputsInfo()) {
                const auto &info = input.second;
                auto blob = GenerateInput(*info);
                inputs.push_back(blob);
            }
            reference_inputs.push_back(inputs);
            reference_outputs.push_back(CalculateRefs());
        }

        for (int i = 0; i < batch_sizes.size(); i++) {
            infer_requests.push_back(executableNetwork.CreateInferRequest());
            auto batch_size = batch_sizes[i];

            auto& infer_request = infer_requests[i];
            infer_request.SetBatch(batch_size);

            inputs.clear();
            for (const auto &input : executableNetwork.GetInputsInfo()) {
                const auto &info = input.second;
                auto blob = GenerateInput(*info);
                infer_request.SetBlob(info->name(), blob);
                inputs.push_back(blob);
            }

            scaled_inputs.push_back(inputs);

            for (int j = 0; j < reference_inputs[i].size(); j++) {
                auto& ref = reference_inputs[i][j];
                auto& actual = scaled_inputs[i][j];

                auto byte_num = ref->byteSize();
                auto ref_ptr = ref->buffer().as<uint8_t*>();
                auto actual_ptr = actual->buffer().as<uint8_t*>();

                for (int k = 0; k < byte_num; k++) {
                    actual_ptr[k] = ref_ptr[k];
                }
            }
        }

        for (auto& infer_request : infer_requests) {
            if (run_async) {
                infer_request.StartAsync();
            } else {
                infer_request.Infer();
            }
        }

        if (run_async) {
            for (auto& infer_request : infer_requests) {
                auto status = infer_request.Wait(10000);
                if (status != InferenceEngine::StatusCode::OK) {
                    GTEST_FAIL() << "Inference request status after wait is not OK";
                }
            }
        }
    }

    void DynamicBatchTest::Validate() {
        for (int i = 0; i < infer_requests.size(); i++) {
            auto outputs = std::vector<InferenceEngine::Blob::Ptr>{};
            for (const auto &output : executableNetwork.GetOutputsInfo()) {
                const auto &name = output.first;
                outputs.push_back(infer_requests[i].GetBlob(name));
            }
            for (int j = 0; j < reference_outputs[i].size(); j++) {
                if (reference_outputs[i][j].second.size() < outputs[j]->byteSize()) {
                    auto actual_ptr = outputs[j]->buffer().as<uint8_t*>();
                    for (int k = reference_outputs[i][j].second.size(); k < outputs[j]->byteSize(); k++) actual_ptr[k] = 0;
                    reference_outputs[i][j].second.resize(outputs[j]->byteSize());
                }
            }
            Compare(reference_outputs[i], outputs);
        }
    }

    void DynamicBatchTest::Run() {
        SKIP_IF_CURRENT_TEST_IS_DISABLED();
        LoadNetwork();
        GenerateInputs();
        Infer();
        Validate();
    }

    TEST_P(DynamicBatchTest, CompareWithRefs) {
        Run();
    };
} // namespace ConfigurationTestsDefinitions
