// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <gpu/gpu_config.hpp>
#include <memory>
#include <string>
#include <utility>
#include <vector>


#include "base/behavior_test_utils.hpp"
#include "common_test_utils/test_common.hpp"
#include "functional_test_utils/blob_utils.hpp"
#include "ov_models/subgraph_builders.hpp"

using namespace ::testing;
using namespace InferenceEngine;

namespace AutoBatchingTests {
using AutoBatchTwoNetsParams = std::tuple<std::string,  // device name
                                          bool,         // get or set blob
                                          size_t,       // number of streams
                                          size_t,       // number of requests
                                          size_t>;      // batch size>

class AutoBatching_Test : public BehaviorTestsUtils::IEPluginTestBase,
                          public testing::WithParamInterface<AutoBatchTwoNetsParams> {
    void SetUp() override {
        std::tie(target_device, use_get_blob, num_streams, num_requests, num_batch) = this->GetParam();
        // Skip test according to plugin specific disabledTestPatterns() (if any)
        SKIP_IF_CURRENT_TEST_IS_DISABLED()
        fn_ptrs = {ngraph::builder::subgraph::makeSingleConv(), ngraph::builder::subgraph::makeMultiSingleConv()};
    };

public:
    static std::string getTestCaseName(const testing::TestParamInfo<AutoBatchTwoNetsParams>& obj) {
        size_t streams, requests, batch;
        bool use_get_blob;
        std::string target_device;
        std::tie(target_device, use_get_blob, streams, requests, batch) = obj.param;
        return target_device + std::string(use_get_blob ? "_get_blob" : "_set_blob") + "_batch_size_" +
               std::to_string(batch) + "_num_streams_" + std::to_string(streams) + "_num_req_" +
               std::to_string(requests);
    }

protected:
    bool use_get_blob;
    size_t num_streams;
    size_t num_requests;
    size_t num_batch;
    std::vector<std::shared_ptr<ngraph::Function>> fn_ptrs;

    void TestAutoBatch() {
        std::vector<InferenceEngine::CNNNetwork> nets;
        for (auto& fn_ptr : fn_ptrs) {
            nets.push_back(CNNNetwork(fn_ptr));
        }

        auto ie = BehaviorTestsUtils::createIECoreWithTemplate();
        std::vector<std::string> outputs;
        std::vector<InferRequest> irs;
        std::vector<std::vector<uint8_t>> ref;
        std::vector<size_t> outElementsCount;

        for (size_t i = 0; i < nets.size(); ++i) {
            auto net = nets[i];
            auto inputs = net.getInputsInfo();
            for (auto n : inputs) {
                n.second->setPrecision(Precision::FP32);
            }
            std::map<std::string, std::string> config;
            // minimize timeout to reduce test time
            config[CONFIG_KEY(AUTO_BATCH_TIMEOUT)] = std::to_string(1);
            auto exec_net_ref = ie.LoadNetwork(net,
                                               std::string(ov::test::utils::DEVICE_BATCH) + ":" + target_device + "(" +
                                                   std::to_string(num_batch) + ")",
                                               config);

            auto network_outputs = net.getOutputsInfo();
            ASSERT_EQ(network_outputs.size(), 1) << " Auto-Batching tests use networks with single output";
            auto output = network_outputs.begin();  // single output
            for (size_t j = 0; j < num_requests; j++) {
                outputs.push_back(output->first);
                outElementsCount.push_back(std::accumulate(begin(fn_ptrs[i]->get_output_shape(0)),
                                                           end(fn_ptrs[i]->get_output_shape(0)),
                                                           (size_t)1,
                                                           std::multiplies<size_t>()));

                auto inf_req = exec_net_ref.CreateInferRequest();
                irs.push_back(inf_req);

                std::vector<std::vector<uint8_t>> inData;
                for (auto n : inputs) {
                    auto blob = FuncTestUtils::createAndFillBlob(n.second->getTensorDesc());
                    if (use_get_blob)
                        memcpy(reinterpret_cast<void*>(inf_req.GetBlob(n.first)->buffer().as<uint8_t*>()),
                               reinterpret_cast<const void*>(blob->cbuffer().as<uint8_t*>()),
                               blob->byteSize());
                    else
                        inf_req.SetBlob(n.first, blob);

                    const auto inBlob = inf_req.GetBlob(n.first);
                    const auto blobSize = inBlob->byteSize();
                    const auto inBlobBuf = inBlob->cbuffer().as<uint8_t*>();
                    inData.push_back(std::vector<uint8_t>(inBlobBuf, inBlobBuf + blobSize));
                }
                if (!use_get_blob) {
                    auto blob = FuncTestUtils::createAndFillBlob(output->second->getTensorDesc());
                    inf_req.SetBlob(output->first, blob);
                }

                auto refOutData = ngraph::helpers::interpreterFunction(fn_ptrs[i], {inData}).front().second;
                ref.push_back(refOutData);
            }
        }

        const int niter = 1;
        for (int i = 0; i < niter; i++) {
            for (auto ir : irs) {
                ir.StartAsync();
            }

            for (auto ir : irs) {
                ir.Wait(InferRequest::RESULT_READY);
            }
        }

        auto thr = FuncTestUtils::GetComparisonThreshold(InferenceEngine::Precision::FP32);
        for (size_t i = 0; i < irs.size(); ++i) {
            const auto& refBuffer = ref[i].data();
            ASSERT_EQ(outElementsCount[i], irs[i].GetBlob(outputs[i])->size());
            FuncTestUtils::compareRawBuffers(irs[i].GetBlob(outputs[i])->buffer().as<float*>(),
                                             reinterpret_cast<const float*>(refBuffer),
                                             outElementsCount[i],
                                             outElementsCount[i],
                                             thr);
        }
    }
};

class AutoBatching_Test_DetectionOutput : public AutoBatching_Test {
public:
    void SetUp() override {
        std::tie(target_device, use_get_blob, num_streams, num_requests, num_batch) = this->GetParam();
        // Skip test according to plugin specific disabledTestPatterns() (if any)
        SKIP_IF_CURRENT_TEST_IS_DISABLED()
        fn_ptrs = {ngraph::builder::subgraph::makeDetectionOutput(), ngraph::builder::subgraph::makeDetectionOutput()};
    };

    static std::string getTestCaseName(const testing::TestParamInfo<AutoBatchTwoNetsParams>& obj) {
        size_t streams, requests, batch;
        bool use_get_blob;
        std::string target_device;
        std::tie(target_device, use_get_blob, streams, requests, batch) = obj.param;
        return "DetectionOutput_HETERO_" + target_device + std::string(use_get_blob ? "_get_blob" : "_set_blob") +
               "_batch_size_" + std::to_string(batch) + "_num_streams_" + std::to_string(streams) + "_num_req_" +
               std::to_string(requests);
    }
};

TEST_P(AutoBatching_Test, compareAutoBatchingToSingleBatch) {
    TestAutoBatch();
}

TEST_P(AutoBatching_Test_DetectionOutput, compareAutoBatchingToSingleBatch) {
    TestAutoBatch();
}

}  // namespace AutoBatchingTests
