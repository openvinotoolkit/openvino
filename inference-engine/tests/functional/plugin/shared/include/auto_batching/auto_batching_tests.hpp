// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <utility>
#include <vector>
#include <memory>

#include <gpu/gpu_config.hpp>
#include <common_test_utils/test_common.hpp>
#include <functional_test_utils/plugin_cache.hpp>

#include "ngraph_functions/subgraph_builders.hpp"
#include "functional_test_utils/blob_utils.hpp"

using namespace ::testing;
using namespace InferenceEngine;

using AutoBatchTwoNetsParams = std::tuple<
        std::string,             // device name
        bool,  // get or set blob
        size_t,  // number of streams
        size_t,  // number of requests
        size_t>; // batch size>


class AutoBatching_Test : public CommonTestUtils::TestsCommon,
                          public testing::WithParamInterface<AutoBatchTwoNetsParams> {
    void SetUp() override {
        std::tie(device_name, use_get_blob, num_streams, num_requests, num_batch) = this->GetParam();
        fn_ptrs = {ngraph::builder::subgraph::makeSingleConv(), ngraph::builder::subgraph::makeMultiSingleConv()};
    };
public:
    static std::string getTestCaseName(const testing::TestParamInfo<AutoBatchTwoNetsParams>& obj) {
        size_t streams, requests, batch;
        bool use_get_blob;
        std::string device_name;
        std::tie(device_name, use_get_blob, streams, requests, batch) = obj.param;
        return device_name + std::string(use_get_blob ? "_get_blob" : "_set_blob") + "_batch_size_" + std::to_string(batch) +
               "_num_streams_" + std::to_string(streams) + "_num_req_" + std::to_string(requests);
    }

protected:
    std::string device_name;
    bool   use_get_blob;
    size_t num_streams;
    size_t num_requests;
    size_t num_batch;
    std::vector<std::shared_ptr<ngraph::Function>> fn_ptrs;
};

TEST_P(AutoBatching_Test, compareAutoBatchingToBatch1) {
    std::vector<InferenceEngine::CNNNetwork> nets;
    for (auto &fn_ptr : fn_ptrs) {
        nets.push_back(CNNNetwork(fn_ptr));
    }

    auto ie = InferenceEngine::Core();
    std::vector<std::string> outputs;
    std::vector<InferRequest> irs;
    std::vector<std::vector<uint8_t>> ref;
    std::vector<int> outElementsCount;

    for (size_t i = 0; i < nets.size(); ++i) {
        auto net = nets[i];

        // we test single inputs networks only
        auto inp = net.getInputsInfo().begin()->second;
        inp->setLayout(Layout::NCHW);
        inp->setPrecision(Precision::FP32);
        std::map<std::string, std::string> config;
        if (device_name.find("GPU") != std::string::npos)
            config[CONFIG_KEY(GPU_THROUGHPUT_STREAMS)] = std::to_string(num_streams);
        if (device_name.find("CPU") != std::string::npos)
            config[CONFIG_KEY(CPU_THROUGHPUT_STREAMS)] = std::to_string(num_streams);
        auto exec_net_ref = ie.LoadNetwork(net, std::string(CommonTestUtils::DEVICE_BATCH) + ":" +
                                                   device_name + "(" + std::to_string(num_batch) + ")",
                                                   config);

        for (int j = 0; j < num_requests; j++) {
            outputs.push_back(net.getOutputsInfo().begin()->first);

            auto inf_req = exec_net_ref.CreateInferRequest();
            irs.push_back(inf_req);

            auto blob = FuncTestUtils::createAndFillBlob(net.getInputsInfo().begin()->second->getTensorDesc());
            if (use_get_blob)
                InferenceEngine::blob_copy(blob, inf_req.GetBlob(net.getInputsInfo().begin()->first));
            else
                inf_req.SetBlob(net.getInputsInfo().begin()->first, blob);
            outElementsCount.push_back(
                    std::accumulate(begin(fn_ptrs[i]->get_output_shape(0)), end(fn_ptrs[i]->get_output_shape(0)), 1,
                                    std::multiplies<size_t>()));
            const auto inBlob = inf_req.GetBlob(net.getInputsInfo().begin()->first);
            const auto blobSize = inBlob->byteSize();
            const auto inBlobBuf = inBlob->cbuffer().as<uint8_t *>();
            std::vector<uint8_t> inData(inBlobBuf, inBlobBuf + blobSize);
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
        const auto &refBuffer = ref[i].data();
        ASSERT_EQ(outElementsCount[i], irs[i].GetBlob(outputs[i])->size());
        FuncTestUtils::compareRawBuffers(irs[i].GetBlob(outputs[i])->buffer().as<float *>(),
                                         reinterpret_cast<const float *>(refBuffer), outElementsCount[i],
                                         outElementsCount[i],
                                         thr);
    }
}