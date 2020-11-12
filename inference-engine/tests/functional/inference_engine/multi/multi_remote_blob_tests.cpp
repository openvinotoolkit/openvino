// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <vector>
#include "multi/multi_helpers.hpp"
#include "common_test_utils/test_constants.hpp"
#include "functional_test_utils/plugin_cache.hpp"

TEST_P(MultiDevice_SupportTest, canCreateContextThenRequestThenBlobsAndInfer) {
    InferenceEngine::CNNNetwork net;
    net = CNNNetwork(fn_ptr);
    net.getInputsInfo().begin()->second->setLayout(Layout::NCHW);
    net.getInputsInfo().begin()->second->setPrecision(Precision::U8);

    auto ie = InferenceEngine::Core();

    auto exec_net = ie.LoadNetwork(net, device_names);
    if (expected_status) {
        InferenceEngine::RemoteContext::Ptr ctx;
        ASSERT_NE(ctx = exec_net.GetContext(), nullptr);
        InferRequest req = exec_net.CreateInferRequest();
        ASSERT_NE((std::shared_ptr<InferenceEngine::IInferRequest>)req, nullptr);
        const InferenceEngine::ConstInputsDataMap inputInfo = exec_net.GetInputsInfo();
        for (auto i : inputInfo) {
            auto rblob = InferenceEngine::make_shared_blob(i.second->getTensorDesc(), ctx);
            rblob->allocate();
            req.SetBlob(i.first, rblob);
        }
        ASSERT_NO_THROW(req.StartAsync());
        ASSERT_EQ(req.Wait(IInferRequest::RESULT_READY), StatusCode::OK);

    } else {
        ASSERT_THROW(exec_net.GetContext(), InferenceEngine::NotImplemented);
    }
}

const std::vector<DevicesNamesAndSupportPair> device_names_and_support_for_remote_blobs {
#if ENABLE_MKL_DNN
        {{CPU}, false}, // CPU via MULTI
    #if ENABLE_CLDNN
        {{GPU, CPU}, true}, // GPU+CPU
        {{CPU, GPU}, true}, // CPU+GPU
    #endif
    #if ENABLE_MYRIAD
        {{CPU, MYRIAD}, false},  // CPU+MYX
    #endif
#endif
#if ENABLE_CLDNN
        {{GPU}, true}, // GPU via MULTI,
#endif
#if ENABLE_MYRIAD
        {{MYRIAD}, false}, // MYX via MULTI
#endif
};

INSTANTIATE_TEST_CASE_P(smoke_RemoteBlobMulti, MultiDevice_SupportTest,
        ::testing::ValuesIn(device_names_and_support_for_remote_blobs), MultiDevice_SupportTest::getTestCaseName);
