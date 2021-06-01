// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <vector>
#include "ie_core.hpp"
#include "base/multi/multi_helpers.hpp"
#include "functional_test_utils/plugin_cache.hpp"

TEST_P(MultiDevice_SupportTest, canCreateContextThenRequestThenBlobsAndInfer) {
    InferenceEngine::CNNNetwork net(fn_ptr);
    net.getInputsInfo().begin()->second->setLayout(InferenceEngine::Layout::NCHW);
    net.getInputsInfo().begin()->second->setPrecision(InferenceEngine::Precision::U8);

    auto ie = PluginCache::get().ie();

    auto exec_net = ie->LoadNetwork(net, device_names);
    if (expected_status) {
        InferenceEngine::RemoteContext::Ptr ctx;
        ASSERT_NE(ctx = exec_net.GetContext(), nullptr);
        InferenceEngine::InferRequest req = exec_net.CreateInferRequest();
        ASSERT_TRUE(req);
        const InferenceEngine::ConstInputsDataMap inputInfo = exec_net.GetInputsInfo();
        for (auto i : inputInfo) {
            auto rblob = InferenceEngine::make_shared_blob(i.second->getTensorDesc(), ctx);
            rblob->allocate();
            req.SetBlob(i.first, rblob);
        }
        ASSERT_NO_THROW(req.StartAsync());
        ASSERT_EQ(req.Wait(InferenceEngine::InferRequest::RESULT_READY), InferenceEngine::StatusCode::OK);

    } else {
        ASSERT_THROW(exec_net.GetContext(), InferenceEngine::NotImplemented);
    }
}
