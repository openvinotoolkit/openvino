// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <vector>
#include "gpu/gpu_config.hpp"
#include "multi/multi_remote_blob_tests.hpp"
#include "common_test_utils/test_constants.hpp"

const std::vector<DevicesNamesAndSupportPair> device_names_and_support_for_remote_blobs {
        {{GPU}, true}, // GPU via MULTI,
#ifdef ENABLE_MKL_DNN
        {{GPU, CPU}, true}, // GPU+CPU
        {{CPU, GPU}, true}, // CPU+GPU
#endif
};

INSTANTIATE_TEST_CASE_P(smoke_RemoteBlobMultiGPU, MultiDevice_SupportTest,
                        ::testing::ValuesIn(device_names_and_support_for_remote_blobs), MultiDevice_SupportTest::getTestCaseName);

TEST_P(MultiDevice_Test, cannotInferRemoteBlobIfNotInitializedForDevice) {
    InferenceEngine::CNNNetwork net(fn_ptr);
    auto ie = PluginCache::get().ie();
    // load a network to the GPU to make sure we have a remote context
    auto exec_net = ie->LoadNetwork(net, GPU);
    auto ctx = exec_net.GetContext();

    const InferenceEngine::ConstInputsDataMap inputInfo = exec_net.GetInputsInfo();
    auto& first_input_name = inputInfo.begin()->first;
    auto& first_input = inputInfo.begin()->second;
    auto rblob = InferenceEngine::make_shared_blob(first_input->getTensorDesc(), ctx);
    rblob->allocate();

    InferenceEngine::ExecutableNetwork exec_net_multi;
    try {
        exec_net_multi = ie->LoadNetwork(net, device_names);
    } catch(...) {
        // device is unavailable (e.g. for the "second GPU" test) or other (e.g. env) issues not related to the test
        return;
    }
    InferenceEngine::InferRequest req = exec_net_multi.CreateInferRequest();
    ASSERT_TRUE(req);
    ASSERT_NO_THROW(req.SetBlob(first_input_name, rblob));
    ASSERT_NO_THROW(req.StartAsync());
    ASSERT_THROW(req.Wait(InferenceEngine::InferRequest::WaitMode::RESULT_READY), InferenceEngine::Exception);
}

const std::vector<DevicesNames> device_names_and_support_for_remote_blobs2 {
#ifdef ENABLE_MKL_DNN
        {CPU},  // stand-alone CPU via MULTI (no GPU), no OCL context
#endif
        {"GPU.1"},  // another GPU (the test will test its presence), different OCL contexts
};

INSTANTIATE_TEST_CASE_P(smoke_RemoteBlobMultiInitializedWithoutGPU, MultiDevice_Test,
                        ::testing::ValuesIn(device_names_and_support_for_remote_blobs2), MultiDevice_Test::getTestCaseName);
