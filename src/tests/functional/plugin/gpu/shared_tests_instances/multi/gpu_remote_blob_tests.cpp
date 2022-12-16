// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <vector>
#include "gpu/gpu_config.hpp"
#include "multi/multi_remote_blob_tests.hpp"
#include "multi/multi_remote_blob_multidevice_test.hpp"
#include "common_test_utils/test_constants.hpp"
#include <openvino/runtime/auto/properties.hpp>

using MultiDevice_RemoteBlobAndBindTest = MultiDevice_Test;

auto device_names_and_support_for_remote_blobs = []() {
    return std::vector<DevicesNamesAndSupportPair>{
        {{GPU}, true},      // GPU via MULTI,
        {{"GPU.0"}, true},  // GPU.0 via MULTI,
#ifdef ENABLE_INTEL_CPU
        {{GPU, CPU}, true},  // GPU+CPU
        {{CPU, GPU}, true},  // CPU+GPU
#endif
    };
};

INSTANTIATE_TEST_SUITE_P(smoke_Multi_RemoteBlobGPU,
                         MultiDevice_SupportTest,
                         ::testing::ValuesIn(device_names_and_support_for_remote_blobs()),
                         MultiDevice_SupportTest::getTestCaseName);

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

TEST_P(MultiDevice_RemoteBlobAndBindTest, testRemoteBlobAndDeviceBind) {
    InferenceEngine::CNNNetwork net(fn_ptr);
    net.getInputsInfo().begin()->second->setLayout(InferenceEngine::Layout::NCHW);
    net.getInputsInfo().begin()->second->setPrecision(InferenceEngine::Precision::U8);
    auto ie = PluginCache::get().ie();

    const std::map<std::string, std::string> multi_prop_config = {
        {InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, CommonTestUtils::DEVICE_GPU},
        {ov::intel_auto::device_bind_buffer.name(), InferenceEngine::PluginConfigParams::YES}};

    auto exec_net = ie->LoadNetwork(net, device_names, multi_prop_config);
    std::shared_ptr<InferenceEngine::RemoteContext> ctx;
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
}

auto device_names_and_support_for_remote_blobs2 = []() {
    return std::vector<DevicesNames>{
#ifdef ENABLE_INTEL_CPU
        {CPU},  // stand-alone CPU via MULTI (no GPU), no OCL context
#endif
        {"GPU.1"},  // another GPU (the test will test its presence), different OCL contexts
    };
};

INSTANTIATE_TEST_SUITE_P(smoke_Multi_RemoteBlobInitializedWithoutGPU,
                         MultiDevice_Test,
                         ::testing::ValuesIn(device_names_and_support_for_remote_blobs2()),
                         MultiDevice_Test::getTestCaseName);

auto multi_device_names_and_support_for_remote_blobs = []() {
    return std::vector<DevicesNames>{
#ifdef ENABLE_INTEL_CPU
        {CPU, "GPU.0"},
        {CPU, "GPU.0", "GPU.1"},  // another GPU (the test will test its presence), different OCL contexts
#endif
        {"GPU.0", "GPU.1"}};
};

INSTANTIATE_TEST_SUITE_P(smoke_Multi_RemoteBlobInitializedWithoutGPU,
                         MultiDeviceMultipleGPU_Test,
                         ::testing::ValuesIn(multi_device_names_and_support_for_remote_blobs()),
                         MultiDeviceMultipleGPU_Test::getTestCaseName);

auto multi_device_names_and_support_for_remote_blobs2 = []() {
    return std::vector<DevicesNames>{{"GPU"}};
};

INSTANTIATE_TEST_SUITE_P(smoke_Multi_RemoteBlobAndDeviceBindBuffer,
                         MultiDevice_RemoteBlobAndBindTest,
                         ::testing::ValuesIn(multi_device_names_and_support_for_remote_blobs2()),
                         MultiDevice_RemoteBlobAndBindTest::getTestCaseName);
