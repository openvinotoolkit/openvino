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

using MultiDevice_Bind_test = MultiDevice_Test;

auto device_names_and_support_for_remote_blobs = []() {
    return std::vector<DevicesNamesAndSupportTuple>{
        {{GPU}, true, {}},      // GPU via MULTI,
        {{"GPU.0"}, true, {}},  // GPU.0 via MULTI,
        {{GPU}, true, {ov::intel_auto::device_bind_buffer(true)}},      // GPU via MULTI,
        {{"GPU.0"}, true, {ov::intel_auto::device_bind_buffer(true)}},  // GPU.0 via MULTI,
#ifdef ENABLE_INTEL_CPU
        {{GPU, CPU}, true, {}},  // GPU+CPU
        {{CPU, GPU}, true, {}},  // CPU+GPU
        {{GPU, CPU}, true, {ov::intel_auto::device_bind_buffer(true)}},  // GPU+CPU
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

    std::map<std::string, std::string> configs;
    for (auto&& value : _properties) {
        configs.emplace(value.first, value.second.as<std::string>());
    }

    InferenceEngine::ExecutableNetwork exec_net_multi;
    try {
        exec_net_multi = ie->LoadNetwork(net, device_names, configs);
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

TEST_P(MultiDevice_Bind_test, oversubsciptionOfInferRequest) {
    InferenceEngine::CNNNetwork net(fn_ptr);
    auto ie = PluginCache::get().ie();
    // load a network to the GPU to make sure we have a remote context
    auto exec_net = ie->LoadNetwork(net, GPU);
    auto ctx = exec_net.GetContext();

    const InferenceEngine::ConstInputsDataMap inputInfo = exec_net.GetInputsInfo();
    auto& first_input = inputInfo.begin()->second;
    auto rblob = InferenceEngine::make_shared_blob(first_input->getTensorDesc(), ctx);
    rblob->allocate();

    std::map<std::string, std::string> configs;
    for (auto&& value : _properties) {
        configs.emplace(value.first, value.second.as<std::string>());
    }

    InferenceEngine::ExecutableNetwork exec_net_multi;
    try {
        exec_net_multi = ie->LoadNetwork(net, device_names, configs);
    } catch(...) {
        // device is unavailable (e.g. for the "second GPU" test) or other (e.g. env) issues not related to the test
        return;
    }

    unsigned int optimalNum = 0;
    try {
        optimalNum = exec_net_multi.GetMetric(METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS)).as<unsigned int>();
    } catch (...) {
        std::cout << "ExecutableNetwork getMetric failed" << std::endl;
        return;
    }

    // test binder mode to throw exception when oversubsciption of infer requests
    InferenceEngine::InferRequest req;
    for (int i = 0; i < optimalNum; i++) {
        req = exec_net_multi.CreateInferRequest();
    }
    ASSERT_ANY_THROW(req = exec_net_multi.CreateInferRequest());
}

auto device_names_and_support_for_remote_blobs2 = []() {
    return std::vector<DevicesNamseAndProperties>{
#ifdef ENABLE_INTEL_CPU
        //{{CPU}, {}},  // stand-alone CPU via MULTI (no GPU), no OCL context
        {{CPU}, {ov::intel_auto::device_bind_buffer(true)}},  // stand-alone CPU via MULTI (no GPU), no OCL context
#endif
        {{"GPU.1"}, {}},  // another GPU (the test will test its presence), different OCL contexts
        {{"GPU.1"}, {ov::intel_auto::device_bind_buffer(true)}},  // another GPU (the test will test its presence), different OCL contexts
    };
};

INSTANTIATE_TEST_SUITE_P(smoke_Multi_RemoteBlobInitializedWithoutGPU,
                         MultiDevice_Test,
                         ::testing::ValuesIn(device_names_and_support_for_remote_blobs2()),
                         MultiDevice_Test::getTestCaseName);

auto device_names_and_support_for_remote_blobs3 = []() {
    return std::vector<DevicesNamseAndProperties>{
#ifdef ENABLE_INTEL_CPU
        {{CPU}, {ov::intel_auto::device_bind_buffer(true)}},  // stand-alone CPU via MULTI (no GPU), no OCL context
#endif
    };
};

INSTANTIATE_TEST_SUITE_P(smoke_Multi_RemoteBlobOversubsciptionInferRequest,
                         MultiDevice_Bind_test,
                         ::testing::ValuesIn(device_names_and_support_for_remote_blobs3()),
                         MultiDevice_Test::getTestCaseName);

auto multi_device_names_and_support_for_remote_blobs = []() {
    return std::vector<DevicesNames>{
#ifdef ENABLE_INTEL_CPU
        {"GPU.0", CPU},
        {"GPU.0", "GPU.1", CPU},  // another GPU (the test will test its presence), different OCL contexts
#endif
        {"GPU.0", "GPU.1"}};
};

INSTANTIATE_TEST_SUITE_P(smoke_Multi_RemoteBlobInitializedWithoutGPU,
                         MultiDeviceMultipleGPU_Test,
                         ::testing::ValuesIn(multi_device_names_and_support_for_remote_blobs()),
                         MultiDeviceMultipleGPU_Test::getTestCaseName);
