// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <chrono>
#include <future>
#include <tuple>
#include <vector>
#include <string>
#include <memory>
#include "ie_extension.h"
#include <condition_variable>
#include "shared_test_classes/base/layer_test_utils.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"
#include "ngraph_functions/builders.hpp"
#include <string>
#include <ie_core.hpp>
#include <thread>
#include <base/behavior_test_utils.hpp>
#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/plugin_cache.hpp"
#include "functional_test_utils/blob_utils.hpp"
#include "ngraph_functions/subgraph_builders.hpp"
#include "shared_test_classes/subgraph/basic_lstm.hpp"

namespace BehaviorTestsDefinitions {
using InferRequestTests = BehaviorTestsUtils::BehaviorTestsBasic;

// Setting empty config to LoadNetwork doesn't throw
TEST_P(InferRequestTests, SetEmptyConfig) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // Create CNNNetwork from ngrpah::Function
    InferenceEngine::CNNNetwork cnnNet(function);
    // Load CNNNetwork to target plugins
    InferenceEngine::ExecutableNetwork execNet;
    std::map<std::string, std::string> config {};
    if (targetDevice.find(CommonTestUtils::DEVICE_AUTO) == std::string::npos &&
        targetDevice.find(CommonTestUtils::DEVICE_MULTI) == std::string::npos &&
        targetDevice.find(CommonTestUtils::DEVICE_HETERO) == std::string::npos) {
        ASSERT_NO_THROW(ie->SetConfig(configuration, targetDevice));
        ASSERT_NO_THROW(execNet = ie->LoadNetwork(cnnNet, targetDevice, config));
    } else {
        ASSERT_NO_THROW(ie->SetConfig(configuration, targetDevice));
        ASSERT_NO_THROW(execNet = ie->LoadNetwork(cnnNet, targetDevice, configuration));
    }
}

// Load correct network to Plugin to get executable network
TEST_P(InferRequestTests, canLoadCorrectNetworkToGetExecutable) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // Create CNNNetwork from ngrpah::Function
    InferenceEngine::CNNNetwork cnnNet(function);
    InferenceEngine::ExecutableNetwork execNet;
    ASSERT_NO_THROW(execNet = ie->LoadNetwork(cnnNet, targetDevice, configuration));
}

TEST_P(InferRequestTests,  CanCreateTwoExeNetworks) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // Create CNNNetwork from ngrpah::Function
    InferenceEngine::CNNNetwork cnnNet(function);
    InferenceEngine::ExecutableNetwork execNet;
    for (auto i = 0; i < 2; i++) {
        ASSERT_NO_THROW(execNet = ie->LoadNetwork(cnnNet, targetDevice, configuration));
        ASSERT_NE(nullptr, cnnNet.getFunction());
    }
}

TEST_P(InferRequestTests, CanCreateInferRequest) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // Create CNNNetwork from ngrpah::Function
    InferenceEngine::CNNNetwork cnnNet(function);
    // Load CNNNetwork to target plugins
    auto execNet = ie->LoadNetwork(cnnNet, targetDevice, configuration);
    // Create InferRequest
    InferenceEngine::InferRequest req;
    ASSERT_NO_THROW(req = execNet.CreateInferRequest());
}

TEST_P(InferRequestTests, failToSetNullptrForInput) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // Create CNNNetwork from ngrpah::Function
    InferenceEngine::CNNNetwork cnnNet(function);
    // Load CNNNetwork to target plugins
    auto execNet = ie->LoadNetwork(cnnNet, targetDevice, configuration);
    // Create InferRequest
    InferenceEngine::InferRequest req;
    ASSERT_NO_THROW(req = execNet.CreateInferRequest());
    InferenceEngine::Blob::Ptr inputBlob = nullptr;
    ASSERT_THROW(req.SetBlob(cnnNet.getInputsInfo().begin()->first, inputBlob),
            InferenceEngine::Exception);
}

TEST_P(InferRequestTests, failToSetEmptyInputBlob) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // Create CNNNetwork from ngrpah::Function
    InferenceEngine::CNNNetwork cnnNet(function);
    // Load CNNNetwork to target plugins
    auto execNet = ie->LoadNetwork(cnnNet, targetDevice, configuration);
    // Create InferRequest
    InferenceEngine::InferRequest req;
    ASSERT_NO_THROW(req = execNet.CreateInferRequest());
    InferenceEngine::Blob::Ptr blob;
    ASSERT_THROW(req.SetBlob(cnnNet.getInputsInfo().begin()->first, blob),
            InferenceEngine::Exception);
}

TEST_P(InferRequestTests, failToSetEmptyOutputBlob) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // Create CNNNetwork from ngrpah::Function
    InferenceEngine::CNNNetwork cnnNet(function);
    // Load CNNNetwork to target plugins
    auto execNet = ie->LoadNetwork(cnnNet, targetDevice, configuration);
    // Create InferRequest
    InferenceEngine::InferRequest req;
    ASSERT_NO_THROW(req = execNet.CreateInferRequest());
    InferenceEngine::Blob::Ptr blob;
    ASSERT_THROW(req.SetBlob(cnnNet.getOutputsInfo().begin()->first, blob),
            InferenceEngine::Exception);
}

TEST_P(InferRequestTests, failToSetNotAllocatedInput) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // Create CNNNetwork from ngrpah::Function
    InferenceEngine::CNNNetwork cnnNet(function);
    // Load CNNNetwork to target plugins
    auto execNet = ie->LoadNetwork(cnnNet, targetDevice, configuration);
    // Create InferRequest
    InferenceEngine::InferRequest req;
    ASSERT_NO_THROW(req = execNet.CreateInferRequest());
    InferenceEngine::Blob::Ptr blob =
            FuncTestUtils::createAndFillBlob(cnnNet.getInputsInfo().begin()->second->getTensorDesc());
    ASSERT_NO_THROW(req.SetBlob(cnnNet.getInputsInfo().begin()->first, blob));
}

TEST_P(InferRequestTests, failToSetNotAllocatedOutput) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // Create CNNNetwork from ngrpah::Function
    InferenceEngine::CNNNetwork cnnNet(function);
    // Load CNNNetwork to target plugins
    auto execNet = ie->LoadNetwork(cnnNet, targetDevice, configuration);
    // Create InferRequest
    InferenceEngine::InferRequest req;
    ASSERT_NO_THROW(req = execNet.CreateInferRequest());
    InferenceEngine::Blob::Ptr blob =
            FuncTestUtils::createAndFillBlob(cnnNet.getOutputsInfo().begin()->second->getTensorDesc());
    ASSERT_NO_THROW(req.SetBlob(cnnNet.getOutputsInfo().begin()->first, blob));
}

TEST_P(InferRequestTests, failToSetBlobWithIncorrectName) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // Create CNNNetwork from ngrpah::Function
    InferenceEngine::CNNNetwork cnnNet(function);
    // Load CNNNetwork to target plugins
    auto execNet = ie->LoadNetwork(cnnNet, targetDevice, configuration);
    // Create InferRequest
    InferenceEngine::InferRequest req;
    const char incorrect_input_name[] = "incorrect_input_name";
    ASSERT_NO_THROW(req = execNet.CreateInferRequest());
    InferenceEngine::Blob::Ptr blob =
            FuncTestUtils::createAndFillBlob(cnnNet.getInputsInfo().begin()->second->getTensorDesc());
    blob->allocate();
    ASSERT_THROW(req.SetBlob(incorrect_input_name, blob),
            InferenceEngine::Exception);
}

TEST_P(InferRequestTests, failToSetInputWithIncorrectSizes) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // Create CNNNetwork from ngrpah::Function
    InferenceEngine::CNNNetwork cnnNet(function);
    // Load CNNNetwork to target plugins
    auto execNet = ie->LoadNetwork(cnnNet, targetDevice, configuration);
    // Create InferRequest
    InferenceEngine::InferRequest req;
    ASSERT_NO_THROW(req = execNet.CreateInferRequest());
    InferenceEngine::Blob::Ptr blob =
            FuncTestUtils::createAndFillBlob(cnnNet.getInputsInfo().begin()->second->getTensorDesc());
    blob->allocate();
    blob->getTensorDesc().getDims()[0]*=2;
    ASSERT_THROW(req.SetBlob(cnnNet.getInputsInfo().begin()->first, blob),
            InferenceEngine::Exception);
}

TEST_P(InferRequestTests, failToSetOutputWithIncorrectSizes) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // Create CNNNetwork from ngrpah::Function
    InferenceEngine::CNNNetwork cnnNet(function);
    // Load CNNNetwork to target plugins
    auto execNet = ie->LoadNetwork(cnnNet, targetDevice, configuration);
    // Create InferRequest
    InferenceEngine::InferRequest req;
    ASSERT_NO_THROW(req = execNet.CreateInferRequest());
    InferenceEngine::Blob::Ptr blob =
            FuncTestUtils::createAndFillBlob(cnnNet.getOutputsInfo().begin()->second->getTensorDesc());
    blob->allocate();
    blob->getTensorDesc().getDims()[0]*=2;
    ASSERT_THROW(req.SetBlob(cnnNet.getOutputsInfo().begin()->first, blob),
            InferenceEngine::Exception);
}

TEST_P(InferRequestTests, canInferWithoutSetAndGetInOut) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // Create CNNNetwork from ngrpah::Function
    InferenceEngine::CNNNetwork cnnNet(function);
    // Load CNNNetwork to target plugins
    auto execNet = ie->LoadNetwork(cnnNet, targetDevice, configuration);
    // Create InferRequest
    InferenceEngine::InferRequest req;
    ASSERT_NO_THROW(req = execNet.CreateInferRequest());
    ASSERT_NO_THROW(req.Infer());
}

TEST_P(InferRequestTests, canProcessDeallocatedInputBlobAfterGetBlob) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // Create CNNNetwork from ngrpah::Function
    InferenceEngine::CNNNetwork cnnNet(function);
    // Load CNNNetwork to target plugins
    auto execNet = ie->LoadNetwork(cnnNet, targetDevice, configuration);
    // Create InferRequest
    InferenceEngine::InferRequest req;
    InferenceEngine::Blob::Ptr blob;
    ASSERT_NO_THROW(req = execNet.CreateInferRequest());
    ASSERT_NO_THROW(blob = req.GetBlob(cnnNet.getInputsInfo().begin()->first));
    ASSERT_NO_THROW(req.Infer());
}

TEST_P(InferRequestTests, canProcessDeallocatedInputBlobAfterGetBlobForAsync) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // Create CNNNetwork from ngrpah::Function
    InferenceEngine::CNNNetwork cnnNet(function);
    // Load CNNNetwork to target plugins
    auto execNet = ie->LoadNetwork(cnnNet, targetDevice, configuration);
    // Create InferRequest
    InferenceEngine::InferRequest req;
    InferenceEngine::Blob::Ptr blob;
    ASSERT_NO_THROW(req = execNet.CreateInferRequest());
    ASSERT_NO_THROW(blob = req.GetBlob(cnnNet.getInputsInfo().begin()->first));
    ASSERT_NO_THROW(req.Infer());
    ASSERT_NO_THROW(req.StartAsync());
}

TEST_P(InferRequestTests, canProcessDeallocatedInputBlobAfterGetAndSetBlob) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // Create CNNNetwork from ngrpah::Function
    InferenceEngine::CNNNetwork cnnNet(function);
    // Load CNNNetwork to target plugins
    auto execNet = ie->LoadNetwork(cnnNet, targetDevice, configuration);
    // Create InferRequest
    InferenceEngine::InferRequest req;
    InferenceEngine::Blob::Ptr blob;
    ASSERT_NO_THROW(req = execNet.CreateInferRequest());
    ASSERT_NO_THROW(blob = req.GetBlob(cnnNet.getInputsInfo().begin()->first));
    ASSERT_NO_THROW(req.SetBlob(cnnNet.getInputsInfo().begin()->first, blob));
    ASSERT_NO_THROW(req.Infer());
    ASSERT_NO_THROW(req.StartAsync());
}

TEST_P(InferRequestTests, canProcessDeallocatedInputBlobAfterSetBlob) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // Create CNNNetwork from ngrpah::Function
    InferenceEngine::CNNNetwork cnnNet(function);
    // Load CNNNetwork to target plugins
    auto execNet = ie->LoadNetwork(cnnNet, targetDevice, configuration);
    // Create InferRequest
    InferenceEngine::InferRequest req;
    InferenceEngine::Blob::Ptr blob =
            FuncTestUtils::createAndFillBlob(cnnNet.getInputsInfo().begin()->second->getTensorDesc());
    ASSERT_NO_THROW(req = execNet.CreateInferRequest());
    blob->allocate();
    ASSERT_NO_THROW(req.SetBlob(cnnNet.getInputsInfo().begin()->first, blob));
    blob->deallocate();
    ASSERT_THROW(req.Infer(), InferenceEngine::Exception);
}

TEST_P(InferRequestTests, canProcessDeallocatedOutputBlobAfterGetBlob) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // Create CNNNetwork from ngrpah::Function
    InferenceEngine::CNNNetwork cnnNet(function);
    // Load CNNNetwork to target plugins
    auto execNet = ie->LoadNetwork(cnnNet, targetDevice, configuration);
    // Create InferRequest
    InferenceEngine::InferRequest req;
    InferenceEngine::Blob::Ptr blob =
            FuncTestUtils::createAndFillBlob(cnnNet.getOutputsInfo().begin()->second->getTensorDesc());
    ASSERT_NO_THROW(req = execNet.CreateInferRequest());
    blob->allocate();
    ASSERT_NO_THROW(req.SetBlob(cnnNet.getOutputsInfo().begin()->first, blob));
    blob->deallocate();
    ASSERT_THROW(req.Infer(), InferenceEngine::Exception);
}

TEST_P(InferRequestTests, canProcessDeallocatedOutputBlobAfterGetBlobForAsync) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // Create CNNNetwork from ngrpah::Function
    InferenceEngine::CNNNetwork cnnNet(function);
    // Load CNNNetwork to target plugins
    auto execNet = ie->LoadNetwork(cnnNet, targetDevice, configuration);
    // Create InferRequest
    InferenceEngine::InferRequest req;
    InferenceEngine::Blob::Ptr blob =
            FuncTestUtils::createAndFillBlob(cnnNet.getOutputsInfo().begin()->second->getTensorDesc());
    ASSERT_NO_THROW(req = execNet.CreateInferRequest());
    blob->allocate();
    ASSERT_NO_THROW(req.SetBlob(cnnNet.getOutputsInfo().begin()->first, blob));
    blob->deallocate();
    ASSERT_THROW(req.Infer(), InferenceEngine::Exception);
    ASSERT_THROW({ req.StartAsync(); req.Wait(); }, InferenceEngine::Exception);
}

TEST_P(InferRequestTests, canProcessDeallocatedOutputBlobAfterGetAndSetBlob) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // Create CNNNetwork from ngrpah::Function
    InferenceEngine::CNNNetwork cnnNet(function);
    // Load CNNNetwork to target plugins
    auto execNet = ie->LoadNetwork(cnnNet, targetDevice, configuration);
    // Create InferRequest
    InferenceEngine::InferRequest req;
    InferenceEngine::Blob::Ptr blob =
            FuncTestUtils::createAndFillBlob(cnnNet.getOutputsInfo().begin()->second->getTensorDesc());
    ASSERT_NO_THROW(req = execNet.CreateInferRequest());
    blob->allocate();
    ASSERT_NO_THROW(blob = req.GetBlob(cnnNet.getOutputsInfo().begin()->first));
    ASSERT_NO_THROW(req.SetBlob(cnnNet.getOutputsInfo().begin()->first, blob));
    blob->deallocate();
    ASSERT_THROW(req.Infer(), InferenceEngine::Exception);
}

TEST_P(InferRequestTests, canProcessDeallocatedOutputBlobAfterSetBlob) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // Create CNNNetwork from ngrpah::Function
    InferenceEngine::CNNNetwork cnnNet(function);
    // Load CNNNetwork to target plugins
    auto execNet = ie->LoadNetwork(cnnNet, targetDevice, configuration);
    // Create InferRequest
    InferenceEngine::InferRequest req;
    InferenceEngine::Blob::Ptr blob =
            FuncTestUtils::createAndFillBlob(cnnNet.getOutputsInfo().begin()->second->getTensorDesc());
    ASSERT_NO_THROW(req = execNet.CreateInferRequest());
    blob->allocate();
    ASSERT_NO_THROW(req.SetBlob(cnnNet.getOutputsInfo().begin()->first, blob));
    blob->deallocate();
    ASSERT_THROW(req.Infer(), InferenceEngine::Exception);
}

TEST_P(InferRequestTests, secondCallGetOutputDoNotReAllocateData) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // Create CNNNetwork from ngrpah::Function
    InferenceEngine::CNNNetwork cnnNet(function);
    // Load CNNNetwork to target plugins
    auto execNet = ie->LoadNetwork(cnnNet, targetDevice, configuration);
    // Create InferRequest
    InferenceEngine::InferRequest req;
    InferenceEngine::Blob::Ptr blob1;
    InferenceEngine::Blob::Ptr blob2;
    ASSERT_NO_THROW(req = execNet.CreateInferRequest());
    ASSERT_NO_THROW(blob1 = req.GetBlob(cnnNet.getInputsInfo().begin()->first));
    ASSERT_NO_THROW(blob2 = req.GetBlob(cnnNet.getInputsInfo().begin()->first));
    ASSERT_EQ(blob1.get(), blob2.get());
}

TEST_P(InferRequestTests, CorrectOneAsyncInferWithGetInOutWithInfWait) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // Create CNNNetwork from ngrpah::Function
    InferenceEngine::CNNNetwork cnnNet(function);
    // Load CNNNetwork to target plugins
    auto execNet = ie->LoadNetwork(cnnNet, targetDevice, configuration);
    // Create InferRequest
    InferenceEngine::InferRequest req;
    InferenceEngine::Blob::Ptr blob =
            FuncTestUtils::createAndFillBlob(cnnNet.getOutputsInfo().begin()->second->getTensorDesc());
    ASSERT_NO_THROW(req = execNet.CreateInferRequest());
    ASSERT_NO_THROW(blob = req.GetBlob(cnnNet.getInputsInfo().begin()->first));
    req.Infer();
    req.StartAsync();
    InferenceEngine::StatusCode sts;
    sts = req.Wait(InferenceEngine::InferRequest::WaitMode::RESULT_READY);
    ASSERT_EQ(InferenceEngine::StatusCode::OK, sts);
    ASSERT_NO_THROW(blob = req.GetBlob(cnnNet.getOutputsInfo().begin()->first));
}

// Plugin correct infer request with allocating input and result BlobMaps inside plugin
TEST_P(InferRequestTests, canStartAsyncInferWithGetInOutWithStatusOnlyWait) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // Create CNNNetwork from ngrpah::Function
    InferenceEngine::CNNNetwork cnnNet(function);
    // Load CNNNetwork to target plugins
    auto execNet = ie->LoadNetwork(cnnNet, targetDevice, configuration);
    // Create InferRequest
    InferenceEngine::InferRequest req;
    InferenceEngine::Blob::Ptr blob =
            FuncTestUtils::createAndFillBlob(cnnNet.getOutputsInfo().begin()->second->getTensorDesc());
    ASSERT_NO_THROW(req = execNet.CreateInferRequest());
    ASSERT_NO_THROW(blob = req.GetBlob(cnnNet.getInputsInfo().begin()->first));
    req.Infer();
    req.StartAsync();
    InferenceEngine::StatusCode sts;
    sts = req.Wait(InferenceEngine::InferRequest::WaitMode::STATUS_ONLY);
    ASSERT_TRUE(sts == InferenceEngine::StatusCode::OK ||
        sts == InferenceEngine::StatusCode::RESULT_NOT_READY);
}

// Plugin correct infer request with allocating input and result BlobMaps inside plugin
TEST_P(InferRequestTests, FailedAsyncInferWithNegativeTimeForWait) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // Create CNNNetwork from ngrpah::Function
    InferenceEngine::CNNNetwork cnnNet(function);
    // Load CNNNetwork to target plugins
    auto execNet = ie->LoadNetwork(cnnNet, targetDevice, configuration);
    // Create InferRequest
    InferenceEngine::InferRequest req;
    InferenceEngine::Blob::Ptr blob =
            FuncTestUtils::createAndFillBlob(cnnNet.getOutputsInfo().begin()->second->getTensorDesc());
    ASSERT_NO_THROW(req = execNet.CreateInferRequest());
    ASSERT_NO_THROW(blob = req.GetBlob(cnnNet.getInputsInfo().begin()->first));
    req.Infer();
    req.StartAsync();
    ASSERT_THROW(req.Wait(-2), InferenceEngine::Exception);
}

TEST_P(InferRequestTests, canRun3SyncRequestsConsistentlyFromThreads) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // Create CNNNetwork from ngrpah::Function
    InferenceEngine::CNNNetwork cnnNet(function);
    // Load CNNNetwork to target plugins
    auto execNet = ie->LoadNetwork(cnnNet, targetDevice, configuration);
    // Create InferRequest
    auto req1 = execNet.CreateInferRequest();
    auto req2 = execNet.CreateInferRequest();
    auto req3 = execNet.CreateInferRequest();


    auto f1 = std::async(std::launch::async, [&] { req1.Infer();});
    auto f2 = std::async(std::launch::async, [&] { req2.Infer();});
    auto f3 = std::async(std::launch::async, [&] { req3.Infer();});

    ASSERT_NO_THROW(f1.get());
    ASSERT_NO_THROW(f2.get());
    ASSERT_NO_THROW(f3.get());
}

TEST_P(InferRequestTests, canRun3AsyncRequestsConsistentlyWithWait) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // Create CNNNetwork from ngrpah::Function
    InferenceEngine::CNNNetwork cnnNet(function);
    // Load CNNNetwork to target plugins
    auto execNet = ie->LoadNetwork(cnnNet, targetDevice, configuration);
    // Create InferRequest
    auto req1 = execNet.CreateInferRequest();
    auto req2 = execNet.CreateInferRequest();
    auto req3 = execNet.CreateInferRequest();

    req1.StartAsync();
    ASSERT_NO_THROW(req1.Wait(InferenceEngine::InferRequest::WaitMode::RESULT_READY));

    req2.Infer();
    ASSERT_NO_THROW(req2.Wait(InferenceEngine::InferRequest::WaitMode::RESULT_READY));

    req3.Infer();
    ASSERT_NO_THROW(req3.Wait(InferenceEngine::InferRequest::WaitMode::RESULT_READY));
}

TEST_P(InferRequestTests, canRun3AsyncRequestsConsistentlyFromThreadsWithoutWait) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // Create CNNNetwork from ngrpah::Function
    InferenceEngine::CNNNetwork cnnNet(function);
    // Load CNNNetwork to target plugins
    auto execNet = ie->LoadNetwork(cnnNet, targetDevice, configuration);
    // Create InferRequest
    auto req1 = execNet.CreateInferRequest();
    auto req2 = execNet.CreateInferRequest();
    auto req3 = execNet.CreateInferRequest();
    InferenceEngine::ResponseDesc response1, response2, response3;
    InferenceEngine::StatusCode sts1, sts2, sts3;

    req1.Infer();
    req2.Infer();
    req3.Infer();

    std::thread t1([&] { req1.StartAsync(); sts1 = req1.Wait(InferenceEngine::InferRequest::WaitMode::RESULT_READY); });
    std::thread t2([&] { req2.StartAsync(); sts2 = req2.Wait(InferenceEngine::InferRequest::WaitMode::RESULT_READY); });
    std::thread t3([&] { req3.StartAsync(); sts3 = req3.Wait(InferenceEngine::InferRequest::WaitMode::RESULT_READY); });

    t1.join();
    t2.join();
    t3.join();

    ASSERT_EQ(static_cast<int>(InferenceEngine::StatusCode::OK), sts1) << response1.msg;
    ASSERT_EQ(static_cast<int>(InferenceEngine::StatusCode::OK), sts2) << response2.msg;
    ASSERT_EQ(static_cast<int>(InferenceEngine::StatusCode::OK), sts3) << response3.msg;
}

TEST_P(InferRequestTests, canWaitWithotStartAsync) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // Create CNNNetwork from ngrpah::Function
    InferenceEngine::CNNNetwork cnnNet(function);
    // Load CNNNetwork to target plugins
    auto execNet = ie->LoadNetwork(cnnNet, targetDevice, configuration);
    // Create InferRequest
    auto req = execNet.CreateInferRequest();
    ASSERT_NO_THROW(req.Wait(InferenceEngine::InferRequest::WaitMode::RESULT_READY));
    ASSERT_NO_THROW(req.Wait(InferenceEngine::InferRequest::WaitMode::STATUS_ONLY));
    ASSERT_NO_THROW(req.Wait(1));
}

TEST_P(InferRequestTests, returnDeviceBusyOnSetBlobAfterAsyncInfer) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    auto&& config = configuration;
    auto itConfig = config.find(CONFIG_KEY(CPU_THROUGHPUT_STREAMS));
    if (itConfig != config.end()) {
        if (itConfig->second != "CPU_THROUGHPUT_AUTO") {
            if (std::stoi(itConfig->second) == 0) {
                GTEST_SKIP() << "Not applicable with disabled streams";
            }
        }
    }
    // Create CNNNetwork from ngrpah::Function
    InferenceEngine::CNNNetwork cnnNet(function);
    // Load CNNNetwork to target plugins
    auto execNet = ie->LoadNetwork(cnnNet, targetDevice, configuration);
    // Create InferRequest
    auto req = execNet.CreateInferRequest();
    auto outputBlob = req.GetBlob(cnnNet.getInputsInfo().begin()->first);
    InferenceEngine::ResponseDesc response;

    InferenceEngine::StatusCode sts;
    sts = req.Wait(InferenceEngine::InferRequest::WaitMode::STATUS_ONLY);
    ASSERT_EQ(InferenceEngine::StatusCode::INFER_NOT_STARTED, sts) << response.msg;
    req.StartAsync();
    sts = req.Wait(InferenceEngine::InferRequest::WaitMode::RESULT_READY);
    ASSERT_EQ(static_cast<int>(InferenceEngine::StatusCode::OK), sts) << response.msg;
    try {
        req.SetBlob(cnnNet.getInputsInfo().begin()->first, outputBlob);
    }
    catch (const std::exception &e) {
        std::cout << "Exception: " << e.what() << std::endl;
    }
    sts = req.Wait(InferenceEngine::InferRequest::WaitMode::STATUS_ONLY);
    ASSERT_TRUE(sts == InferenceEngine::StatusCode::OK ||
    sts == InferenceEngine::StatusCode::RESULT_NOT_READY) << response.msg;
}

TEST_P(InferRequestTests, returnDeviceBusyOnGetBlobAfterAsyncInfer) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // Create CNNNetwork from ngrpah::Function
    InferenceEngine::CNNNetwork cnnNet(function);
    // Load CNNNetwork to target plugins
    auto execNet = ie->LoadNetwork(cnnNet, targetDevice, configuration);
    // Create InferRequest
    auto req = execNet.CreateInferRequest();
    auto outputBlob = req.GetBlob(cnnNet.getInputsInfo().begin()->first);
    InferenceEngine::ResponseDesc response;
    InferenceEngine::StatusCode sts;
    req.StartAsync();
    sts = req.Wait(InferenceEngine::InferRequest::WaitMode::RESULT_READY);
    ASSERT_EQ(static_cast<int>(InferenceEngine::StatusCode::OK), sts) << response.msg;
    try {
        req.SetBlob(cnnNet.getInputsInfo().begin()->first, outputBlob);
    }
    catch (const std::exception &e) {
        std::cout << "Exception" << e.what() << std::endl;
    }
}

TEST_P(InferRequestTests, returnDeviceBusyOnGetPerformanceCountAfterAsyncInfer) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // Create CNNNetwork from ngrpah::Function
    InferenceEngine::CNNNetwork cnnNet(function);
    // Load CNNNetwork to target plugins
    auto execNet = ie->LoadNetwork(cnnNet, targetDevice, configuration);
    // Create InferRequest
    auto req = execNet.CreateInferRequest();
    auto outputBlob = req.GetBlob(cnnNet.getInputsInfo().begin()->first);
    InferenceEngine::ResponseDesc response;
    InferenceEngine::StatusCode sts;
    req.StartAsync();
    sts = req.Wait(InferenceEngine::InferRequest::WaitMode::RESULT_READY);
    ASSERT_EQ(static_cast<int>(InferenceEngine::StatusCode::OK), sts) << response.msg;

    std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> perfMap;

    try {
        perfMap = req.GetPerformanceCounts();
    }
    catch (const std::exception &e) {
        std::cout << "Exception" << e.what() << std::endl;
    }
}

class InferRequestTestsResultNotReady : public InferRequestTests {
};

TEST_P(InferRequestTestsResultNotReady, ReturnResultNotReadyFromWaitInAsyncModeForTooSmallTimeout) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // Create CNNNetwork from ngraph::Function
    // return ngrpah::Function
    // GetNetwork(3000, 380) make inference around 20ms on GNA SW
    // so increases chances for getting RESULT_NOT_READY
    function = SubgraphTestsDefinitions::Basic_LSTM_S::GetNetwork(300, 38);
    InferenceEngine::CNNNetwork cnnNet(function);
    // Load CNNNetwork to target plugins
    auto execNet = ie->LoadNetwork(cnnNet, targetDevice, configuration);
    // Create InferRequest
    InferenceEngine::InferRequest req;
    ASSERT_NO_THROW(req = execNet.CreateInferRequest());
    InferenceEngine::StatusCode sts = InferenceEngine::StatusCode::OK;
    std::promise<std::chrono::system_clock::time_point> callbackTimeStamp;
    auto callbackTimeStampFuture = callbackTimeStamp.get_future();
    // add a callback to the request and capture the timestamp
    req.SetCompletionCallback([&]() {
        callbackTimeStamp.set_value(std::chrono::system_clock::now());
        });
    req.StartAsync();
    ASSERT_NO_THROW(sts = req.Wait(InferenceEngine::InferRequest::WaitMode::STATUS_ONLY));
    // get timestamp taken AFTER return from the Wait(STATUS_ONLY)
    const auto afterWaitTimeStamp = std::chrono::system_clock::now();
    // IF the callback timestamp is larger than the afterWaitTimeStamp
    // then we should observe RESULT_NOT_READY
    if (afterWaitTimeStamp < callbackTimeStampFuture.get()) {
        ASSERT_TRUE(sts == InferenceEngine::StatusCode::RESULT_NOT_READY);
    }
    ASSERT_NO_THROW(req.Wait(InferenceEngine::InferRequest::WaitMode::RESULT_READY));
}
}  // namespace BehaviorTestsDefinitions
