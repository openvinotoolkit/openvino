// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <thread>
#include <future>

#include "base/behavior_test_utils.hpp"
#include "shared_test_classes/subgraph/basic_lstm.hpp"

namespace BehaviorTestsDefinitions {
using namespace BehaviorTestsUtils;

TEST_P(InferRequestTests, CanCreateInferRequest) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // Create InferRequest
    InferenceEngine::InferRequest req;
    ASSERT_NO_THROW(req = execNet.CreateInferRequest());
}

TEST_P(InferRequestTests, failToSetNullptrForInput) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // Create InferRequest
    InferenceEngine::InferRequest req;
    ASSERT_NO_THROW(req = execNet.CreateInferRequest());
    InferenceEngine::Blob::Ptr inputBlob = nullptr;
    ASSERT_THROW(req.SetBlob(cnnNet.getInputsInfo().begin()->first, inputBlob),
            InferenceEngine::Exception);
}

TEST_P(InferRequestTests, failToSetNullptrForOutput) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // Create InferRequest
    InferenceEngine::InferRequest req;
    ASSERT_NO_THROW(req = execNet.CreateInferRequest());
    InferenceEngine::Blob::Ptr outputBlob = nullptr;
    ASSERT_THROW(req.SetBlob(cnnNet.getOutputsInfo().begin()->first, outputBlob),
                 InferenceEngine::Exception);
}

TEST_P(InferRequestTests, failToSetUninitializedInputBlob) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // Create InferRequest
    InferenceEngine::InferRequest req;
    ASSERT_NO_THROW(req = execNet.CreateInferRequest());
    InferenceEngine::Blob::Ptr blob;
    ASSERT_THROW(req.SetBlob(cnnNet.getInputsInfo().begin()->first, blob),
            InferenceEngine::Exception);
}

TEST_P(InferRequestTests, failToSetUninitializedOutputBlob) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // Create InferRequest
    InferenceEngine::InferRequest req;
    ASSERT_NO_THROW(req = execNet.CreateInferRequest());
    InferenceEngine::Blob::Ptr blob;
    ASSERT_THROW(req.SetBlob(cnnNet.getOutputsInfo().begin()->first, blob),
            InferenceEngine::Exception);
}

TEST_P(InferRequestTests, setNotAllocatedInput) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // Create InferRequest
    InferenceEngine::InferRequest req;
    ASSERT_NO_THROW(req = execNet.CreateInferRequest());
    InferenceEngine::Blob::Ptr blob =
            FuncTestUtils::createAndFillBlob(cnnNet.getInputsInfo().begin()->second->getTensorDesc());
    ASSERT_NO_THROW(req.SetBlob(cnnNet.getInputsInfo().begin()->first, blob));
}

TEST_P(InferRequestTests, setNotAllocatedOutput) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
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
    // Create InferRequest
    InferenceEngine::InferRequest req;
    ASSERT_NO_THROW(req = execNet.CreateInferRequest());
    InferenceEngine::Blob::Ptr blob =
            FuncTestUtils::createAndFillBlob(cnnNet.getInputsInfo().begin()->second->getTensorDesc());
    blob->allocate();
    blob->getTensorDesc().getDims()[0] *= 2;
    ASSERT_THROW(req.SetBlob(cnnNet.getInputsInfo().begin()->first, blob),
            InferenceEngine::Exception);
}

TEST_P(InferRequestTests, failToSetOutputWithIncorrectSizes) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // Create InferRequest
    InferenceEngine::InferRequest req;
    ASSERT_NO_THROW(req = execNet.CreateInferRequest());
    InferenceEngine::Blob::Ptr blob =
            FuncTestUtils::createAndFillBlob(cnnNet.getOutputsInfo().begin()->second->getTensorDesc());
    blob->allocate();
    blob->getTensorDesc().getDims()[0] *= 2;
    ASSERT_THROW(req.SetBlob(cnnNet.getOutputsInfo().begin()->first, blob),
            InferenceEngine::Exception);
}

// + SetInout, + SetOutput + Set In/Out (+ Async)
TEST_P(InferRequestTests, canInferWithoutSetAndGetInOut) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // Create InferRequest
    InferenceEngine::InferRequest req;
    ASSERT_NO_THROW(req = execNet.CreateInferRequest());
    ASSERT_NO_THROW(req.Infer());
}

TEST_P(InferRequestTests, canProcessDeallocatedInputBlobAfterGetBlob) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
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
    // Create InferRequest
    InferenceEngine::InferRequest req;
    InferenceEngine::Blob::Ptr blob;
    ASSERT_NO_THROW(req = execNet.CreateInferRequest());
    ASSERT_NO_THROW(blob = req.GetBlob(cnnNet.getInputsInfo().begin()->first));
    ASSERT_NO_THROW(req.SetBlob(cnnNet.getInputsInfo().begin()->first, blob));
    ASSERT_NO_THROW(req.Infer());
    ASSERT_NO_THROW(req.StartAsync());
}

TEST_P(InferRequestTests, canProcessDeallocatedInputBlobAfterSetBlobSync) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
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

TEST_P(InferRequestTests, canProcessDeallocatedInputBlobAfterSetBlobAsync) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // Create InferRequest
    InferenceEngine::InferRequest req;
    InferenceEngine::Blob::Ptr blob =
            FuncTestUtils::createAndFillBlob(cnnNet.getInputsInfo().begin()->second->getTensorDesc());
    ASSERT_NO_THROW(req = execNet.CreateInferRequest());
    blob->allocate();
    ASSERT_NO_THROW(req.SetBlob(cnnNet.getInputsInfo().begin()->first, blob));
    blob->deallocate();
    ASSERT_THROW({ req.StartAsync(); req.Wait(); }, InferenceEngine::Exception);
}

TEST_P(InferRequestTests, canProcessDeallocatedOutputBlobAfterGetBlob) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
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
    ASSERT_THROW({ req.StartAsync(); req.Wait(); }, InferenceEngine::Exception);
}

TEST_P(InferRequestTests, canProcessDeallocatedOutputBlobAfterSetBlob) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
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

TEST_P(InferRequestTests, secondCallGetInputDoNotReAllocateData) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // Create InferRequest
    InferenceEngine::InferRequest req;
    InferenceEngine::Blob::Ptr blob1;
    InferenceEngine::Blob::Ptr blob2;
    ASSERT_NO_THROW(req = execNet.CreateInferRequest());
    ASSERT_NO_THROW(blob1 = req.GetBlob(cnnNet.getInputsInfo().begin()->first));
    ASSERT_NO_THROW(blob2 = req.GetBlob(cnnNet.getInputsInfo().begin()->first));
    ASSERT_EQ(blob1.get(), blob2.get());
}

TEST_P(InferRequestTests, secondCallGetOutputDoNotReAllocateData) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // Create InferRequest
    InferenceEngine::InferRequest req;
    InferenceEngine::Blob::Ptr blob1;
    InferenceEngine::Blob::Ptr blob2;
    ASSERT_NO_THROW(req = execNet.CreateInferRequest());
    ASSERT_NO_THROW(blob1 = req.GetBlob(cnnNet.getOutputsInfo().begin()->first));
    ASSERT_NO_THROW(blob2 = req.GetBlob(cnnNet.getOutputsInfo().begin()->first));
    ASSERT_EQ(blob1.get(), blob2.get());
}

TEST_P(InferRequestTests, secondCallGetOutputAfterInferSync) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // Create InferRequest
    InferenceEngine::InferRequest req;
    InferenceEngine::Blob::Ptr inBlob = FuncTestUtils::createAndFillBlob(cnnNet.getInputsInfo().begin()->second->getTensorDesc());
    InferenceEngine::Blob::Ptr blob1;
    InferenceEngine::Blob::Ptr blob2;
    ASSERT_NO_THROW(req = execNet.CreateInferRequest());
    ASSERT_NO_THROW(req.SetBlob(cnnNet.getInputsInfo().begin()->first, inBlob));
    ASSERT_NO_THROW(req.Infer());
    ASSERT_NO_THROW(blob1 = req.GetBlob(cnnNet.getOutputsInfo().begin()->first));
    ASSERT_NO_THROW(blob2 = req.GetBlob(cnnNet.getOutputsInfo().begin()->first));
    ASSERT_EQ(blob1.get(), blob2.get());
}

TEST_P(InferRequestTests, secondCallGetOutputAfterInferAsync) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // Create InferRequest
    InferenceEngine::InferRequest req;
    InferenceEngine::Blob::Ptr inBlob = FuncTestUtils::createAndFillBlob(cnnNet.getInputsInfo().begin()->second->getTensorDesc());
    InferenceEngine::Blob::Ptr blob1;
    InferenceEngine::Blob::Ptr blob2;
    ASSERT_NO_THROW(req = execNet.CreateInferRequest());
    ASSERT_NO_THROW(req.SetBlob(cnnNet.getInputsInfo().begin()->first, inBlob));
    ASSERT_NO_THROW({ req.StartAsync(); req.Wait(); });
    ASSERT_NO_THROW(blob1 = req.GetBlob(cnnNet.getOutputsInfo().begin()->first));
    ASSERT_NO_THROW(blob2 = req.GetBlob(cnnNet.getOutputsInfo().begin()->first));
    ASSERT_EQ(blob1.get(), blob2.get());
}

TEST_P(InferRequestTests, CorrectOneAsyncInferWithGetInOutWithInfWait) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // Create InferRequest
    InferenceEngine::InferRequest req;
    InferenceEngine::Blob::Ptr blob =
            FuncTestUtils::createAndFillBlob(cnnNet.getOutputsInfo().begin()->second->getTensorDesc());
    ASSERT_NO_THROW(req = execNet.CreateInferRequest());
    ASSERT_NO_THROW(blob = req.GetBlob(cnnNet.getInputsInfo().begin()->first));
    // Infer + InferAsync?
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
    // Create InferRequest
    InferenceEngine::InferRequest req;
    InferenceEngine::Blob::Ptr blob =
            FuncTestUtils::createAndFillBlob(cnnNet.getOutputsInfo().begin()->second->getTensorDesc());
    ASSERT_NO_THROW(req = execNet.CreateInferRequest());
    ASSERT_NO_THROW(blob = req.GetBlob(cnnNet.getInputsInfo().begin()->first));
    // iefode: Infer + InferAsync?
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
    // Create InferRequest
    InferenceEngine::InferRequest req;
    InferenceEngine::Blob::Ptr blob =
            FuncTestUtils::createAndFillBlob(cnnNet.getOutputsInfo().begin()->second->getTensorDesc());
    ASSERT_NO_THROW(req = execNet.CreateInferRequest());
    ASSERT_NO_THROW(blob = req.GetBlob(cnnNet.getInputsInfo().begin()->first));
    // iefode: Infer + InferAsync?
    req.Infer();
    req.StartAsync();
    ASSERT_THROW(req.Wait(-2), InferenceEngine::Exception);
}

TEST_P(InferRequestTests, canRun3SyncRequestsConsistentlyFromThreads) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
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

TEST_P(InferRequestTests, canRun3AsyncRequestsConsistentlyFromThreadsWithoutWait) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // Create InferRequest
    auto req1 = execNet.CreateInferRequest();
    auto req2 = execNet.CreateInferRequest();
    auto req3 = execNet.CreateInferRequest();
    InferenceEngine::StatusCode sts1, sts2, sts3;

    // iefode: Infer???
    req1.Infer();
    req2.Infer();
    req3.Infer();

    std::thread t1([&] { req1.StartAsync(); sts1 = req1.Wait(InferenceEngine::InferRequest::WaitMode::RESULT_READY); });
    std::thread t2([&] { req2.StartAsync(); sts2 = req2.Wait(InferenceEngine::InferRequest::WaitMode::RESULT_READY); });
    std::thread t3([&] { req3.StartAsync(); sts3 = req3.Wait(InferenceEngine::InferRequest::WaitMode::RESULT_READY); });

    t1.join();
    t2.join();
    t3.join();

    ASSERT_EQ(InferenceEngine::StatusCode::OK, sts1);
    ASSERT_EQ(InferenceEngine::StatusCode::OK, sts2);
    ASSERT_EQ(InferenceEngine::StatusCode::OK, sts3);
}

TEST_P(InferRequestTests, canRun3AsyncRequestsConsistentlyWithWait) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
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

TEST_P(InferRequestTests, canWaitWithotStartAsync) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // Create InferRequest
    auto req = execNet.CreateInferRequest();
    ASSERT_NO_THROW(req.Wait(InferenceEngine::InferRequest::WaitMode::RESULT_READY));
    ASSERT_NO_THROW(req.Wait(InferenceEngine::InferRequest::WaitMode::STATUS_ONLY));
    ASSERT_NO_THROW(req.Wait(1));
}

// iefode: what is the idea? I didn't catch it
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

    // Create InferRequest
    auto req = execNet.CreateInferRequest();
    auto outputBlob = req.GetBlob(cnnNet.getInputsInfo().begin()->first);
    InferenceEngine::StatusCode sts;
    sts = req.Wait(InferenceEngine::InferRequest::WaitMode::STATUS_ONLY);
    ASSERT_EQ(InferenceEngine::StatusCode::INFER_NOT_STARTED, sts);
    req.StartAsync();
    sts = req.Wait(InferenceEngine::InferRequest::WaitMode::RESULT_READY);
    ASSERT_EQ(InferenceEngine::StatusCode::OK, sts);
    try {
        req.SetBlob(cnnNet.getInputsInfo().begin()->first, outputBlob);
    } catch (const std::exception &e) {
        std::cout << "Exception: " << e.what() << std::endl;
    }
    sts = req.Wait(InferenceEngine::InferRequest::WaitMode::STATUS_ONLY);
    ASSERT_TRUE(sts == InferenceEngine::StatusCode::OK || sts == InferenceEngine::StatusCode::RESULT_NOT_READY);
}

TEST_P(InferRequestTests, returnDeviceBusyOnGetBlobAfterAsyncInfer) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // Create InferRequest
    auto req = execNet.CreateInferRequest();
    auto outputBlob = req.GetBlob(cnnNet.getInputsInfo().begin()->first);
    InferenceEngine::StatusCode sts;
    req.StartAsync();
    sts = req.Wait(InferenceEngine::InferRequest::WaitMode::RESULT_READY);
    ASSERT_EQ(InferenceEngine::StatusCode::OK, sts);
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
    // Create InferRequest
    auto req = execNet.CreateInferRequest();
    auto outputBlob = req.GetBlob(cnnNet.getInputsInfo().begin()->first);
    InferenceEngine::StatusCode sts;
    req.StartAsync();
    sts = req.Wait(InferenceEngine::InferRequest::WaitMode::RESULT_READY);
    ASSERT_EQ(InferenceEngine::StatusCode::OK, sts);

    std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> perfMap;

    try {
        perfMap = req.GetPerformanceCounts();
    }
    catch (const std::exception &e) {
        std::cout << "Exception" << e.what() << std::endl;
    }
}
}  // namespace BehaviorTestsDefinitions
