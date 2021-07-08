// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <thread>
#include <future>

#include "base/behavior_test_utils.hpp"
#include "shared_test_classes/subgraph/basic_lstm.hpp"

namespace BehaviorTestsDefinitions {
using InferRequestIOBBlobTest = BehaviorTestsUtils::InferRequestTests;

TEST_P(InferRequestIOBBlobTest, CanCreateInferRequest) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // Create InferRequest
    InferenceEngine::InferRequest req;
    ASSERT_NO_THROW(req = execNet.CreateInferRequest());
}

TEST_P(InferRequestIOBBlobTest, failToSetNullptrForInput) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // Create InferRequest
    InferenceEngine::InferRequest req;
    ASSERT_NO_THROW(req = execNet.CreateInferRequest());
    InferenceEngine::Blob::Ptr inputBlob = nullptr;
    ASSERT_THROW(req.SetBlob(cnnNet.getInputsInfo().begin()->first, inputBlob),
            InferenceEngine::Exception);
}

TEST_P(InferRequestIOBBlobTest, failToSetNullptrForOutput) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // Create InferRequest
    InferenceEngine::InferRequest req;
    ASSERT_NO_THROW(req = execNet.CreateInferRequest());
    InferenceEngine::Blob::Ptr outputBlob = nullptr;
    ASSERT_THROW(req.SetBlob(cnnNet.getOutputsInfo().begin()->first, outputBlob),
                 InferenceEngine::Exception);
}

TEST_P(InferRequestIOBBlobTest, failToSetUninitializedInputBlob) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // Create InferRequest
    InferenceEngine::InferRequest req;
    ASSERT_NO_THROW(req = execNet.CreateInferRequest());
    InferenceEngine::Blob::Ptr blob;
    ASSERT_THROW(req.SetBlob(cnnNet.getInputsInfo().begin()->first, blob),
            InferenceEngine::Exception);
}

TEST_P(InferRequestIOBBlobTest, failToSetUninitializedOutputBlob) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // Create InferRequest
    InferenceEngine::InferRequest req;
    ASSERT_NO_THROW(req = execNet.CreateInferRequest());
    InferenceEngine::Blob::Ptr blob;
    ASSERT_THROW(req.SetBlob(cnnNet.getOutputsInfo().begin()->first, blob),
            InferenceEngine::Exception);
}

TEST_P(InferRequestIOBBlobTest, setNotAllocatedInput) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // Create InferRequest
    InferenceEngine::InferRequest req;
    ASSERT_NO_THROW(req = execNet.CreateInferRequest());
    InferenceEngine::Blob::Ptr blob =
            FuncTestUtils::createAndFillBlob(cnnNet.getInputsInfo().begin()->second->getTensorDesc());
    ASSERT_NO_THROW(req.SetBlob(cnnNet.getInputsInfo().begin()->first, blob));
}

TEST_P(InferRequestIOBBlobTest, setNotAllocatedOutput) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // Create InferRequest
    InferenceEngine::InferRequest req;
    ASSERT_NO_THROW(req = execNet.CreateInferRequest());
    InferenceEngine::Blob::Ptr blob =
            FuncTestUtils::createAndFillBlob(cnnNet.getOutputsInfo().begin()->second->getTensorDesc());
    ASSERT_NO_THROW(req.SetBlob(cnnNet.getOutputsInfo().begin()->first, blob));
}

TEST_P(InferRequestIOBBlobTest, getAfterSetInputDoNotChangeInput) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // Create InferRequest
    InferenceEngine::InferRequest req = execNet.CreateInferRequest();
    std::shared_ptr<InferenceEngine::Blob> inputBlob = FuncTestUtils::createAndFillBlob(
            cnnNet.getInputsInfo().begin()->second->getTensorDesc());
    ASSERT_NO_THROW(req.SetBlob(cnnNet.getInputsInfo().begin()->first, inputBlob));
    std::shared_ptr<InferenceEngine::Blob> actualBlob = nullptr;
    ASSERT_NO_THROW(actualBlob = req.GetBlob(cnnNet.getInputsInfo().begin()->first));

    ASSERT_TRUE(actualBlob);
    ASSERT_FALSE(actualBlob->buffer() == nullptr);
    ASSERT_EQ(inputBlob.get(), actualBlob.get());

    ASSERT_TRUE(cnnNet.getInputsInfo().begin()->second->getTensorDesc() == actualBlob->getTensorDesc());
}

TEST_P(InferRequestIOBBlobTest, getAfterSetInputDoNotChangeOutput) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // Create InferRequest
    InferenceEngine::InferRequest req = execNet.CreateInferRequest();
    std::shared_ptr<InferenceEngine::Blob> inputBlob = FuncTestUtils::createAndFillBlob(
            cnnNet.getOutputsInfo().begin()->second->getTensorDesc());
    ASSERT_NO_THROW(req.SetBlob(cnnNet.getOutputsInfo().begin()->first, inputBlob));
    std::shared_ptr<InferenceEngine::Blob> actualBlob;
    ASSERT_NO_THROW(actualBlob = req.GetBlob(cnnNet.getOutputsInfo().begin()->first));
    ASSERT_EQ(inputBlob.get(), actualBlob.get());

    ASSERT_TRUE(actualBlob);
    ASSERT_FALSE(actualBlob->buffer() == nullptr);
    ASSERT_EQ(inputBlob.get(), actualBlob.get());

    ASSERT_TRUE(cnnNet.getOutputsInfo().begin()->second->getTensorDesc() == actualBlob->getTensorDesc());
}

TEST_P(InferRequestIOBBlobTest, failToSetBlobWithIncorrectName) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // Create InferRequest
    InferenceEngine::InferRequest req;
    const char incorrect_input_name[] = "incorrect_input_name";
    ASSERT_NO_THROW(req = execNet.CreateInferRequest());
    InferenceEngine::Blob::Ptr blob =
            FuncTestUtils::createAndFillBlob(cnnNet.getInputsInfo().begin()->second->getTensorDesc());
    blob->allocate();
    ASSERT_THROW(req.SetBlob(incorrect_input_name, blob), InferenceEngine::Exception);
}

TEST_P(InferRequestIOBBlobTest, failToSetInputWithIncorrectSizes) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // Create InferRequest
    InferenceEngine::InferRequest req;
    ASSERT_NO_THROW(req = execNet.CreateInferRequest());
    InferenceEngine::Blob::Ptr blob =
            FuncTestUtils::createAndFillBlob(cnnNet.getInputsInfo().begin()->second->getTensorDesc());
    blob->allocate();
    blob->getTensorDesc().getDims()[0] *= 2;
    ASSERT_THROW(req.SetBlob(cnnNet.getInputsInfo().begin()->first, blob), InferenceEngine::Exception);
}

TEST_P(InferRequestIOBBlobTest, failToSetOutputWithIncorrectSizes) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // Create InferRequest
    InferenceEngine::InferRequest req;
    ASSERT_NO_THROW(req = execNet.CreateInferRequest());
    InferenceEngine::Blob::Ptr blob =
            FuncTestUtils::createAndFillBlob(cnnNet.getOutputsInfo().begin()->second->getTensorDesc());
    blob->allocate();
    blob->getTensorDesc().getDims()[0] *= 2;
    ASSERT_THROW(req.SetBlob(cnnNet.getOutputsInfo().begin()->first, blob), InferenceEngine::Exception);
}

TEST_P(InferRequestIOBBlobTest, canInferWithoutSetAndGetInOutSync) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // Create InferRequest
    InferenceEngine::InferRequest req;
    ASSERT_NO_THROW(req = execNet.CreateInferRequest());
    ASSERT_NO_THROW(req.Infer());
}

TEST_P(InferRequestIOBBlobTest, canInferWithoutSetAndGetInOutAsync) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // Create InferRequest
    InferenceEngine::InferRequest req;
    ASSERT_NO_THROW(req = execNet.CreateInferRequest());
    ASSERT_NO_THROW(req.StartAsync());
}

TEST_P(InferRequestIOBBlobTest, canProcessDeallocatedInputBlobAfterGetBlob) {
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

TEST_P(InferRequestIOBBlobTest, canProcessDeallocatedInputBlobAfterGetAndSetBlob) {
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

TEST_P(InferRequestIOBBlobTest, canProcessDeallocatedInputBlobAfterSetBlobSync) {
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

TEST_P(InferRequestIOBBlobTest, canProcessDeallocatedInputBlobAfterSetBlobAsync) {
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

TEST_P(InferRequestIOBBlobTest, canProcessDeallocatedOutputBlobAfterSetBlob) {
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

TEST_P(InferRequestIOBBlobTest, canProcessDeallocatedOutputBlobAfterGetAndSetBlob) {
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

TEST_P(InferRequestIOBBlobTest, secondCallGetInputDoNotReAllocateData) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // Create InferRequest
    InferenceEngine::InferRequest req;
    InferenceEngine::Blob::Ptr blob1, blob2;
    ASSERT_NO_THROW(req = execNet.CreateInferRequest());
    ASSERT_NO_THROW(blob1 = req.GetBlob(cnnNet.getInputsInfo().begin()->first));
    ASSERT_NO_THROW(blob2 = req.GetBlob(cnnNet.getInputsInfo().begin()->first));
    ASSERT_EQ(blob1.get(), blob2.get());
}

TEST_P(InferRequestIOBBlobTest, secondCallGetOutputDoNotReAllocateData) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // Create InferRequest
    InferenceEngine::InferRequest req;
    InferenceEngine::Blob::Ptr blob1, blob2;
    ASSERT_NO_THROW(req = execNet.CreateInferRequest());
    ASSERT_NO_THROW(blob1 = req.GetBlob(cnnNet.getOutputsInfo().begin()->first));
    ASSERT_NO_THROW(blob2 = req.GetBlob(cnnNet.getOutputsInfo().begin()->first));
    ASSERT_EQ(blob1.get(), blob2.get());
}

TEST_P(InferRequestIOBBlobTest, secondCallGetInputAfterInferSync) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // Create InferRequest
    InferenceEngine::InferRequest req;
    InferenceEngine::Blob::Ptr blob1, blob2;
    ASSERT_NO_THROW(req = execNet.CreateInferRequest());
    ASSERT_NO_THROW(req.Infer());
    ASSERT_NO_THROW({ req.StartAsync(); req.Wait(); });
    ASSERT_NO_THROW(blob1 = req.GetBlob(cnnNet.getInputsInfo().begin()->first));
    ASSERT_NO_THROW(blob2 = req.GetBlob(cnnNet.getInputsInfo().begin()->first));
    ASSERT_EQ(blob1.get(), blob2.get());
}

TEST_P(InferRequestIOBBlobTest, secondCallGetOutputAfterInferSync) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // Create InferRequest
    InferenceEngine::InferRequest req;
    InferenceEngine::Blob::Ptr blob1, blob2;
    ASSERT_NO_THROW(req = execNet.CreateInferRequest());
    ASSERT_NO_THROW(req.Infer());
    ASSERT_NO_THROW({ req.StartAsync(); req.Wait(); });
    ASSERT_NO_THROW(blob1 = req.GetBlob(cnnNet.getOutputsInfo().begin()->first));
    ASSERT_NO_THROW(blob2 = req.GetBlob(cnnNet.getOutputsInfo().begin()->first));
    ASSERT_EQ(blob1.get(), blob2.get());
}

TEST_P(InferRequestIOBBlobTest, canSetInputBlobForInferRequest) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // Create InferRequest
    InferenceEngine::InferRequest req;
    ASSERT_NO_THROW(req = execNet.CreateInferRequest());
    InferenceEngine::Blob::Ptr inputBlob =
            FuncTestUtils::createAndFillBlob(cnnNet.getInputsInfo().begin()->second->getTensorDesc());
    ASSERT_NO_THROW(req.SetBlob(cnnNet.getInputsInfo().begin()->first, inputBlob));
    InferenceEngine::Blob::Ptr actualBlob;
    ASSERT_NO_THROW(actualBlob = req.GetBlob(cnnNet.getInputsInfo().begin()->first));
    ASSERT_EQ(inputBlob, actualBlob);
}

TEST_P(InferRequestIOBBlobTest, canSetOutputBlobForInferRequest) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // Create InferRequest
    InferenceEngine::InferRequest req = execNet.CreateInferRequest();
    std::shared_ptr<InferenceEngine::Blob> outputBlob = FuncTestUtils::createAndFillBlob(
            cnnNet.getOutputsInfo().begin()->second->getTensorDesc());
    ASSERT_NO_THROW(req.SetBlob(cnnNet.getOutputsInfo().begin()->first, outputBlob));
    std::shared_ptr<InferenceEngine::Blob> actualBlob;
    ASSERT_NO_THROW(actualBlob = req.GetBlob(cnnNet.getOutputsInfo().begin()->first));
    ASSERT_EQ(outputBlob.get(), actualBlob.get());
}

TEST_P(InferRequestIOBBlobTest, canInferWithSetInOutBlobs) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // Create InferRequest
    InferenceEngine::InferRequest req;
    ASSERT_NO_THROW(req = execNet.CreateInferRequest());
    InferenceEngine::Blob::Ptr inputBlob =
            FuncTestUtils::createAndFillBlob(cnnNet.getInputsInfo().begin()->second->getTensorDesc());
    req.SetBlob(cnnNet.getInputsInfo().begin()->first, inputBlob);
    InferenceEngine::Blob::Ptr outputBlob =
            FuncTestUtils::createAndFillBlob(cnnNet.getInputsInfo().begin()->second->getTensorDesc());
    req.SetBlob(cnnNet.getInputsInfo().begin()->first, outputBlob);
    ASSERT_NO_THROW(req.Infer());
}

TEST_P(InferRequestIOBBlobTest, canInferWithGetIn) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // Create InferRequest
    InferenceEngine::InferRequest req;
    ASSERT_NO_THROW(req = execNet.CreateInferRequest());
    InferenceEngine::Blob::Ptr inputBlob = req.GetBlob(cnnNet.getInputsInfo().begin()->first);
    ASSERT_NO_THROW(req.Infer());
    InferenceEngine::StatusCode sts;
    ASSERT_NO_THROW({ req.StartAsync(); sts = req.Wait(InferenceEngine::InferRequest::WaitMode::RESULT_READY); });
    ASSERT_EQ(InferenceEngine::StatusCode::OK, sts);
    ASSERT_NO_THROW(InferenceEngine::Blob::Ptr outputBlob = req.GetBlob(cnnNet.getOutputsInfo().begin()->first));
}

TEST_P(InferRequestIOBBlobTest, canInferWithGetOut) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // Create InferRequest
    InferenceEngine::InferRequest req;
    ASSERT_NO_THROW(req = execNet.CreateInferRequest());
    InferenceEngine::Blob::Ptr inputBlob = req.GetBlob(cnnNet.getOutputsInfo().begin()->first);
    ASSERT_NO_THROW(req.Infer());
    InferenceEngine::StatusCode sts;
    ASSERT_NO_THROW({ req.StartAsync(); sts = req.Wait(InferenceEngine::InferRequest::WaitMode::RESULT_READY); });
    ASSERT_EQ(InferenceEngine::StatusCode::OK, sts);
    ASSERT_NO_THROW(InferenceEngine::Blob::Ptr outputBlob = req.GetBlob(cnnNet.getOutputsInfo().begin()->first));
}

// Set Blob with incorrect precision, with layout and so on

}  // namespace BehaviorTestsDefinitions
