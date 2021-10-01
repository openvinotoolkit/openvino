// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <vector>
#include <string>
#include <memory>
#include "ie_extension.h"
#include <condition_variable>
#include "shared_test_classes/base/layer_test_utils.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"
#include "ngraph_functions/builders.hpp"
#include <ie_core.hpp>
#include <base/behavior_test_utils.hpp>
#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/plugin_cache.hpp"
#include "functional_test_utils/blob_utils.hpp"
#include "ngraph_functions/subgraph_builders.hpp"
#include "behavior/infer_request_input.hpp"

namespace BehaviorTestsDefinitions {
using InferRequestInputTests = BehaviorTestsUtils::BehaviorTestsBasic;

TEST_P(InferRequestInputTests, canSetInputBlobForSyncRequest) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // Create CNNNetwork from ngrpah::Function
    InferenceEngine::CNNNetwork cnnNet(function);
    // Load CNNNetwork to target plugins
    auto execNet = ie->LoadNetwork(cnnNet, targetDevice, configuration);
    // Create InferRequest
    InferenceEngine::InferRequest req;
    ASSERT_NO_THROW(req = execNet.CreateInferRequest());

    // Collect input blob info
    const auto  inputsInfo     = cnnNet.getInputsInfo();
    const auto& blobName       = inputsInfo.begin()->first;
    const auto& blobTensorDesc = inputsInfo.begin()->second->getTensorDesc();
    // Set input blob
    InferenceEngine::Blob::Ptr inputBlob = FuncTestUtils::createAndFillBlob(blobTensorDesc);
    ASSERT_NO_THROW(req.SetBlob(blobName, inputBlob));

    // Get input blob
    InferenceEngine::Blob::Ptr actualBlob;
    ASSERT_NO_THROW(actualBlob = req.GetBlob(blobName));

    // Compare the blobs
    ASSERT_EQ(inputBlob, actualBlob);
}

TEST_P(InferRequestInputTests, canInferWithSetInOut) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // Create CNNNetwork from ngrpah::Function
    InferenceEngine::CNNNetwork cnnNet(function);
    // Load CNNNetwork to target plugins
    auto execNet = ie->LoadNetwork(cnnNet, targetDevice, configuration);
    // Create InferRequest
    InferenceEngine::InferRequest req;
    ASSERT_NO_THROW(req = execNet.CreateInferRequest());

    // Collect input blob info
    const auto  inputsInfo          = cnnNet.getInputsInfo();
    const auto& inputBlobName       = inputsInfo.begin()->first;
    const auto& inputBlobTensorDesc = inputsInfo.begin()->second->getTensorDesc();
    // Set input blob
    InferenceEngine::Blob::Ptr inputBlob = FuncTestUtils::createAndFillBlob(inputBlobTensorDesc);
    req.SetBlob(inputBlobName, inputBlob);

    // Collect output blob info
    const auto  outputsInfo          = cnnNet.getOutputsInfo();
    const auto& outputBlobName       = outputsInfo.begin()->first;
    const auto& outputBlobTensorDesc = outputsInfo.begin()->second->getTensorDesc();
    // Set output blob
    InferenceEngine::Blob::Ptr outputBlob = FuncTestUtils::createAndFillBlob(outputBlobTensorDesc);
    req.SetBlob(outputBlobName, outputBlob); 

    // Infer
    ASSERT_NO_THROW(req.Infer());
}

TEST_P(InferRequestInputTests, canGetInputBlob_deprecatedAPI) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // Create CNNNetwork from ngrpah::Function
    InferenceEngine::CNNNetwork cnnNet(function);
    // Load CNNNetwork to target plugins
    auto execNet = ie->LoadNetwork(cnnNet, targetDevice, configuration);
    // Create InferRequest
    InferenceEngine::InferRequest req;
    ASSERT_NO_THROW(req = execNet.CreateInferRequest());

    // Collect input blob info
    const auto  inputsInfo     = cnnNet.getInputsInfo();
    const auto& blobName       = inputsInfo.begin()->first;
    const auto& blobTensorDesc = inputsInfo.begin()->second->getTensorDesc();
    const auto& blobPrecision  = inputsInfo.begin()->second->getPrecision();
    const auto& blobDims       = blobTensorDesc.getDims();

    // Get input blob
    InferenceEngine::Blob::Ptr actualBlob;
    ASSERT_NO_THROW(actualBlob = req.GetBlob(blobName));
    ASSERT_TRUE(actualBlob) << "Plugin didn't allocate input blobs";
    ASSERT_FALSE(actualBlob->buffer() == nullptr) << "Plugin didn't allocate input blobs";
    const auto& tensorDescription = actualBlob->getTensorDesc();
    const auto& dims = tensorDescription.getDims();

    ASSERT_TRUE(blobDims == dims)
        << "Input blob dimensions don't match network input";
    ASSERT_EQ(blobPrecision, tensorDescription.getPrecision())
        << "Input blob precision doesn't match network input";
}

TEST_P(InferRequestInputTests, getAfterSetInputDoNotChangeInput) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // Create CNNNetwork from ngrpah::Function
    InferenceEngine::CNNNetwork cnnNet(function);
    // Load CNNNetwork to target plugins
    auto execNet = ie->LoadNetwork(cnnNet, targetDevice, configuration);
    // Create InferRequest
    InferenceEngine::InferRequest req = execNet.CreateInferRequest();

    // Collect input blob info
    const auto  inputsInfo     = cnnNet.getInputsInfo();
    const auto& blobName       = inputsInfo.begin()->first;
    const auto& blobTensorDesc = inputsInfo.begin()->second->getTensorDesc();

    // Set blob
    InferenceEngine::Blob::Ptr inputBlob = FuncTestUtils::createAndFillBlob(blobTensorDesc);
    ASSERT_NO_THROW(req.SetBlob(blobName, inputBlob));

    // Get blob
    InferenceEngine::Blob::Ptr actualBlob;
    ASSERT_NO_THROW(actualBlob = req.GetBlob(blobName));

    // Compare blobs
    ASSERT_EQ(inputBlob.get(), actualBlob.get());
}

TEST_P(InferRequestInputTests, canInferWithGetInOut) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // Create CNNNetwork from ngrpah::Function
    InferenceEngine::CNNNetwork cnnNet(function);
    // Load CNNNetwork to target plugins
    auto execNet = ie->LoadNetwork(cnnNet, targetDevice, configuration);
    // Create InferRequest
    InferenceEngine::InferRequest req;
    ASSERT_NO_THROW(req = execNet.CreateInferRequest());

    // Get input blob
    const auto inputsInfo  = cnnNet.getInputsInfo();
    const auto& inputBlobName = inputsInfo.begin()->first;
    InferenceEngine::Blob::Ptr inputBlob = req.GetBlob(inputBlobName);

    // Get output blob
    const auto outputsInfo = cnnNet.getOutputsInfo();
    const auto& outputBlobName = outputsInfo.begin()->first;
    InferenceEngine::Blob::Ptr outputBlob = req.GetBlob(outputBlobName);

    // Infer
    ASSERT_NO_THROW(req.Infer());
}

TEST_P(InferRequestInputTests, canStartAsyncInferWithGetInOut) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // Create CNNNetwork from ngrpah::Function
    InferenceEngine::CNNNetwork cnnNet(function);
    // Load CNNNetwork to target plugins
    auto execNet = ie->LoadNetwork(cnnNet, targetDevice, configuration);
    // Create InferRequest
    InferenceEngine::InferRequest req;
    ASSERT_NO_THROW(req = execNet.CreateInferRequest());

    // Get input blob name
    const auto inputsInfo  = cnnNet.getInputsInfo();
    const auto& inputBlobName = inputsInfo.begin()->first;
    // Get output blob name
    const auto outputsInfo = cnnNet.getOutputsInfo();
    const auto& outputBlobName = outputsInfo.begin()->first;

    // Async Infer
    InferenceEngine::Blob::Ptr inputBlob = req.GetBlob(inputBlobName); 
    InferenceEngine::StatusCode sts;
    ASSERT_NO_THROW(req.Infer());
    ASSERT_NO_THROW(req.StartAsync());
    sts = req.Wait(500);
    ASSERT_EQ(InferenceEngine::StatusCode::OK, sts);
    InferenceEngine::Blob::Ptr outputBlob = req.GetBlob(outputBlobName);
}

}  // namespace BehaviorTestsDefinitions
