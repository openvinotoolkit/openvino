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
#include "behavior/infer_request_output.hpp"

namespace BehaviorTestsDefinitions {
using InferRequestOutputTests = BehaviorTestsUtils::BehaviorTestsBasic;

TEST_P(InferRequestOutputTests, canGetOutputBlobForSyncRequest) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // Create CNNNetwork from ngrpah::Function
    InferenceEngine::CNNNetwork cnnNet(function);
    // Load CNNNetwork to target plugins
    auto execNet = ie->LoadNetwork(cnnNet, targetDevice, configuration);
    // Create InferRequest
    InferenceEngine::InferRequest req;
    ASSERT_NO_THROW(req = execNet.CreateInferRequest());

    // Collect output blob info
    const auto  outputsInfo    = cnnNet.getOutputsInfo();
    const auto& blobName       = outputsInfo.begin()->first;
    const auto& blobTensorDesc = outputsInfo.begin()->second->getTensorDesc();
    
    // Set output blob
    InferenceEngine::Blob::Ptr OutputBlob = FuncTestUtils::createAndFillBlob(blobTensorDesc);
    ASSERT_NO_THROW(req.SetBlob(blobName, OutputBlob));
    
    // Get output blob
    InferenceEngine::Blob::Ptr actualBlob;
    ASSERT_NO_THROW(actualBlob = req.GetBlob(blobName));

    // Compare blobs
    ASSERT_EQ(OutputBlob, actualBlob);
}

TEST_P(InferRequestOutputTests, canInferWithSetInOut) {
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

TEST_P(InferRequestOutputTests, canGetOutputBlob_deprecatedAPI) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // Create CNNNetwork from ngrpah::Function
    InferenceEngine::CNNNetwork cnnNet(function);
    // Load CNNNetwork to target plugins
    auto execNet = ie->LoadNetwork(cnnNet, targetDevice, configuration);
    // Create InferRequest
    InferenceEngine::InferRequest req;
    ASSERT_NO_THROW(req = execNet.CreateInferRequest());
    
    // Collect output blob info
    const auto  outputsInfo    = cnnNet.getOutputsInfo();
    const auto& blobName       = outputsInfo.begin()->first;
    const auto& blobTensorDesc = outputsInfo.begin()->second->getTensorDesc();
    const auto& blobPrecision  = outputsInfo.begin()->second->getPrecision();
    const auto& blobDims       = blobTensorDesc.getDims();

    // Get output blob
    InferenceEngine::Blob::Ptr actualBlob;
    ASSERT_NO_THROW(actualBlob = req.GetBlob(blobName));
    ASSERT_TRUE(actualBlob) << "Plugin didn't allocate Output blobs";
    ASSERT_FALSE(actualBlob->buffer() == nullptr) << "Plugin didn't allocate Output blobs";
    const auto& tensorDescription = actualBlob->getTensorDesc();
    const auto& dims = tensorDescription.getDims();

    ASSERT_TRUE(blobDims == dims)
        << "Output blob dimensions don't match network Output";
    ASSERT_EQ(blobPrecision, tensorDescription.getPrecision())
        << "Output blob precision doesn't match network Output";
}

TEST_P(InferRequestOutputTests, getOutputAfterSetOutputDoNotChangeOutput) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // Create CNNNetwork from ngrpah::Function
    InferenceEngine::CNNNetwork cnnNet(function);
    // Load CNNNetwork to target plugins
    auto execNet = ie->LoadNetwork(cnnNet, targetDevice, configuration);
    // Create InferRequest
    InferenceEngine::InferRequest req;
    ASSERT_NO_THROW(req = execNet.CreateInferRequest());

    // Collect output blob info
    const auto outputsInfo = cnnNet.getOutputsInfo();
    const auto& blobName = outputsInfo.begin()->first;
    const auto& blobTensorDesc = outputsInfo.begin()->second->getTensorDesc();
    
    // Set output blob 
    InferenceEngine::Blob::Ptr outputBlob = FuncTestUtils::createAndFillBlob(blobTensorDesc);
    ASSERT_NO_THROW(req.SetBlob(blobName, outputBlob));
    
    // Get output blob
    InferenceEngine::Blob::Ptr actualBlob;
    ASSERT_NO_THROW(actualBlob = req.GetBlob(blobName));

    // Compare blobs
    ASSERT_EQ(outputBlob.get(), actualBlob.get());
}

TEST_P(InferRequestOutputTests, canInferWithGetInOut) {
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

TEST_P(InferRequestOutputTests, canStartAsyncInferWithGetInOut) {
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
    const auto inputsInfo = cnnNet.getInputsInfo();
    const auto& inputBlobName = inputsInfo.begin()->first;
    // Collect output blob info
    const auto outputsInfo = cnnNet.getOutputsInfo();
    const auto& outputBlobName = outputsInfo.begin()->first;

    // start AsyncInfer
    InferenceEngine::Blob::Ptr inputBlob = req.GetBlob(inputBlobName);
    ASSERT_NO_THROW(req.Infer());
    ASSERT_NO_THROW(req.StartAsync());
    ASSERT_NO_THROW(req.Wait());
    InferenceEngine::Blob::Ptr outputBlob = req.GetBlob(outputBlobName);
}

}  // namespace BehaviorTestsDefinitions
