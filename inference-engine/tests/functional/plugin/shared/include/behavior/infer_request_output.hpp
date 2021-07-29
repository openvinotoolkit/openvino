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

TEST_P(InferRequestOutputTests, canGetInputBlobForSyncRequest) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // Create CNNNetwork from ngrpah::Function
    InferenceEngine::CNNNetwork cnnNet(function);
    // Load CNNNetwork to target plugins
    auto execNet = ie->LoadNetwork(cnnNet, targetDevice, configuration);
    // Create InferRequest
    InferenceEngine::InferRequest req;
    ASSERT_NO_THROW(req = execNet.CreateInferRequest());
    InferenceEngine::Blob::Ptr OutputBlob =
            FuncTestUtils::createAndFillBlob(cnnNet.getOutputsInfo().begin()->second->getTensorDesc());
    ASSERT_NO_THROW(req.SetBlob(cnnNet.getOutputsInfo().begin()->first, OutputBlob));
    InferenceEngine::Blob::Ptr actualBlob;
    ASSERT_NO_THROW(actualBlob = req.GetBlob(cnnNet.getOutputsInfo().begin()->first));
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
    InferenceEngine::Blob::Ptr inputBlob =
            FuncTestUtils::createAndFillBlob(cnnNet.getInputsInfo().begin()->second->getTensorDesc());
    req.SetBlob(cnnNet.getInputsInfo().begin()->first, inputBlob);
    InferenceEngine::Blob::Ptr outputBlob =
            FuncTestUtils::createAndFillBlob(cnnNet.getInputsInfo().begin()->second->getTensorDesc());
    req.SetBlob(cnnNet.getInputsInfo().begin()->first, outputBlob);
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
    std::shared_ptr<InferenceEngine::Blob> actualBlob;

    ASSERT_NO_THROW(actualBlob = req.GetBlob(cnnNet.getOutputsInfo().begin()->first));
    ASSERT_TRUE(actualBlob) << "Plugin didn't allocate Output blobs";
    ASSERT_FALSE(actualBlob->buffer() == nullptr) << "Plugin didn't allocate Output blobs";

    auto tensorDescription = actualBlob->getTensorDesc();
    auto dims = tensorDescription.getDims();
    ASSERT_TRUE(cnnNet.getOutputsInfo().begin()->second->getTensorDesc().getDims() == dims)
                                << "Output blob dimensions don't match network Output";

    ASSERT_EQ(execNet.GetInputsInfo().begin()->second->getPrecision(), tensorDescription.getPrecision())
                                << "Output blob precision don't match network Output";
}

TEST_P(InferRequestOutputTests, getOutputAfterSetOutputDoNotChangeOutput) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // Create CNNNetwork from ngrpah::Function
    InferenceEngine::CNNNetwork cnnNet(function);
    // Load CNNNetwork to target plugins
    auto execNet = ie->LoadNetwork(cnnNet, targetDevice, configuration);
    // Create InferRequest
    InferenceEngine::InferRequest req = execNet.CreateInferRequest();
    std::shared_ptr<InferenceEngine::Blob> OutputBlob = FuncTestUtils::createAndFillBlob(
            cnnNet.getOutputsInfo().begin()->second->getTensorDesc());
    ASSERT_NO_THROW(req.SetBlob(cnnNet.getOutputsInfo().begin()->first, OutputBlob));
    std::shared_ptr<InferenceEngine::Blob> actualBlob;
    ASSERT_NO_THROW(actualBlob = req.GetBlob(cnnNet.getOutputsInfo().begin()->first));
    ASSERT_EQ(OutputBlob.get(), actualBlob.get());
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
    InferenceEngine::Blob::Ptr inputBlob = req.GetBlob(cnnNet.getInputsInfo().begin()->first);
    InferenceEngine::Blob::Ptr outputBlob = req.GetBlob(cnnNet.getOutputsInfo().begin()->first);
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
    InferenceEngine::Blob::Ptr inputBlob = req.GetBlob(cnnNet.getInputsInfo().begin()->first);
    ASSERT_NO_THROW(req.Infer());
    ASSERT_NO_THROW(req.StartAsync());
    ASSERT_NO_THROW(req.Wait());
    InferenceEngine::Blob::Ptr outputBlob = req.GetBlob(cnnNet.getOutputsInfo().begin()->first);
}
}  // namespace BehaviorTestsDefinitions