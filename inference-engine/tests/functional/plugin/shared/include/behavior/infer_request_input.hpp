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
    InferenceEngine::Blob::Ptr inputBlob =
            FuncTestUtils::createAndFillBlob(cnnNet.getInputsInfo().begin()->second->getTensorDesc());
    ASSERT_NO_THROW(req.SetBlob(cnnNet.getInputsInfo().begin()->first, inputBlob));
    InferenceEngine::Blob::Ptr actualBlob;
    ASSERT_NO_THROW(actualBlob = req.GetBlob(cnnNet.getInputsInfo().begin()->first));
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
    InferenceEngine::Blob::Ptr inputBlob =
            FuncTestUtils::createAndFillBlob(cnnNet.getInputsInfo().begin()->second->getTensorDesc());
    req.SetBlob(cnnNet.getInputsInfo().begin()->first, inputBlob);
    InferenceEngine::Blob::Ptr outputBlob =
            FuncTestUtils::createAndFillBlob(cnnNet.getInputsInfo().begin()->second->getTensorDesc());
    req.SetBlob(cnnNet.getInputsInfo().begin()->first, outputBlob);
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
    std::shared_ptr<InferenceEngine::Blob> actualBlob;

    ASSERT_NO_THROW(actualBlob = req.GetBlob(cnnNet.getInputsInfo().begin()->first));
    ASSERT_TRUE(actualBlob) << "Plugin didn't allocate input blobs";
    ASSERT_FALSE(actualBlob->buffer() == nullptr) << "Plugin didn't allocate input blobs";

    auto tensorDescription = actualBlob->getTensorDesc();
    auto dims = tensorDescription.getDims();
    ASSERT_TRUE(cnnNet.getInputsInfo().begin()->second->getTensorDesc().getDims() == dims)
                                << "Input blob dimensions don't match network input";

    ASSERT_EQ(execNet.GetInputsInfo().begin()->second->getPrecision(), tensorDescription.getPrecision())
                                << "Input blob precision don't match network input";
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
    std::shared_ptr<InferenceEngine::Blob> inputBlob = FuncTestUtils::createAndFillBlob(
            cnnNet.getInputsInfo().begin()->second->getTensorDesc());
    ASSERT_NO_THROW(req.SetBlob(cnnNet.getInputsInfo().begin()->first, inputBlob));
    std::shared_ptr<InferenceEngine::Blob> actualBlob;
    ASSERT_NO_THROW(actualBlob = req.GetBlob(cnnNet.getInputsInfo().begin()->first));
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
    InferenceEngine::Blob::Ptr inputBlob = req.GetBlob(cnnNet.getInputsInfo().begin()->first);
    InferenceEngine::Blob::Ptr outputBlob = req.GetBlob(cnnNet.getOutputsInfo().begin()->first);
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
    InferenceEngine::Blob::Ptr inputBlob = req.GetBlob(cnnNet.getInputsInfo().begin()->first);
    InferenceEngine::StatusCode sts;
    ASSERT_NO_THROW(req.Infer());
    ASSERT_NO_THROW(req.StartAsync());
    sts = req.Wait(500);
    ASSERT_EQ(InferenceEngine::StatusCode::OK, sts);
    InferenceEngine::Blob::Ptr outputBlob = req.GetBlob(cnnNet.getOutputsInfo().begin()->first);
}
}  // namespace BehaviorTestsDefinitions