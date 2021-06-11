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
using InferRequestDynamicTests = BehaviorTestsUtils::BehaviorTestsBasic;

TEST_P(InferRequestDynamicTests, InferDynamicNetworkWithoutSetShape) {
    const std::string param_name = "Param_1";
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // Create CNNNetwork from ngrpah::Function
    InferenceEngine::CNNNetwork cnnNet(function);
    std::map<std::string, ngraph::PartialShape> shapes;
    shapes[param_name] = {ngraph::Dimension::dynamic(), 1, 32, 32};
    cnnNet.reshape(shapes);
    // Load CNNNetwork to target plugins
    auto execNet = ie->LoadNetwork(cnnNet, targetDevice, configuration);
    // Create InferRequest
    InferenceEngine::InferRequest req;
    InferenceEngine::Blob::Ptr blob;
    ASSERT_NO_THROW(req = execNet.CreateInferRequest());
    ASSERT_THROW(blob = req.GetBlob(cnnNet.getInputsInfo().begin()->first), InferenceEngine::Exception);
}

TEST_P(InferRequestDynamicTests, InferDynamicNetworkWithGetBlob) {
    const std::string param_name = "Param_1";
    const InferenceEngine::SizeVector refShape = {1, 1, 32, 32};
    const InferenceEngine::SizeVector refOutShape = {1, 116};
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // Create CNNNetwork from ngrpah::Function
    InferenceEngine::CNNNetwork cnnNet(function);
    std::map<std::string, ngraph::PartialShape> shapes;
    shapes[param_name] = {ngraph::Dimension::dynamic(), 1, 32, 32};
    cnnNet.reshape(shapes);
    // Load CNNNetwork to target plugins
    auto execNet = ie->LoadNetwork(cnnNet, targetDevice, configuration);
    // Create InferRequest
    InferenceEngine::InferRequest req;
    InferenceEngine::Blob::Ptr blob;
    ASSERT_NO_THROW(req = execNet.CreateInferRequest());
    ASSERT_NO_THROW(req.SetShape(param_name, {1, 1, 32, 32}));
    ASSERT_NO_THROW(blob = req.GetBlob(cnnNet.getInputsInfo().begin()->first));
    ASSERT_EQ(blob->getTensorDesc().getDims(), refShape);
    req.Infer();
    req.StartAsync();
    InferenceEngine::StatusCode sts;
    sts = req.Wait(InferenceEngine::InferRequest::WaitMode::RESULT_READY);
    ASSERT_EQ(InferenceEngine::StatusCode::OK, sts);
    ASSERT_NO_THROW(blob = req.GetBlob(cnnNet.getOutputsInfo().begin()->first));
    ASSERT_EQ(blob->getTensorDesc().getDims(), refOutShape);
}

TEST_P(InferRequestDynamicTests, InferDynamicNetworkWithSetBlob) {
    const std::string param_name = "Param_1";
    const InferenceEngine::SizeVector refShape = {1, 1, 32, 32};
    const InferenceEngine::SizeVector refOutShape = {1, 116};
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // Create CNNNetwork from ngrpah::Function
    InferenceEngine::CNNNetwork cnnNet(function);
    std::map<std::string, ngraph::PartialShape> shapes;
    shapes[param_name] = {ngraph::Dimension::dynamic(), 1, 32, 32};
    cnnNet.reshape(shapes);
    // Load CNNNetwork to target plugins
    auto execNet = ie->LoadNetwork(cnnNet, targetDevice, configuration);
    // Create InferRequest
    InferenceEngine::InferRequest req;
    InferenceEngine::Blob::Ptr blob = make_blob_with_precision({InferenceEngine::Precision::FP32, {1, 1, 32, 32}, InferenceEngine::Layout::NCHW});
    blob->allocate();
    ASSERT_NO_THROW(req = execNet.CreateInferRequest());
    ASSERT_NO_THROW(req.SetBlob(cnnNet.getInputsInfo().begin()->first, blob));
    ASSERT_EQ(blob->getTensorDesc().getDims(), refShape);
    req.Infer();
    req.StartAsync();
    InferenceEngine::StatusCode sts;
    sts = req.Wait(InferenceEngine::InferRequest::WaitMode::RESULT_READY);
    ASSERT_EQ(InferenceEngine::StatusCode::OK, sts);
    ASSERT_NO_THROW(blob = req.GetBlob(cnnNet.getOutputsInfo().begin()->first));
    ASSERT_EQ(blob->getTensorDesc().getDims(), refOutShape);
}

}  // namespace BehaviorTestsDefinitions
