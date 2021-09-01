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
#include <ngraph/opsets/opset6.hpp>
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

class InferRequestDynamicTests : public BehaviorTestsUtils::BehaviorTestsBasic {
public:
    void SetUp()  override {
        std::tie(netPrecision, targetDevice, configuration) = this->GetParam();
        function = ngraph::builder::subgraph::makeSplitConvConcat();
    }
};

TEST_P(InferRequestDynamicTests, InferDynamicNetworkWithoutSetShape) {
    const std::string param_name = "Param_1";
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // Create CNNNetwork from ngrpah::Function
    InferenceEngine::CNNNetwork cnnNet(function);
    std::map<std::string, ngraph::PartialShape> shapes;
    shapes[param_name] = {ngraph::Dimension::dynamic(), 4, 20, 20};
    cnnNet.reshape(shapes);
    // Load CNNNetwork to target plugins
    auto execNet = ie->LoadNetwork(cnnNet, targetDevice, configuration);
    // Create InferRequest
    InferenceEngine::InferRequest req;
    InferenceEngine::Blob::Ptr blob;
    ASSERT_NO_THROW(req = execNet.CreateInferRequest());
    ASSERT_NO_THROW(blob = req.GetBlob(cnnNet.getInputsInfo().begin()->first));
}

TEST_P(InferRequestDynamicTests, InferDynamicNetworkBoundWithoutSetShape) {
    const std::string param_name = "Param_1";
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // Create CNNNetwork from ngrpah::Function
    InferenceEngine::CNNNetwork cnnNet(function);
    std::map<std::string, ngraph::PartialShape> shapes;
    shapes[param_name] = {ngraph::Dimension(0, 5), 4, 20, 20};
    cnnNet.reshape(shapes);
    // Load CNNNetwork to target plugins
    auto execNet = ie->LoadNetwork(cnnNet, targetDevice, configuration);
    // Create InferRequest
    InferenceEngine::InferRequest req;
    InferenceEngine::Blob::Ptr blob;
    ASSERT_NO_THROW(req = execNet.CreateInferRequest());
    ASSERT_NO_THROW(blob = req.GetBlob(cnnNet.getInputsInfo().begin()->first));
}


TEST_P(InferRequestDynamicTests, InferDynamicNetworkWithGetBlob) {
    const std::string param_name = "Param_1";
    const InferenceEngine::SizeVector refShape = {1, 4, 20, 20};
    const InferenceEngine::SizeVector refOutShape = {1, 10, 18, 18};
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // Create CNNNetwork from ngrpah::Function
    InferenceEngine::CNNNetwork cnnNet(function);
    std::map<std::string, ngraph::PartialShape> shapes;
    shapes[param_name] = {ngraph::Dimension::dynamic(), 4, 20, 20};
    cnnNet.reshape(shapes);
    // Load CNNNetwork to target plugins
    auto execNet = ie->LoadNetwork(cnnNet, targetDevice, configuration);
    // Create InferRequest
    InferenceEngine::InferRequest req;
    InferenceEngine::Blob::Ptr blob;
    ASSERT_NO_THROW(req = execNet.CreateInferRequest());
    //ASSERT_NO_THROW(req.SetShape(param_name, {1, 4, 20, 20}));
    ASSERT_NO_THROW(blob = req.GetBlob(cnnNet.getInputsInfo().begin()->first));
    ASSERT_NO_THROW(blob->setShape({1, 4, 20, 20}));
    ASSERT_EQ(blob->getTensorDesc().getDims(), refShape);
    req.Infer();
    req.StartAsync();
    InferenceEngine::StatusCode sts;
    sts = req.Wait(InferenceEngine::InferRequest::WaitMode::RESULT_READY);
    ASSERT_EQ(InferenceEngine::StatusCode::OK, sts);
    ASSERT_NO_THROW(blob = req.GetBlob(cnnNet.getOutputsInfo().begin()->first));
    ASSERT_EQ(blob->getTensorDesc().getDims(), refOutShape);
}

TEST_P(InferRequestDynamicTests, InferUpperBoundNetworkWithGetBlob) {
    const std::string param_name = "Param_1";
    const InferenceEngine::SizeVector refShape = {1, 4, 20, 20};
    const InferenceEngine::SizeVector refOutShape = {1, 10, 18, 18};
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // Create CNNNetwork from ngrpah::Function
    InferenceEngine::CNNNetwork cnnNet(function);
    std::map<std::string, ngraph::PartialShape> shapes;
    shapes[param_name] = {ngraph::Dimension(0, 19), 4, 20, 20};
    cnnNet.reshape(shapes);
    // Load CNNNetwork to target plugins
    auto execNet = ie->LoadNetwork(cnnNet, targetDevice, configuration);
    // Create InferRequest
    InferenceEngine::InferRequest req;
    InferenceEngine::Blob::Ptr blob;
    ASSERT_NO_THROW(req = execNet.CreateInferRequest());
    //ASSERT_NO_THROW(req.SetShape(param_name, {1, 4, 20, 20}));
    ASSERT_NO_THROW(blob = req.GetBlob(cnnNet.getInputsInfo().begin()->first));
    ASSERT_NO_THROW(blob->setShape({1, 4, 20, 20}));
    ASSERT_EQ(blob->getTensorDesc().getDims(), refShape);
    req.Infer();
    req.StartAsync();
    InferenceEngine::StatusCode sts;
    sts = req.Wait(InferenceEngine::InferRequest::WaitMode::RESULT_READY);
    ASSERT_EQ(InferenceEngine::StatusCode::OK, sts);
    ASSERT_NO_THROW(blob = req.GetBlob(cnnNet.getOutputsInfo().begin()->first));
    ASSERT_EQ(blob->getTensorDesc().getDims(), refOutShape);
}

TEST_P(InferRequestDynamicTests, InferOutOfRangeShapeNetworkWithGetBlobLower) {
    const std::string param_name = "Param_1";
    const InferenceEngine::SizeVector refShape = {1, 4, 20, 20};
    const InferenceEngine::SizeVector refOutShape = {1, 10, 18, 18};
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // Create CNNNetwork from ngrpah::Function
    InferenceEngine::CNNNetwork cnnNet(function);
    std::map<std::string, ngraph::PartialShape> shapes;
    shapes[param_name] = {ngraph::Dimension(2, 3), 4, 20, 20};
    cnnNet.reshape(shapes);
    // Load CNNNetwork to target plugins
    auto execNet = ie->LoadNetwork(cnnNet, targetDevice, configuration);
    // Create InferRequest
    InferenceEngine::InferRequest req;
    InferenceEngine::Blob::Ptr blob;
    ASSERT_NO_THROW(req = execNet.CreateInferRequest());
    ASSERT_NO_THROW(blob = req.GetBlob(cnnNet.getInputsInfo().begin()->first));
    ASSERT_NO_THROW(blob->setShape({1, 4, 20, 20}));
    // Plugin may or may not throw in case if input tensor has dimensions that are out of bounds
    //ASSERT_THROW(req.Infer(), InferenceEngine::Exception);
}

TEST_P(InferRequestDynamicTests, InferOutOfRangeShapeNetworkWithGetBlobUpper) {
    const std::string param_name = "Param_1";
    const InferenceEngine::SizeVector refShape = {1, 4, 20, 20};
    const InferenceEngine::SizeVector refOutShape = {1, 10, 18, 18};
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // Create CNNNetwork from ngrpah::Function
    InferenceEngine::CNNNetwork cnnNet(function);
    std::map<std::string, ngraph::PartialShape> shapes;
    shapes[param_name] = {ngraph::Dimension(1, 2), 4, 20, 20};
    cnnNet.reshape(shapes);
    // Load CNNNetwork to target plugins
    auto execNet = ie->LoadNetwork(cnnNet, targetDevice, configuration);
    // Create InferRequest
    InferenceEngine::InferRequest req;
    InferenceEngine::Blob::Ptr blob;
    ASSERT_NO_THROW(req = execNet.CreateInferRequest());
    ASSERT_NO_THROW(blob = req.GetBlob(cnnNet.getInputsInfo().begin()->first));
    ASSERT_NO_THROW(blob->setShape({3, 4, 20, 20}));
    // Plugin may or may not throw in case if input tensor has dimensions that are out of bounds
    // ASSERT_THROW(req.Infer(), InferenceEngine::Exception);
}

TEST_P(InferRequestDynamicTests, InferDynamicNetworkWithGetBlob2times) {
    const std::string param_name = "Param_1";
    const InferenceEngine::SizeVector refShape = {1, 4, 20, 20};
    const InferenceEngine::SizeVector refShape2 = {2, 4, 20, 20};
    const InferenceEngine::SizeVector refOutShape = {1, 10, 18, 18};
    const InferenceEngine::SizeVector refOutShape2 = {2, 10, 18, 18};
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // Create CNNNetwork from ngrpah::Function
    InferenceEngine::CNNNetwork cnnNet(function);
    std::map<std::string, ngraph::PartialShape> shapes;
    shapes[param_name] = {ngraph::Dimension::dynamic(), 4, 20, 20};
    cnnNet.reshape(shapes);
    // Load CNNNetwork to target plugins
    auto execNet = ie->LoadNetwork(cnnNet, targetDevice, configuration);
    // Create InferRequest
    InferenceEngine::InferRequest req;
    InferenceEngine::Blob::Ptr blob;
    ASSERT_NO_THROW(req = execNet.CreateInferRequest());
    ASSERT_NO_THROW(blob = req.GetBlob(cnnNet.getInputsInfo().begin()->first));
    ASSERT_NO_THROW(blob->setShape(refShape));
    ASSERT_EQ(blob->getTensorDesc().getDims(), refShape);
    req.Infer();
    req.StartAsync();
    InferenceEngine::StatusCode sts;
    sts = req.Wait(InferenceEngine::InferRequest::WaitMode::RESULT_READY);
    ASSERT_EQ(InferenceEngine::StatusCode::OK, sts);
    ASSERT_NO_THROW(blob = req.GetBlob(cnnNet.getOutputsInfo().begin()->first));
    ASSERT_EQ(blob->getTensorDesc().getDims(), refOutShape);

    ASSERT_NO_THROW(blob = req.GetBlob(cnnNet.getInputsInfo().begin()->first));
    ASSERT_NO_THROW(blob->setShape(refShape2));
    ASSERT_EQ(blob->getTensorDesc().getDims(), refShape2);
    req.Infer();
    req.StartAsync();
    sts = req.Wait(InferenceEngine::InferRequest::WaitMode::RESULT_READY);
    ASSERT_EQ(InferenceEngine::StatusCode::OK, sts);
    ASSERT_NO_THROW(blob = req.GetBlob(cnnNet.getOutputsInfo().begin()->first));
    ASSERT_EQ(blob->getTensorDesc().getDims(), refOutShape2);
}


TEST_P(InferRequestDynamicTests, GetSameBlob2times) {
    const std::string param_name = "Param_1";
    const InferenceEngine::SizeVector refShape = {1, 4, 20, 20};
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // Create CNNNetwork from ngrpah::Function
    InferenceEngine::CNNNetwork cnnNet(function);
    std::map<std::string, ngraph::PartialShape> shapes;
    shapes[param_name] = {ngraph::Dimension::dynamic(), 4, 20, 20};
    cnnNet.reshape(shapes);
    // Load CNNNetwork to target plugins
    auto execNet = ie->LoadNetwork(cnnNet, targetDevice, configuration);
    // Create InferRequest
    InferenceEngine::InferRequest req;
    InferenceEngine::Blob::Ptr blob;
    ASSERT_NO_THROW(req = execNet.CreateInferRequest());
    ASSERT_NO_THROW(blob = req.GetBlob(cnnNet.getInputsInfo().begin()->first));
    ASSERT_NO_THROW(blob->setShape(refShape));
    ASSERT_EQ(blob->getTensorDesc().getDims(), refShape);
    ASSERT_NO_THROW(blob = req.GetBlob(cnnNet.getInputsInfo().begin()->first));
    ASSERT_EQ(blob->getTensorDesc().getDims(), refShape);
}

TEST_P(InferRequestDynamicTests, InferDynamicNetworkWithSetBlob) {
    const std::string param_name = "Param_1";
    const InferenceEngine::SizeVector refShape = {1, 4, 20, 20};
    const InferenceEngine::SizeVector refOutShape = {1, 10, 18, 18};
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // Create CNNNetwork from ngrpah::Function
    InferenceEngine::CNNNetwork cnnNet(function);
    std::map<std::string, ngraph::PartialShape> shapes;
    shapes[param_name] = {ngraph::Dimension::dynamic(), 4, 20, 20};
    cnnNet.reshape(shapes);
    // Load CNNNetwork to target plugins
    auto execNet = ie->LoadNetwork(cnnNet, targetDevice, configuration);
    // Create InferRequest
    InferenceEngine::InferRequest req;
    InferenceEngine::Blob::Ptr blob = make_blob_with_precision({InferenceEngine::Precision::FP32, refShape, InferenceEngine::Layout::NCHW});
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

TEST_P(InferRequestDynamicTests, InferDynamicNetworkWithSetBlob2times) {
    const std::string param_name = "Param_1";
    const InferenceEngine::SizeVector refShape = {1, 4, 20, 20};
    const InferenceEngine::SizeVector refShape2 = {2, 4, 20, 20};
    const InferenceEngine::SizeVector refOutShape = {1, 10, 18, 18};
    const InferenceEngine::SizeVector refOutShape2 = {2, 10, 18, 18};
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // Create CNNNetwork from ngrpah::Function
    InferenceEngine::CNNNetwork cnnNet(function);
    std::map<std::string, ngraph::PartialShape> shapes;
    shapes[param_name] = {ngraph::Dimension::dynamic(), 4, 20, 20};
    cnnNet.reshape(shapes);
    // Load CNNNetwork to target plugins
    auto execNet = ie->LoadNetwork(cnnNet, targetDevice, configuration);
    // Create InferRequest
    InferenceEngine::InferRequest req;
    InferenceEngine::Blob::Ptr blob = make_blob_with_precision({InferenceEngine::Precision::FP32, refShape, InferenceEngine::Layout::NCHW});
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

    blob = make_blob_with_precision({InferenceEngine::Precision::FP32, refShape2, InferenceEngine::Layout::NCHW});
    blob->allocate();
    ASSERT_NO_THROW(req.SetBlob(cnnNet.getInputsInfo().begin()->first, blob));
    ASSERT_EQ(blob->getTensorDesc().getDims(), refShape2);
    req.Infer();
    req.StartAsync();
    sts = req.Wait(InferenceEngine::InferRequest::WaitMode::RESULT_READY);
    ASSERT_EQ(InferenceEngine::StatusCode::OK, sts);
    ASSERT_NO_THROW(blob = req.GetBlob(cnnNet.getOutputsInfo().begin()->first));
    ASSERT_EQ(blob->getTensorDesc().getDims(), refOutShape2);
}

}  // namespace BehaviorTestsDefinitions
