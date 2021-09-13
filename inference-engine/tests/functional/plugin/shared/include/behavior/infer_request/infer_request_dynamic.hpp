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

// TODO [mandrono]: move current test case inside CPU plug-in and return the original tests
namespace BehaviorTestsDefinitions {

typedef std::tuple<
        std::shared_ptr<ngraph::Function>,                                 // ngraph function
        std::vector<std::pair<std::vector<size_t>, std::vector<size_t>>>,  // input/expected output shapes per inference
        std::string,                                                       // Device name
        std::map<std::string, std::string>                                 // Config
> InferRequestDynamicParams;

class InferRequestDynamicTests : public testing::WithParamInterface<InferRequestDynamicParams>,
                                 public CommonTestUtils::TestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<InferRequestDynamicParams> obj) {
        std::shared_ptr<ngraph::Function> func;
        std::vector<std::pair<std::vector<size_t>, std::vector<size_t>>> inOutShapes;
        std::string targetDevice;
        std::map<std::string, std::string> configuration;
        std::tie(func, inOutShapes, targetDevice, configuration) = obj.param;
        std::ostringstream result;
        result << "function=" << func->get_friendly_name() << "_";
        result << "inOutShape=(";
        for (const auto& inOutShape : inOutShapes) {
            result << "(" << CommonTestUtils::vec2str(inOutShape.first) << "_" << CommonTestUtils::vec2str(inOutShape.second) << ")";
        }
        result << ")_";
        result << "targetDevice=" << targetDevice;
        if (!configuration.empty()) {
            for (auto& configItem : configuration) {
                result << "configItem=" << configItem.first << "_" << configItem.second << "_";
            }
        }
        return result.str();
    }

protected:
    void SetUp() override {
        SKIP_IF_CURRENT_TEST_IS_DISABLED()
        std::tie(function, inOutShapes, targetDevice, configuration) = this->GetParam();
    }

    void TearDown() override {
        if (!configuration.empty()) {
            PluginCache::get().reset();
        }
        function.reset();
    }

    std::shared_ptr<InferenceEngine::Core> ie = PluginCache::get().ie();
    std::shared_ptr<ngraph::Function> function;
    std::string targetDevice;
    std::map<std::string, std::string> configuration;
    std::vector<std::pair<std::vector<size_t>, std::vector<size_t>>> inOutShapes;
};

TEST_P(InferRequestDynamicTests, InferDynamicNetworkWithoutSetShape) {
    const std::string param_name = "Param_1";
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
    const InferenceEngine::SizeVector refShape = inOutShapes[0].first;
    const InferenceEngine::SizeVector refOutShape = inOutShapes[0].second;
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
    const InferenceEngine::SizeVector refShape = inOutShapes[0].first;
    const InferenceEngine::SizeVector refOutShape = inOutShapes[0].second;
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
    const InferenceEngine::SizeVector refShape = inOutShapes[0].first;
    const InferenceEngine::SizeVector refOutShape = inOutShapes[0].second;
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
    const InferenceEngine::SizeVector refShape = inOutShapes[0].first;
    const InferenceEngine::SizeVector refOutShape = inOutShapes[0].second;
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
    const InferenceEngine::SizeVector refShape = inOutShapes[0].first;
    const InferenceEngine::SizeVector refShape2 = inOutShapes[1].first;
    const InferenceEngine::SizeVector refOutShape = inOutShapes[0].second;
    const InferenceEngine::SizeVector refOutShape2 = inOutShapes[1].second;
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
    const InferenceEngine::SizeVector refShape = inOutShapes[0].first;
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
    const InferenceEngine::SizeVector refShape = inOutShapes[0].first;
    const InferenceEngine::SizeVector refOutShape = inOutShapes[0].second;
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
    const InferenceEngine::SizeVector refShape = inOutShapes[0].first;
    const InferenceEngine::SizeVector refShape2 = inOutShapes[1].first;
    const InferenceEngine::SizeVector refOutShape = inOutShapes[0].second;
    const InferenceEngine::SizeVector refOutShape2 = inOutShapes[1].second;
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