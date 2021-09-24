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
#include "/home/maximandronov/test_repo/openvino/inference-engine/tests/ie_test_utils/functional_test_utils/include/functional_test_utils/blob_utils.hpp"

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

template<typename vecElementType>
inline std::string vec2str(const std::vector<vecElementType> &vec) {
    if (!vec.empty()) {
        std::ostringstream result;
        result << "(";
        std::copy(vec.begin(), vec.end() - 1, std::ostream_iterator<vecElementType>(result, "."));
        result << vec.back() << ")";
        return result.str();
    }
    return std::string("()");
}

TEST_P(InferRequestDynamicTests, CPU_ONLY) {
    const std::string param_name_1 = "Param_1";
    const std::string param_name_2 = "Param_2";
    // const std::string param_name_3 = "Param_3";

    InferenceEngine::Precision netPrecision = InferenceEngine::Precision::FP32;
    std::vector<std::vector<size_t>> inputShapes{{8, 6, 3}/*, {2}*/};
    auto inType = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

    ngraph::ParameterVector params(1); // = ngraph::builder::makeParams(inType, inputShapes);
    params[0] = std::make_shared<ngraph::op::Parameter>(inType, ngraph::Shape(inputShapes[0]));
    // params[1] = std::make_shared<ngraph::op::Parameter>(ov::element::i32, ngraph::Shape(inputShapes[1]));
    auto constanta = std::make_shared<ngraph::opset1::Constant>(ngraph::element::Type_t::i32, ngraph::Shape{2}, std::vector<int>{1, 2});
    params[0]->set_friendly_name(param_name_1);
    // params[1]->set_friendly_name(param_name_2);

    auto paramOuts = ngraph::helpers::convert2OutputVector(
            ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));

    const std::shared_ptr<ngraph::Node> reshape = std::make_shared<ngraph::opset1::Unsqueeze>(paramOuts[0], constanta);
    const ngraph::ResultVector results{std::make_shared<ngraph::opset3::Result>(reshape)};
    function = std::make_shared<ngraph::Function>(results, params, "Squeeze");

    // Create CNNNetwork from ngrpah::Function
    InferenceEngine::CNNNetwork cnnNet(function);
    // cnnNet.serialize("/home/maximandronov/test_repo/openvino/models/before.xml");
    std::map<std::string, ngraph::PartialShape> shapes;
    shapes[param_name_1] = {ngraph::Dimension::dynamic(), ngraph::Dimension::dynamic(), ngraph::Dimension::dynamic()};
    // shapes[param_name_2] = {ngraph::Dimension::dynamic(), ngraph::Dimension::dynamic(), ngraph::Dimension::dynamic(), ngraph::Dimension::dynamic(), ngraph::Dimension::dynamic()};
    // shapes[param_name_3] = {ngraph::Dimension::dynamic()};
    cnnNet.reshape(shapes);
    // cnnNet.serialize("/home/maximandronov/test_repo/openvino/models/after.xml");
    // Load CNNNetwork to target plugins
    auto execNet = ie->LoadNetwork(cnnNet, targetDevice, configuration);
    // Create InferRequest
    InferenceEngine::InferRequest req;
    InferenceEngine::Blob::Ptr blob1 = FuncTestUtils::createAndFillBlob(InferenceEngine::TensorDesc{InferenceEngine::Precision::FP32, inputShapes[0], InferenceEngine::Layout::BLOCKED});
    // InferenceEngine::Blob::Ptr blob2 = FuncTestUtils::createAndFillBlob(InferenceEngine::TensorDesc{InferenceEngine::Precision::I32, inputShapes[1], InferenceEngine::Layout::BLOCKED});
    // int32_t *ptr = blob2->buffer().as<int32_t *>();
    // ptr[0] = 2;
    // ptr[1] = 4;
    // InferenceEngine::Blob::Ptr blob3 = FuncTestUtils::createAndFillBlob(InferenceEngine::TensorDesc{InferenceEngine::Precision::FP32, inputShapes[2], InferenceEngine::Layout::BLOCKED});

    req = execNet.CreateInferRequest();
    req.SetBlob(param_name_1, blob1);
    const float *in1 = blob1->cbuffer().as<const float *>();
    for (size_t i = 0; i < blob1->size(); i++)
        std::cout << in1[i] << " ";
    std::cout << std::endl;
    // req.SetBlob(param_name_2, blob2);
    // req.SetBlob(param_name_3, blob3);
    // ASSERT_EQ(blob->getTensorDesc().getDims(), inputShapes);
    std::cout << "IN: " << vec2str(blob1->getTensorDesc().getDims()) << std::endl;
    req.Infer();
    InferenceEngine::StatusCode sts;
    sts = req.Wait(InferenceEngine::InferRequest::WaitMode::RESULT_READY);
    ASSERT_EQ(InferenceEngine::StatusCode::OK, sts);
    InferenceEngine::Blob::Ptr blob = req.GetBlob(cnnNet.getOutputsInfo().begin()->first);
    std::cout << "OUT: " << vec2str(blob->getTensorDesc().getDims()) << std::endl;
    std::cout << blob->getTensorDesc().getPrecision() << std::endl;
    const float *out1 = blob->cbuffer().as<const float *>();
    for (size_t i = 0; i < blob->size(); i++)
        std::cout << out1[i] << " ";
    std::cout << std::endl;
    // ASSERT_EQ(blob->getTensorDesc().getDims(), refOutShape);

    std::vector<std::vector<size_t>> inputShapes2{/*{5, 10, 1},*/ {2, 5, 1, }/*, {12}*/};
    blob1 = FuncTestUtils::createAndFillBlob(InferenceEngine::TensorDesc{InferenceEngine::Precision::FP32, inputShapes2[0], InferenceEngine::Layout::BLOCKED});
    // ptr[0] = 1;
    // ptr[1] = 3;
    // blob2 = FuncTestUtils::createAndFillBlob(InferenceEngine::TensorDesc{InferenceEngine::Precision::FP32, inputShapes2[1], InferenceEngine::Layout::BLOCKED});
    // blob3 = FuncTestUtils::createAndFillBlob(InferenceEngine::TensorDesc{InferenceEngine::Precision::FP32, inputShapes2[2], InferenceEngine::Layout::BLOCKED});
    req.SetBlob(param_name_1, blob1);
    const float *in2 = blob1->cbuffer().as<const float *>();
    for (size_t i = 0; i < blob1->size(); i++)
        std::cout << in2[i] << " ";
    std::cout << std::endl;
    // req.SetBlob(param_name_2, blob2);
    // req.SetBlob(param_name_3, blob3);
    // ASSERT_EQ(blob->getTensorDesc().getDims(), inputShape2);
    std::cout << "IN: " << vec2str(blob1->getTensorDesc().getDims()) << std::endl;
    req.Infer();
    sts = req.Wait(InferenceEngine::InferRequest::WaitMode::RESULT_READY);
    ASSERT_EQ(InferenceEngine::StatusCode::OK, sts);
    blob = req.GetBlob(cnnNet.getOutputsInfo().begin()->first);
    std::cout << "OUT: " << vec2str(blob->getTensorDesc().getDims()) << std::endl;
    std::cout << blob->getTensorDesc().getPrecision() << std::endl;
    const float *out2 = blob->cbuffer().as<const float *>();
    for (size_t i = 0; i < blob->size(); i++)
        std::cout << out2[i] << " ";
    std::cout << std::endl;
    // ASSERT_EQ(blob->getTensorDesc().getDims(), refOutShape2);
}

}  // namespace BehaviorTestsDefinitions