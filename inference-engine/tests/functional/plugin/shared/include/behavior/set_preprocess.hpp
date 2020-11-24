// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include <ie_core.hpp>
#include <blob_factory.hpp>
#include "common_test_utils/test_assertions.hpp"
#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/plugin_cache.hpp"
#include "functional_test_utils/layer_test_utils.hpp"
#include "functional_test_utils/blob_utils.hpp"
#include "ie_preprocess.hpp"
#include "functional_test_utils/behavior_test_utils.hpp"

namespace BehaviorTestsDefinitions {
using PreprocessTest = BehaviorTestsUtils::BehaviorTestsBasic;

TEST_P(PreprocessTest, SetPreProcessToInputInfo) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // Create CNNNetwork from ngrpah::Function
    InferenceEngine::CNNNetwork cnnNet(function);

    auto &preProcess = cnnNet.getInputsInfo().begin()->second->getPreProcess();
    preProcess.setResizeAlgorithm(InferenceEngine::ResizeAlgorithm::RESIZE_BILINEAR);
    // Load CNNNetwork to target plugins
    auto execNet = ie->LoadNetwork(cnnNet, targetDevice, configuration);
    // Create InferRequest
    auto req = execNet.CreateInferRequest();
    {
        InferenceEngine::ConstInputsDataMap inputsMap = execNet.GetInputsInfo();
        const auto &name = inputsMap.begin()->second->name();
        const InferenceEngine::PreProcessInfo *info = &req.GetPreProcess(name.c_str());
        ASSERT_EQ(info->getResizeAlgorithm(), InferenceEngine::ResizeAlgorithm::RESIZE_BILINEAR);
        ASSERT_PREPROCESS_INFO_EQ(preProcess, *info);
    }
}

TEST_P(PreprocessTest, SetPreProcessToInferRequest) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // Create CNNNetwork from ngrpah::Function
    InferenceEngine::CNNNetwork cnnNet(function);

    auto &preProcess = cnnNet.getInputsInfo().begin()->second->getPreProcess();
    preProcess.setResizeAlgorithm(InferenceEngine::ResizeAlgorithm::RESIZE_BILINEAR);
    // Load CNNNetwork to target plugins
    auto execNet = ie->LoadNetwork(cnnNet, targetDevice, configuration);
    // Create InferRequest
    auto req = execNet.CreateInferRequest();
    InferenceEngine::ConstInputsDataMap inputsMap = execNet.GetInputsInfo();
    const auto &name = inputsMap.begin()->second->name();
    auto inputBlob = FuncTestUtils::createAndFillBlob(
            cnnNet.getInputsInfo().begin()->second->getTensorDesc());
    req.SetBlob(cnnNet.getInputsInfo().begin()->first, inputBlob);
    {
        const InferenceEngine::PreProcessInfo *info = &req.GetPreProcess(name.c_str());
        ASSERT_EQ(cnnNet.getInputsInfo().begin()->second->getPreProcess().getResizeAlgorithm(),
                  info->getResizeAlgorithm());
    }
}

TEST_P(PreprocessTest, SetMeanImagePreProcess) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    std::shared_ptr<ngraph::Function> ngraph;
    {
        ngraph::PartialShape shape({1, 3, 10, 10});
        ngraph::element::Type type(ngraph::element::Type_t::f32);
        auto param = std::make_shared<ngraph::op::Parameter>(type, shape);
        param->set_friendly_name("param");
        auto relu = std::make_shared<ngraph::op::Relu>(param);
        relu->set_friendly_name("relu");
        auto result = std::make_shared<ngraph::op::Result>(relu);
        result->set_friendly_name("result");

        ngraph::ParameterVector params = {param};
        ngraph::ResultVector results = {result};

        ngraph = std::make_shared<ngraph::Function>(results, params);
    }

    // Create CNNNetwork from ngrpah::Function
    InferenceEngine::CNNNetwork cnnNet(ngraph);

    auto &preProcess = cnnNet.getInputsInfo().begin()->second->getPreProcess();
    preProcess.init(3);
    for (size_t i = 0; i < 3; i++) {
        preProcess[i]->meanData = make_blob_with_precision(InferenceEngine::TensorDesc(InferenceEngine::Precision::FP32,
                                                                                       {10, 10},
                                                                                       InferenceEngine::Layout::HW));
        preProcess[i]->meanData->allocate();
        auto lockedMem = preProcess[i]->meanData->buffer();
        auto* data = lockedMem.as<float *>();
        for (size_t j = 0; j < 100; j++) {
            data[j] = 0;
            data[j] -= i * 100 + j;
        }
    }
    preProcess.setVariant(InferenceEngine::MEAN_IMAGE);
    // Load CNNNetwork to target plugins
    auto execNet = ie->LoadNetwork(cnnNet, targetDevice, configuration);
    // Create InferRequest
    auto req = execNet.CreateInferRequest();
    auto inBlob = req.GetBlob("param");

    // Fill input
    {
        auto locketMem = inBlob->buffer();
        auto *inData = locketMem.as<float*>();
        for (size_t i = 0; i < inBlob->size(); i++)
            inData[i] = i;
    }

    req.Infer();

    // Check output
    auto outBlob = req.GetBlob(cnnNet.getOutputsInfo().begin()->first);
    {
        auto inMem = inBlob->cbuffer();
        const auto* inData = inMem.as<const float*>();
        auto outMem = outBlob->cbuffer();
        const auto* outData = outMem.as<const float*>();
        ASSERT_EQ(inBlob->size(), outBlob->size());
        for (size_t i = 0; i < inBlob->size(); i++)
            ASSERT_EQ(inData[i] + inData[i], outData[i]);
    }
}

TEST_P(PreprocessTest, SetMeanValuePreProcess) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    std::shared_ptr<ngraph::Function> ngraph;
    {
        ngraph::PartialShape shape({1, 3, 10, 10});
        ngraph::element::Type type(ngraph::element::Type_t::f32);
        auto param = std::make_shared<ngraph::op::Parameter>(type, shape);
        param->set_friendly_name("param");
        auto relu = std::make_shared<ngraph::op::Relu>(param);
        relu->set_friendly_name("relu");
        auto result = std::make_shared<ngraph::op::Result>(relu);
        result->set_friendly_name("result");

        ngraph::ParameterVector params = {param};
        ngraph::ResultVector results = {result};

        ngraph = std::make_shared<ngraph::Function>(results, params);
    }

    // Create CNNNetwork from ngrpah::Function
    InferenceEngine::CNNNetwork cnnNet(ngraph);

    auto &preProcess = cnnNet.getInputsInfo().begin()->second->getPreProcess();
    preProcess.init(3);
    preProcess[0]->meanValue = -5;
    preProcess[1]->meanValue = -5;
    preProcess[2]->meanValue = -5;
    preProcess[0]->stdScale = 1;
    preProcess[1]->stdScale = 1;
    preProcess[2]->stdScale = 1;
    preProcess.setVariant(InferenceEngine::MEAN_VALUE);
    // Load CNNNetwork to target plugins
    auto execNet = ie->LoadNetwork(cnnNet, targetDevice, configuration);
    // Create InferRequest
    auto req = execNet.CreateInferRequest();
    auto inBlob = req.GetBlob("param");

    // Fill input
    {
        auto locketMem = inBlob->buffer();
        auto *inData = locketMem.as<float*>();
        for (size_t i = 0; i < inBlob->size(); i++)
            inData[i] = i;
    }

    req.Infer();

    // Check output
    auto outBlob = req.GetBlob(cnnNet.getOutputsInfo().begin()->first);
    {
        auto inMem = inBlob->cbuffer();
        const auto* inData = inMem.as<const float*>();
        auto outMem = outBlob->cbuffer();
        const auto* outData = outMem.as<const float*>();
        ASSERT_EQ(inBlob->size(), outBlob->size());
        for (size_t i = 0; i < inBlob->size(); i++)
            ASSERT_EQ(inData[i]+5, outData[i]);
    }
}

TEST_P(PreprocessTest, ReverseInputChannelsPreProcess) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    std::shared_ptr<ngraph::Function> ngraph;
    {
        ngraph::PartialShape shape({1, 3, 10, 10});
        ngraph::element::Type type(ngraph::element::Type_t::f32);
        auto param = std::make_shared<ngraph::op::Parameter>(type, shape);
        param->set_friendly_name("param");
        auto relu = std::make_shared<ngraph::op::Relu>(param);
        relu->set_friendly_name("relu");
        auto result = std::make_shared<ngraph::op::Result>(relu);
        result->set_friendly_name("result");

        ngraph::ParameterVector params = {param};
        ngraph::ResultVector results = {result};

        ngraph = std::make_shared<ngraph::Function>(results, params);
    }

    // Create CNNNetwork from ngrpah::Function
    InferenceEngine::CNNNetwork cnnNet(ngraph);

    auto &preProcess = cnnNet.getInputsInfo().begin()->second->getPreProcess();
    preProcess.setColorFormat(InferenceEngine::ColorFormat::RGB);
    // Load CNNNetwork to target plugins
    auto execNet = ie->LoadNetwork(cnnNet, targetDevice, configuration);
    // Create InferRequest
    auto req = execNet.CreateInferRequest();
    auto inBlob = req.GetBlob("param");

    // Fill input
    {
        auto locketMem = inBlob->buffer();
        auto *inData = locketMem.as<float*>();
        for (size_t i = 0; i < inBlob->size(); i++)
            inData[i] = i;
    }

    req.Infer();

    // Check output
    auto outBlob = req.GetBlob(cnnNet.getOutputsInfo().begin()->first);
    {
        auto inMem = inBlob->cbuffer();
        const auto* inData = inMem.as<const float*>();
        auto outMem = outBlob->cbuffer();
        const auto* outData = outMem.as<const float*>();
        ASSERT_EQ(inBlob->size(), outBlob->size());
        for (size_t i = 0; i < 3; i++)
            for (size_t j = 0; j < 100; j++) {
                // BGR to RGB
                if (!i) {
                    ASSERT_EQ(inData[j], outData[200 + j]);
                } else if (i == j) {
                    ASSERT_EQ(inData[100 + j], outData[100 + j]);
                } else {
                    ASSERT_EQ(inData[200 + j], outData[j]);
                }
            }
    }
}

TEST_P(PreprocessTest, SetScalePreProcess) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    std::shared_ptr<ngraph::Function> ngraph;
    {
        ngraph::PartialShape shape({1, 3, 10, 10});
        ngraph::element::Type type(ngraph::element::Type_t::f32);
        auto param = std::make_shared<ngraph::op::Parameter>(type, shape);
        param->set_friendly_name("param");
        auto relu = std::make_shared<ngraph::op::Relu>(param);
        relu->set_friendly_name("relu");
        auto result = std::make_shared<ngraph::op::Result>(relu);
        result->set_friendly_name("result");

        ngraph::ParameterVector params = {param};
        ngraph::ResultVector results = {result};

        ngraph = std::make_shared<ngraph::Function>(results, params);
    }

    // Create CNNNetwork from ngrpah::Function
    InferenceEngine::CNNNetwork cnnNet(ngraph);

    auto &preProcess = cnnNet.getInputsInfo().begin()->second->getPreProcess();
    preProcess.init(3);
    preProcess[0]->stdScale = 2;
    preProcess[1]->stdScale = 2;
    preProcess[2]->stdScale = 2;
    preProcess[0]->meanValue = 0;
    preProcess[1]->meanValue = 0;
    preProcess[2]->meanValue = 0;
    preProcess.setVariant(InferenceEngine::MEAN_VALUE);
    // Load CNNNetwork to target plugins
    auto execNet = ie->LoadNetwork(cnnNet, targetDevice, configuration);
    // Create InferRequest
    auto req = execNet.CreateInferRequest();
    auto inBlob = req.GetBlob("param");

    // Fill input
    {
        auto locketMem = inBlob->buffer();
        auto *inData = locketMem.as<float*>();
        for (size_t i = 0; i < inBlob->size(); i++)
            inData[i] = i;
    }

    req.Infer();

    // Check output
    auto outBlob = req.GetBlob(cnnNet.getOutputsInfo().begin()->first);
    {
        auto inMem = inBlob->cbuffer();
        const auto* inData = inMem.as<const float*>();
        auto outMem = outBlob->cbuffer();
        const auto* outData = outMem.as<const float*>();
        ASSERT_EQ(inBlob->size(), outBlob->size());
        for (size_t i = 0; i < inBlob->size(); i++)
            ASSERT_EQ(inData[i]*2, outData[i]);
    }
}

}  // namespace BehaviorTestsDefinitions
