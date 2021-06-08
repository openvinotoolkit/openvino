// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include <ie_core.hpp>
#include <blob_factory.hpp>
#include "common_test_utils/test_assertions.hpp"
#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/plugin_cache.hpp"
#include "shared_test_classes/base/layer_test_utils.hpp"
#include "functional_test_utils/blob_utils.hpp"
#include "ie_preprocess.hpp"
#include "base/behavior_test_utils.hpp"
#include "ie_ngraph_utils.hpp"

namespace BehaviorTestsDefinitions {
using PreprocessTest = BehaviorTestsUtils::BehaviorTestsBasic;

TEST_P(PreprocessTest, SetPreProcessToInputInfo) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // Create CNNNetwork from ngraph::Function
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
    // Create CNNNetwork from ngraph::Function
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

TEST_P(PreprocessTest, SetMeanImagePreProcessGetBlob) {
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

    // Create CNNNetwork from ngraph::Function
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
        auto lockedMem = inBlob->buffer();
        auto *inData = lockedMem.as<float*>();
        for (size_t i = 0; i < inBlob->size(); i++)
            inData[i] = static_cast<float>(i);
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
        for (size_t i = 0; i < inBlob->size(); i++) {
            ASSERT_EQ(inData[i] + inData[i], outData[i]);
        }
    }
}

TEST_P(PreprocessTest, SetMeanImagePreProcessSetBlob) {
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

    // Create CNNNetwork from ngraph::Function
    InferenceEngine::CNNNetwork cnnNet(ngraph);

    auto &preProcess = cnnNet.getInputsInfo().begin()->second->getPreProcess();
    preProcess.init(3);
    for (size_t i = 0; i < 3; i++) {
        auto meanData = make_blob_with_precision(
            InferenceEngine::TensorDesc(InferenceEngine::Precision::FP32,  {10, 10},
            InferenceEngine::Layout::HW));
        meanData->allocate();
        auto lockedMem = meanData->buffer();
        auto* data = lockedMem.as<float *>();
        for (size_t j = 0; j < 100; j++) {
            data[j] = 0;
            data[j] -= i * 100 + j;
        }
        ASSERT_NO_THROW(preProcess.setMeanImageForChannel(meanData, i));
    }
    preProcess.setVariant(InferenceEngine::MEAN_IMAGE);
    // Load CNNNetwork to target plugins
    auto execNet = ie->LoadNetwork(cnnNet, targetDevice, configuration);
    // Create InferRequest
    auto req = execNet.CreateInferRequest();

    auto inBlob = make_blob_with_precision(cnnNet.getInputsInfo().begin()->second->getTensorDesc());
    inBlob->allocate();
    req.SetBlob("param", inBlob);

    // Fill input
    {
        auto lockedMem = inBlob->buffer();
        auto *inData = lockedMem.as<float*>();
        for (size_t i = 0; i < inBlob->size(); i++)
            inData[i] = static_cast<float>(i);
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

TEST_P(PreprocessTest, SetMeanValuePreProcessGetBlob) {
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

    // Create CNNNetwork from ngraph::Function
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
        auto lockedMem = inBlob->buffer();
        auto *inData = lockedMem.as<float*>();
        for (size_t i = 0; i < inBlob->size(); i++)
            inData[i] = static_cast<float>(i);
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
        for (size_t i = 0; i < inBlob->size(); i++) {
            ASSERT_EQ(inData[i] + 5, outData[i]);
        }
    }
}

TEST_P(PreprocessTest, SetMeanValuePreProcessSetBlob) {
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

    // Create CNNNetwork from ngraph::Function
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

    auto inBlob = make_blob_with_precision(cnnNet.getInputsInfo().begin()->second->getTensorDesc());
    inBlob->allocate();
    req.SetBlob("param", inBlob);

    // Fill input
    {
        auto lockedMem = inBlob->buffer();
        auto *inData = lockedMem.as<float*>();
        for (size_t i = 0; i < inBlob->size(); i++)
            inData[i] = static_cast<float>(i);
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
            ASSERT_EQ(inData[i] + 5, outData[i]);
    }
}

TEST_P(PreprocessTest, ReverseInputChannelsPreProcessGetBlob) {
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

    // Create CNNNetwork from ngraph::Function
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
        auto lockedMem = inBlob->buffer();
        auto *inData = lockedMem.as<float*>();
        for (size_t i = 0; i < inBlob->size(); i++)
            inData[i] = static_cast<float>(i);
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

TEST_P(PreprocessTest, ReverseInputChannelsPreProcessSetBlob) {
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

    // Create CNNNetwork from ngraph::Function
    InferenceEngine::CNNNetwork cnnNet(ngraph);

    auto &preProcess = cnnNet.getInputsInfo().begin()->second->getPreProcess();
    preProcess.setColorFormat(InferenceEngine::ColorFormat::RGB);
    // Load CNNNetwork to target plugins
    auto execNet = ie->LoadNetwork(cnnNet, targetDevice, configuration);
    // Create InferRequest
    auto req = execNet.CreateInferRequest();

    auto inBlob = make_blob_with_precision(cnnNet.getInputsInfo().begin()->second->getTensorDesc());
    inBlob->allocate();
    req.SetBlob("param", inBlob);

    // Fill input
    {
        auto lockedMem = inBlob->buffer();
        auto *inData = lockedMem.as<float*>();
        for (size_t i = 0; i < inBlob->size(); i++)
            inData[i] = static_cast<float>(i);
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

TEST_P(PreprocessTest, SetScalePreProcessGetBlob) {
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

    // Create CNNNetwork from ngraph::Function
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
        auto lockedMem = inBlob->buffer();
        auto *inData = lockedMem.as<float*>();
        for (size_t i = 0; i < inBlob->size(); i++)
            inData[i] = static_cast<float>(i);
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
        for (size_t i = 0; i < inBlob->size(); i++) {
            ASSERT_EQ(inData[i] / 2, outData[i]);
        }
    }
}

TEST_P(PreprocessTest, SetScalePreProcessSetBlob) {
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

    // Create CNNNetwork from ngraph::Function
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

    auto inBlob = make_blob_with_precision(cnnNet.getInputsInfo().begin()->second->getTensorDesc());
    inBlob->allocate();
    req.SetBlob("param", inBlob);

    // Fill input
    {
        auto lockedMem = inBlob->buffer();
        auto *inData = lockedMem.as<float*>();
        for (size_t i = 0; i < inBlob->size(); i++)
            inData[i] = static_cast<float>(i);
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
            ASSERT_EQ(inData[i] / 2, outData[i]);
    }
}

typedef std::tuple<
        InferenceEngine::Precision,         // Network precision
        InferenceEngine::Precision,         // Set input precision
        InferenceEngine::Precision,         // Set output precision
        InferenceEngine::Layout,            // Network layout - always NCHW
        InferenceEngine::Layout,            // Set input layout
        InferenceEngine::Layout,            // Set output layout
        bool,                               // SetBlob or GetBlob for input blob
        bool,                               // SetBlob or GetBlob for output blob
        std::string,                        // Device name
        std::map<std::string, std::string>  // Config
> PreprocessConversionParams;

class PreprocessConversionTest : public testing::WithParamInterface<PreprocessConversionParams>,
                                 public CommonTestUtils::TestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<PreprocessConversionParams> obj) {
        InferenceEngine::Precision netPrecision, iPrecision, oPrecision;
        InferenceEngine::Layout netLayout, iLayout, oLayout;
        bool setInputBlob, setOutputBlob;
        std::string targetDevice;
        std::map<std::string, std::string> configuration;
        std::tie(netPrecision, iPrecision, oPrecision,
                 netLayout, iLayout, oLayout,
                 setInputBlob, setOutputBlob,
                 targetDevice, configuration) = obj.param;
        std::ostringstream result;
        result << "netPRC=" << netPrecision.name() << "_";
        result << "iPRC=" << iPrecision.name() << "_";
        result << "oPRC=" << oPrecision.name() << "_";
        result << "netLT=" << netLayout << "_";
        result << "iLT=" << iLayout << "_";
        result << "oLT=" << oLayout << "_";
        result << "setIBlob=" << setInputBlob << "_";
        result << "setOBlob=" << setOutputBlob << "_";
        result << "targetDevice=" << targetDevice;
        if (!configuration.empty()) {
            for (auto& configItem : configuration) {
                result << "configItem=" << configItem.first << "_" << configItem.second << "_";
            }
        }
        return result.str();
    }

    static InferenceEngine::Layout getOppositeLayout(InferenceEngine::Layout l) {
        if (InferenceEngine::Layout::NCHW == l) {
            return InferenceEngine::Layout::NHWC;
        } else if (InferenceEngine::Layout::NHWC == l) {
            return InferenceEngine::Layout::NCHW;
        }
        return InferenceEngine::Layout::ANY;
    }

    static InferenceEngine::Precision getOppositePrecision(InferenceEngine::Precision p) {
        if (InferenceEngine::Precision::U8 == p) {
            return InferenceEngine::Precision::FP32;
        } else if (InferenceEngine::Precision::FP32 == p) {
            return InferenceEngine::Precision::U8;
        }
        return InferenceEngine::Precision::UNSPECIFIED;
    }

    void SetUp()  override {
        std::tie(netPrecision, iPrecision, oPrecision,
                 netLayout, iLayout, oLayout,
                 setInputBlob, setOutputBlob,
                 targetDevice, configuration) = this->GetParam();
    }

    void TearDown() override {
        if (!configuration.empty()) {
            PluginCache::get().reset();
        }
    }

    std::shared_ptr<InferenceEngine::Core> ie = PluginCache::get().ie();
    InferenceEngine::Precision netPrecision, iPrecision, oPrecision;
    InferenceEngine::Layout netLayout, iLayout, oLayout;
    bool setInputBlob, setOutputBlob;
    std::string targetDevice;
    std::map<std::string, std::string> configuration;
};

TEST_P(PreprocessConversionTest, Infer) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    std::shared_ptr<ngraph::Function> ngraph;
    unsigned int shape_size = 9, channels = 3, batch = 1, offset = 0;
    {
        ngraph::PartialShape shape({batch, channels, shape_size, shape_size});
        ngraph::element::Type type(InferenceEngine::details::convertPrecision(netPrecision));
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

    // Create CNNNetwork from ngraph::Function
    InferenceEngine::CNNNetwork cnnNet(ngraph);

    cnnNet.getInputsInfo().begin()->second->setPrecision(iPrecision);
    cnnNet.getInputsInfo().begin()->second->setLayout(iLayout);
    cnnNet.getOutputsInfo().begin()->second->setPrecision(oPrecision);
    cnnNet.getOutputsInfo().begin()->second->setLayout(oLayout);

    // Load CNNNetwork to target plugins
    auto execNet = ie->LoadNetwork(cnnNet, targetDevice, configuration);
    // Create InferRequest
    auto req = execNet.CreateInferRequest();

    // unsigned int stride = shape_size + offset;
    // std::vector<float> blobData(batch * channels * stride * stride, 0);
    // InferenceEngine::BlockingDesc blockDesc({ batch, shape_size, shape_size, channels },
    //     { 0, 2, 3, 1 },
    //       0,
    //     { 0, 0, 0, 0 },
    //     { channels * stride * stride, channels * stride, channels, 1 });
    // InferenceEngine::TensorDesc desc(
    //         InferenceEngine::Precision::FP32,
    //         { batch, channels, shape_size, shape_size }, blockDesc);
    (void)offset;

    InferenceEngine::Blob::Ptr inBlob = nullptr, outBlob = nullptr;

    if (setInputBlob) {
        inBlob = make_blob_with_precision(cnnNet.getInputsInfo().begin()->second->getTensorDesc());
        inBlob->allocate();
        req.SetBlob("param", inBlob);
    } else {
        inBlob = req.GetBlob("param");
    }

    if (setOutputBlob) {
        outBlob = make_blob_with_precision(cnnNet.getOutputsInfo().begin()->second->getTensorDesc());
        outBlob->allocate();
        req.SetBlob("relu", outBlob);
    } else {
        outBlob = req.GetBlob("relu");
    }

    // Fill input
    {
        auto lockedMem = inBlob->buffer();
        auto desc = inBlob->getTensorDesc();

        if (desc.getPrecision() == InferenceEngine::Precision::FP32) {
            auto *inData = lockedMem.as<float*>();
            for (size_t i = 0; i < inBlob->size(); i++)
                inData[desc.offset(i)] = static_cast<float>(i);
        } else if (desc.getPrecision() == InferenceEngine::Precision::U8) {
            auto *inData = lockedMem.as<std::uint8_t*>();
            for (size_t i = 0; i < inBlob->size(); i++)
                inData[desc.offset(i)] = static_cast<std::uint8_t>(i);
        } else {
            ASSERT_TRUE(false);
        }
    }

    req.Infer();

    // Check output
    {
        auto outMem = outBlob->cbuffer();
        auto desc = outBlob->getTensorDesc();

        if (desc.getPrecision() == InferenceEngine::Precision::FP32) {
            const auto* outData = outMem.as<const float *>();
            ASSERT_EQ(inBlob->size(), outBlob->size());
            for (size_t i = 0; i < inBlob->size(); i++)
                ASSERT_EQ(i, outData[desc.offset(i)]) << i;
        } else if (desc.getPrecision() == InferenceEngine::Precision::U8) {
            const auto* outData = outMem.as<const std::uint8_t *>();
            ASSERT_EQ(inBlob->size(), outBlob->size());
            for (size_t i = 0; i < inBlob->size(); i++)
                ASSERT_EQ(i, outData[desc.offset(i)]) << i;
        } else {
            ASSERT_TRUE(false);
        }
    }
}

typedef std::tuple<
        InferenceEngine::Precision,         // Network precision
        bool,                               // Change input precision
        bool,                               // Change output precision
        InferenceEngine::Layout,            // Network layout - always NCHW
        bool,                               // Change input layout
        bool,                               // Change output layout
        bool,                               // SetBlob or GetBlob for input blob
        bool,                               // SetBlob or GetBlob for output blob
        std::string,                        // Device name
        std::map<std::string, std::string>  // Config
> PreprocessSetBlobCheckParams;

class PreprocessDynamicallyInSetBlobTest : public testing::WithParamInterface<PreprocessSetBlobCheckParams>,
                                   public CommonTestUtils::TestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<PreprocessSetBlobCheckParams> obj) {
        InferenceEngine::Precision netPrecision;
        InferenceEngine::Layout netLayout;
        bool changeIPrecision, changeOPrecision;
        bool changeILayout, changeOLayout;
        bool setInputBlob, setOutputBlob;
        std::string targetDevice;
        std::map<std::string, std::string> configuration;
        std::tie(netPrecision, changeIPrecision, changeOPrecision,
                 netLayout, changeILayout, changeOLayout,
                 setInputBlob, setOutputBlob,
                 targetDevice, configuration) = obj.param;
        std::ostringstream result;
        result << "netPRC=" << netPrecision.name() << "_";
        result << "iPRC=" << changeIPrecision << "_";
        result << "oPRC=" << changeOPrecision << "_";
        result << "netLT=" << netLayout << "_";
        result << "iLT=" << changeILayout << "_";
        result << "oLT=" << changeOLayout << "_";
        result << "setIBlob=" << setInputBlob << "_";
        result << "setOBlob=" << setOutputBlob << "_";
        result << "targetDevice=" << targetDevice;
        if (!configuration.empty()) {
            for (auto& configItem : configuration) {
                result << "configItem=" << configItem.first << "_" << configItem.second << "_";
            }
        }
        return result.str();
    }

    InferenceEngine::Layout getOppositeLayout(InferenceEngine::Layout l) {
        if (InferenceEngine::Layout::NCHW == l) {
            return InferenceEngine::Layout::NHWC;
        } else if (InferenceEngine::Layout::NHWC == l) {
            return InferenceEngine::Layout::NCHW;
        }
        return InferenceEngine::Layout::ANY;
    }

    InferenceEngine::Precision getOppositePrecision(InferenceEngine::Precision p) {
        if (InferenceEngine::Precision::U8 == p) {
            return InferenceEngine::Precision::FP32;
        } else if (InferenceEngine::Precision::FP32 == p) {
            return InferenceEngine::Precision::U8;
        }
        return InferenceEngine::Precision::UNSPECIFIED;
    }

    void SetUp()  override {
        std::tie(netPrecision, changeIPrecision, changeOPrecision,
                 netLayout, changeILayout, changeOLayout,
                 setInputBlob, setOutputBlob,
                 targetDevice, configuration) = this->GetParam();
    }

    void TearDown() override {
        if (!configuration.empty()) {
            PluginCache::get().reset();
        }
    }

    std::shared_ptr<InferenceEngine::Core> ie = PluginCache::get().ie();
    InferenceEngine::Precision netPrecision;
    bool changeIPrecision, changeOPrecision;
    InferenceEngine::Layout netLayout;
    bool changeILayout, changeOLayout;
    bool setInputBlob, setOutputBlob;
    std::string targetDevice;
    std::map<std::string, std::string> configuration;
};

TEST_P(PreprocessDynamicallyInSetBlobTest, Infer) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    std::shared_ptr<ngraph::Function> ngraph;
    unsigned int shape_size = 9, channels = 3, batch = 1;
    {
        ngraph::PartialShape shape({batch, channels, shape_size, shape_size});
        ngraph::element::Type type(InferenceEngine::details::convertPrecision(netPrecision));
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

    // Create CNNNetwork from ngraph::Function
    InferenceEngine::CNNNetwork cnnNet(ngraph);

    // Load CNNNetwork to target plugins
    auto execNet = ie->LoadNetwork(cnnNet, targetDevice, configuration);
    auto req = execNet.CreateInferRequest();
    InferenceEngine::Blob::Ptr inBlob = nullptr, outBlob = nullptr;

    // create input blob

    auto recreateInputBlob = [&] (InferenceEngine::Blob::Ptr & _inBlob) {
        auto desc = cnnNet.getInputsInfo().begin()->second->getTensorDesc();
        desc = InferenceEngine::TensorDesc(
            changeIPrecision ? getOppositePrecision(desc.getPrecision()) : desc.getPrecision(),
            desc.getDims(),
            changeILayout ? getOppositeLayout(desc.getLayout()) : desc.getLayout());
        auto tempBlob = make_blob_with_precision(desc);
        tempBlob->allocate();

        _inBlob = std::move(tempBlob);
    };

    if (setInputBlob) {
        recreateInputBlob(inBlob);
        if (changeIPrecision) {
            EXPECT_THROW(req.SetBlob("param", inBlob), InferenceEngine::ParameterMismatch);
            // fallback
            inBlob = req.GetBlob("param");
        } else {
            EXPECT_NO_THROW(req.SetBlob("param", inBlob));
        }
    } else {
        inBlob = req.GetBlob("param");
        recreateInputBlob(inBlob);
    }

    // Fill input
    {
        auto lockedMem = inBlob->buffer();
        auto desc = inBlob->getTensorDesc();

        if (desc.getPrecision() == InferenceEngine::Precision::FP32) {
            auto *inData = lockedMem.as<float*>();
            for (size_t i = 0; i < inBlob->size(); i++)
                inData[desc.offset(i)] = static_cast<float>(i);
        } else if (desc.getPrecision() == InferenceEngine::Precision::U8) {
            auto *inData = lockedMem.as<std::uint8_t*>();
            for (size_t i = 0; i < inBlob->size(); i++)
                inData[desc.offset(i)] = static_cast<std::uint8_t>(i);
        } else {
            ASSERT_TRUE(false);
        }
    }

    // create output blob

    auto recreateOutputBlob = [&] (InferenceEngine::Blob::Ptr & _outBlob) {
        auto desc = cnnNet.getOutputsInfo().begin()->second->getTensorDesc();
        desc = InferenceEngine::TensorDesc(
            changeOPrecision ? getOppositePrecision(desc.getPrecision()) : desc.getPrecision(),
            desc.getDims(),
            changeOLayout ? getOppositeLayout(desc.getLayout()) : desc.getLayout());
        auto tempBlob = make_blob_with_precision(desc);
        tempBlob->allocate();

        _outBlob = std::move(tempBlob);
    };

    if (setOutputBlob) {
        recreateOutputBlob(outBlob);
        if (changeOPrecision) {
            ASSERT_THROW(req.SetBlob("relu", outBlob), InferenceEngine::ParameterMismatch);
            // fallback
            outBlob = req.GetBlob("relu");
        } else {
            ASSERT_NO_THROW(req.SetBlob("relu", outBlob));
        }
    } else {
        outBlob = req.GetBlob("relu");
        recreateOutputBlob(outBlob);
    }

    if (setOutputBlob && setInputBlob) {
        ASSERT_NO_THROW(req.Infer());
    } else {
        // TODO: if blob from GetBlob is re-created, no checks are performed
        // should be "GetBlob re-creation error mismatch"
        // EXPECT_THROW(req.Infer(), InferenceEngine::Exception);

        ASSERT_NO_THROW(req.Infer());
    }

    // Check output
    {
        auto outMem = outBlob->cbuffer();
        auto desc = outBlob->getTensorDesc();

        if (desc.getPrecision() == InferenceEngine::Precision::FP32) {
            const auto* outData = outMem.as<const float *>();
            ASSERT_EQ(inBlob->size(), outBlob->size());
            for (size_t i = 0; i < inBlob->size(); i++)
                ASSERT_EQ(i, outData[desc.offset(i)]) << i;
        } else if (desc.getPrecision() == InferenceEngine::Precision::U8) {
            const auto* outData = outMem.as<const std::uint8_t *>();
            ASSERT_EQ(inBlob->size(), outBlob->size());
            for (size_t i = 0; i < inBlob->size(); i++)
                ASSERT_EQ(i, outData[desc.offset(i)]) << i;
        } else {
            ASSERT_TRUE(false);
        }
    }
}

}  // namespace BehaviorTestsDefinitions
