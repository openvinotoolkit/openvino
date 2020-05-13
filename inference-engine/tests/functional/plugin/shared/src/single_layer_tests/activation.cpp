// Copyright (C) 2020 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#include <tuple>
#include <string>
#include <vector>
#include <functional>
#include <cmath>
#include <memory>
#include <functional_test_utils/skip_tests_config.hpp>

#include "ie_core.hpp"
#include "ie_precision.hpp"

#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/blob_utils.hpp"
#include "functional_test_utils/plugin_cache.hpp"
#include "functional_test_utils/layer_test_utils.hpp"

#include "single_layer_tests/activation.hpp"

namespace LayerTestsDefinitions {

std::string ActivationLayerTest::getTestCaseName(const testing::TestParamInfo<activationParams> &obj) {
    InferenceEngine::Precision inputPrecision, netPrecision;
    InferenceEngine::SizeVector inputShapes;
    std::string targetDevice;
    ngraph::helpers::ActivationTypes activationType;
    std::tie(activationType,
             inputPrecision,
             netPrecision,
             inputShapes,
             targetDevice) = obj.param;

    std::ostringstream result;
    const char separator = '_';
    result << activationNames[activationType] << separator;
    result << "IS=" << CommonTestUtils::vec2str(inputShapes) << separator;
    result << "inPRC=" << inputPrecision.name() << separator;
    result << "netPRC=" << netPrecision.name() << separator;
    result << "targetDevice=" << targetDevice;
    return result.str();
}

void ActivationLayerTest::SetUp() {
    std::tie(activationType, inputPrecision, netPrecision, inputShapes, targetDevice) = GetParam();
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    auto params = ngraph::builder::makeParams(ngPrc, {inputShapes});
    auto activation = ngraph::builder::makeActivation(params[0], ngPrc, activationType);
    fnPtr = std::make_shared<ngraph::Function>(ngraph::NodeVector{activation}, params);
}

TEST_P(ActivationLayerTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    InferenceEngine::CNNNetwork cnnNet(fnPtr);
    setNetInOutPrecision(cnnNet, inputPrecision);
    std::string inputName = cnnNet.getInputsInfo().begin()->first;
    std::string outputName = cnnNet.getOutputsInfo().begin()->first;
    auto ie = PluginCache::get().ie();
    auto execNet = ie->LoadNetwork(cnnNet, targetDevice);
    auto a = execNet.GetInputsInfo();
    auto req = execNet.CreateInferRequest();

    uint32_t data_range = 20;
    int32_t data_start_from = -10;
    if (!inputPrecision.isSigned()) {
        data_range = 15;
        data_start_from = 0;
    }
    if (activationType == ngraph::helpers::ActivationTypes::Exp && targetDevice == CommonTestUtils::DEVICE_GNA) {
        const double max_result_on_GNA = 15.9;
        const double exp_inverse = std::round(std::log(max_result_on_GNA));
        if (inputPrecision.isSigned()) {
            data_range = exp_inverse * 2.0;
            data_start_from = -exp_inverse;
        } else {
            data_range = exp_inverse;
            data_start_from = 0;
        }
    }

    InferenceEngine::Blob::Ptr inBlob = FuncTestUtils::createAndFillBlob({inputPrecision,
                                                                          inputShapes,
                                                                          InferenceEngine::TensorDesc::getLayoutByDims(
                                                                                  inputShapes)},
                                                                         data_range,
                                                                         data_start_from,
                                                                                          32768);
    req.SetBlob(inputName, inBlob);
    req.Infer();
    auto outBlob = req.GetBlob(outputName);
    std::vector<const float *> inRawData;
    InferenceEngine::Blob::Ptr castedBlob = inBlob;
    if (inputPrecision != InferenceEngine::Precision::FP32) {
        castedBlob = FuncTestUtils::copyBlobWithCast<InferenceEngine::Precision::FP32>(inBlob);
    }
    inRawData.push_back(castedBlob->cbuffer().as<float *>());

    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    convertFuncToF32(fnPtr, netPrecision);
    auto refOutData = ngraph::helpers::inferFnWithInterp<ngraph::element::Type_t::f32>(fnPtr, inRawData);
    float thr1, thr2;
    FuncTestUtils::GetComparisonThreshold(netPrecision, thr1, thr2);

    size_t outElementsCount = std::accumulate(begin(fnPtr->get_output_shape(0)), end(fnPtr->get_output_shape(0)), 1,
                                              std::multiplies<size_t>());
    FuncTestUtils::compareRawBuffers(outBlob->cbuffer().as<float *>(), *refOutData[0],
                                                     outElementsCount, outElementsCount,
                                                     FuncTestUtils::CompareType::ABS_AND_REL,
                                                     thr1, thr2);
    fnPtr.reset();
    if (targetDevice.find(CommonTestUtils::DEVICE_GPU) != std::string::npos) {
        PluginCache::get().reset();
    }
}

}  // namespace LayerTestsDefinitions
