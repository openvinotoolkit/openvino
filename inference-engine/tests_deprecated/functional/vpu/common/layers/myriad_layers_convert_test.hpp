// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cmath>

#include <blob_factory.hpp>
#include "myriad_layers_tests.hpp"
#include "myriad_layers_reference_functions.hpp"

#define ERROR_BOUND (5.e-4f)

using namespace InferenceEngine;

typedef std::pair<Precision, Precision> PrecisionPair;
typedef std::tuple<InferenceEngine::SizeVector, PrecisionPair> ConvertIOTestParam;
typedef std::tuple<InferenceEngine::SizeVector, Precision> ConvertWithFP16TestParam;

class myriadLayersTestsIOConvert_nightly: public myriadLayersTests_nightly,
                                          public testing::WithParamInterface<ConvertIOTestParam> {
};

TEST_P(myriadLayersTestsIOConvert_nightly, TestsIOConvert)
{
    const auto& param = ::testing::WithParamInterface<ConvertIOTestParam>::GetParam();
    const auto& inputDims = std::get<0>(param);
    const auto& precisions = std::get<1>(param);
    const auto& inputPrecision = precisions.first;
    const auto& outputPrecision = precisions.second;

    SetInputTensors({inputDims});
    SetOutputTensors({inputDims});

    makeSingleLayerNetwork(LayerInitParams("Copy"),
                NetworkInitParams()
                .inputPrecision(inputPrecision)
                .outputPrecision(outputPrecision));
    ASSERT_TRUE(Infer());

    auto tensorDesc = InferenceEngine::TensorDesc(
        outputPrecision, _outputMap.begin()->second->getTensorDesc().getDims(),
        _outputMap.begin()->second->getTensorDesc().getLayout());
    auto refBlob = make_blob_with_precision(outputPrecision, tensorDesc);
    refBlob->allocate();

    ref_convert(_inputMap.begin()->second, refBlob);

    CompareCommonAbsolute(_outputMap.begin()->second, refBlob, ERROR_BOUND);
}

class myriadLayersTestsConvertWithFP16_nightly: public myriadLayersTests_nightly,
                                        public testing::WithParamInterface<ConvertWithFP16TestParam> {
};

TEST_P(myriadLayersTestsConvertWithFP16_nightly, TestsConvertWithFP16)
{
    const auto& param = ::testing::WithParamInterface<ConvertWithFP16TestParam>::GetParam();
    const auto& inputDims = std::get<0>(param);
    const auto& internalPrecision = std::get<1>(param);
    const auto defaultPrecision = Precision::FP16;

    std::map<std::string, std::string> convertToInternalPrecisionParams = {
        {"precision", std::to_string(internalPrecision)}
    };
    std::map<std::string, std::string> convertFromInternalPrecisionParams = {
        {"precision", std::to_string(defaultPrecision)}
    };

    auto convertLayerToTestPrecisionParams = LayerInitParams("Convert")
            .params(convertToInternalPrecisionParams)
            .name("convert_to")
            .in({inputDims})
            .out({inputDims})
            .outPrecision(internalPrecision);

    auto convertLayerFromTestPrecisionParams = LayerInitParams("Convert")
            .params(convertFromInternalPrecisionParams)
            .name("convert_from")
            .in({inputDims})
            .out({inputDims})
            .outPrecision(defaultPrecision);

    _testNet.addLayer(convertLayerToTestPrecisionParams, ref_convert_wrap);
    _testNet.addLayer(convertLayerFromTestPrecisionParams, ref_convert_wrap);

    ASSERT_TRUE(generateNetAndInfer(NetworkInitParams()
            .inputPrecision(defaultPrecision)
            .outputPrecision(defaultPrecision)
            .runRefGraph(true)));

    CompareCommonAbsolute(_outputMap.begin()->second, getReferenceOutput(), ERROR_BOUND);
}

std::vector<InferenceEngine::SizeVector> inputsDims = {
    {       224, 224 },
    {    3, 224, 224 },
    { 1, 1, 224, 224 },
    { 1, 1, 416, 416 },
    { 1, 1,  62,  62 },
    { 1, 1, 227, 227 },
    { 1, 3, 224, 224 },

    // 5D case
    { 2, 2, 3, 224, 224 },
};

std::vector<PrecisionPair> precisionsIO = {
    {Precision::U8,   Precision::FP16},
    {Precision::FP32, Precision::FP16},
    {Precision::FP16, Precision::FP32}
};

std::vector<Precision> withFP16Precisions = {
    Precision::I32,
    Precision::FP32,
};
