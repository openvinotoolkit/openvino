// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cmath>

#include <blob_factory.hpp>
#include "myriad_layers_tests.hpp"
#include "myriad_layers_reference_functions.hpp"

#define ERROR_BOUND (5.e-4f)

using namespace InferenceEngine;

PRETTY_PARAM(CustomConfig, std::string);

typedef std::pair<Precision, Precision> PrecisionPair;
typedef std::tuple<SizeVector, PrecisionPair, CustomConfig> ConvertIOTestParam;
typedef std::tuple<InferenceEngine::SizeVector, Precision> ConvertWithFP16TestParam;

static CustomConfig s_CustomConfig = {
#ifdef VPU_HAS_CUSTOM_KERNELS
    {getIELibraryPath() + "/vpu_custom_kernels/customLayerBindings.xml"}
#endif
};

typedef myriadLayerTestBaseWithParam<ConvertIOTestParam> myriadLayersTestsIOConvert_smoke;

TEST_P(myriadLayersTestsIOConvert_smoke, TestsIOConvert) {
    const SizeVector& dims = std::get<0>(GetParam());
    const PrecisionPair& precision = std::get<1>(GetParam());
    const std::string& customConfig = std::get<2>(GetParam());
    const auto& inputPrecision = precision.first;
    const auto& outputPrecision = precision.second;

    if(!customConfig.empty() && !CheckMyriadX()) {
        GTEST_SKIP()<<"Custom layers for MYRIAD2 not supported";
    }
    _config[InferenceEngine::MYRIAD_CUSTOM_LAYERS] = customConfig;

    _config[InferenceEngine::MYRIAD_DISABLE_CONVERT_STAGES] = CONFIG_VALUE(YES);

    SetInputTensors({dims});
    SetOutputTensors({dims});

    std::map<std::string, std::string> params = {
        {"precision", std::to_string(outputPrecision)},
        {"scale", std::to_string(1.0)},  // scale and bias are needed for custom layer
        {"bias", std::to_string(0.0)}
    };

    ASSERT_NO_FATAL_FAILURE(makeSingleLayerNetwork(
        LayerInitParams("Convert")
            .params(params)
            .in({dims})
            .out({dims})
            .outPrecision(outputPrecision),
        NetworkInitParams()
            .layoutPreference(vpu::LayoutPreference::ChannelMajor)
            .inputPrecision(inputPrecision)
            .outputPrecision(outputPrecision)
            .lockLayout(true)));

    ASSERT_TRUE(Infer());

    ASSERT_NO_FATAL_FAILURE(ref_convert(_inputMap.begin()->second, _refBlob));

    CompareCommonAbsolute(_outputMap.begin()->second, _refBlob, ERROR_BOUND);
}

typedef myriadLayerTestBaseWithParam<ConvertWithFP16TestParam> myriadLayersTestsConvertWithFP16_smoke;

TEST_P(myriadLayersTestsConvertWithFP16_smoke, TestsConvertWithFP16)
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

std::vector<SizeVector> inputsDims = {
    // TODO: rewrite to ngraph to have reshape functionality
    // {       224, 224 },
    // {    3, 224, 224 },
    { 1, 1, 224, 224 },
    { 1, 1, 416, 416 },
    { 1, 1,  62,  62 },
    { 1, 1, 227, 227 },
    { 1, 3, 224, 224 },

    // TODO: rewrite to ngraph to have reshape functionality
    // 5D case
    // { 2, 2, 3, 224, 224 },
};

std::vector<SizeVector> inputsDims4D = {
    {{ 1, 1, 224, 224 }},
    {{ 1, 1, 416, 416 }},
    {{ 1, 1,  62,  62 }},
    {{ 1, 1, 227, 227 }},
    {{ 1, 3, 224, 224 }},
    {{ 1, 3, 360, 480 }},
};

std::vector<PrecisionPair> precisionsIO = {
    {Precision::U8,   Precision::FP16},
    {Precision::FP32, Precision::FP16},
    {Precision::FP16, Precision::FP32},
    {Precision::I32, Precision::U8}
};

std::vector<Precision> withFP16Precisions = {
    Precision::I32,
    Precision::FP32,
};
