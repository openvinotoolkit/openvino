// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "myriad_layers_tests.hpp"
#include <algorithm>

using namespace InferenceEngine;

static const float ERROR_BOUND = 0.0f;

struct OneHotParams {
    SizeVector dims;
    unsigned int depth;
    std::vector<int> axis;
    std::vector<float> on_value;
    std::vector<float> off_value;
};

PRETTY_PARAM(oneHot_test_params, OneHotParams);

void ref_oneHot(const InferenceEngine::Blob::Ptr src,
                InferenceEngine::Blob::Ptr dst,
                const unsigned int depth,
                const int axis,
                const float onValue,
                const float offValue)
{
    const ie_fp16 onValueFp16 = PrecisionUtils::f32tof16(onValue);
    const ie_fp16 offValueFp16 = PrecisionUtils::f32tof16(offValue);
    const auto* srcPtr = src->buffer().as<const int32_t*>();
    auto* dstPtr = dst->buffer().as<ie_fp16*>();
    const TensorDesc srcDesc = src->getTensorDesc();

    auto inputDims = srcDesc.getDims();
    std::reverse(inputDims.begin(), inputDims.end());
    const int actualAxis = (axis == -1) ? 0 : inputDims.size() - axis;

    const int prefixSize = std::accumulate(inputDims.cbegin(), inputDims.cbegin() + actualAxis, 1, std::multiplies<int>());
    const int suffixSize = src->size() / prefixSize;

    size_t dstOffset = 0;
    for (int suffixIdx = 0; suffixIdx < suffixSize; suffixIdx++) {
        for (int depthIdx = 0; depthIdx < depth; depthIdx++) {
            for (int prefixIdx = 0; prefixIdx < prefixSize; prefixIdx++) {
                const int idx = suffixIdx * prefixSize + prefixIdx;
                const size_t v = static_cast<size_t>(srcPtr[idx]);
                dstPtr[dstOffset++] = (v == depthIdx) ? onValueFp16 : offValueFp16;
            }
        }
    }
}

typedef myriadLayerTestBaseWithParam<oneHot_test_params> myriadLayerTestOneHot_nightly;

TEST_P(myriadLayerTestOneHot_nightly, OneHot) {
    _config[VPU_CONFIG_KEY(DETECT_NETWORK_BATCH)] = CONFIG_VALUE(NO);

    OneHotParams testParams = GetParam();

    int axis = -1;
    unsigned int depth = testParams.depth;
    float onValue = 1.0f;
    float offValue = 0.0f;
    SizeVector inputDims = testParams.dims;

    std::map<std::string, std::string> params;
    params["depth"] = std::to_string(depth);
    if (!testParams.axis.empty()) {
        axis = testParams.axis[0];
        params["axis"] = std::to_string(axis);
    }
    if (!testParams.on_value.empty()) {
        onValue = testParams.on_value[0];
        params["on_value"] = std::to_string(onValue);
    }
    if (!testParams.off_value.empty()) {
        offValue = testParams.off_value[0];
        params["off_value"] = std::to_string(offValue);
    }

    auto outputDims = inputDims;
    int actualAxis = axis == -1 ? inputDims.size() : axis;
    outputDims.insert(outputDims.begin() + actualAxis, depth);

    SetInputTensors({inputDims});
    SetOutputTensors({outputDims});
    makeSingleLayerNetwork(
        LayerInitParams("OneHot").params(params),
        NetworkInitParams().inputPrecision(Precision::I32).lockLayout(true));

    auto inputBlob = _inputMap.begin()->second;
    auto outputBlob = _outputMap.begin()->second;

    ASSERT_TRUE(Infer());

    ref_oneHot(inputBlob, _refBlob, depth, axis, onValue, offValue);
    CompareCommonAbsolute(outputBlob, _refBlob, ERROR_BOUND);
}
