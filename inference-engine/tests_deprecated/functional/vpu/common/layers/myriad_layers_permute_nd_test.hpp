// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "myriad_layers_tests.hpp"
#include "vpu/model/data_desc.hpp"

#include <iostream>

#define ERROR_BOUND 0

using namespace InferenceEngine;

namespace {

void calculateRefBlob(const Blob::Ptr& src, Blob::Ptr dst, const SizeVector& permutationVector) {
    const auto precision = src->getTensorDesc().getPrecision();
    switch (precision)
    {
        case InferenceEngine::Precision::I32:
            ref_Permute<int>(src, dst, permutationVector);
            break;
        case InferenceEngine::Precision::FP16:
            ref_Permute<ie_fp16>(src, dst, permutationVector);
            break;
        default: THROW_IE_EXCEPTION << "Unsupported precision";
    }
}

template <typename T>
void genRefData(Blob::Ptr blob) {
    ASSERT_NE(blob, nullptr);

    const auto  dataPtr    = blob->buffer().as<T*>();
    const auto& tensorDims = blob->getTensorDesc().getDims();
    const auto  precision  = blob->getTensorDesc().getPrecision();
    const auto  numDims    = tensorDims.size();

    std::function<T(float)> calculate;
    if (precision == InferenceEngine::Precision::I32) {
        calculate = [](float counter){return counter * 4; };
    }
    else if (precision == InferenceEngine::Precision::FP16) {
        calculate = [](float counter) { return PrecisionUtils::f32tof16(counter); };
    }
    else {
        calculate = [](float counter){return counter; };
    }

    SizeVector current_index(numDims);

    float counter = 0;
    const auto data_size = std::accumulate(tensorDims.begin(), tensorDims.end(), size_t{1}, std::multiplies<size_t>{});
    for (auto i = 0; i < data_size; ++i) {
        dataPtr[i] = calculate(counter);
        counter += 0.25f;
    }
}

}

using PermuteNDParams = std::tuple<InferenceEngine::SizeVector,  // input tensor sizes
                                   InferenceEngine::SizeVector,  // permutation vector
                                   IRVersion,
                                   InferenceEngine::Precision>;

class myriadLayersPermuteNDTests_nightly:
    public myriadLayersTests_nightly,
    public testing::WithParamInterface<PermuteNDParams> {
};


TEST_P(myriadLayersPermuteNDTests_nightly, Permute) {
    const auto& testParams = GetParam();
    const auto& inputTensorSizes   = std::get<0>(testParams);
    const auto& permutationVector  = std::get<1>(testParams);
    _irVersion                     = std::get<2>(testParams);
    const auto  precision          = std::get<3>(testParams);

    const auto numDims = inputTensorSizes.size();
    SizeVector outputTensorSizes(numDims);
    for (size_t i = 0; i < numDims; i++) {
        outputTensorSizes[i] = inputTensorSizes[permutationVector[i]];
    }

    const auto order = std::accumulate(std::next(permutationVector.begin()), permutationVector.end(),
                                       std::to_string(permutationVector.front()),
                                       [](std::string& res, size_t s) {
                                           return res + "," + std::to_string(s);
                                       });

    const std::map<std::string, std::string> layerParams{{"order", order}};

    _config[VPU_CONFIG_KEY(DETECT_NETWORK_BATCH)] = CONFIG_VALUE(NO);

    switch (precision)
    {
        case InferenceEngine::Precision::I32:
              _genDataCallback = genRefData<int>;
              break;
        case InferenceEngine::Precision::FP16:
              _genDataCallback = genRefData<ie_fp16>;
              break;
        default:
              VPU_THROW_EXCEPTION << "Unsupported precision";
    }

    _testNet.addLayer(LayerInitParams(_irVersion == IRVersion::v10 ? "Transpose" : "Permute")
             .params(layerParams)
             .in({inputTensorSizes})
             .out({outputTensorSizes}));

    ASSERT_TRUE(generateNetAndInfer(NetworkInitParams()
                                    .useHWOpt(CheckMyriadX())
                                    .runRefGraph(false)
                                    .inputPrecision(precision)
                                    .outputPrecision(precision)));

    const auto& inputBlob = _inputMap.begin()->second;
    const auto& outputBlob = _outputMap.begin()->second;

    calculateRefBlob(inputBlob, _refBlob, permutationVector);
    CompareCommonAbsolute(outputBlob, _refBlob, ERROR_BOUND);
}

static const std::vector<InferenceEngine::SizeVector> s_inTensors_2D = {
    {17, 19},
    { 7,  8},
    { 12, 2}
};

static const std::vector<InferenceEngine::SizeVector> s_permuteTensors_2D = {
    {0, 1},
    {1, 0},
};

static const std::vector<InferenceEngine::SizeVector> s_inTensors_3D = {
    {36, 17, 19},
    { 2,  7,  8},
    {196, 12, 2}
};
static const std::vector<InferenceEngine::SizeVector> s_permuteTensors_3D = {
    {0, 1, 2},
    {0, 2, 1},
    {2, 1, 0},
    {1, 0, 2}
};

static const std::vector<InferenceEngine::SizeVector> s_inTensors_4D = {
    {1, 36, 17, 19},
    {1,  2,  7,  8},
    {1, 196, 12, 2}
};
static const std::vector<InferenceEngine::SizeVector> s_permuteTensors_4D = {
    {0, 1, 2, 3},
    {0, 1, 3, 2},
    {0, 2, 1, 3},
    {0, 2, 3, 1},
    {0, 3, 1, 2},
    {0, 3, 2, 1}
};

static const std::vector<InferenceEngine::SizeVector> s_inTensors_5D = {
    {1, 36, 17, 19, 23},
    {1,  2,  7,  8,  9},
    {1, 196, 12, 2,  5}
};

static const std::vector<InferenceEngine::SizeVector> s_permuteTensors_5D = {
    {0, 1, 2, 3, 4},
    {0, 2, 1, 3, 4},
    {0, 3, 4, 1, 2},
    {0, 2, 3, 4, 1},
    {0, 2, 1, 3, 4},
    {0, 1, 3, 2, 4},
    {0, 1, 3, 4, 2},
    {0, 3, 1, 2, 4},
    {0, 1, 3, 2, 4},
    {0, 1, 2, 4, 3},
    {0, 4, 1, 2, 3},
    {0, 1, 4, 2, 3},
    {0, 1, 2, 4, 3}
};

static const std::vector<InferenceEngine::Precision> s_permutePrecisions = {
        InferenceEngine::Precision::I32,
        InferenceEngine::Precision::FP16,
};