// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "myriad_layers_tests.hpp"

#include <blob_factory.hpp>

#include <algorithm>
#include <random>

using namespace InferenceEngine;

class myriadLayerTestNonZero_smoke: public myriadLayersTests_nightly,
                                      public testing::WithParamInterface<SizeVector> {
public:
    void testNonZero(vpu::LayoutPreference preference, Precision precision);

protected:
    static void GenRandomNonZeroData(Blob::Ptr& blob) {
        std::mt19937 generator(DEFAULT_SEED_VALUE);

        const auto getRandomValue = [&generator]() {
            // Each third value will be a zero for test NonZero functionality
            return generator() % 3 ? float(generator()) / float(generator.max()) * 255.f : 0.f;
        };

        size_t count = blob->size();
        if (blob->getTensorDesc().getPrecision() == Precision::U8) {
            auto blobPtr = InferenceEngine::as<MemoryBlob>(blob)->rwmap().as<uint8_t*>();
            for (size_t idx = 0; idx < count; ++idx) {
                blobPtr[idx] = static_cast<uint8_t>(getRandomValue());
            }
        } else if (blob->getTensorDesc().getPrecision() == Precision::I32) {
            auto blobPtr = InferenceEngine::as<MemoryBlob>(blob)->rwmap().as<int32_t*>();
            for (size_t idx = 0; idx < count; ++idx) {
                blobPtr[idx] = static_cast<int32_t>(getRandomValue());
            }
        } else {
            auto blobPtr = InferenceEngine::as<MemoryBlob>(blob)->rwmap().as<ie_fp16*>();
            for (size_t idx = 0; idx < count; ++idx) {
                blobPtr[idx] = PrecisionUtils::f32tof16(getRandomValue());
            }
        }
    }

    static void CompareNonZero(const InferenceEngine::Blob::Ptr& outputIndicesBlob,
                               const InferenceEngine::Blob::Ptr& refIndicesBlob,
                               const InferenceEngine::Blob::Ptr& outputDimsBlob,
                               const InferenceEngine::Blob::Ptr& refDimsBlob) {
        const auto outputIndicesPtr = InferenceEngine::as<MemoryBlob>(
                outputIndicesBlob)->rmap().as<const int32_t*>();
        const auto refIndicesPtr = InferenceEngine::as<MemoryBlob>(
                refIndicesBlob)->rmap().as<const int32_t*>();
        const auto outputDimsPtr = InferenceEngine::as<MemoryBlob>(
                outputDimsBlob)->rmap().as<const int32_t*>();
        const auto refDimsPtr = InferenceEngine::as<MemoryBlob>(
                refDimsBlob)->rmap().as<const int32_t*>();

        ASSERT_EQ(outputDimsPtr[0], refDimsPtr[0]);
        ASSERT_EQ(outputDimsPtr[1], refDimsPtr[1]);

        const auto totalDimsSize = refIndicesBlob->getTensorDesc().getDims()[1];

        for (int axis = 0; axis < outputDimsPtr[0]; ++axis) {
            for (int i = 0; i < outputDimsPtr[1]; ++i) {
                const auto idx = i + axis * totalDimsSize;
                ASSERT_EQ(outputIndicesPtr[idx], refIndicesPtr[idx]);
            }
        }
    }
};

void myriadLayerTestNonZero_smoke::testNonZero(vpu::LayoutPreference preference, Precision precision) {
    _config[InferenceEngine::MYRIAD_DETECT_NETWORK_BATCH] = CONFIG_VALUE(NO);

    const auto& inputDims = GetParam();
    const size_t numDims = inputDims.size();
    const size_t inputTotalSize = std::accumulate(inputDims.begin(), inputDims.end(), 1,
                                                  std::multiplies<size_t>());

    const SizeVector outIndicesDims = {numDims, inputTotalSize};
    const SizeVector outDimsDims = {outIndicesDims.size()};

    SetInputTensors({inputDims});
    SetOutputTensors({outIndicesDims, outDimsDims});

    makeSingleLayerNetwork(LayerInitParams("StaticShapeNonZero"),
                           NetworkInitParams()
                                   .inputPrecision(precision)
                                   .outputPrecision(Precision::I32)
                                   .layoutPreference(preference));

    auto inputBlob = _inputMap.begin()->second;
    auto outputIndicesBlob = _outputMap.begin()->second;
    auto outputDimsBlob = (++_outputMap.begin())->second;

    inputBlob->getTensorDesc().setLayout(
            vpu::deviceLayout(inputBlob->getTensorDesc().getLayout(), preference));
    inputBlob->getTensorDesc().setPrecision(precision);
    GenRandomNonZeroData(inputBlob);

    auto refIndicesBlob = make_blob_with_precision(outputIndicesBlob->getTensorDesc());
    auto refDimsBlob = make_blob_with_precision(outputDimsBlob->getTensorDesc());
    refIndicesBlob->allocate();
    refDimsBlob->allocate();
    ref_nonZero(inputBlob, refIndicesBlob, refDimsBlob);

    ASSERT_TRUE(Infer());

    CompareNonZero(outputIndicesBlob, refIndicesBlob, outputDimsBlob, refDimsBlob);
}

TEST_P(myriadLayerTestNonZero_smoke, NonZero) {
    testNonZero(vpu::LayoutPreference::ChannelMajor, Precision::FP16);
}

TEST_P(myriadLayerTestNonZero_smoke, NonZeroNHWC) {
    testNonZero(vpu::LayoutPreference::ChannelMinor, Precision::FP16);
}

TEST_P(myriadLayerTestNonZero_smoke, NonZeroI32) {
    testNonZero(vpu::LayoutPreference::ChannelMajor, Precision::I32);
}

TEST_P(myriadLayerTestNonZero_smoke, NonZeroU8) {
    testNonZero(vpu::LayoutPreference::ChannelMajor, Precision::U8);
}

std::vector<InferenceEngine::SizeVector> inputDims = {
        { 7 },
        { 1000 },
        { 3, 5 },
        { 65, 33 },
        { 33, 65 },
        { 1, 1000 },
        { 223, 217, 21 },
        { 3, 4, 5, 1 },
        { 3, 4, 1, 5, 1 }
};
