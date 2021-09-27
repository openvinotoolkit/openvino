// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpu_case_common.hpp"
#include "ie_memcpy.h"

using RawResultsTestVpuParam = WithParamInterface<std::tuple<
        PluginDevicePair,
        Precision,
        Batch,
        DoReshape>>;

std::vector <float> operator + (std::vector <float> && l,
    const std::vector <float> & r);

//------------------------------------------------------------------------------
// class VpuNoDetectionRegression
//------------------------------------------------------------------------------

class VpuNoRawResultsRegression : public VpuNoRegressionBase,
                                  public RawResultsTestVpuParam {
public:
    //Operations
    static std::string getTestCaseName(
            TestParamInfo<RawResultsTestVpuParam ::ParamType> param);

    template <class Ctx, class Reader>
    void intoBatch(const Ctx & ctx, int batch, const Reader & rdr) ;

    template <class Ctx>
    void readInput(const Ctx & ctx, bool rgb = false);

    template <class Ctx>
    void readInputDistance(const Ctx & ctx);

    template <class Ctx>
    InferenceEngine::Blob::Ptr skipNegatives(const Ctx & ctx);

    template <class Ctx>
    InferenceEngine::Blob::Ptr dumpToBlob(const Ctx & ctx);

    template <class Ctx>
    void readInputForLPR(const Ctx & ctx);

    std::vector<float> fromBinaryFile(std::string inputTensorBinary);

protected:
    //Operations
    void SetUp() override;
    virtual void InitConfig() override;

    template <class T>
    static T Times(int n, const T & container);

    bool loadImage(const std::string &imageFilename, const InferenceEngine::Blob::Ptr &blob,
        bool bgr = true, InferenceEngine::Layout layout = InferenceEngine::Layout::NCHW);
    bool generateSeqIndLPR(InferenceEngine::Blob::Ptr &seq_ind);
    bool loadTensorDistance(InferenceEngine::Blob::Ptr blob1, const std::vector<float> &input1);
};


//------------------------------------------------------------------------------
// Implementation of template methods of class VpuNoRawResultsRegression
//------------------------------------------------------------------------------

template <class Ctx, class Reader>
void VpuNoRawResultsRegression::intoBatch(const Ctx & ctx, int batch, const Reader & rdr) {
    const auto & input = ctx.currentInputs();
    std::this_thread::sleep_for(std::chrono::milliseconds(10000));
    rdr();

    for (int i = 1; i < batch; i++) {
        auto p = const_cast<uint8_t*>(input->cbuffer().template as<const uint8_t*>());
        size_t dstSize = input->byteSize() - i * input->byteSize() / batch;
        ie_memcpy(p + i * input->byteSize() / batch, dstSize, p, input->byteSize() / batch);
    }
}

template <class Ctx>
void VpuNoRawResultsRegression::readInput(const Ctx & ctx, bool rgb) {

    loadImage(ctx.currentInputFile(), ctx.currentInputs(), rgb);
}

template <class Ctx>
void VpuNoRawResultsRegression::readInputDistance(const Ctx & ctx) {
    auto input = ctx.currentInputs();
    auto fileName = ctx.currentInputFile();

    std::string inputTensorBinary = TestDataHelpers::get_data_path() + "/vpu/InputEmbedding.bin";

    std::vector <float> inTensor = fromBinaryFile(inputTensorBinary);

    loadTensorDistance(std::const_pointer_cast<InferenceEngine::Blob>(input), inTensor);
}

template <class Ctx>
InferenceEngine::Blob::Ptr VpuNoRawResultsRegression::skipNegatives(const Ctx & ctx) {
    std::vector <float> result;
    for (auto output : ctx.outputs()) {
        for (auto x : *std::dynamic_pointer_cast<TBlob<float>>(output.second)) {
            if (x >= 0) {
                result.push_back(x);
            }
        }
    }

    return make_shared_blob<float>({Precision::FP32, C}, &result[0]);
}

template <class Ctx>
InferenceEngine::Blob::Ptr VpuNoRawResultsRegression::dumpToBlob(const Ctx & ctx) {

    std::vector <float> result;
    for (auto output : ctx.outputs()) {
        for (auto x : *std::dynamic_pointer_cast<TBlob<float>>(output.second)) {
            result.push_back(x);
        }
    }
    return make_shared_blob<float>({Precision::FP32, C}, &result[0]);
}

template <class Ctx>
void VpuNoRawResultsRegression::readInputForLPR(const Ctx & ctx) {

    if (ctx.getInputIdx() == 0) {
        const auto input = ctx.currentInputs();
        auto fileName = ctx.fileNames()[0];
        loadImage(fileName, input, true);
    } else if (ctx.getInputIdx() == 1) {
        auto seq_ind = ctx.currentInputs();
        generateSeqIndLPR(seq_ind);
    } else {
        IE_THROW() << "incorrect input index for LPRNET: " << ctx.getInputIdx();
    }
}

template <class T>
inline T VpuNoRawResultsRegression::Times(int n, const T & container) {
    T out;
    for (size_t i =0; i < n; i++) {
        out.insert(out.end(), container.begin(), container.end());
    }
    return out;
}

