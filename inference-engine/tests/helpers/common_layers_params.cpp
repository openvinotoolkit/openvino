// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <common_layers_params.hpp>
#include "tests_common.hpp"

void getConvOutShape(const std::vector<size_t>& inShape,
                    const conv_common_params& params,
                    std::vector<size_t>& outShape) {
    outShape.resize(inShape.size(), 1lu);
    outShape[0] = inShape[0];
    outShape[1] = params.out_c;
    size_t in_size = inShape.size();
    for (int i = 0; i < params.kernel.size() && i + 2 < outShape.size(); i++) {
        outShape[i + 2] =
            (inShape[i + 2] + params.pads_begin[i] + params.pads_end[i] - (params.kernel[i] - 1lu) * params.dilation[i] - 1lu) /
            params.stride[i] + 1lu;
    }
};

void getPoolOutShape(const std::vector<size_t>& inShape,
                    const pool_common_params& params,
                    std::vector<size_t>& outShape) {
    outShape.resize(inShape.size(), 1lu);
    outShape[0] = inShape[0];
    outShape[1] = inShape[1];
    size_t in_size = inShape.size();
    for (int i = 0; i < params.kernel.size() && i + 2 < outShape.size(); i++) {
        outShape[i + 2] = (inShape[i + 2] + params.pads_begin[i] + params.pads_end[i] - params.kernel[i]) / params.stride[i] + 1lu;
    }
};

size_t getConvWeightsSize(const std::vector<size_t>& inShape,
                    const conv_common_params& params,
                    const std::string& precision) {
    size_t res = 0lu;
    if (params.group != 0lu && params.out_c != 0lu && params.kernel.size() != 0lu) {
        size_t type_size = 1lu;
        if (precision == "FP32")
            type_size = sizeof(float);

        int weights_size = type_size * inShape[1] * params.out_c / params.group;
        for (size_t i = 0lu; i < params.kernel.size(); i++) {
            weights_size *= params.kernel[i];
        }
    }
    return res;
}

size_t getConvBiasesSize(const conv_common_params& params,
                        const std::string& precision) {
    size_t type_size = 1lu;
    if (precision == "FP32")
        type_size = sizeof(float);
    return params.with_bias ? type_size * params.out_c : 0lu;
}

InferenceEngine::TBlob<uint8_t>::Ptr getConvWeightsBlob(const std::vector<size_t>& inShape,
                                                        const conv_common_params& params) {
    size_t blob_size = getConvWeightsSize(inShape, params, "FP32");
    if (params.with_bias)
        blob_size += params.out_c * sizeof(float);
    InferenceEngine::TBlob<uint8_t> *weights =
            new InferenceEngine::TBlob<uint8_t>({ InferenceEngine::Precision::U8, {blob_size}, InferenceEngine::C });
    weights->allocate();
    TestsCommon::fill_data((float *) weights->buffer(), weights->size() / sizeof(float));

    return InferenceEngine::TBlob<uint8_t>::Ptr(weights);
}

InferenceEngine::TBlob<uint8_t>::Ptr getWeightsBlob(size_t size,
                                    const std::string& precision) {
    size_t type_size = 1lu;
    if (precision == "FP32")
        type_size = sizeof(float);
    InferenceEngine::TBlob<uint8_t> *weights =
            new InferenceEngine::TBlob<uint8_t>({ InferenceEngine::Precision::U8, {size * type_size}, InferenceEngine::C });
    weights->allocate();
    TestsCommon::fill_data((float *) weights->buffer(), weights->size() / sizeof(float));

    return InferenceEngine::TBlob<uint8_t>::Ptr(weights);
}

void fillStatistic(Statistic& out, size_t size, float min, float max) {
    float ampl = (max - min) / 4.f;
    float center1 = min + ampl;
    float center2 = max - ampl;
    out.min.resize(size);
    out.max.resize(size);
    TestsCommon::fill_data_sine(out.min.data(), size, center1, ampl, 1);
    TestsCommon::fill_data_sine(out.max.data(), size, center2, ampl, 1);
}
