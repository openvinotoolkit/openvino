// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "myriad_layers_tests.hpp"


#define ERROR_BOUND 1e-3f

static void refLRN(const InferenceEngine::Blob::Ptr src,
                         InferenceEngine::Blob::Ptr dst,
                         uint32_t local_size,
                         float alpha,
                         float beta,
                         float k) {
    ASSERT_NE(src, nullptr);
    ASSERT_NE(dst, nullptr);
    const uint16_t *src_data = src->buffer();
    uint16_t *dst_data = dst->buffer();
    ASSERT_NE(src_data, nullptr);
    ASSERT_NE(dst_data, nullptr);
    int32_t IW = 1;
    int32_t IH = 1;
    int32_t IC = 1;
    get_dims(src, IW, IH, IC);
    for (uint32_t h = 0; h < IH; h++) {
        for (uint32_t w = 0; w < IW; w++) {
            for (uint32_t c = 0; c < IC; c++) {
                uint32_t oidx = c + w * IC + h * IC * IW;
                uint32_t sz = local_size;
                int32_t c_start = c - sz / 2;
                int32_t c_end = c_start + sz;
                c_start = std::max(c_start, 0);
                c_end   = std::min(c_end, (int32_t)IC);
                float sum = 0.0;
                for (int32_t c1 = c_start; c1 < c_end; c1++) {
                    uint32_t idx = c1 + w * IC + h * IC * IW;
                    float s =InferenceEngine::PrecisionUtils::f16tof32(src_data[idx]);
                    sum += s * s;
                }
                float norm_coef = powf(k + alpha * sum / sz, -beta);

                dst_data[oidx] = InferenceEngine::PrecisionUtils::f32tof16(norm_coef *
                                        InferenceEngine::PrecisionUtils::f16tof32(src_data[oidx]));
           }
        }
    }
}

static void refInnerLRN(const InferenceEngine::Blob::Ptr src,
                         InferenceEngine::Blob::Ptr dst,
                         uint32_t local_size,
                         float alpha,
                         float beta,
                         float k) {
    ASSERT_NE(src, nullptr);
    ASSERT_NE(dst, nullptr);
    const uint16_t *src_data = src->buffer();
    uint16_t *dst_data = dst->buffer();
    ASSERT_NE(src_data, nullptr);
    ASSERT_NE(dst_data, nullptr);
    int32_t IW = 1;
    int32_t IH = 1;
    int32_t IC = 1;
    get_dims(src, IW, IH, IC);
    for (uint32_t h = 0; h < IH; h++) {
        for (uint32_t w = 0; w < IW; w++) {
            for (uint32_t c = 0; c < IC; c++) {
                uint32_t oidx = c + w * IC + h * IC * IW;
                uint32_t sz = local_size;
                int32_t h_start = h - sz / 2;
                int32_t h_end = h + sz / 2;
                int32_t w_start = w - sz / 2;
                int32_t w_end = w + sz / 2;
                h_start = std::max(h_start, 0);
                h_end   = std::min(h_end, (int32_t)IH - 1);
                w_start = std::max(w_start, 0);
                w_end   = std::min(w_end, (int32_t)IW - 1);
                float sum = 0;
                for (int32_t h1 = h_start; h1 <= h_end; h1++) {
                    for (int32_t w1 = w_start; w1 <= w_end; w1++) {
                        uint32_t idx = c + w1 * IC + h1 * IC * IW;
                        float s = InferenceEngine::PrecisionUtils::f16tof32(src_data[idx]);
                        sum += s * s;
                    }
                }
                float norm_coef = powf(k + alpha * sum / (float)(sz * sz), -beta);

                dst_data[oidx] = InferenceEngine::PrecisionUtils::f32tof16(norm_coef *
                                        InferenceEngine::PrecisionUtils::f16tof32(src_data[oidx]));
            }
        }
    }
}

PRETTY_PARAM(local_size, uint32_t)
PRETTY_PARAM(k_val, float)
PRETTY_PARAM(alpha, float)
PRETTY_PARAM(beta,  float)

typedef myriadLayerTestBaseWithParam<std::tuple<Dims, local_size, k_val, alpha, beta>> myriadLayersTestsLRN_nightly;

TEST_P(myriadLayersTestsLRN_nightly, LRN) {
    tensor_test_params dims = std::get<0>(GetParam());
    uint32_t local_v = std::get<1>(GetParam());
    float k          = std::get<2>(GetParam());
    float alpha_val  = std::get<3>(GetParam());
    float beta_val   = std::get<4>(GetParam());

    SetInputTensor(dims);
    SetOutputTensor(dims);

    std::map<std::string, std::string> layer_params = {
        {"alpha",     std::to_string(alpha_val)},
        {"beta",      std::to_string(beta_val)},
        {"local-size", std::to_string(local_v)},
        {"k", std::to_string(k)},
        {"region", "Across"},
    };

    ASSERT_NO_FATAL_FAILURE(makeSingleLayerNetwork(LayerInitParams("Norm").params(layer_params)));

    ASSERT_TRUE(Infer());
    auto src = _inputMap.begin()->second;
    auto dst = _outputMap.begin()->second;
    refLRN(src, _refBlob, local_v, alpha_val, beta_val, k);
    
    CompareCommonAbsolute(dst, _refBlob, ERROR_BOUND);
}

TEST_P(myriadLayersTestsLRN_nightly, InnerLRN) {
    tensor_test_params dims = std::get<0>(GetParam());
    uint32_t local_v = std::get<1>(GetParam());
    float k          = std::get<2>(GetParam());
    float alpha_val  = std::get<3>(GetParam());
    float beta_val   = std::get<4>(GetParam());

    SetInputTensor(dims);
    SetOutputTensor(dims);

    std::map<std::string, std::string> layer_params = {
        {"alpha",     std::to_string(alpha_val)},
        {"beta",      std::to_string(beta_val)},
        {"local-size", std::to_string(local_v)},
        {"k", std::to_string(k)},
        {"region", "Same"},
    };

    ASSERT_NO_FATAL_FAILURE(makeSingleLayerNetwork(LayerInitParams("Norm").params(layer_params)));

    ASSERT_TRUE(Infer());
    auto src = _inputMap.begin()->second;
    auto dst = _outputMap.begin()->second;
    refInnerLRN(src, _refBlob, local_v, alpha_val, beta_val, k);
    
    CompareCommonAbsolute(dst, _refBlob, ERROR_BOUND);
}

static std::vector<Dims> s_LRNTensors = {
    {{1, 4, 16, 32}},
    {{1, 8, 20, 36}},
};

static std::vector<local_size> s_LRNlocal_size = {
    3, 5, /*1*/ // local_size = 1 is committed because mvTensor returns "junk" values in some output positions, but InnerLRN return correct values
};

static std::vector<alpha> s_LRNalpha = {
    9.9999997e-05f,
};

static std::vector<beta> s_LRNbeta = {
    0.75,
};

static std::vector<k_val> s_LRN_K = {
    1, 3, 5, 7
};
