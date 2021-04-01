// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "myriad_layers_tests.hpp"

#define ERROR_BOUND (1.e-2f)
#define WEIGHTS_BOUND (12.f)

using namespace InferenceEngine;

struct bn_test_params {
    tensor_test_params in;
    float epsilon;
    friend std::ostream& operator<<(std::ostream& os, bn_test_params const& tst)
    {
        return os << tst.in
                  << ", epsilon=" << tst.epsilon;
    };
};

void ref_batch_normalization(const InferenceEngine::Blob::Ptr src,
                      const uint16_t *weights, const size_t weightsSize,
                      InferenceEngine::Blob::Ptr dst, float eps)
{
    ASSERT_NE(src, nullptr);
    ASSERT_NE(weights, nullptr);
    ASSERT_NE(dst, nullptr);

    const auto& src_dims = src->getTensorDesc().getDims();

    size_t IW = src_dims[3];
    size_t IH = src_dims[2];
    size_t IC = src_dims[1];

    std::vector<float> new_weights(IC);
    std::vector<float> new_bias(IC);

    const uint16_t *src_data = src->buffer();
    const uint16_t *weights_data = weights;
    const uint16_t *bias_data = weights_data + IC;
    uint16_t *dst_data = dst->buffer();
    for (size_t ic = 0; ic < IC; ic++) {
        float val = PrecisionUtils::f16tof32(weights_data[ic]) + eps;
        val = 1.0f/sqrt(val);
        new_weights[ic] = val;
        new_bias[ic] = -val * PrecisionUtils::f16tof32(bias_data[ic]);
    }
    for (size_t ic = 0; ic < IC; ic++) {
        float val = new_bias[ic];
        for (size_t kh = 0; kh < IH; kh++) {
            for (size_t  kw = 0; kw < IW; kw++) {
                size_t iidx = ic + kw * IC + kh * IC * IW;
                float res = val + PrecisionUtils::f16tof32(src_data[iidx]) * new_weights[ic];
                dst_data[iidx] = PrecisionUtils::f32tof16(res);
            }
        }
    }
}

class myriadLayersTestsBatchNormalization_smoke: public myriadLayersTests_nightly,
                           public testing::WithParamInterface<bn_test_params> {
};

TEST_P(myriadLayersTestsBatchNormalization_smoke, TestsBatchNorm)
{
    bn_test_params p = ::testing::WithParamInterface<bn_test_params>::GetParam();
    size_t sz_weights = p.in.c;
    size_t sz_bias = p.in.c;
    size_t sz = sz_weights + sz_bias;
    InferenceEngine::TBlob<uint8_t>::Ptr weights_ptr = InferenceEngine::TBlob<uint8_t>::Ptr(GenWeights(sz, -WEIGHTS_BOUND, WEIGHTS_BOUND));
    uint16_t* weights = weights_ptr->data().as<uint16_t*>();
    for (int i = 0; i < sz_weights; ++i) {
        /* weights are variations so all gains should be >= 0 */
        weights[i] = PrecisionUtils::f32tof16(fabs(PrecisionUtils::f16tof32(weights[i])));
    }
    IN_OUT_desc inpt = {{p.in.n, p.in.c, p.in.h, p.in.w}};
    SetInputTensors(inpt);
    SetOutputTensors(inpt);
    std::map<std::string, std::string> params;
    params["epsilon"] = std::to_string(p.epsilon);

    ASSERT_NO_FATAL_FAILURE(makeSingleLayerNetwork(LayerInitParams("BatchNormalization")
                                        .params(params)
                                        .weights(sz_weights)
                                        .biases(sz_bias),
                                        {},
                                        weights_ptr));
    ASSERT_TRUE(Infer());
    ref_batch_normalization(_inputMap.begin()->second, weights, sz, _refBlob, p.epsilon);
    CompareCommonWithNorm(_outputMap.begin()->second, _refBlob, ERROR_BOUND);
}
