// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cmath>
#include "myriad_layers_tests.hpp"

#include <sstream>

template <typename T>
std::string to_string_with_precision(const T a_value, const int n = 6)
{
    std::ostringstream out;
    out.precision(n);
    out << std::fixed << a_value;
    return out.str();
}

using namespace InferenceEngine;

#define ERROR_BOUND 1e-3f

static void refNormalize(const Blob::Ptr src,
                         Blob::Ptr dst,
                         ie_fp16* weights_data,
                         int across_spatial,
                         int channel_shared,
                         float eps) {
    ASSERT_EQ(Layout::NHWC, src->getTensorDesc().getLayout());

    auto src_data = src->buffer().as<const uint16_t*>();
    auto dst_data = dst->buffer().as<uint16_t*>();

    const auto& dims = src->getTensorDesc().getDims();
    auto N = dims[0];
    auto C = dims[1];
    auto H = dims[2];
    auto W = dims[3];

    for (size_t n = 0; n < N; ++n) {
        auto psrc = src_data + n * (C * H * W);
        auto pdst = dst_data + n * (C * H * W);

        if (across_spatial) {
            float norm = eps;
            for (size_t i = 0; i < C * H * W; ++i) {
                auto src_val = PrecisionUtils::f16tof32(psrc[i]);
                norm += src_val * src_val;
            }
            norm = 1.0f / std::sqrt(norm);

            for (size_t hw = 0; hw < H * W; ++hw) {
                for (size_t c = 0 ; c < C; ++c) {
                    auto ind = hw * C + c;

                    if (channel_shared) {
                        auto w = PrecisionUtils::f16tof32(weights_data[0]);
                        auto dst_val = PrecisionUtils::f16tof32(psrc[ind]) * norm * w;
                        pdst[ind] = PrecisionUtils::f32tof16(dst_val);
                    }
                    else {
                        auto w = PrecisionUtils::f16tof32(weights_data[c]);
                        auto dst_val = PrecisionUtils::f16tof32(psrc[ind]) * norm * w;
                        pdst[ind] = PrecisionUtils::f32tof16(dst_val);
                    }
                }
            }
        }
        else {
            for (int hw = 0; hw < H * W; ++hw) {
                float norm = eps;
                for (size_t c = 0; c < C; ++c) {
                    auto ind = hw * C + c;
                    auto src_val = PrecisionUtils::f16tof32(psrc[ind]);
                    norm += src_val * src_val;
                }
                norm = 1.0f / std::sqrt(norm);

                for (size_t c = 0; c < C; ++c) {
                    auto ind = hw * C + c;

                    if (channel_shared) {
                        auto w = PrecisionUtils::f16tof32(weights_data[0]);
                        auto dst_val = PrecisionUtils::f16tof32(psrc[ind]) * norm * w;
                        pdst[ind] = PrecisionUtils::f32tof16(dst_val);
                    }
                    else {
                        auto w = PrecisionUtils::f16tof32(weights_data[c]);
                        auto dst_val = PrecisionUtils::f16tof32(psrc[ind]) * norm * w;
                        pdst[ind] = PrecisionUtils::f32tof16(dst_val);
                    }
                }
            }
        }
    }
}

PRETTY_PARAM(AcrossSpatial, bool)
PRETTY_PARAM(ChannelSharedNormalize, bool)
PRETTY_PARAM(EPS, float)

typedef myriadLayerTestBaseWithParam<std::tuple<Dims, AcrossSpatial, ChannelSharedNormalize, EPS>> myriadLayersTestsNormalize_smoke;

TEST_P(myriadLayersTestsNormalize_smoke, Normalize) {
    tensor_test_params dims = std::get<0>(GetParam());
    int across_spatial = std::get<1>(GetParam());
    int channel_shared = std::get<2>(GetParam());
    float eps = std::get<3>(GetParam());

    SetInputTensor(dims);
    SetOutputTensor(dims);

    std::map<std::string, std::string> layer_params = {
        {"across_spatial",  std::to_string(across_spatial)},
        {"channel_shared",  std::to_string(channel_shared)},
        {"eps",             to_string_with_precision(eps, 10)}
    };

    size_t num_weights = 0;
    if (channel_shared) {
        num_weights = 1;
    }
    else {
        num_weights = dims.c;
    }
    TBlob<uint8_t>::Ptr weights(GenWeights(num_weights));

    ASSERT_NO_FATAL_FAILURE(makeSingleLayerNetwork(LayerInitParams("Normalize")
                                        .params(layer_params)
                                        .weights(weights->byteSize() / sizeof (uint16_t)),
                                        {},
                                        weights));

    ASSERT_TRUE(Infer());

    auto src = _inputMap.begin()->second;
    auto dst = _outputMap.begin()->second;
    auto weights_data = weights->data().as<ie_fp16*>();

    refNormalize(src, _refBlob, weights_data, across_spatial, channel_shared, eps);

    CompareCommonAbsolute(dst, _refBlob, ERROR_BOUND);
}
