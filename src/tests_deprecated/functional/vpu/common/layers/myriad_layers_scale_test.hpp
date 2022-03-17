// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include "myriad_layers_tests.hpp"
#include "vpu/model/data_desc.hpp"

#define ERROR_BOUND (5.e-3f)

using namespace InferenceEngine;

namespace {
    bool iter(SizeVector& in, SizeVector& out)
    {
        bool flag = true;
        for(int i = 0; i < out.size(); i++) {
            if(in[i] < out[i] - 1) {
                in[i]++;
                break;
            } else {
                if(i == out.size() - 1) {
                    flag = false;
                    break;
                }
                in[i] = 0;
            }
        }
        return flag;
    }

    int calcOffset(SizeVector& in, SizeVector& out)
    {
        int offset = in.back();
        for(int i = in.size() - 2; i >= 0; i--) {
            int mul = in[i];
            for(int j = i + 1; j < out.size(); j++)
                mul *= out[j];
            offset += mul;
        }
        return offset;
    }
}

void ref_scale(const InferenceEngine::Blob::Ptr src,
                      const uint16_t *weights,
                      InferenceEngine::Blob::Ptr dst,
                      bool bias) {
    ASSERT_NE(src, nullptr);
    ASSERT_NE(dst, nullptr);

    SizeVector in_size;
    SizeVector out_size;
    in_size = src->getTensorDesc().getDims();
    out_size = dst->getTensorDesc().getDims();
    Layout layout = src->getTensorDesc().getLayout();
    int dims = in_size.size();
    int dimC = dimToIeInd(vpu::Dim::C, dims);
    SizeVector curr_size(dims);
    const uint16_t *src_data = src->buffer();
    const uint16_t *bias_data = weights + in_size[dimC];
    uint16_t *dst_data = dst->buffer();
    // TODO: investigate this case
    if (layout == NCHW || layout == NHWC) {
        size_t N1 = out_size[0];
        size_t C1 = out_size[1];
        size_t H1 = out_size[2];
        size_t W1 = out_size[3];
        for (size_t n = 0; n < N1; n++) {
            for (size_t c = 0; c < C1; c++) {
                float val = 0.0f;
                if (bias)
                    val = PrecisionUtils::f16tof32(bias_data[c]);
                for (size_t h = 0; h < H1; h++) {
                    for (size_t w = 0; w < W1; w++) {
                        size_t iidx = layout == NCHW ?
                                           w + h * W1 + c * W1 * H1 + n * W1 * H1 * C1 :
                                           c + w * C1 + h * C1 * W1 + n * W1 * H1 * C1;
                        float res = val + PrecisionUtils::f16tof32(src_data[iidx]) *
                                PrecisionUtils::f16tof32(weights[c]);
                        dst_data[iidx] = PrecisionUtils::f32tof16(res);
                    }
                }
            }
        }
    } else {
        do {
            float val = 0.0f;
            if (bias)
                val = PrecisionUtils::f16tof32(bias_data[curr_size[dimC]]);
            float res = val + PrecisionUtils::f16tof32(src_data[calcOffset(curr_size, in_size)]) *
                              PrecisionUtils::f16tof32(weights[curr_size[dimC]]);
            dst_data[calcOffset(curr_size, out_size)] = PrecisionUtils::f32tof16(res);
        } while(iter(curr_size, out_size));
    }
}

typedef std::tuple<SizeVector, bool> TestScaleShift;

class myriadLayersTestsScale_smoke: public myriadLayersTests_nightly,
                              public testing::WithParamInterface<TestScaleShift> {
};

TEST_P(myriadLayersTestsScale_smoke, TestsScale)
{
    _config[InferenceEngine::MYRIAD_DETECT_NETWORK_BATCH] = CONFIG_VALUE(NO);

    SizeVector p = std::get<0>(::testing::WithParamInterface<TestScaleShift>::GetParam());
    bool biasAdd = std::get<1>(::testing::WithParamInterface<TestScaleShift>::GetParam());
    auto dims = p.size();
    int dimC = dimToIeInd(vpu::Dim::C, dims);
    size_t sz_weights = p[dimC];
    size_t sz_bias = p[dimC] * biasAdd;
    size_t sz = sz_weights + sz_bias;
    InferenceEngine::TBlob<uint8_t>::Ptr weights_ptr = InferenceEngine::TBlob<uint8_t>::Ptr(GenWeights(sz));
    uint16_t* weights = weights_ptr->data().as<uint16_t*>();
    IN_OUT_desc inpt = {p};
    SetInputTensors(inpt);
    SetOutputTensors(inpt);
    ASSERT_NO_FATAL_FAILURE(makeSingleLayerNetwork(LayerInitParams("ScaleShift")
                                        .weights(sz_weights)
                                        .biases(sz_bias),
                                        {},
                                        weights_ptr));
    ASSERT_TRUE(Infer());
    ref_scale(_inputMap.begin()->second, weights, _refBlob, biasAdd);
    CompareCommonAbsolute(_outputMap.begin()->second, _refBlob, ERROR_BOUND);
}

static std::vector<SizeVector> s_inputScaleTensors = {
    {{1, 16, 8}},              //     CHW
    {{2, 4, 8, 16}},           //    NCHW
    {{2, 2, 44, 88, 16}},      //   NCDHW
    {{2, 2, 2, 16, 32, 32}},   //  6DNCWH
    {{3, 4, 3, 2, 12, 7, 7}},  // 76DNCHW
};

static std::vector<bool> s_inputBiasScale = {
    false,
    true
};
