// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu/model/data_desc.hpp"
#include "myriad_layers_tests.hpp"

#define ERROR_BOUND (1.e-3f)

using namespace InferenceEngine;

namespace {
    bool iter(SizeVector& in, SizeVector& out) {
        bool flag = true;
        for(int t = 0; t < out.size(); t++) {
            int i = out.size() - 1 - t;
            if(in[i] < out[i] - 1) {
                in[i]++;
                break;
            } else {
                if(i == 0) {
                    flag = false;
                    break;
                }
                in[i] = 0;
            }
        }
        return flag;
    }

    int calcOffset(SizeVector& in, SizeVector& out) {
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

void ref_bias(const InferenceEngine::Blob::Ptr src1,
              const InferenceEngine::Blob::Ptr src2,
              InferenceEngine::Blob::Ptr dst) {
    ASSERT_NE(src1, nullptr);
    ASSERT_NE(src2, nullptr);
    ASSERT_NE(dst, nullptr);

    SizeVector in_size;
    SizeVector out_size;
    in_size = src1->getTensorDesc().getDims();
    out_size = dst->getTensorDesc().getDims();
    Layout layout = src1->getTensorDesc().getLayout();
    const uint16_t *src_data = src1->buffer();
    const uint16_t *bias_data = src2->buffer();
    uint16_t *dst_data = dst->buffer();

    // TODO: investigate this case
    if (layout == NCHW || layout == NHWC) {
        size_t N1 = out_size[0];
        size_t C1 = out_size[1];
        size_t H1 = out_size[2];
        size_t W1 = out_size[3];
        for (size_t n = 0; n < N1; n++) {
            for (size_t c = 0; c < C1; c++) {
                float val = PrecisionUtils::f16tof32(bias_data[c]);
                for (size_t h = 0; h < H1; h++) {
                    for (size_t w = 0; w < W1; w++) {
                        size_t iidx = layout == NCHW ?
                                           w + h * W1 + c * W1 * H1 + n * W1 * H1 * C1 : 
                                           c + w * C1 + h * C1 * W1 + n * W1 * H1 * C1;
                        float res = val + PrecisionUtils::f16tof32(src_data[iidx]);
                        dst_data[iidx] = PrecisionUtils::f32tof16(res);
                    }
                }
            }
        }
    } else {
        int dims = out_size.size();
        int dimC = dimToIeInd(vpu::Dim::C, dims);
        SizeVector curr_size(dims);
        do {
            float val = PrecisionUtils::f16tof32(bias_data[curr_size[dimC]]);
            float res = val + PrecisionUtils::f16tof32(src_data[calcOffset(curr_size, in_size)]);
            dst_data[calcOffset(curr_size, out_size)] = PrecisionUtils::f32tof16(res);
        } while(iter(curr_size, out_size));
    }
}

class myriadLayersTestsBias_smoke: public myriadLayersTests_nightly,
                             public testing::WithParamInterface<InferenceEngine::SizeVector> {
};

TEST_P(myriadLayersTestsBias_smoke, TestsBias) {
    _config[InferenceEngine::MYRIAD_DETECT_NETWORK_BATCH] = CONFIG_VALUE(NO);
    auto input_dim = GetParam();
    InferenceEngine::SizeVector input_dim1;
    auto dims = input_dim.size();
    int dimC = dimToIeInd(vpu::Dim::C, dims);
    input_dim1.push_back(input_dim[dimC]);
    SetInputTensors({input_dim, input_dim1});
    SetOutputTensors({input_dim});
    ASSERT_NO_FATAL_FAILURE(makeSingleLayerNetwork(LayerInitParams("Bias")));

    ASSERT_TRUE(Infer());
    ASSERT_EQ(_inputMap.size(), 2);
    ASSERT_EQ(_outputMap.size(), 1);
    auto iter = _inputMap.begin();
    auto first_input = iter->second;
    ++iter;
    auto second_input = iter->second;
    ref_bias(first_input, second_input, _refBlob);
    CompareCommonAbsolute(_outputMap.begin()->second, _refBlob, ERROR_BOUND);
}

static std::vector<InferenceEngine::SizeVector> s_biasDims = {
    {4, 10, 8, 4, 4},
    {10, 8, 4, 4},
    {32, 8, 16}
};
