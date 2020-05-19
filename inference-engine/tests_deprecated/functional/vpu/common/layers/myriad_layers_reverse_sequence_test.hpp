// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "myriad_layers_tests.hpp"
#include <algorithm>

using std::tuple;
using std::get;

#define f32Tof16 PrecisionUtils::f32tof16
#define f16Tof32 PrecisionUtils::f16tof32

using namespace InferenceEngine;

struct reverse_sequence_test_params {
    SizeVector dims;
    int seq_axis;
    int batch_axis;
    friend std::ostream& operator<<(std::ostream& os, reverse_sequence_test_params const& tst)
    {
        os << "dims = (";
        for (int i = 0; i < tst.dims.size()-1; i++)
            os << tst.dims[i] << ", ";
        os << tst.dims[tst.dims.size()-1] << ")";
        return os << ", " <<
                  " sequence axis = " << tst.seq_axis
                  << ", batch axis = " << tst.batch_axis;
    };
};

PRETTY_PARAM(ReverseSequence, reverse_sequence_test_params);
typedef myriadLayerTestBaseWithParam<std::tuple<ReverseSequence, IRVersion>> myriadLayerReverseSequence_smoke;

static int nchw_to_nhwc(InferenceEngine::SizeVector dims, int ind)
{
    int ind3 = ind % dims[3];
    int ind2 = (ind / dims[3]) % dims[2];
    int ind1 = ((ind / dims[3]) / dims[2]) % dims[1];
    int ind0 = ((ind / dims[3]) / dims[2]) / dims[1];
    return ind1 + ind3 * dims[1] + ind2 * dims[1] * dims[3] + ind0 * dims[1] * dims[3] * dims[2];
}

static void ref_reverse_sequence(
        const Blob::Ptr& src,
        const Blob::Ptr& seq_lengths,
        Blob::Ptr& dst,
        int seq_axis,
        int batch_axis
) {
    const ie_fp16* src_data = src->cbuffer().as<const ie_fp16*>();
    const ie_fp16* seq_lengths_data = static_cast<ie_fp16*>(seq_lengths->cbuffer().as<ie_fp16*>());
    ie_fp16* dst_data = static_cast<ie_fp16*>(dst->cbuffer().as<ie_fp16*>());

    InferenceEngine::SizeVector src_dims = src->getTensorDesc().getDims();
    InferenceEngine::SizeVector srcStrides = src->getTensorDesc().getBlockingDesc().getStrides();

    if (seq_axis < 0)
        seq_axis += src_dims.size();

    if (seq_axis < 0 || seq_axis >= src_dims.size())
        FAIL() << "Incorrect 'seq_axis' parameters dimensions and axis number!";

    if (batch_axis < 0)
        batch_axis += src_dims.size();

    if (batch_axis < 0 || batch_axis >= src_dims.size())
        FAIL() << "Incorrect 'batch_axis' parameters dimensions and axis number!";

    for (size_t i = 0; i < src_dims[batch_axis]; i++) {
        if (f16Tof32(seq_lengths_data[i]) > src_dims[seq_axis])
        {
            FAIL() << "Incorrect input 'seq_lengths' values!";
        }
    }
    size_t work_amount_dst = srcStrides[0] * src_dims[0];
    InferenceEngine::SizeVector counters(src_dims.size(), 0);
    Layout layout = src->getTensorDesc().getLayout();
    for (size_t iwork = 0; iwork < work_amount_dst; ++iwork) {
        size_t srcStride = 1;
        int i;
        size_t src_idx;
        for (i = src_dims.size() - 1, src_idx = 0; i >= 0; i--) {
            size_t idx = counters[i];

            if (i == seq_axis && idx < f16Tof32(seq_lengths_data[counters[batch_axis]])) {
                idx = f16Tof32(seq_lengths_data[counters[batch_axis]]) - idx - 1;
            }

            src_idx += idx * srcStride;

            srcStride *= src_dims[i];
        }

        if (layout == NHWC)
            dst_data[nchw_to_nhwc(src_dims, iwork)] = src_data[nchw_to_nhwc(src_dims, src_idx)];
        else
            dst_data[iwork] = src_data[src_idx];

        for (int j = src_dims.size() - 1; j >= 0; j--) {
            counters[j] = (counters[j] + 1) % src_dims[j];
            if (counters[j] != 0) break;
        }
    }
}

TEST_P(myriadLayerReverseSequence_smoke, ReverseSequence) {
    _config[VPU_CONFIG_KEY(DETECT_NETWORK_BATCH)] = CONFIG_VALUE(NO);

    reverse_sequence_test_params input_dims = std::get<0>(GetParam());
    _irVersion = std::get<1>(GetParam());
    auto dims = input_dims.dims;
    auto seq_axis = input_dims.seq_axis;
    auto batch_axis = input_dims.batch_axis;

    SetInputTensors({dims, {dims[batch_axis]}});
    SetOutputTensors({dims});

    std::map<std::string, std::string> layer_params = {
              {"seq_axis", std::to_string(seq_axis)}
            , {"batch_axis", std::to_string(batch_axis)}
    };
    ASSERT_NO_FATAL_FAILURE(makeSingleLayerNetwork(LayerInitParams("ReverseSequence").params(layer_params)));

    /* input tensor generating */
    auto pInputBlob = _inputMap.begin();
    Blob::Ptr inputBlob = pInputBlob->second;
    ie_fp16 *src_data = static_cast<ie_fp16*>(inputBlob->buffer());
    for (int i = 0; i < inputBlob->size(); i++) {
        src_data[i] = f32Tof16(float(i % 10000));
    }
    pInputBlob++;
    Blob::Ptr lengthSequenceBlob = pInputBlob->second;
    ie_fp16* len_sequence = static_cast<ie_fp16*>(lengthSequenceBlob->buffer());

    for (int i = 0; i < dims[batch_axis]; i++) {
        len_sequence[i] = f32Tof16(static_cast<float>(rand() % ((dims[seq_axis]) + 1)));
    }

    ref_reverse_sequence(inputBlob, lengthSequenceBlob, _refBlob, seq_axis, batch_axis);

    ASSERT_TRUE(Infer());

    auto outputBlob = _outputMap.begin()->second;

    CompareCommonAbsolute(outputBlob, _refBlob, 0);
}
