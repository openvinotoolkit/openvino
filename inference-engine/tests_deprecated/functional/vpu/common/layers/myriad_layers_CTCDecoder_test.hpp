// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cmath>
#include "myriad_layers_tests.hpp"

using namespace InferenceEngine;

#define ERROR_BOUND 0.2f

PRETTY_PARAM(CustomConfig, std::string);
PRETTY_PARAM(HwOptimization, bool);

typedef myriadLayerTestBaseWithParam<std::tuple<Dims, HwOptimization, IRVersion, CustomConfig>>
    myriadCTCDecoderLayerTests_smoke;

void refCTCDecoder(const Blob::Ptr src, const Blob::Ptr seq_ind, Blob::Ptr dst) {
    ie_fp16 *src_data = static_cast<ie_fp16*>(src->buffer());
    ie_fp16 *src_seq_inp = static_cast<ie_fp16*>(seq_ind->buffer());
    ie_fp16 *output_sequences = static_cast<ie_fp16*>(dst->buffer());
    ASSERT_NE(src_data, nullptr);
    ASSERT_NE(src_seq_inp, nullptr);
    ASSERT_NE(output_sequences, nullptr);

    const auto& dims = src->getTensorDesc().getDims();
    size_t in_width      = dims[dims.size() - 1];
    size_t in_height     = dims[dims.size() - 2];
    size_t in_channels   = dims[dims.size() - 3];

    size_t T_ = in_channels;
    size_t N_ = in_height;
    size_t C_ = in_width;

    std::vector<int> seq_ind_data(88);
    seq_ind_data[0] = 0;
    for(int i = 1; i < 88; i++) {
        seq_ind_data[i] = (int)(PrecisionUtils::f16tof32(src_seq_inp[i]));
    }

    // Fill output_sequences with -1
    for (size_t ii = 0; ii < T_; ii++) {
        output_sequences[ii] = PrecisionUtils::f32tof16(-1.0);
    }
    size_t output_index = 0;

    // Caffe impl
    for(size_t n = 0; n < N_; ++n) {
        int prev_class_idx = -1;

        for (size_t t = 0; /* check at end */; ++t) {
            // get maximum probability and its index
            int max_class_idx = 0;
            ie_fp16* probs;
            ie_fp16 max_prob;

            probs = src_data + t*C_;
            max_prob = probs[0];
            ++probs;

            for (size_t c = 1; c < C_; ++c, ++probs) {
                if (*probs > max_prob) {
                    max_class_idx = c;
                    max_prob = *probs;
                }
            }

            //if (max_class_idx != blank_index_
            //        && !(merge_repeated_&& max_class_idx == prev_class_idx))
            if (max_class_idx < (int)C_-1 && !(1 && max_class_idx == prev_class_idx)) {
                output_sequences[output_index] =  PrecisionUtils::f32tof16((float)max_class_idx);
                output_index++;
            }

            prev_class_idx = max_class_idx;

            // Assume sequence_indicators is always 1
//             if (t + 1 == T_)
            if (t + 1 == T_ || seq_ind_data[t + 1] == 0) {
                break;
            }
        }
    }
}

TEST_P(myriadCTCDecoderLayerTests_smoke, CTCGreedyDecoder) {
    const tensor_test_params dims = std::get<0>(GetParam());
    const bool hwOptimization = std::get<1>(GetParam());
    _irVersion = std::get<2>(GetParam());
    const std::string customConfig = std::get<3>(GetParam());

    if (!customConfig.empty() && !CheckMyriadX()) {
		GTEST_SKIP() << "Custom layers for MYRIAD2 not supported";
	}

    _config[VPU_CONFIG_KEY(CUSTOM_LAYERS)] = customConfig;

    const auto inputTensors = IN_OUT_desc{{dims.c, dims.h, dims.w}, {dims.h, dims.c}};
    const auto outputTensors = IN_OUT_desc{{1, 1, dims.h, dims.c}};

    SetInputTensors(inputTensors);
    SetOutputTensors(outputTensors);

    std::map<std::string, std::string> params;
    params["ctc_merge_repeated"] = "1";

    ASSERT_NO_FATAL_FAILURE(makeSingleLayerNetwork(LayerInitParams("CTCGreedyDecoder").params(params),
                                                   NetworkInitParams()
                                                           .useHWOpt(hwOptimization)
                                                           .layoutPreference(vpu::LayoutPreference::ChannelMajor)
                                                           .lockLayout(true)));

    auto dataBlob = _inputMap.begin()->second;
    auto seqIndBlob = std::next(_inputMap.begin())->second;

    auto seqIndFp16 = seqIndBlob->buffer().as<uint16_t *>();
    seqIndFp16[0] = PrecisionUtils::f32tof16(0.0);
    for (size_t i = 1; i < seqIndBlob->size(); ++i) {
        seqIndFp16[i] = PrecisionUtils::f32tof16(1.0);
    }

    std::string inputTensorBinary = TestDataHelpers::get_data_path() + "/vpu/InputGreedyDecoderMyriadCHW.bin";
    ASSERT_TRUE(fromBinaryFile(inputTensorBinary, dataBlob));

    ASSERT_TRUE(Infer());

    refCTCDecoder(dataBlob, seqIndBlob, _refBlob);

    CompareCommonAbsolute(_outputMap.begin()->second, _refBlob, 0.0);
}

static std::vector<CustomConfig> s_CustomConfig = {
        {""},
#ifdef VPU_HAS_CUSTOM_KERNELS
        getIELibraryPath() + "/vpu_custom_kernels/customLayerBindings.xml"
#endif
};
