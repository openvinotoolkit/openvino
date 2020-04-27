// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include "ie_layouts.h"
#include "myriad_layers_tests.hpp"
#include <vpu/private_plugin_config.hpp>
#include "myriad_layers_reference_functions.hpp"
#include "ie_memcpy.h"

using namespace InferenceEngine;

typedef std::vector<int32_t> IndicesVector;
typedef myriadLayerTestBaseWithParam<std::tuple<InferenceEngine::SizeVector, IndicesVector>> myriadLayersTestsUnsqueeze;

static void ref_unsqueeze(const InferenceEngine::Blob::Ptr src,
                        InferenceEngine::Blob::Ptr dst) {
    ASSERT_NE(src, nullptr);
    ASSERT_NE(dst, nullptr);
    int32_t OW = 1;
    int32_t OH = 1;
    int32_t OC = 1;
    int32_t ON = 1;
    int32_t IW = 1;
    int32_t IH = 1;
    int32_t IC = 1;
    int32_t I_N = 1;

    get_ndims(src, IW, IH, IC, I_N);
    get_ndims(dst, OW, OH, OC, ON);

    ASSERT_EQ(IW * IH * IC * I_N, OW * OH * OC * ON);

    const ie_fp16 *src_data = src->buffer();
    ie_fp16 *dst_data = dst->buffer();
    size_t src_size = src->size();
    size_t dst_size = dst->size();

    ASSERT_NE(src_data, nullptr);
    ASSERT_NE(dst_data, nullptr);
    ASSERT_EQ(src_size, dst_size);

    ie_memcpy(dst_data, dst_size * sizeof(ie_fp16), src_data, src_size * sizeof(ie_fp16));
}

TEST_P (myriadLayersTestsUnsqueeze, Unsqueeze){
    auto input = std::get<0>(GetParam());
    auto indices = std::get<1>(GetParam());

    std::string in_dims{};
    std::string out_dims{};

    InferenceEngine::SizeVector output = input;

    std::sort(indices.begin(), indices.end());

    for (auto index : indices) {
        ASSERT_LE(index, output.size());
        output.insert(output.begin() + index, 1);
    }

    for (auto in_dim : input) {
        in_dims += R"V0G0N(
                        <dim>
)V0G0N"
                            + std::to_string(in_dim) +
R"V0G0N(
                        </dim>
)V0G0N";
    }

    for (auto out_dim : output) {
        out_dims += R"V0G0N(
                        <dim>
)V0G0N"
                            + std::to_string(out_dim) +
R"V0G0N(
                        </dim>
)V0G0N";
    }

    std::string UNSQUEEZE_MODEL = R"V0G0N(
        <net name="UNSQUEEZE_MODEL" version="2" batch="1">
            <layers>
                <layer id="0" name="input" precision="FP16" type="Input">
                <output>
                    <port id="0">
)V0G0N"
                    + in_dims +
R"V0G0N(
                    </port>
                </output>
                </layer>
                <layer id="1" name="indices" precision="FP16" type="Const">
                    <output>
                        <port id="1">
                            <dim>
)V0G0N"
                                + std::to_string(indices.size()) +
R"V0G0N(
                            </dim>
                        </port>
                    </output>
                    <blobs>
                        <custom offset="0" size=")V0G0N"
                                          + std::to_string(indices.size() * sizeof(ie_fp16)) +
                                          R"V0G0N("/>
                    </blobs>
                </layer>
                <layer id="2" name="unsqueeze" precision="FP16" type="Unsqueeze">
                    <input>
                        <port id="0">
)V0G0N"
                        + in_dims +
R"V0G0N(
                        </port>
                        <port id="1">
                            <dim>
)V0G0N"
                                + std::to_string(indices.size()) +
R"V0G0N(
                            </dim>
                        </port>
                    </input>
                    <output>
                        <port id="1">
)V0G0N"
                        + out_dims +
R"V0G0N(
                        </port>
                    </output>
                </layer>
            </layers>
            <edges>
                <edge from-layer="0" from-port="0" to-layer="2" to-port="0"/>
                <edge from-layer="1" from-port="1" to-layer="2" to-port="1"/>
            </edges>
        </net>
)V0G0N";

    InferenceEngine::StatusCode st = InferenceEngine::OK;
    InferenceEngine::ResponseDesc resp;

    InferenceEngine::TBlob<uint8_t> *weights_raw = new InferenceEngine::TBlob<uint8_t>(
        {InferenceEngine::Precision::U8,
         {indices.size() * sizeof(ie_fp16)},
         InferenceEngine::C});
    weights_raw->allocate();
    ie_fp16 *inputBlobRawDataFp16 = weights_raw->data().as<ie_fp16 *>();

    for (size_t index = 0; index < indices.size(); ++index) {
        inputBlobRawDataFp16[index] = InferenceEngine::PrecisionUtils::f32tof16(indices[index]);
    }

    TBlob<uint8_t>::Ptr weights(weights_raw);
    ASSERT_NO_THROW(readNetwork(UNSQUEEZE_MODEL, weights));
    createInferRequest(NetworkInitParams().useHWOpt(true));

    ASSERT_TRUE(Infer());

    ref_unsqueeze(_inputMap.begin()->second, _refBlob);
    auto outBlob = _outputMap.begin()->second;

    const auto& outDims = outBlob->getTensorDesc().getDims();
    const auto& refDims = _refBlob->getTensorDesc().getDims();
    ASSERT_EQ(outDims.size(), refDims.size());
    for (size_t i = 0; i < outDims.size(); i++) {
        ASSERT_EQ(outDims[i], refDims[i]);
    }

    const ie_fp16 *out_data = outBlob->buffer();
    const ie_fp16 *ref_data = _refBlob->buffer();
    size_t out_size = outBlob->size();
    size_t ref_size = _refBlob->size();
    ASSERT_EQ(out_size, ref_size);
    for (size_t i = 0; i < out_size; i++) {
        ASSERT_EQ(out_data[i], ref_data[i]);
    }
}

static std::vector<InferenceEngine::SizeVector> s_squeezeTensors = {
        {{3}, {1}, {1, 3}, {3, 1}}
};

static std::vector<IndicesVector> s_squeezeIndices = {
        {{0, 2}, {0}, {1}, {0, 1}, {1, 2}}
};
