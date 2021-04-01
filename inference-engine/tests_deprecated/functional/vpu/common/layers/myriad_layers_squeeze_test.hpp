// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include "ie_layouts.h"
#include "myriad_layers_tests.hpp"
#include <vpu/private_plugin_config.hpp>
#include "myriad_layers_reference_functions.hpp"
#include "ie_memcpy.h"
#include "tests_vpu_common.hpp"

using namespace InferenceEngine;

typedef std::vector<int32_t> IndicesVector;

static void ref_squeeze(const InferenceEngine::Blob::Ptr src,
                              InferenceEngine::Blob::Ptr dst,
                        const SizeVector output) {
    ASSERT_NE(src, nullptr);
    ASSERT_NE(dst, nullptr);

    ASSERT_EQ(src->size(), dst->size());

    const ie_fp16 *src_data = src->buffer().as<ie_fp16*>();
    ie_fp16 *dst_data = dst->buffer().as<ie_fp16*>();

    size_t src_size = src->size();
    size_t dst_size = dst->size();

    ASSERT_NE(src_data, nullptr);
    ASSERT_NE(dst_data, nullptr);
    ASSERT_EQ(src_size, dst_size);

    dst->getTensorDesc().setDims(output);
    ie_memcpy(dst_data, dst_size * sizeof(ie_fp16), src_data, src_size * sizeof(ie_fp16));
}

PRETTY_PARAM(layoutPreference, vpu::LayoutPreference)


static void GenerateOutput(SizeVector& output, const IndicesVector indices,
                     const SizeVector input,   const int32_t keep_at_least_1d) {
    auto indicesCopy = indices;
    for (auto &index : indicesCopy) {
        if (index < 0)
            index += input.size();
        ASSERT_LT(abs((int)index), input.size());
        ASSERT_EQ(input[index], 1);
    }

    for (size_t k = 0; k < input.size(); k++) {
        if (std::find(indicesCopy.begin(), indicesCopy.end(), k) == indicesCopy.end()) {
            output.push_back(input[k]);
        }
    }

    if (output.size() == 0) {
        if (keep_at_least_1d) {
            output.push_back({ 1 });
        } else {
            output.push_back({ 0 });
        }
    }
}

static std::string DimToString(SizeVector dimVector) {
    std::string outString;
    for (auto dim : dimVector) {
        outString += "<dim>" + std::to_string(dim) + "</dim>\n";
    }
    return outString;
}

static std::string GenerateSqueezeNN(const SizeVector& inputDims, const SizeVector& outputDims,
                                     const std::vector<int32_t>& indices, const int keep_at_least_1d) {
    std::string model =  R"V0G0N(
        <net name="SQUEEZE_MODEL" version="2" batch="1">
            <layers>
                <layer id="0" name="input" precision="FP16" type="Input">
                <output>
                    <port id="0">
                    __IN_DIMS__
                    </port>
                </output>
                </layer>
                <layer id="1" name="indices" precision="FP16" type="Const">
                    <output>
                        <port id="1">
                            <dim>
                            __IND_SIZE__
                            </dim>
                        </port>
                    </output>
                    <blobs>
                        <custom offset="0" size="__IND_SIZE_OFFSET__"/>
                    </blobs>
                </layer>
                <layer id="2" name="squeeze" precision="FP16" type="Squeeze">
                    <data keep_at_least_1d="__KEEP_1D__"/>
                    <input>
                        <port id="0">
                        __IN_DIMS__
                        </port>
                        <port id="1">
                            <dim>
                            __IND_SIZE__
                            </dim>
                        </port>
                    </input>
                    <output>
                        <port id="1">
                        __OUT_DIMS__
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
     REPLACE_WITH_STR(model, "__IN_DIMS__", DimToString(inputDims));
     REPLACE_WITH_STR(model, "__OUT_DIMS__", DimToString(outputDims));
     REPLACE_WITH_STR(model, "__IND_SIZE__", std::to_string(indices.size()));
     REPLACE_WITH_STR(model, "__IND_SIZE_OFFSET__", std::to_string(indices.size()* sizeof(ie_fp16)));
     REPLACE_WITH_STR(model, "__KEEP_1D__", std::to_string(keep_at_least_1d));

     return model;
}

static  InferenceEngine::TBlob<uint8_t>* GenerateWeightBlob(const IndicesVector& indices) {
    InferenceEngine::TBlob<uint8_t> *weights_raw = new InferenceEngine::TBlob<uint8_t>(
            {InferenceEngine::Precision::U8,
                    {indices.size() * sizeof(ie_fp16)},
                    InferenceEngine::Layout :: C});
    weights_raw->allocate();
    ie_fp16 *inputBlobRawDataFp16 = weights_raw->data().as<ie_fp16 *>();
    for (size_t index = 0; index < indices.size(); ++index) {
        inputBlobRawDataFp16[index] = InferenceEngine::PrecisionUtils::f32tof16(indices[index]);
    }
    return weights_raw;
}

class myriadLayersTestsSqueezeBase : public
        myriadLayerTestBaseWithParam<std::tuple<InferenceEngine::SizeVector, IndicesVector, int32_t, layoutPreference>>
{
protected:
    virtual void InitBody()
    {
        auto input = std::get<0>(GetParam());
        auto indices = std::get<1>(GetParam());
        auto keep_at_least_1d = std::get<2>(GetParam());
        auto layoutPreference = std::get<3>(GetParam());

        SizeVector output;
        GenerateOutput(output, indices, input, keep_at_least_1d);
        TBlob<uint8_t>::Ptr weights(GenerateWeightBlob(indices));
        std::string SQUEEZE_MODEL_FORMATTED = GenerateSqueezeNN(input, output, indices, keep_at_least_1d);

        ASSERT_NO_THROW(readNetwork(SQUEEZE_MODEL_FORMATTED, weights));
        createInferRequest(NetworkInitParams().useHWOpt(true).layoutPreference(layoutPreference).lockLayout(true));

        ASSERT_TRUE(Infer());

        ref_squeeze(_inputMap.begin()->second, _refBlob, output);
        CompareCommonAbsolute(_outputMap.begin()->second, _refBlob, 0);
    }
};

class myriadLayersTestsSqueezeTC1_smoke : public myriadLayersTestsSqueezeBase
{
};

class myriadLayersTestsSqueezeTC2_smoke : public myriadLayersTestsSqueezeBase
{
};

class myriadLayersTestsSqueezeTC3_smoke : public myriadLayersTestsSqueezeBase
{
};

class myriadLayersTestsSqueezeTC4_smoke : public myriadLayersTestsSqueezeBase
{
};

class myriadLayersTestsSqueezeTC5_smoke : public myriadLayersTestsSqueezeBase
{
};

TEST_P(myriadLayersTestsSqueezeTC1_smoke, Squeeze) {
    DISABLE_IF(!CheckMyriadX());
    InitBody();
}

TEST_P(myriadLayersTestsSqueezeTC2_smoke, Squeeze) {
    DISABLE_IF(!CheckMyriadX());
    InitBody();
}

TEST_P(myriadLayersTestsSqueezeTC3_smoke, Squeeze) {
    DISABLE_IF(!CheckMyriadX());
    InitBody();
}

TEST_P(myriadLayersTestsSqueezeTC4_smoke, Squeeze) {
    DISABLE_IF(!CheckMyriadX());
    InitBody();
}

TEST_P(myriadLayersTestsSqueezeTC5_smoke, Squeeze) {
    DISABLE_IF(!CheckMyriadX());
    InitBody();
}

static std::vector<InferenceEngine::SizeVector> s_squeezeTensorsTC1 = {
    {{1, 3, 1}, {1, 1, 1}}
};

static std::vector<IndicesVector> s_squeezeIndicesTC1 = {
    {{0, 2}, {0}, {2}, {-3}, {-1}, {-3, -1}}
};

static std::vector<InferenceEngine::SizeVector> s_squeezeTensorsTC2 = {
    {{3, 1, 2}}
};

static std::vector<IndicesVector> s_squeezeIndicesTC2 = {
    {{1, -2}, {-2, 1}}
};

static std::vector<InferenceEngine::SizeVector> s_squeezeTensorsTC3 = {
        {{3, 1, 2, 3}}
};

static std::vector<IndicesVector> s_squeezeIndicesTC3 = {
        {{1, -3}, {-3, 1}}
};

static std::vector<InferenceEngine::SizeVector> s_squeezeTensorsTC4 = {
        {{3, 1, 2, 1}}
};

static std::vector<IndicesVector> s_squeezeIndicesTC4 = {
        {{1}, {3}, {1, 3}, {3, 1}, {-1}, {-3}, {-3, -1}}
};

static std::vector<InferenceEngine::SizeVector> s_squeezeTensorsTC5 = {
        {{1, 13, 1, 1, 33}},
};

static std::vector<IndicesVector> s_squeezeIndicesTC5 = {
        {0}, {3}, {0, 2}, {0, 3}, {2, 3}, {-5, -2, -3},
};

static std::vector<int32_t> s_squeezeKeepAtLeast1D = {
    0, 1
};

static std::vector<layoutPreference> s_squeezeLayoutPreference = {
        vpu::LayoutPreference::ChannelMajor,
};
