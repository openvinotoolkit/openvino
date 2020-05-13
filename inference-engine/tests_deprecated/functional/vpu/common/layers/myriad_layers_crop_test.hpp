// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "myriad_layers_tests.hpp"

using namespace InferenceEngine;

#define ERROR_BOUND (0.0f)

PRETTY_PARAM(crop_axis, InferenceEngine::SizeVector)
PRETTY_PARAM(offset, InferenceEngine::SizeVector)
PRETTY_PARAM(dim, InferenceEngine::SizeVector)
PRETTY_PARAM(crop_begin, InferenceEngine::SizeVector)
PRETTY_PARAM(crop_end, InferenceEngine::SizeVector)

typedef myriadLayerTestBaseWithParam<std::tuple<Dims, Dims, crop_axis, offset, dim >> myriadLayerCropOneInputAndDim_smoke;
typedef myriadLayerTestBaseWithParam<std::tuple<Dims, Dims, crop_axis, crop_begin, crop_end >> myriadLayerCropOneInput_smoke;
typedef myriadLayerTestBaseWithParam<std::tuple<Dims, Dims, Dims, crop_axis, offset >> myriadLayerCropTwoInputs_smoke;

static void ref_crop(const Blob::Ptr src,
                     Blob::Ptr dst,
                     InferenceEngine::SizeVector& axis,
                     InferenceEngine::SizeVector& offset) {

    ASSERT_NE(src, nullptr);
    ASSERT_NE(dst, nullptr);
    int32_t IW;
    int32_t IH;
    int32_t IC;
    int32_t OW = 1;
    int32_t OH = 1;
    int32_t OC = 1;

    get_dims(src, IW, IH, IC);
    get_dims(dst, OW, OH, OC);
    int32_t IW_off = 0;
    int32_t IH_off = 0;
    int32_t IC_off = 0;

    for (size_t i = 0; i < axis.size(); ++i) {
        switch(axis[i]){
            case 1:
                IC_off = offset[i];
                break;
            case 2:
                IH_off = offset[i];
                break;
            case 3:
                IW_off = offset[i];
                break;
        }
    }
    auto real_W = std::min(OW, IW - IW_off);
    auto real_H = std::min(OH, IH - IH_off);
    auto real_C = std::min(OC, IC - IC_off);
    const uint16_t *src_data = src->buffer();
    uint16_t *dst_data = dst->buffer();
    for (int32_t w = 0; w < real_W; ++w) {
        for (int32_t h = 0; h < real_H; ++h) {
            for (int32_t c = 0; c < real_C; ++c) {
                int32_t inp_ind = (c + IC_off) + IC * ((w + IW_off) + (h + IH_off)* IW);
                int32_t out_ind = c + OC * (w  + h * OW);
                dst_data[out_ind] = src_data[inp_ind];
            }
        }
    }
}

TEST_P(myriadLayerCropOneInputAndDim_smoke, CropWithOneInputAndDim) {
    auto param = GetParam();
    tensor_test_params tensor1 = std::get<0>(param);
    tensor_test_params tensor2 = std::get<1>(param);
    InferenceEngine::SizeVector axis_val = std::get<2>(param);
    InferenceEngine::SizeVector offsets = std::get<3>(param);
    InferenceEngine::SizeVector dims = std::get<4>(param);
    InferenceEngine::SizeVector input_dim1 = {tensor1.n, tensor1.c, tensor1.h, tensor1.w};
    InferenceEngine::SizeVector input_dim2 = {tensor2.n, tensor2.c, tensor2.h, tensor2.w};
    ASSERT_EQ(axis_val.size(), offsets.size());
    ASSERT_EQ(axis_val.size(), dims.size());
    char prm[256];
    char val[256];
    std::string axis;
    std::string offset;
    std::string dim;
    std::map<std::string, std::string> params;

    for (size_t i = 0; i < axis_val.size(); ++i) {
        axis += std::to_string(axis_val[i]) +",";
        offset += std::to_string(offsets[i]) +",";
        dim += std::to_string(dims[i]) +",";
    }
    params["dim"] = dim;
    params["axis"] = axis;
    params["offset"] = offset;
    SetInputTensors({input_dim1});
    SetOutputTensors({input_dim2});
    ASSERT_NO_FATAL_FAILURE(makeSingleLayerNetwork(LayerInitParams("Crop").params(params), NetworkInitParams().layoutPreference(vpu::LayoutPreference::ChannelMinor)));
    ASSERT_TRUE(Infer());
    ref_crop(_inputMap.begin()->second, _refBlob, axis_val, offsets);
    CompareCommonAbsolute(_outputMap.begin()->second, _refBlob, ERROR_BOUND);
}

TEST_P(myriadLayerCropOneInput_smoke, CropWithOneInput) {
    auto param = GetParam();
    tensor_test_params tensor1 = std::get<0>(param);
    tensor_test_params tensor2 = std::get<1>(param);
    InferenceEngine::SizeVector axis_val = std::get<2>(param);
    InferenceEngine::SizeVector crop_begin_val = std::get<3>(param);
    InferenceEngine::SizeVector crop_end_val = std::get<4>(param);
    InferenceEngine::SizeVector input_dim1 = {tensor1.n, tensor1.c, tensor1.h, tensor1.w};
    InferenceEngine::SizeVector input_dim2 = {tensor2.n, tensor2.c, tensor2.h, tensor2.w};

    ASSERT_EQ(axis_val.size(), crop_begin_val.size());
    ASSERT_EQ(axis_val.size(), crop_end_val.size());
    std::string axis;
    std::string crop_begin;
    std::string dim;
    std::map<std::string, std::string> params;

    for (size_t i = 0; i < axis_val.size(); ++i) {
        axis += std::to_string(axis_val[i]) +",";
        crop_begin += std::to_string(crop_begin_val[i]) +",";
    }

    params["axis"] = axis;
    params["offset"] = crop_begin;
    SetInputTensors({input_dim1});
    SetOutputTensors({input_dim2});
    ASSERT_NO_FATAL_FAILURE(makeSingleLayerNetwork(LayerInitParams("Crop").params(params), NetworkInitParams().layoutPreference(vpu::LayoutPreference::ChannelMinor)));
    ASSERT_TRUE(Infer());
    InferenceEngine::SizeVector sum;

    ref_crop(_inputMap.begin()->second, _refBlob, axis_val, crop_begin_val);
    CompareCommonAbsolute(_outputMap.begin()->second, _refBlob, ERROR_BOUND);
}

TEST_P(myriadLayerCropTwoInputs_smoke, CropWithTwoInputs) {
    auto param = GetParam();
    tensor_test_params tensor1 = std::get<0>(param);
    tensor_test_params tensor2 = std::get<1>(param);
    tensor_test_params tensor3 = std::get<2>(param);
    InferenceEngine::SizeVector axis_val = std::get<3>(param);
    InferenceEngine::SizeVector offsets = std::get<4>(param);
    InferenceEngine::SizeVector input_dim1 = {tensor1.n, tensor1.c, tensor1.h, tensor1.w};
    InferenceEngine::SizeVector input_dim2 = {tensor2.n, tensor2.c, tensor2.h, tensor2.w};
    InferenceEngine::SizeVector output_dim3 = {tensor3.n, tensor3.c, tensor3.h, tensor3.w};

    ASSERT_EQ(axis_val.size(), offsets.size());
    std::string axis;
    std::string offset;
    std::map<std::string, std::string> params;

    for (size_t i = 0; i < axis_val.size(); ++i) {
        axis += std::to_string(axis_val[i]) +",";
        offset += std::to_string(offsets[i]) +",";
    }

    params["axis"] = axis;
    params["offset"] = offset;
    SetInputTensors({input_dim1, input_dim2});
    SetOutputTensors({output_dim3});
    ASSERT_NO_FATAL_FAILURE(makeSingleLayerNetwork(LayerInitParams("Crop").params(params), NetworkInitParams().layoutPreference(vpu::LayoutPreference::ChannelMinor)));
    ASSERT_TRUE(Infer());
    ref_crop(_inputMap.begin()->second, _refBlob, axis_val, offsets);
    CompareCommonAbsolute(_outputMap.begin()->second, _refBlob, ERROR_BOUND);
}

static std::vector<Dims> s_tileTensors1 = {
    {{1, 4, 16, 32}},
    {{1, 8, 20, 36}},
};

static std::vector<Dims> s_tileTensors2 = {
    {{1, 2, 12, 26}},
};

static std::vector<crop_axis> s_tileCropAxis = {
    {{1, 2, 3}},
};

static std::vector<offset> s_tileOffset = {
    {{2, 4, 6}},
    {{2, 2, 2}},
};

static std::vector<dim> s_tileDim = {
    {{2, 12, 26}},
};

static std::vector<crop_begin> s_tileCropBegin= {
    {{2, 2, 3}},
};

static std::vector<crop_end> s_tileCropEnd = {
    {{2, 2, 6}},
};
