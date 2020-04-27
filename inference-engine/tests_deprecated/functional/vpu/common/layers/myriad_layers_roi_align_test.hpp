// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include "myriad_layers_reference_functions.hpp"
#include "myriad_layers_tests.hpp"

#include <random>

using namespace InferenceEngine;

#define ERROR_BOUND (2.5e-3f)

struct roi_align_param {
    int         in_net_w;
    int         in_net_h;
    uint32_t    pooled_w;
    uint32_t    pooled_h;
    int         sampling_ratio;
    float       spatial_scale;

    friend std::ostream& operator<<(std::ostream& os, roi_align_param const& tst)
    {
        return os << "input net width = " << tst.in_net_w
                  << ", input net height = " << tst.in_net_h
                  << ", pooled_w = " << tst.pooled_w
                  << ", pooled_h = " << tst.pooled_h
                  << ", sampling_ratio = " << tst.sampling_ratio
                  << ", spatial_scale = " << tst.spatial_scale;
    };
};

PRETTY_PARAM(roi_align_mode, std::string);
PRETTY_PARAM(number_rois, uint32_t);

using ROIAlignTestParams = std::tuple<Dims, roi_align_param, number_rois, roi_align_mode>;
typedef myriadLayerTestBaseWithParam<ROIAlignTestParams> myriadLayersTestsROIAlign_nightly;

const int roi_cols = 4;

static void genROIs(InferenceEngine::Blob::Ptr rois,
                    const roi_align_param& params,
                    const uint32_t num_rois) {
    auto roisBlob_data = rois->buffer().as<ie_fp16*>();
    const int max_range_width = params.in_net_w * 4 / 5;
    const int max_range_height = params.in_net_h * 4 / 5;

    float scale_width  = (float)params.in_net_w;
    float scale_height = (float)params.in_net_h;

    std::mt19937 gen(145781);

    std::uniform_int_distribution<> dis_x0(0, max_range_width - 1);
    std::uniform_int_distribution<> dis_y0(0, max_range_height - 1);
    for (int i = 0; i < num_rois; i++) {
        int x0 = dis_x0(gen);
        std::uniform_int_distribution<> dis_x1(0, (params.in_net_w - x0 - 1) - 1);
        int x1 = x0 + dis_x1(gen) + 1;

        int y0 = dis_y0(gen);
        std::uniform_int_distribution<> dis_y1(0, (params.in_net_h - y0 - 1) - 1);
        int y1 = y0 + dis_y1(gen) + 1;

        roisBlob_data[i * roi_cols + 0] = PrecisionUtils::f32tof16(x0 / scale_width);
        roisBlob_data[i * roi_cols + 1] = PrecisionUtils::f32tof16(y0 / scale_height);
        roisBlob_data[i * roi_cols + 2] = PrecisionUtils::f32tof16(x1 / scale_width);
        roisBlob_data[i * roi_cols + 3] = PrecisionUtils::f32tof16(y1 / scale_height);
    }
}

static void genBatchIndices(InferenceEngine::Blob::Ptr batch_indices,
                            const uint32_t num_rois,
                            const uint32_t num_batches) {
    int32_t* batch_indices_data = batch_indices->buffer().as<int32_t*>();

    std::mt19937 gen(145781);
    std::uniform_int_distribution<> dis_index(0, num_batches - 1);
    for (int i = 0; i < num_rois; i++) {
        batch_indices_data[i] = dis_index(gen);
    }
}

static std::string getModel(const int batches, const int channels, const int height, const int width,
                            const int pooled_h, const int pooled_w, const float spatial_scale,
                            const int sampling_ratio, const int num_rois, const std::string mode) {
    std::string model = R"V0G0N(
                <net name="testROIAlign" version="7">
                    <layers>
                        <layer id="0" name="feature_map" precision="FP16" type="Input">
                            <output>
                                <port id="0">
                                    <dim>__BATCHES__</dim>
                                    <dim>__CHANNELS__</dim>
                                    <dim>__HEIGHT__</dim>
                                    <dim>__WIDTH__</dim>
                                </port>
                            </output>
                        </layer>
                        <layer id="1" name="boxes" precision="FP16" type="Input">
                            <output>
                                <port id="0">
                                    <dim>__NUM_ROIS__</dim>
                                    <dim>4</dim>
                                </port>
                            </output>
                        </layer>
                        <layer id="2" name="batch_indices" precision="I32" type="Input">
                            <output>
                                <port id="0">
                                    <dim>__NUM_ROIS__</dim>
                                </port>
                            </output>
                        </layer>
                        <layer id="3" name="roi_align" type="ROIAlign">
                            <data pooled_h="__POOLED_H__" pooled_w="__POOLED_W__" spatial_scale="__SPATIAL_SCALE__" sampling_ratio="__SAMPLING_RATIO__" mode="__MODE__"/>
                            <input>
                                <port id="0">
                                    <dim>__BATCHES__</dim>
                                    <dim>__CHANNELS__</dim>
                                    <dim>__HEIGHT__</dim>
                                    <dim>__WIDTH__</dim>
                                </port>
                                <port id="1">
                                    <dim>__NUM_ROIS__</dim>
                                    <dim>4</dim>
                                </port>
                                <port id="2">
                                    <dim>__NUM_ROIS__</dim>
                                </port>
                            </input>
                            <output>
                                <port id="0">
                                    <dim>__NUM_ROIS__</dim>
                                    <dim>__CHANNELS__</dim>
                                    <dim>__POOLED_H__</dim>
                                    <dim>__POOLED_W__</dim>
                                </port>
                            </output>
                        </layer>
                    </layers>
                    <edges>
                        <edge from-layer="0" from-port="0" to-layer="3" to-port="0"/>
                        <edge from-layer="1" from-port="0" to-layer="3" to-port="1"/>
                        <edge from-layer="2" from-port="0" to-layer="3" to-port="2"/>
                    </edges>
                </net>
            )V0G0N";

    REPLACE_WITH_STR(model, "__NUM_ROIS__", std::to_string(num_rois));
    REPLACE_WITH_STR(model, "__BATCHES__",  std::to_string(batches));
    REPLACE_WITH_STR(model, "__CHANNELS__", std::to_string(channels));
    REPLACE_WITH_STR(model, "__POOLED_H__", std::to_string(pooled_h));
    REPLACE_WITH_STR(model, "__POOLED_W__", std::to_string(pooled_w));
    REPLACE_WITH_STR(model, "__HEIGHT__",   std::to_string(height));
    REPLACE_WITH_STR(model, "__WIDTH__",    std::to_string(width));
    REPLACE_WITH_STR(model, "__SPATIAL_SCALE__",  std::to_string(spatial_scale));
    REPLACE_WITH_STR(model, "__SAMPLING_RATIO__", std::to_string(sampling_ratio));
    REPLACE_WITH_STR(model, "__MODE__", mode);

    return model;
}

TEST_P(myriadLayersTestsROIAlign_nightly, ROIAlign) {
    _config[VPU_CONFIG_KEY(DETECT_NETWORK_BATCH)] = CONFIG_VALUE(NO);

    const tensor_test_params dims_layer_in = std::get<0>(GetParam());
    const roi_align_param test_params      = std::get<1>(GetParam());
    const uint32_t num_rois                = std::get<2>(GetParam());
    const std::string mode_str             = std::get<3>(GetParam());

    const uint32_t num_batches = dims_layer_in.n;
    const uint32_t pooled_h = test_params.pooled_h;
    const uint32_t pooled_w = test_params.pooled_w;
    const float spatial_scale = test_params.spatial_scale;

    const auto model = getModel(num_batches, dims_layer_in.c, dims_layer_in.h, dims_layer_in.w,
                                pooled_h, pooled_w, spatial_scale,
                                test_params.sampling_ratio, num_rois, mode_str);

    ASSERT_NO_THROW(readNetwork(model));

    const auto& network = _cnnNetwork;
    _inputsInfo = network.getInputsInfo();
    _inputsInfo["boxes"]->setPrecision(Precision::FP16);
    _inputsInfo["feature_map"]->setPrecision(Precision::FP16);
    _inputsInfo["batch_indices"]->setPrecision(Precision::I32);

     _outputsInfo = network.getOutputsInfo();
    _outputsInfo["roi_align"]->setPrecision(Precision::FP16);

    StatusCode st = OK;
    ASSERT_NO_THROW(st = _vpuPluginPtr->LoadNetwork(_exeNetwork, network, _config, &_resp));
    ASSERT_EQ(StatusCode::OK, st) << _resp.msg;
    ASSERT_NE(_exeNetwork, nullptr) << _resp.msg;

    ASSERT_NO_THROW(st = _exeNetwork->CreateInferRequest(_inferRequest, &_resp));
    ASSERT_EQ(StatusCode::OK, st) << _resp.msg;

    Blob::Ptr roisBlob;
    ASSERT_NO_THROW(st = _inferRequest->GetBlob("boxes", roisBlob, &_resp));
    ASSERT_EQ(StatusCode::OK, st) << _resp.msg;
    genROIs(roisBlob, test_params, num_rois);

    Blob::Ptr featureMapBlob;
    ASSERT_NO_THROW(st = _inferRequest->GetBlob("feature_map", featureMapBlob, &_resp));
    ASSERT_EQ(StatusCode::OK, st) << _resp.msg;
    GenRandomData(featureMapBlob);

    Blob::Ptr batchIndicesBlob;
    ASSERT_NO_THROW(st = _inferRequest->GetBlob("batch_indices", batchIndicesBlob, &_resp));
    ASSERT_EQ(StatusCode::OK, st) << _resp.msg;
    genBatchIndices(batchIndicesBlob, num_rois, num_batches);

    ASSERT_NO_THROW(st = _inferRequest->Infer(&_resp));
    ASSERT_EQ(StatusCode::OK, st) << _resp.msg;

    Blob::Ptr outputBlob;
    ASSERT_NO_THROW(st = _inferRequest->GetBlob("roi_align", outputBlob, &_resp));
    ASSERT_EQ(StatusCode::OK, st) << _resp.msg;

    Blob::Ptr refOutputBlob = make_shared_blob<float>({Precision::FP32,
                                                      outputBlob->getTensorDesc().getDims(),
                                                      outputBlob->getTensorDesc().getLayout()});
    refOutputBlob->allocate();

    ref_ROIAlign(featureMapBlob,
                 roisBlob,
                 batchIndicesBlob,

                 refOutputBlob,

                 test_params.sampling_ratio,
                 pooled_h, pooled_w,

                 num_rois,
                 spatial_scale,
                 mode_str);

    CompareCommonAbsolute(refOutputBlob, outputBlob, ERROR_BOUND);
}

static std::vector<Dims> s_ROIAlignLayerInput = {
    {{5, 256, 160, 157}},
};

static std::vector<roi_align_param> s_ROIAlignLayerParam = {
    {{640, 640, 7, 9, 2, 1.4f}},
};

static std::vector<number_rois> s_ROIAlignNumROIs = {
    53
};

static std::vector<roi_align_mode> s_ROIAlignMode = {
        std::string("avg"),
        std::string("max")
};