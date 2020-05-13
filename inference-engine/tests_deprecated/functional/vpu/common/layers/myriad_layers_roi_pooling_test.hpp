// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include "myriad_layers_tests.hpp"

using namespace InferenceEngine;

#define NUM_ELEM_ROIS (5)
#define ERROR_BOUND (2.5e-3f)
#define DIV_THEN_CEIL(x, y)  (((x) + (y) - 1) / (y))

struct ROIPoolingParams {
    int         in_net_w;
    int         in_net_h;
    uint32_t    pooled_w;
    uint32_t    pooled_h;
    float       spatial_scales;
};

PRETTY_PARAM(roi_pooling_param, ROIPoolingParams);

static inline void PrintTo(const ROIPoolingParams& param, ::std::ostream* os)
{
    ROIPoolingParams data = param;
    *os << "roi_pooling_param: " << data.in_net_w << ", " << data.in_net_h << ", " << data.pooled_w << ", " << data.pooled_h << ", " << data.spatial_scales;
}

typedef enum {
    roi_pooling_max = 0,
    roi_pooling_bilinear =1
} t_ROIPooling_method;

PRETTY_PARAM(roi_pooling_method, t_ROIPooling_method);

static inline void PrintTo(const t_ROIPooling_method& param, ::std::ostream* os)
{
    t_ROIPooling_method data = param;
    *os << "roi_pooling_method: " << (data == roi_pooling_bilinear? "bilinear" : "max");
}

using ROIPoolingTestParams = std::tuple<Dims, roi_pooling_param, uint32_t, roi_pooling_method, IRVersion>;

class myriadLayersTestsROIPooling_smoke: public myriadLayerTestBaseWithParam<ROIPoolingTestParams> {
public:
    void genROIs(InferenceEngine::Blob::Ptr rois,
                 const ROIPoolingParams& params,
                 const uint32_t num_rois,
                 const t_ROIPooling_method method) {

        ie_fp16 *roisBlob_data = rois->buffer().as<ie_fp16*>();
        const int max_range_width = params.in_net_w * 4 / 5;
        const int max_range_height = params.in_net_h * 4 / 5;

        float scale_width = 1.0f;
        float scale_height = 1.0f;
        if (method == roi_pooling_bilinear) {
            scale_width = (params.in_net_w);
            scale_height = (params.in_net_h);
        }

        for (int i = 0; i < num_rois; i++)
        {
            int x0 = std::rand() % max_range_width;
            int x1 = x0 + (std::rand() % (params.in_net_w - x0 - 1)) + 1;
            int y0 = std::rand() % max_range_height;
            int y1 = y0 + (std::rand() % (params.in_net_h - y0 - 1)) + 1;

            roisBlob_data[i * NUM_ELEM_ROIS + 0] = PrecisionUtils::f32tof16(0);
            roisBlob_data[i * NUM_ELEM_ROIS + 1] = PrecisionUtils::f32tof16(x0 / scale_width);
            roisBlob_data[i * NUM_ELEM_ROIS + 2] = PrecisionUtils::f32tof16(y0 / scale_height);
            roisBlob_data[i * NUM_ELEM_ROIS + 3] = PrecisionUtils::f32tof16(x1 / scale_width);
            roisBlob_data[i * NUM_ELEM_ROIS + 4] = PrecisionUtils::f32tof16(y1 / scale_height);
        }
    }

    void refROIPooling(const InferenceEngine::Blob::Ptr src,
                                const InferenceEngine::Blob::Ptr rois,
                                InferenceEngine::Blob::Ptr dst,
                                const int num_rois,
                                const ROIPoolingParams& params,
                                const tensor_test_params& in,
                                const t_ROIPooling_method method) {
        const ie_fp16* bottom3d = src->cbuffer().as<ie_fp16 *>();
        const ie_fp16* roi2d = rois->cbuffer().as<ie_fp16 *>();
        ie_fp16* top4d = dst->buffer().as<ie_fp16 *>();
        const int R = num_rois;
        const int C = in.c;
        const int H = in.h;
        const int W = in.w;
        const int pooled_h = params.pooled_h;
        const int pooled_w = params.pooled_w;
        const float spatial_scale = params.spatial_scales;
        const int top_area = pooled_h * pooled_w;
        const int top_volume = C * pooled_h * pooled_w;
        if (method == roi_pooling_max) //  generate GT for roi_pooling_max
        {
            for (int r = 0; r < R; ++r) {
                // RoI in the bottom plane
                const int x1 = std::round(PrecisionUtils::f16tof32(roi2d[r * NUM_ELEM_ROIS + 1]) * spatial_scale);
                const int y1 = std::round(PrecisionUtils::f16tof32(roi2d[r * NUM_ELEM_ROIS + 2]) * spatial_scale);
                const int x2 = std::round(PrecisionUtils::f16tof32(roi2d[r * NUM_ELEM_ROIS + 3]) * spatial_scale);
                const int y2 = std::round(PrecisionUtils::f16tof32(roi2d[r * NUM_ELEM_ROIS + 4]) * spatial_scale);
                const int roi_W = x2 - x1 + 1;
                const int roi_H = y2 - y1 + 1;

                for (int h = 0; h < pooled_h; ++h) {
                    for (int w = 0; w < pooled_w; ++w) {
                        const int hb_start  = std::min(H-1, std::max(0, y1 + (h * roi_H) / pooled_h));
                        const int hb_end    = std::min(H-1, std::max(0, y1 + DIV_THEN_CEIL((h + 1) * roi_H, pooled_h)));
                        const int wb_start  = std::min(W-1, std::max(0, x1 + (w * roi_W) / pooled_w));
                        const int wb_end    = std::min(W-1, std::max(0, x1 + DIV_THEN_CEIL((w + 1) * roi_W, pooled_w)));

                        // Usually Myriad data order is top[h][w][r][c]
                        // But the roipooling output data order is top[r][c][h][w]
                        const int plane = pooled_w * pooled_h;
                        const int top_index = (h * pooled_w) + (w) + (r * C * plane);

                        // if the bottom region is empty,
                        if (hb_start >= hb_end || wb_start >= wb_end) {
                            for (int c = 0; c < C; ++c) {
                                top4d[top_index + c * plane] = 0;
                            }
                            continue;
                        }

                        // if the bottom region is not empty,
                        //   top[r][c][h][w] = "max in the region"
                        for (int c = 0; c < C; ++c) {
                            // Myriad data order is different: bottom[h][w][c]
                            const ie_fp16* p_bottom3d = bottom3d + c;
                            int max_idx = hb_start * W * C + wb_start * C;
                            for (int hb = hb_start; hb < hb_end; ++hb) {
                                for (int wb = wb_start; wb < wb_end; ++wb) {
                                    // Data order is different
                                    const int this_idx = hb * W * C + wb * C;
                                    float this_value = PrecisionUtils::f16tof32(p_bottom3d[this_idx]);
                                    float max_value = PrecisionUtils::f16tof32(p_bottom3d[max_idx]);
                                    max_idx = (this_value > max_value) ? this_idx : max_idx;
                                }
                            }
                            top4d[top_index + c * plane] = p_bottom3d[max_idx];
                        } // endfor c
                    }
                } // endfor h, w
            } // endfor r
        } else { //  generate GT for roi_pooling_bilinear
            for (int r = 0; r < R; ++r) {
                float roi_start_w_ = PrecisionUtils::f16tof32(roi2d[r * NUM_ELEM_ROIS + 1]);//Normalized coordinates
                float roi_start_h_ = PrecisionUtils::f16tof32(roi2d[r * NUM_ELEM_ROIS + 2]);
                float roi_end_w_   = PrecisionUtils::f16tof32(roi2d[r * NUM_ELEM_ROIS + 3]);
                float roi_end_h_   = PrecisionUtils::f16tof32(roi2d[r * NUM_ELEM_ROIS + 4]);

                float height_scale = (roi_end_h_ - roi_start_h_) * (H - 1) / (pooled_h - 1);
                float width_scale  = (roi_end_w_ - roi_start_w_) * (W - 1) / (pooled_w - 1);

                for (int c = 0; c < C; ++c) {
                    const ie_fp16* p_bottom3d = bottom3d + c;
                    for (int ph = 0; ph < pooled_h; ++ph) {
                        for (int pw = 0; pw < pooled_w; ++pw) {
                            float in_y = (ph * height_scale + roi_start_h_ * (H - 1));
                            float in_x = (pw * width_scale  + roi_start_w_ * (W - 1));

                            // Usually Myriad data order is top[h][w][r][c]
                            // But the roipooling output data order is top[r][c][h][w]
                            const int top_index = (pw) + (ph * pooled_w) + (c * top_area) + (r * C * top_area);
                            if (in_y < 0 || in_y > H - 1 || in_x < 0 || in_x > W - 1) {
                                top4d[top_index] = 0;
                            } else {
                                int top_y_index    = static_cast<int>(floorf(in_y));
                                int bottom_y_index = static_cast<int>(ceilf(in_y));
                                int left_x_index   = static_cast<int>(floorf(in_x));
                                int right_x_index  = static_cast<int>(ceilf(in_x));

                                if (right_x_index > W - 1)
                                    right_x_index = W - 1;

                                if (bottom_y_index > H - 1)
                                    bottom_y_index = H - 1;

                                const float top_left     = PrecisionUtils::f16tof32(p_bottom3d[top_y_index * W * C + left_x_index * C]);
                                const float top_right    = PrecisionUtils::f16tof32(p_bottom3d[top_y_index * W * C + right_x_index * C]);
                                const float bottom_left  = PrecisionUtils::f16tof32(p_bottom3d[bottom_y_index * W * C + left_x_index * C]);
                                const float bottom_right = PrecisionUtils::f16tof32(p_bottom3d[bottom_y_index * W * C + right_x_index * C]);

                                const float top    = top_left + (top_right - top_left) * (in_x - left_x_index);
                                const float bottom = bottom_left + (bottom_right - bottom_left) * (in_x - left_x_index);

                                top4d[top_index] = PrecisionUtils::f32tof16(top + (bottom - top) * (in_y - top_y_index));
                            }
                        }
                    }
                }
            }
        }
    }
    using myriadLayersTests_nightly::makeSingleLayerNetwork;
    void makeSingleLayerNetwork(const std::map<std::string, std::string>& params,
                     const ROIPoolingParams& test_params,
                     const uint32_t num_rois,
                     const t_ROIPooling_method method)
    {
        makeSingleLayerNetwork(LayerInitParams("ROIPooling").params(params),
                               NetworkInitParams().createInference(false));
        createInferRequest(test_params, num_rois, method);
    }
    void createInferRequest(const ROIPoolingParams& params,
                            const uint32_t num_rois,
                            const t_ROIPooling_method method)
    {
        ASSERT_NO_THROW(_inputsInfo = _cnnNetwork.getInputsInfo());
        ASSERT_TRUE(_inputsInfo.size() == 2);
        _inputsInfo.begin()->second->setLayout(NHWC);
        for (auto inputInfo : _inputsInfo) {
            inputInfo.second->setPrecision(Precision::FP16);
        }

        ASSERT_NO_THROW(_outputsInfo = _cnnNetwork.getOutputsInfo());
        ASSERT_TRUE(_outputsInfo.size() == 1);
        for (auto outputInfo : _outputsInfo) {
            outputInfo.second->setPrecision(Precision::FP16);
            outputInfo.second->setLayout(NCHW);
        }

        InferenceEngine::StatusCode st = InferenceEngine::StatusCode::GENERAL_ERROR;
        ASSERT_NO_THROW(st = _vpuPluginPtr->LoadNetwork(_exeNetwork, _cnnNetwork, {}, &_resp));
        ASSERT_NE(_exeNetwork, nullptr) << _resp.msg;
        ASSERT_NO_THROW(_exeNetwork->CreateInferRequest(_inferRequest, &_resp)) << _resp.msg;
        ASSERT_EQ((int) InferenceEngine::StatusCode::OK, st) << _resp.msg;
        ASSERT_NE(_inferRequest, nullptr) << _resp.msg;

        ASSERT_NO_THROW(_inputsInfo = _cnnNetwork.getInputsInfo());
        ASSERT_NO_THROW(_outputsInfo = _cnnNetwork.getOutputsInfo());
        for (auto inpt : _inputsInfo)
        {
            InferenceEngine::Layout layout = inpt.second->getTensorDesc().getLayout();

            Blob::Ptr data;
            ASSERT_NO_THROW(st = _inferRequest->GetBlob(inpt.first.c_str(), data, &_resp));
            ASSERT_EQ(StatusCode::OK, st) << _resp.msg;

            SetSeed(3);

            if (inpt.first == _inputsInfo.begin()->first)
            {
                GenRandomData(data);
            }
            else
            {
                genROIs(data, params, num_rois, method);
            }
            _inputMap[inpt.first] = data;
        }

        Blob::Ptr data;
        ASSERT_NO_THROW(st = _inferRequest->GetBlob(_outputsInfo.begin()->first.c_str(), data, &_resp));
        ASSERT_EQ(StatusCode::OK, st) << _resp.msg;
        _outputMap[_outputsInfo.begin()->first] = data;
    }
};

TEST_P(myriadLayersTestsROIPooling_smoke, ROIPooling) {
    tensor_test_params dims_layer_in = std::get<0>(GetParam());
    ROIPoolingParams test_params = std::get<1>(GetParam());
    const uint32_t num_rois = std::get<2>(GetParam());
    const t_ROIPooling_method method = (t_ROIPooling_method)(std::get<3>(GetParam()));
    _irVersion = std::get<4>(GetParam());
    IN_OUT_desc input_tensors, output_tensors;
    input_tensors.push_back({1, dims_layer_in.c, dims_layer_in.h, dims_layer_in.w});
    input_tensors.push_back({num_rois, NUM_ELEM_ROIS});
    output_tensors.push_back({num_rois, dims_layer_in.c, test_params.pooled_h, test_params.pooled_w});

    SetInputTensors(input_tensors);
    SetOutputTensors(output_tensors);

    std::map<std::string, std::string> layer_params = {
        {"pooled_w",        std::to_string(test_params.pooled_w)},
        {"pooled_h",        std::to_string(test_params.pooled_h)},
        {"spatial_scale",  std::to_string(test_params.spatial_scales)},
        {"method",          (method == roi_pooling_bilinear? "bilinear" : "max")},
    };

    ASSERT_NO_FATAL_FAILURE(makeSingleLayerNetwork(layer_params, test_params, num_rois, method));

    ASSERT_TRUE(Infer());

    //Verify result
    auto src  = _inputMap.begin()->second;
    auto rois = std::next(_inputMap.begin())->second;
    auto dst  = _outputMap.begin()->second;

    auto _refBlob = make_shared_blob<ie_fp16>(dst->getTensorDesc());
    _refBlob->allocate();

    refROIPooling(src, rois, _refBlob, num_rois, test_params, dims_layer_in, method);

    CompareCommonAbsolute(dst, _refBlob, ERROR_BOUND);
}

static std::vector<Dims> s_ROIPoolingLayerInput = {
    {{1, 1,   14, 14}},
    {{1, 2,   14, 14}},
    {{1, 256, 14, 14}},
};

static std::vector<roi_pooling_param> s_ROIPoolingLayerParam = {
    {{224, 224, 7, 7, 0.0625}},
};

static std::vector<uint32_t> s_ROIPoolingNumRois = {
    1, 10, 30, 50, 100 
};

static std::vector<roi_pooling_method> s_ROIPoolingMethod = {
        roi_pooling_max,
        roi_pooling_bilinear,
};
