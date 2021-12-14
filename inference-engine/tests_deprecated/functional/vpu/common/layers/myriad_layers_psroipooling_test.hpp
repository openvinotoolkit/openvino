// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "myriad_layers_tests.hpp"

#define ERROR_BOUND (1.2e-2f)
#define DIV_THEN_CEIL(a, b) (((a) + (b) - 1) / (b))

using namespace InferenceEngine;

struct PSROIPoolingParams {
    int in_width;
    int in_height;
    uint32_t group_size;
    uint32_t output_dim;
    float spatial_scale;
};

PRETTY_PARAM(psroipooling_param, PSROIPoolingParams);

static inline void PrintTo(const PSROIPoolingParams& param, ::std::ostream* os)
{
    PSROIPoolingParams data = param;
    *os << "psroipooling_param: " << data.in_width << ", " << data.in_height << ", " << data.group_size << ", " << data.output_dim << "," << data.spatial_scale;
}

using PSROIPoolingTestParams = std::tuple<Dims, psroipooling_param, uint32_t>;

class myriadLayersTestsPSROIPooling_smoke: public myriadLayerTestBaseWithParam<PSROIPoolingTestParams> {
public:
    void genROIs(InferenceEngine::Blob::Ptr rois,
                 const PSROIPoolingParams& params,
                 const uint32_t num_rois) {
        ie_fp16 *roisBlob_data = rois->buffer().as<ie_fp16*>();
        const int max_range_width = params.in_width * 4 / 5;
        const int max_range_height = params.in_height * 4 / 5;
        for (int i = 0; i < num_rois; i++)
        {
            int x0 = std::rand() % max_range_width;
            int x1 = x0 + (std::rand() % (params.in_width - x0 - 1)) + 1;
            int y0 = std::rand() % max_range_height;
            int y1 = y0 + (std::rand() % (params.in_height - y0 - 1)) + 1;

            roisBlob_data[i * 5 + 0] = PrecisionUtils::f32tof16(0);
            roisBlob_data[i * 5 + 1] = PrecisionUtils::f32tof16(x0);
            roisBlob_data[i * 5 + 2] = PrecisionUtils::f32tof16(y0);
            roisBlob_data[i * 5 + 3] = PrecisionUtils::f32tof16(x1);
            roisBlob_data[i * 5 + 4] = PrecisionUtils::f32tof16(y1);
        }
    }

    void refPSROIPooling(const InferenceEngine::Blob::Ptr src,
                         const InferenceEngine::Blob::Ptr rois,
                         InferenceEngine::Blob::Ptr dst,
                         const int num_rois,
                         const PSROIPoolingParams& params,
                         const tensor_test_params& in) {
        const int group_size = params.group_size;
        const float spatial_scale = params.spatial_scale;
        const int pooled_height = params.group_size;
        const int pooled_width = params.group_size;

        const int channels = in.c;
        const int height = in.h;
        const int width = in.w;

        const int nn = num_rois;
        const int nc = params.output_dim;
        const int nh = params.group_size;
        const int nw = params.group_size;

        ie_fp16* dst_data = dst->buffer().as<ie_fp16 *>();
        const ie_fp16* bottom_data_beginning = src->cbuffer().as<ie_fp16 *>();
        const ie_fp16* bottom_rois_beginning = rois->cbuffer().as<ie_fp16 *>();

        for (int n = 0; n < nn; ++n)
        {
            const ie_fp16* bottom_rois = bottom_rois_beginning + n * 5;
            int roi_batch_ind = static_cast<int>(bottom_rois[0]);
            float roi_start_w = round(PrecisionUtils::f16tof32(bottom_rois[1])) * spatial_scale;
            float roi_start_h = round(PrecisionUtils::f16tof32(bottom_rois[2])) * spatial_scale;
            float roi_end_w = round(PrecisionUtils::f16tof32(bottom_rois[3]) + 1.0f) * spatial_scale;
            float roi_end_h = round(PrecisionUtils::f16tof32(bottom_rois[4]) + 1.0f) * spatial_scale;

            float roi_width = std::max(roi_end_w - roi_start_w, 0.1f);
            float roi_height = std::max(roi_end_h - roi_start_h, 0.1f);

            int top_roi_offset = n * nc * nh * nw;
            for (int c = 0; c < nc; ++c)
            {
                int top_plane_offset = top_roi_offset + c * nh * nw;
                for (int h = 0; h < nh; ++h)
                {
                    int top_row_offset = top_plane_offset + h * nw;
                    for (int w = 0; w < nw; ++w)
                    {
                        const int index = top_row_offset + w;
                        dst_data[index] = 0;

                        int hstart = std::min(height, std::max(0, static_cast<int>(floor(roi_start_h + (h * roi_height) / pooled_height))));
                        int hend = std::min(height, std::max(0, static_cast<int>(ceil(roi_start_h + (h + 1) * roi_height / pooled_height))));
                        int wstart = std::min(width, std::max(0, static_cast<int>(floor(roi_start_w + (w * roi_width) / pooled_width))));
                        int wend = std::min(width, std::max(0, static_cast<int>(ceil(roi_start_w + (w + 1) * roi_width / pooled_width))));

                        float bin_area = (hend - hstart) * (wend - wstart);
                        if (bin_area)
                        {
                            int gc = (c * group_size + h) * group_size + w;
                            const ie_fp16* bottom_data =
                                    bottom_data_beginning + ((roi_batch_ind * channels + gc) * height * width);

                            float out_sum = 0.0f;
                            for (int hh = hstart; hh < hend; ++hh)
                                for (int ww = wstart; ww < wend; ++ww)
                                    out_sum += PrecisionUtils::f16tof32(bottom_data[hh * width + ww]);

                            dst_data[index] = PrecisionUtils::f32tof16(out_sum / bin_area);
                        }
                    }
                }
            }
        }
    }
    using myriadLayersTests_nightly::makeSingleLayerNetwork;
    void makeSingleLayerNetwork(const std::map<std::string, std::string>& params,
                     const PSROIPoolingParams& test_params,
                     const uint32_t num_rois) {
        makeSingleLayerNetwork(LayerInitParams("PSROIPooling").params(params),
                               NetworkInitParams().createInference(false));
        createInferRequest(test_params, num_rois);
    }
    void createInferRequest(const PSROIPoolingParams& params,
                            const uint32_t num_rois) {
        ASSERT_NO_THROW(_inputsInfo = _cnnNetwork.getInputsInfo());
        ASSERT_TRUE(_inputsInfo.size() == 2);
        _inputsInfo.begin()->second->setLayout(NHWC);
        for (auto inputInfo : _inputsInfo) {
            inputInfo.second->setPrecision(Precision::FP16);
            if (inputInfo.first == "input0") {
                inputInfo.second->setLayout(NCHW);
            }
        }

        ASSERT_NO_THROW(_outputsInfo = _cnnNetwork.getOutputsInfo());
        ASSERT_TRUE(_outputsInfo.size() == 1);
        for (auto outputInfo : _outputsInfo) {
            outputInfo.second->setPrecision(Precision::FP16);
            outputInfo.second->setLayout(NCHW);
        }

        ASSERT_NO_THROW(_exeNetwork = _vpuPluginPtr->LoadNetwork(_cnnNetwork));
        ASSERT_NO_THROW(_inferRequest = _exeNetwork.CreateInferRequest());

        ASSERT_NO_THROW(_inputsInfo = _cnnNetwork.getInputsInfo());
        ASSERT_NO_THROW(_outputsInfo = _cnnNetwork.getOutputsInfo());
        SetSeed(DEFAULT_SEED_VALUE);

        for (auto inputInfo: _inputsInfo)
        {
            InferenceEngine::SizeVector inputDims = inputInfo.second->getTensorDesc().getDims();

            Blob::Ptr data;
            ASSERT_NO_THROW(data = _inferRequest.GetBlob(inputInfo.first.c_str()));
            
            if (inputInfo.first == _inputsInfo.begin()->first)
                GenRandomData(data);
            else
                genROIs(data, params, num_rois);

            _inputMap[inputInfo.first] = data;
        }

        Blob::Ptr data;
        ASSERT_NO_THROW(data = _inferRequest.GetBlob(_outputsInfo.begin()->first.c_str()));
        _outputMap[_outputsInfo.begin()->first] = data;
    }

};

static std::vector<Dims> s_PSROIPoolingLayerInput = {
        {{1, 1029, 14, 14}},
};

static std::vector<psroipooling_param> s_PSROIPoolingLayerParam = {
        {{224, 224, 7, 21, 0.0625}},
};

static std::vector<uint32_t> s_PSROIPoolingNumROIs = {
        1, 10, 30, 50, 100, 300
};

TEST_P(myriadLayersTestsPSROIPooling_smoke, PSROIPooling) {
#ifdef _WIN32
    GTEST_SKIP() << "Disabled for Windows. Issue-13239";
#endif
    tensor_test_params dims_layer_in = std::get<0>(GetParam());
    PSROIPoolingParams test_params = std::get<1>(GetParam());
    const uint32_t num_rois = std::get<2>(GetParam());
    IN_OUT_desc input_tensors, output_tensors;
    input_tensors.push_back({1, dims_layer_in.c, dims_layer_in.h, dims_layer_in.w});
    input_tensors.push_back({num_rois, 5});
    output_tensors.push_back({num_rois,  test_params.output_dim, test_params.group_size, test_params.group_size});

    SetInputTensors(input_tensors);
    SetOutputTensors(output_tensors);

    std::map<std::string, std::string> layer_params = {
        {"group_size", std::to_string(test_params.group_size)},
        {"output_dim", std::to_string(test_params.output_dim)},
        {"spatial_scale", std::to_string(test_params.spatial_scale)},
    };

    ASSERT_NO_FATAL_FAILURE(makeSingleLayerNetwork(layer_params, test_params, num_rois));

    ASSERT_TRUE(Infer());

    auto src = _inputMap.begin()->second;
    auto rois = std::next(_inputMap.begin())->second;
    auto dst = _outputMap.begin()->second;

    InferenceEngine::TBlob<ie_fp16>::Ptr _refBlob = make_shared_blob<ie_fp16>(dst->getTensorDesc());
    _refBlob->allocate();

    refPSROIPooling(src, rois, _refBlob, num_rois, test_params, dims_layer_in);

    CompareCommonAbsolute(dst, _refBlob, ERROR_BOUND);
}
