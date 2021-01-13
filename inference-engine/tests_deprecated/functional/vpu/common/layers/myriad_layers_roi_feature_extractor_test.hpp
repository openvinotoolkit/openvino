// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include "myriad_layers_reference_functions.hpp"
#include "myriad_layers_tests.hpp"

using namespace InferenceEngine;

#define NUM_ELEM_ROIS (4)
#define ERROR_BOUND (2.5e-3f)

struct roi_feature_extractor_param {
    int         in_net_w;
    int         in_net_h;
    uint32_t    output_size;
    int         sampling_ratio;

    friend std::ostream& operator<<(std::ostream& os, roi_feature_extractor_param const& tst)
    {
        return os << "input net width = " << tst.in_net_w
                  << ", input net height = " << tst.in_net_h
                  << ", output_size = " << tst.output_size
                  << ", sampling_ratio = " << tst.sampling_ratio;
    };
};

PRETTY_PARAM(number_rois, uint32_t);

using ROIFeatureExtractorTestParams = std::tuple<Dims, roi_feature_extractor_param, number_rois>;

typedef myriadLayerTestBaseWithParam<ROIFeatureExtractorTestParams> myriadLayersTestsROIFeatureExtractor_smoke;

static void genROIs(InferenceEngine::Blob::Ptr rois,
                    const roi_feature_extractor_param& params,
                    const uint32_t num_rois) {
    ie_fp16 *roisBlob_data = rois->buffer().as<ie_fp16*>();
    const int max_range_width = params.in_net_w * 4 / 5;
    const int max_range_height = params.in_net_h * 4 / 5;

    float scale_width = (float)params.in_net_w;
    float scale_height = (float)params.in_net_h;

    for (int i = 0; i < num_rois; i++) {
        int x0 = std::rand() % max_range_width;
        int x1 = x0 + (std::rand() % (params.in_net_w - x0 - 1)) + 1;
        int y0 = std::rand() % max_range_height;
        int y1 = y0 + (std::rand() % (params.in_net_h - y0 - 1)) + 1;

        roisBlob_data[i * NUM_ELEM_ROIS + 0] = PrecisionUtils::f32tof16(x0);
        roisBlob_data[i * NUM_ELEM_ROIS + 1] = PrecisionUtils::f32tof16(y0);
        roisBlob_data[i * NUM_ELEM_ROIS + 2] = PrecisionUtils::f32tof16(x1);
        roisBlob_data[i * NUM_ELEM_ROIS + 3] = PrecisionUtils::f32tof16(y1);
    }
}

TEST_P(myriadLayersTestsROIFeatureExtractor_smoke, ROIFeatureExtractor) {
    tensor_test_params dims_layer_in = std::get<0>(GetParam());
    roi_feature_extractor_param test_params = std::get<1>(GetParam());
    const uint32_t num_rois = std::get<2>(GetParam());

    bool use_output_rois = true;
    const int levels_num = 4;

    _config[InferenceEngine::MYRIAD_DETECT_NETWORK_BATCH] = CONFIG_VALUE(NO);

    IN_OUT_desc input_tensors, output_tensors;
    input_tensors.push_back({num_rois, NUM_ELEM_ROIS});
    for (int i = 0; i < levels_num; i++) {
        input_tensors.push_back({1, dims_layer_in.c, dims_layer_in.h / (1 << i), dims_layer_in.w / (1 << i)});
    }
    output_tensors.push_back({num_rois, dims_layer_in.c, test_params.output_size, test_params.output_size});
    // adding output ROIs
    if (use_output_rois)
        output_tensors.push_back({num_rois, NUM_ELEM_ROIS});

    SetInputTensors(input_tensors);
    SetOutputTensors(output_tensors);

    std::vector<int> pyramid_scales = {4, 8, 16, 32, 64};
    std::string pyramid_scales_str = "";
    for (auto i = 0; i < pyramid_scales.size(); i++) {
        pyramid_scales_str += std::to_string(pyramid_scales[i]);
        if (i != pyramid_scales.size() - 1) pyramid_scales_str += ",";
    }

    std::map<std::string, std::string> layer_params = {
        {"output_size",     std::to_string(test_params.output_size)},
        {"sampling_ratio",  std::to_string(test_params.sampling_ratio)},
        {"pyramid_scales",  pyramid_scales_str},
    };

    makeSingleLayerNetwork(LayerInitParams("ExperimentalDetectronROIFeatureExtractor").params(layer_params));

    /* Input data generating */
    for (auto blob : _inputMap) {
        if (blob.second == _inputMap.begin()->second) {
            genROIs(blob.second, test_params, num_rois);
        } else {
            GenRandomData(blob.second);
        }
    }

    std::vector<InferenceEngine::Blob::Ptr> refInputBlobs;
    std::vector<InferenceEngine::Blob::Ptr> refOutputBlobs;
    for (auto blob : _inputMap) {
        refInputBlobs.push_back(blob.second);
    }
    for (auto blob : _outputMap) {
        auto refOutputBlob = make_shared_blob<float>({Precision::FP32,
                                                      blob.second->getTensorDesc().getDims(),
                                                      blob.second->getTensorDesc().getLayout()});
        refOutputBlob->allocate();
        refOutputBlobs.push_back(refOutputBlob);
    }
    ref_ROIFeatureExtractor(refInputBlobs,
                            refOutputBlobs[0],
                            use_output_rois ? refOutputBlobs[1] : nullptr,
                            pyramid_scales,
                            test_params.sampling_ratio,
                            test_params.output_size,
                            test_params.output_size);

    ASSERT_TRUE(Infer());

    auto dst0 = _outputMap.begin()->second;
    CompareCommonAbsolute(dst0, refOutputBlobs[0], ERROR_BOUND);
    if (use_output_rois) {
        auto dst1 = (++_outputMap.begin())->second;
        CompareCommonAbsolute(dst1, refOutputBlobs[1], ERROR_BOUND);
    }
}

static std::vector<Dims> s_ROIFeatureExtractorLayerInput = {
    {{1, 256, 160, 160}},
};

static std::vector<roi_feature_extractor_param> s_ROIFeatureExtractorLayerParam = {
    {{640, 640, 7, 2}},
};

static std::vector<number_rois> s_ROIFeatureExtractorNumROIs = {
    50
};
