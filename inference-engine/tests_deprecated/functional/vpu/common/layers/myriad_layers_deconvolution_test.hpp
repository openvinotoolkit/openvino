// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <deconv_ref.hpp>
#include "myriad_layers_tests.hpp"
#include "common_layers_params.hpp"

using std::tuple;
using std::get;

using namespace InferenceEngine;


static void refDeconvolution(const Blob::Ptr src, Blob::Ptr dst
        , const ie_fp16* weights_data, const ie_fp16* bias_data
        , param_size &kernel, param_size &stride, param_size &pad, size_t group) {
    CommonTestUtils::conv_common_params params;
    params.kernel.insert(X_AXIS, kernel.x);
    params.kernel.insert(Y_AXIS, kernel.y);
    params.stride.insert(X_AXIS, stride.x);
    params.stride.insert(Y_AXIS, stride.y);
    params.pads_begin.insert(X_AXIS, pad.x);
    params.pads_begin.insert(Y_AXIS, pad.y);
    params.group = group;
    ref_deconv_common<ie_fp16>({ src }, *dst.get(), weights_data, 0, bias_data, 0, params);
}

PRETTY_PARAM(kernel, param_size)
PRETTY_PARAM(stride, param_size)

PRETTY_PARAM(pad, param_size)
PRETTY_PARAM(pad_end, param_size)

PRETTY_PARAM(out_channels, int)
PRETTY_PARAM(group, int)
PRETTY_PARAM(layoutPreference, vpu::LayoutPreference)
PRETTY_PARAM(hw_optimization, bool)

typedef myriadLayerTestBaseWithParam<tuple<DimsInput, kernel, stride, pad
        , out_channels, group, layoutPreference, hw_optimization >> myriadLayerDeconvolution_smoke;

typedef myriadLayerTestBaseWithParam<tuple<DimsInput, kernel, stride, pad, pad_end
        , out_channels, group, layoutPreference, hw_optimization >> myriadLayerDeconvolution_asymm_pad;

TEST_P(myriadLayerDeconvolution_smoke, Deconvolution) {
    tensor_test_params input_dims = get<0>(GetParam());
    param_size kernel = get<1>(GetParam());
    param_size stride = get<2>(GetParam());
    param_size pad = get<3>(GetParam());
    size_t out_channels = get<4>(GetParam());
    size_t group = get<5>(GetParam());
    auto layoutPreference = get<6>(GetParam());
    bool hw_optimization = get<7>(GetParam());

    if(hw_optimization && !CheckMyriadX()) {
        GTEST_SKIP_("Skip test with hw_optimization=On for Myriad2\n");
    }

    if (input_dims.n > 1)
        _config[InferenceEngine::MYRIAD_DETECT_NETWORK_BATCH] = CONFIG_VALUE(NO);
    else
        _config[InferenceEngine::MYRIAD_DETECT_NETWORK_BATCH] = CONFIG_VALUE(YES);

    size_t out_w = stride.x * (input_dims.w - 1) + kernel.x - 2 * pad.x;
    size_t out_h = stride.y * (input_dims.h - 1) + kernel.y - 2 * pad.y;

    tensor_test_params output_dims = {input_dims.n, out_channels, out_h, out_w};

    SetInputTensor(input_dims);
    SetOutputTensor(output_dims);

    size_t num_weights = kernel.x * kernel.y * (input_dims.c / group) * output_dims.c;
    size_t num_bias = output_dims.c;

    InferenceEngine::TBlob<uint8_t>::Ptr weights_ptr =
            InferenceEngine::TBlob<uint8_t>::Ptr(GenWeights(num_weights + num_bias));
    ie_fp16* weights = weights_ptr->data().as<ie_fp16*>();
    ie_fp16* bias = weights + num_weights;

    std::map<std::string, std::string> layer_params = {
              {"kernel-x", std::to_string(kernel.x)}
            , {"kernel-y", std::to_string(kernel.y)}
            , {"stride-x", std::to_string(stride.x)}
            , {"stride-y", std::to_string(stride.y)}

            , {"pad-x", std::to_string(pad.x)}
            , {"pad-y", std::to_string(pad.y)}

            , {"output", std::to_string(out_channels)}
            , {"group", std::to_string(group)}
    };

    ASSERT_NO_FATAL_FAILURE(makeSingleLayerNetwork(LayerInitParams("Deconvolution")
                                        .params(layer_params)
                                        .weights(num_weights)
                                        .biases(num_bias),
                                        NetworkInitParams().layoutPreference(layoutPreference)
                                        .useHWOpt(hw_optimization),
                                        weights_ptr));

    auto inputBlob = _inputMap.begin()->second;
    SetFirstInputToRange(-0.9f, 0.9f);

    ASSERT_TRUE(Infer());

    auto outputBlob = _outputMap.begin()->second;

    refDeconvolution(inputBlob, _refBlob, weights, bias, kernel, stride, pad, group);

    float maxerr = 0.00075 * (input_dims.c / group) * kernel.x * kernel.y;
    CompareCommonAbsolute(outputBlob, _refBlob, maxerr);
}

TEST_P(myriadLayerDeconvolution_asymm_pad, Deconvolution) {
    tensor_test_params input_dims = get<0>(GetParam());
    param_size kernel = get<1>(GetParam());
    param_size stride = get<2>(GetParam());
    param_size pad = get<3>(GetParam());
    param_size pad_end = get<4>(GetParam());
    size_t out_channels = get<5>(GetParam());
    size_t group = get<6>(GetParam());
    auto layoutPreference = get<7>(GetParam());
    bool hw_optimization = get<8>(GetParam());

    if(hw_optimization && !CheckMyriadX()) {
        GTEST_SKIP_("Skip test with hw_optimization=On for Myriad2\n");
    }

    if (input_dims.n > 1)
        _config[InferenceEngine::MYRIAD_DETECT_NETWORK_BATCH] = CONFIG_VALUE(NO);
    else
        _config[InferenceEngine::MYRIAD_DETECT_NETWORK_BATCH] = CONFIG_VALUE(YES);

    size_t out_w = stride.x * (input_dims.w - 1) + kernel.x - (pad.x + pad_end.x);
    size_t out_h = stride.y * (input_dims.h - 1) + kernel.y - (pad.y + pad_end.y);

    tensor_test_params output_dims = {input_dims.n, out_channels, out_h, out_w};

    SetInputTensor(input_dims);
    SetOutputTensor(output_dims);

    size_t num_weights = kernel.x * kernel.y * (input_dims.c / group) * output_dims.c;
    size_t num_bias = output_dims.c;

    InferenceEngine::TBlob<uint8_t>::Ptr weights_ptr =
            InferenceEngine::TBlob<uint8_t>::Ptr(GenWeights(num_weights + num_bias));
    ie_fp16* weights = weights_ptr->data().as<ie_fp16*>();
    ie_fp16* bias = weights + num_weights;

    std::map<std::string, std::string> layer_params = {
              {"kernel-x", std::to_string(kernel.x)}
            , {"kernel-y", std::to_string(kernel.y)}
            , {"stride-x", std::to_string(stride.x)}
            , {"stride-y", std::to_string(stride.y)}

            , {"pad-x", std::to_string(pad.x)}
            , {"pad-y", std::to_string(pad.y)}

            , {"pad-r", std::to_string(pad_end.x)}
            , {"pad-b", std::to_string(pad_end.y)}

            , {"output", std::to_string(out_channels)}
            , {"group", std::to_string(group)}
    };

    ASSERT_NO_FATAL_FAILURE(makeSingleLayerNetwork(LayerInitParams("Deconvolution")
                                        .params(layer_params)
                                        .weights(num_weights)
                                        .biases(num_bias),
                                        NetworkInitParams().layoutPreference(layoutPreference)
                                        .useHWOpt(hw_optimization),
                                        weights_ptr));

    auto inputBlob = _inputMap.begin()->second;
    SetFirstInputToRange(-0.9f, 0.9f);

    ASSERT_TRUE(Infer());

    auto outputBlob = _outputMap.begin()->second;

    refDeconvolution(inputBlob, _refBlob, weights, bias, kernel, stride, pad, group);

    float maxerr = 0.00075 * (input_dims.c / group) * kernel.x * kernel.y;
    CompareCommonAbsolute(outputBlob, _refBlob, maxerr);
}
