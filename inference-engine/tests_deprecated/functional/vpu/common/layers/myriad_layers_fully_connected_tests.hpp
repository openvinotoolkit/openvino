// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "myriad_layers_tests.hpp"
#include "myriad_layers_reference_functions.hpp"

using namespace InferenceEngine;

class myriadLayersTestsFullyConnected_nightly: public myriadLayersTests_nightly,
                           public testing::WithParamInterface<fcon_test_params> {
};

typedef std::tuple<InferenceEngine::SizeVector, uint32_t> IR3_FC_params;
class myriadLayersTestsFullyConnectedBatch_nightly: public myriadLayersTests_nightly,
                           public testing::WithParamInterface<IR3_FC_params> {
};

TEST_P(myriadLayersTestsFullyConnected_nightly, TestsFullyConnected)
{
    fcon_test_params p = ::testing::WithParamInterface<fcon_test_params>::GetParam();

    size_t sz_weights = p.in.c * p.in.h * p.in.w * p.out_c;
    size_t sz_bias = 0;
    size_t sz = sz_weights + sz_bias;
    InferenceEngine::TBlob<uint8_t>::Ptr weights_ptr = InferenceEngine::TBlob<uint8_t>::Ptr(GenWeights(sz));
    uint16_t* weights = weights_ptr->data().as<uint16_t*>();
    SetInputTensors({{p.in.n, p.in.c, p.in.h, p.in.w}});
    SetOutputTensors({{1, p.out_c}});
    ASSERT_NO_FATAL_FAILURE(makeSingleLayerNetwork(LayerInitParams("FullyConnected")
                                        .params({{"out-size", std::to_string(p.out_c)}})
                                        .weights(sz_weights)
                                        .biases(sz_bias),
                                        NetworkInitParams().layoutPreference(vpu::LayoutPreference::ChannelMinor),
                                        weights_ptr));
    ASSERT_TRUE(Infer());

    ref_innerproduct(_inputMap.begin()->second, _refBlob, weights,  sz_weights, nullptr, sz_bias, p.out_c);
    CompareCommonAbsolute(_outputMap.begin()->second, _refBlob, p.error_bound);
}

static void constWeightsRange1(uint16_t* ptr, size_t weightsSize) {
    ASSERT_NE(ptr, nullptr);
    float shft = 0.125f;
    float val = 0.125f;
    for (size_t count = 0 ; count < weightsSize; ++count) {
        ptr[count] = PrecisionUtils::f32tof16(val);
        val += shft;
        if (val >0.9f)
            val = -0.9f;
    }
}

static void genTestData1(InferenceEngine::Blob::Ptr blob) {
    ASSERT_NE(blob, nullptr);
    Layout layout = blob->getTensorDesc().getLayout();
    SizeVector dims = blob->getTensorDesc().getDims();
    ie_fp16* ptr = blob->buffer().as<ie_fp16*>();
    if (layout == NCHW || layout == NHWC) {
        size_t N = dims[0];
        size_t C = dims[1];
        size_t H = dims[2];
        size_t W = dims[3];
        float counter = 0.025f;
        for (size_t n = 0; n < N; n++) {
            for (size_t c = 0; c < C; c++) {
                for (size_t h = 0; h < H; h++) {
                    for (size_t w = 0; w < W; w++) {
                        size_t actualIdx = layout == NCHW ?
                                           w + h * W + c * W * H + n * W * H * C : c + w * C + h * C * W +
                                                                                   n * W * H * C;
                        ptr[actualIdx] = PrecisionUtils::f32tof16(counter);
                        counter += 0.025f;
                        if (counter > 0.99990f)
                            counter = -1.0f;
                    }
                }
            }
        }
    } else {
        ASSERT_TRUE(false);
    }
}


TEST_P(myriadLayersTestsFullyConnectedBatch_nightly, TestsFullyConnected)
{
    auto p = ::testing::WithParamInterface<IR3_FC_params>::GetParam();
    auto input_tensor = std::get<0>(p);
    uint32_t out_size = std::get<1>(p);
    int32_t IW = 0;
    int32_t IH = 0;
    int32_t IC = 0;
    int32_t I_N = 0;

    std::map<std::string, std::string> params;
    params["out-size"] = std::to_string(out_size);
    get_dims(input_tensor, IW, IH, IC, I_N);
    InferenceEngine::SizeVector output_tensor = {(size_t)I_N, (size_t)out_size};
    if (I_N > 1)
        _config[VPU_CONFIG_KEY(DETECT_NETWORK_BATCH)] = CONFIG_VALUE(NO);
    else
        _config[VPU_CONFIG_KEY(DETECT_NETWORK_BATCH)] = CONFIG_VALUE(YES);

    size_t sz_weights = IC * IH * IW * out_size;
    size_t sz_bias = 0;
    size_t sz = sz_weights + sz_bias;
    _genDataCallback = genTestData1;
    _testNet.addLayer(LayerInitParams("FullyConnected")
             .params(params)
             .weights(sz_weights).fillWeights(constWeightsRange1)
             .biases(sz_bias).fillBiases(constWeightsRange1)
             .in({input_tensor})
             .out({output_tensor}),
             ref_innerproduct_wrap);
    ASSERT_TRUE(generateNetAndInfer(NetworkInitParams().useHWOpt( CheckMyriadX() ) ));
    CompareCommonAbsolute(_outputMap.begin()->second, getReferenceOutput(), 0.02);
}

class myriadLayersTestsFullyConnectedPVA_nightly: public myriadLayersTests_nightly,
                           public testing::WithParamInterface<IR3_FC_params> {
};

TEST_P(myriadLayersTestsFullyConnectedPVA_nightly, TestsFullyConnected)
{
    auto p = ::testing::WithParamInterface<IR3_FC_params>::GetParam();
    auto input_tensor = std::get<0>(p);
    uint32_t out_size = std::get<1>(p);
    int32_t IW = 0;
    int32_t IH = 0;
    int32_t IC = 0;
    int32_t I_N = 0;

    std::map<std::string, std::string> params;
    params["out-size"] = std::to_string(out_size);
    get_dims(input_tensor, IW, IH, IC, I_N);
    InferenceEngine::SizeVector output_tensor = {(size_t)I_N, (size_t)out_size};
    if (I_N > 1)
        _config[VPU_CONFIG_KEY(DETECT_NETWORK_BATCH)] = CONFIG_VALUE(NO);
    else
        _config[VPU_CONFIG_KEY(DETECT_NETWORK_BATCH)] = CONFIG_VALUE(YES);

    size_t sz_weights = IC * IH * IW * out_size;
    size_t sz_bias = 0;
    size_t sz = sz_weights + sz_bias;
    _genDataCallback = genTestData1;
    _testNet.addLayer(LayerInitParams("FullyConnected")
             .params(params)
             .weights(sz_weights).fillWeights(constWeightsRange1)
             .biases(sz_bias).fillBiases(constWeightsRange1)
             .in({input_tensor})
             .out({output_tensor}),
             ref_innerproduct_wrap
             );
    std::map<std::string, std::string> reshape_params = {
                {"axis", "0"}
              , {"dim", "0,0"}
              , {"num_axes", "-1"}
    };
    size_t in_n = I_N;
    _testNet.addLayer(LayerInitParams("Reshape")
             .params(reshape_params)
             .in({output_tensor})
             .out({{in_n, out_size}}),
             ref_reshape_wrap);
    size_t last_sz = 8;
    std::map<std::string, std::string> fc_params;
    fc_params["out-size"] = std::to_string(last_sz);
    _testNet.addLayer(LayerInitParams("FullyConnected")
             .params(fc_params)
             .weights(out_size * last_sz).fillWeights(constWeightsRange1)
             .biases(0)
             .in({{in_n, out_size}})
             .out({{in_n, last_sz}}),
             ref_innerproduct_wrap);

    ASSERT_TRUE(generateNetAndInfer(NetworkInitParams().useHWOpt( CheckMyriadX())));
    CompareCommonAbsolute(_outputMap.begin()->second, getReferenceOutput(), 0.07);
}

static std::vector<fcon_test_params> s_fcTestParams = {
    {{1,    1, 16,  8},    8, 0.02f},
    {{1,    1,  8, 16},    8, 0.02f},
    {{1,    1,  8, 16},    4, 0.02f},
    {{1,    4,  8, 16},    4, 0.065f},
    {{1,   16, 16, 16},   16, 0.36f},
    {{1,   16,  8,  8},    8, 0.065f},
    {{1,  512,  7,  7}, 4096, 0.4f},
    {{1, 4096,  1,  1}, 4096, 0.1f},   // AlexNet layer
    {{1, 4096,  1,  1}, 1000, 0.1f},   // AlexNet layer
    {{1, 1024,  1,  1}, 1000, 0.1f},   // GoogleNet layer
    {{1,   71,  1, 78}, 6248, 0.065f}, // LPR-0001 layer
    {{1, 1024,  7,  7}, 2048, 0.5f},
    {{1,  576,  1,  1},  128, 0.02f},
    {{1, 1152,  1,  1},  128, 0.032f},
};


static const std::vector<InferenceEngine::SizeVector> s_fcTestBatchParams = {
    {10, 8, 3,  3}
};

static const std::vector<uint32_t> s_fcTestBatchOutSizes = {
    12
};

static const std::vector<InferenceEngine::SizeVector> s_fcTestPVAParams = {
    {2, 2, 7,  7}
};

static const std::vector<uint32_t> s_fcTestPVAOutSizes = {
    16
};
