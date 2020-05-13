// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cmath>
#include "myriad_layers_tests.hpp"
#include "myriad_layers_reference_functions.hpp"


#define ERROR_BOUND (1.2e-2f)

using namespace InferenceEngine;

extern const char POOLING_MAX[] = "max";
extern const char POOLING_AVG[] = "avg";


class myriadLayersTestsMax_smoke: public PoolingTest<POOLING_MAX>
{
};

class myriadLayersTestsMaxOverlappedByKernel_smoke: public PoolingTestPad4<POOLING_MAX, true>
{
};

class myriadLayersTestsMaxPad4_smoke: public PoolingTestPad4<POOLING_MAX>
{
};

class myriadLayersTestsGlobalMax_smoke: public GlobalPoolingTest<POOLING_MAX>
{
};

class myriadLayersTestsAvg_smoke: public PoolingTest<POOLING_AVG>
{
};

class myriadLayersTestsAvgOverlappedByKernel_smoke: public PoolingTestPad4<POOLING_AVG, true>
{
};

class myriadLayersTestsAvgPad4_smoke: public PoolingTestPad4<POOLING_AVG>
{
};

class myriadLayersTestsGlobalAvg_smoke: public GlobalPoolingTest<POOLING_AVG>
{
};

/* IR version 3 tests, main difference is a changes in padding parameters definitions */
/*                   input tensor,               kernel,     stride,    pads_begin, pads_end,  auto_pad,     exclude_pad  method */
typedef std::tuple<InferenceEngine::SizeVector, param_size, param_size, param_size, param_size, const char*, const char*, const char*> IR3_PoolParams;

class myriadLayers_IR3_PoolingTests_smoke: public myriadLayersTests_nightly, /*input tensor, kernel, stride, pads_begin, pads_end, out_channel, group */
                                             public testing::WithParamInterface<IR3_PoolParams> {
};

static void genTestData(InferenceEngine::Blob::Ptr blob) {
    ASSERT_NE(blob, nullptr);
    Layout layout = blob->getTensorDesc().getLayout();
    SizeVector dims = blob->getTensorDesc().getDims();

    ie_fp16* ptr = blob->buffer().as<ie_fp16*>();
    if (layout == NCHW || layout == NHWC) {
        size_t N = dims[0];
        size_t C = dims[1];
        size_t H = dims[2];
        size_t W = dims[3];

        float counter = 0.125f;
        for (size_t n = 0; n < N; n++) {
            for (size_t c = 0; c < C; c++) {
                for (size_t h = 0; h < H; h++) {
                    for (size_t w = 0; w < W; w++) {
                        size_t actualIdx = layout == NCHW ?
                                           w + h * W + c * W * H + n * W * H * C : c + w * C + h * C * W +
                                                                                   n * W * H * C;
                        ptr[actualIdx] = PrecisionUtils::f32tof16(counter);
                        counter += 0.025f;
                        if (counter > 5.0f) {
                            counter = -0.5f;
                        }
                    }
                }
            }
        }
    } else {
        ASSERT_TRUE(false);
    }
}


TEST_P(myriadLayers_IR3_PoolingTests_smoke, Pooling) {
    std::map<std::string, std::string> params;
    InferenceEngine::SizeVector output_tensor;
    int32_t IW = 0;
    int32_t IH = 0;
    int32_t IC = 0;
    int32_t I_N = 0;
    size_t  group = 0;
    /*input tensor,               kernel,     stride,    pads_begin, pads_end,  auto_pad,     exclude_pad  method */
    auto p = ::testing::WithParamInterface<IR3_PoolParams>::GetParam();
    auto input_tensor       = std::get<0>(p);
    param_size kernel       = std::get<1>(p);
    param_size stride       = std::get<2>(p);
    param_size pads_begin   = std::get<3>(p);
    param_size pads_end     = std::get<4>(p);
    const char* auto_pad    = std::get<5>(p);
    const std::string exclude_pad = std::get<6>(p);
    const std::string method      = std::get<7>(p);

    get_dims(input_tensor, IW, IH, IC, I_N);
    if (I_N > 1)
        _config[VPU_CONFIG_KEY(DETECT_NETWORK_BATCH)] = CONFIG_VALUE(NO);
    else
        _config[VPU_CONFIG_KEY(DETECT_NETWORK_BATCH)] = CONFIG_VALUE(YES);
    int32_t OW = 1;
    int32_t OH = 1;
    int32_t OC = 1;
    int32_t ON = 1;
    if (strncmp(auto_pad, "same_upper", strlen(auto_pad)) == 0) {
        OW = input_tensor[3]/2;
        OH = input_tensor[2]/2;
        OC = input_tensor[1];
        ON = input_tensor[0];
    } else {
        ASSERT_TRUE(false);
    }
    if (kernel.x == 4 && kernel.y == 4) {
        /* particular case  for Faster-RCNN */
        OW = input_tensor[3] / kernel.x;
        OH = input_tensor[2] / kernel.y;
        OC = input_tensor[1];
        ON = input_tensor[0];
    }

    gen_dims(output_tensor, input_tensor.size(), OW, OH, OC, ON);

    std::string padsB   = gen_param(pads_begin);
    std::string padsE   = gen_param(pads_end);
    std::string strides = gen_param(stride);
    std::string kern    = gen_param(kernel);

    std::map<std::string, std::string> layer_params = {
              {"kernel",      kern}
            , {"strides",     strides}
            , {"pads_begin",  padsB}
            , {"pads_end",    padsE}
            , {"auto_pad",    auto_pad}
            , {"exclude_pad", exclude_pad}
            , {"pool-method",      method}
    };
    if (kernel.x == 4 && kernel.y == 4) {
        layer_params.erase("auto_pad");
        layer_params["rounding-type"] = "ceil";
    }
    _genDataCallback = genTestData;

    _testNet.addLayer(LayerInitParams("Pooling")
             .params(layer_params)
             .in({input_tensor})
             .out({output_tensor}),
             ref_pooling_wrap);
    ASSERT_TRUE(generateNetAndInfer(NetworkInitParams().useHWOpt(CheckMyriadX())));
    float maxerr = 0.0001f;
    CompareCommonAbsolute(_outputMap.begin()->second, getReferenceOutput(), maxerr);
}

class myriadLayers_IR3_BatchPoolingTests_smoke: public myriadLayersTests_nightly, /*input tensor, kernel, stride, pads_begin, pads_end, out_channel, group */
                                                  public testing::WithParamInterface<IR3_PoolParams> {
};

TEST_P(myriadLayers_IR3_BatchPoolingTests_smoke, Pooling) {
    std::map<std::string, std::string> params;
    InferenceEngine::SizeVector output_tensor;
    int32_t IW = 0;
    int32_t IH = 0;
    int32_t IC = 0;
    int32_t I_N = 0;
    size_t  group = 0;
    /*input tensor,               kernel,     stride,    pads_begin, pads_end,  auto_pad,     exclude_pad  method */
    auto p = ::testing::WithParamInterface<IR3_PoolParams>::GetParam();
    auto input_tensor       = std::get<0>(p);
    param_size kernel       = std::get<1>(p);
    param_size stride       = std::get<2>(p);
    param_size pads_begin   = std::get<3>(p);
    param_size pads_end     = std::get<4>(p);
    const char* auto_pad    = std::get<5>(p);
    const std::string exclude_pad = std::get<6>(p);
    const std::string method      = std::get<7>(p);

    get_dims(input_tensor, IW, IH, IC, I_N);
    if (I_N > 1)
        _config[VPU_CONFIG_KEY(DETECT_NETWORK_BATCH)] = CONFIG_VALUE(NO);
    else
        _config[VPU_CONFIG_KEY(DETECT_NETWORK_BATCH)] = CONFIG_VALUE(YES);
    int32_t OW = 1;
    int32_t OH = 1;
    int32_t OC = 1;
    int32_t ON = 1;
    if (strncmp(auto_pad, "same_upper", strlen(auto_pad)) == 0) {
        OW = input_tensor[3]/2;
        OH = input_tensor[2]/2;
        OC = input_tensor[1];
        ON = input_tensor[0];
    }
    if (kernel.x == 1 && kernel.y == 1 &&
        pads_begin.x == 0 && pads_begin.y == 0 &&
        pads_end.x == 0 && pads_end.y == 0) {
        OW = input_tensor[3];
        OH = input_tensor[2];
        OC = input_tensor[1];
        ON = input_tensor[0];
    }
    if (kernel.x == 2 && kernel.y == 2 && stride.x == 1 && stride.y == 1) {
        OW = input_tensor[3];
        OH = input_tensor[2];
        OC = input_tensor[1];
        ON = input_tensor[0];
    }
    gen_dims(output_tensor, input_tensor.size(), OW, OH, OC, ON);

    std::string padsB   = gen_param(pads_begin);
    std::string padsE   = gen_param(pads_end);
    std::string strides = gen_param(stride);
    std::string kern    = gen_param(kernel);

    std::map<std::string, std::string> layer_params = {
              {"kernel",      kern}
            , {"strides",     strides}
            , {"pads_begin",  padsB}
            , {"pads_end",    padsE}
            , {"auto_pad",    auto_pad}
            , {"exclude_pad", exclude_pad}
            , {"pool-method",      method}
    };
    _genDataCallback = genTestData;
    /*
    */
    _testNet.addLayer(LayerInitParams("Pooling")
             .params(layer_params)
             .in({input_tensor})
             .out({output_tensor}),
             ref_pooling_wrap);
    ASSERT_TRUE(generateNetAndInfer(NetworkInitParams().useHWOpt(CheckMyriadX())));
    float maxerr = 0.0001f;
    CompareCommonAbsolute(_outputMap.begin()->second, getReferenceOutput(), maxerr);
}

static const std::vector<const char*> s_poolingAutoPad = {
        "same_upper"
};

static const std::vector<const char*> s_poolingExcludePad = {
        "true"
};

static const std::vector<const char*> s_poolingMethod = {
        "max"
};

TEST_P(myriadLayersTestsMax_smoke, MaxPooling)
{
    ASSERT_TRUE(generateNetAndInfer(NetworkInitParams().layoutPreference(_layout_preference)));
    CompareCommonAbsolute(_outputMap.begin()->second, getReferenceOutput(), ERROR_BOUND);
}

TEST_P(myriadLayersTestsMaxOverlappedByKernel_smoke, MaxPooling)
{
    ASSERT_TRUE(generateNetAndInfer(NetworkInitParams().layoutPreference(_layout_preference)));
    CompareCommonAbsolute(_outputMap.begin()->second, getReferenceOutput(), ERROR_BOUND);
}

TEST_P(myriadLayersTestsMaxPad4_smoke, MaxPoolingPad4)
{
    ASSERT_TRUE(generateNetAndInfer(NetworkInitParams().layoutPreference(_layout_preference)));
    auto refBlob = getReferenceOutput();
    CompareCommonAbsolute(_outputMap.begin()->second, refBlob, ERROR_BOUND);
}

TEST_P(myriadLayersTestsAvg_smoke, AvgPooling)
{
    ASSERT_TRUE(generateNetAndInfer(NetworkInitParams().layoutPreference(_layout_preference)));
    CompareCommonAbsolute(_outputMap.begin()->second, getReferenceOutput(), ERROR_BOUND);
}

TEST_P(myriadLayersTestsAvgOverlappedByKernel_smoke, AvgPooling)
{
    ASSERT_TRUE(generateNetAndInfer(NetworkInitParams().layoutPreference(_layout_preference)));
    CompareCommonAbsolute(_outputMap.begin()->second, getReferenceOutput(), ERROR_BOUND);
}

TEST_P(myriadLayersTestsAvgPad4_smoke, AvgPoolingPad4)
{
    ASSERT_TRUE(generateNetAndInfer(NetworkInitParams().layoutPreference(_layout_preference)));
    auto refBlob = getReferenceOutput();
    CompareCommonAbsolute(_outputMap.begin()->second, refBlob, ERROR_BOUND);
}

TEST_P(myriadLayersTestsGlobalMax_smoke, GlobalMaxPooling)
{
    ASSERT_TRUE(generateNetAndInfer(NetworkInitParams()));
    auto refBlob = getReferenceOutput();
    CompareCommonAbsolute(_outputMap.begin()->second, refBlob, ERROR_BOUND);
}

TEST_P(myriadLayersTestsGlobalAvg_smoke, GlobalAvgPooling)
{
    if(_pad_val.x != 0 || _pad_val.y != 0) {
        GTEST_SKIP() << "paddings should not be exist for GlobalAvgPool";
    }

    ASSERT_TRUE(generateNetAndInfer(NetworkInitParams()));
    auto refBlob = getReferenceOutput();
    CompareCommonAbsolute(_outputMap.begin()->second, refBlob, ERROR_BOUND);
}

static std::vector<pooling_layer_params> s_poolingLayerParams_k3x3 = {
        {{3, 3}, {1, 1}, {1, 1}},
};

const std::vector<InferenceEngine::SizeVector> g_poolingInputPad4 = {
        {{1, 3,  224,  224}}
};

const std::vector<param_size> g_poolingKernelPad4 = {
        {4, 4},
        {6, 6},
        {8, 8},
};

const std::vector<param_size> g_poolingStridePad4 = {
        {1, 1},
};

const std::vector<paddings4> g_poolingPad4 = {
        {0, 0, 2, 0},
        {1, 2, 3, 2},
        {2, 2, 0, 0},
};

const std::vector<GlobalPoolingTestParam> g_GlobalPoolingInput = {
#if 0 // temporary OFF because of HACKS for rfcn #ifdef MORE_DIMENSIONS // 4DGP
        {{2,  8,    7,  7}},
#endif
        {GlobalPoolingTestParam{{1,  128,  2,  2}, { 3,  3}, {1, 1}, {2, 2}}},
        {GlobalPoolingTestParam{{1, 1024, 64, 32}, {32, 64}, {0, 0}, {1, 1}}},
        {GlobalPoolingTestParam{{1, 2048,  8,  8}, { 8,  8}, {0, 0}, {1, 1}}},
        {GlobalPoolingTestParam{{1, 2048,  7,  7}, { 7,  7}, {0, 0}, {1, 1}}},
        {GlobalPoolingTestParam{{1, 1000, 15, 15}, {15, 15}, {0, 0}, {1, 1}}},
        {GlobalPoolingTestParam{{1, 1000, 14, 14}, {14, 14}, {0, 0}, {1, 1}}},
        {GlobalPoolingTestParam{{1, 1000, 12, 12}, {12, 12}, {0, 0}, {1, 1}}},
        {GlobalPoolingTestParam{{1,  8,    7,  7}, { 7,  7}, {0, 0}, {1, 1}}},
        {GlobalPoolingTestParam{{1,  2,    7,  7}, { 7,  7}, {0, 0}, {1, 1}}},
        {GlobalPoolingTestParam{{1,  8,    7,  7}, { 7,  7}, {0, 0}, {1, 1}}},
        {GlobalPoolingTestParam{{1,  1000, 2,  3}, { 3,  2}, {0, 0}, {1, 1}}},
};
