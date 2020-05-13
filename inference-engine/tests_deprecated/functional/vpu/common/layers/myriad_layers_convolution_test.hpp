// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "myriad_layers_tests.hpp"
#include "myriad_layers_reference_functions.hpp"
#include "weights_for_convolution_test.h"

#include "conv_ref.hpp"

using std::tuple;
using std::get;

using namespace InferenceEngine;

PRETTY_PARAM(kernel, param_size);
PRETTY_PARAM(stride, param_size);
PRETTY_PARAM(pad, param_size);
PRETTY_PARAM(out_channels, int);
PRETTY_PARAM(group, int);
PRETTY_PARAM(dilation_factor, param_size);
PRETTY_PARAM(layoutPreference, vpu::LayoutPreference);

typedef myriadLayerTestBaseWithParam<tuple<DimsInput, kernel, stride, pad
        , out_channels, group, dilation_factor, layoutPreference >> myriadLayerConvolution_smoke;

typedef myriadLayerTestBaseWithParam<tuple<DimsInput, DimsOutput, kernel, stride, pad
        , group, dilation_factor, layoutPreference >> myriadLayerConvolutionTensorFlow_smoke;

TEST_P(myriadLayerConvolution_smoke, Convolution) {
    tensor_test_params input_dims = get<0>(GetParam());
    param_size kernel = get<1>(GetParam());
    param_size stride = get<2>(GetParam());
    param_size pad = get<3>(GetParam());
    size_t out_channels = get<4>(GetParam());
    size_t group = get<5>(GetParam());
    param_size dilation_factor = get<6>(GetParam());
    vpu::LayoutPreference layoutPreference = get<7>(GetParam());

    size_t out_w = (input_dims.w + 2 * pad.x - dilation_factor.x * (kernel.x - 1) - 1 + stride.x) / stride.x;
    size_t out_h = (input_dims.h + 2 * pad.y - dilation_factor.y * (kernel.y - 1) - 1 + stride.y) / stride.y;

    tensor_test_params output_dims = {1, out_channels, out_h, out_w};

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
            , {"dilation-x", std::to_string(dilation_factor.x)}
            , {"dilation-y", std::to_string(dilation_factor.y)}
    };
    ASSERT_NO_FATAL_FAILURE(makeSingleLayerNetwork(LayerInitParams("Convolution")
                                        .params(layer_params)
                                        .weights(num_weights)
                                        .biases(num_bias),
                                        {},
                                        weights_ptr));
    SetFirstInputToRange(-0.9f, 0.9f);

    ASSERT_TRUE(Infer());
    auto inputBlob = _inputMap.begin()->second;
    auto outputBlob = _outputMap.begin()->second;

    ref_convolution(inputBlob, _refBlob, weights, bias, kernel, stride, pad, group, dilation_factor);

    float maxerr = 0;

    if (group == 1)
        maxerr = 0.00055 * input_dims.c * kernel.x * kernel.y;
    else // TODO: currently dephConv is slightly less accurate
        maxerr = 0.00066 * (input_dims.c / group) * kernel.x * kernel.y;

    CompareCommonAbsolute(outputBlob, _refBlob, maxerr);
}

TEST_P(myriadLayerConvolutionTensorFlow_smoke, Convolution) {
    tensor_test_params input_dims = get<0>(GetParam());
    tensor_test_params output_dims = get<1>(GetParam());
    param_size kernel = get<2>(GetParam());
    param_size stride = get<3>(GetParam());
    param_size pad = get<4>(GetParam());
    size_t group = get<5>(GetParam());
    param_size dilation_factor = get<6>(GetParam());

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
            , {"output", std::to_string(output_dims.c)}
            , {"group", std::to_string(group)}
            , {"dilation-x", std::to_string(dilation_factor.x)}
            , {"dilation-y", std::to_string(dilation_factor.y)}
    };
    ASSERT_NO_FATAL_FAILURE(makeSingleLayerNetwork(LayerInitParams("Convolution")
                                        .params(layer_params)
                                        .weights(num_weights)
                                        .biases(num_bias),
                                        {},
                                        weights_ptr));
    SetFirstInputToRange(-0.9f, 0.9f);
    ASSERT_TRUE(Infer());
    auto inputBlob = _inputMap.begin()->second;
    auto outputBlob = _outputMap.begin()->second;

    ref_convolution(inputBlob, _refBlob, weights, bias, kernel, stride, pad, group, dilation_factor);

    float maxerr = 0;

    maxerr = 0.00055 * (input_dims.c / group) * kernel.x * kernel.y;

    CompareCommonAbsolute(outputBlob, _refBlob, maxerr);
}

void FillWeights(uint16_t* ptr, size_t weightsSize) {
    ASSERT_NE(ptr, nullptr);
    auto szW = sizeof(s_3X3X3YOLO_Weights)/sizeof(s_3X3X3YOLO_Weights[0]);
    ASSERT_EQ(szW, weightsSize);
    auto sz = szW;
    size_t indx = 0;
    for (; indx < szW; ++indx) {
        ptr[indx] = PrecisionUtils::f32tof16(s_3X3X3YOLO_Weights[indx]);
    }
}
void FillBiases(uint16_t* ptr, size_t biasSize) {
    ASSERT_NE(ptr, nullptr);
    auto szB = sizeof(s_3X3X3YOLO_Biases)/sizeof(s_3X3X3YOLO_Biases[0]);
    ASSERT_EQ(szB, biasSize);
    auto sz = szB;
    size_t indx = 0;
    for (; indx < sz; ++indx) {
        ptr[indx] = PrecisionUtils::f32tof16(s_3X3X3YOLO_Biases[indx]);
    }
}

void loadConstData(InferenceEngine::Blob::Ptr blob) {
    /* input blob has predefined size and CHW layout */
    ASSERT_NE(blob, nullptr);
    uint16_t *inputBlobRawDataFp16 = static_cast<uint16_t *>(blob->buffer());
    ASSERT_NE(inputBlobRawDataFp16, nullptr);

    for (int indx = 0; indx < blob->size(); indx++) {
        inputBlobRawDataFp16[indx] = PrecisionUtils::f32tof16(128.0);
    }
}

class myriadLayers_3X3X3_ConstInput_smoke: public ConvolutionTest<vpu::LayoutPreference>{
};

TEST_P(myriadLayers_3X3X3_ConstInput_smoke, Convolution) {
    auto p = ::testing::WithParamInterface<std::tuple<InferenceEngine::SizeVector, param_size, param_size, param_size, uint32_t, uint32_t, vpu::LayoutPreference>>::GetParam();
    const auto layoutPreference = std::get<6>(p);

    _testNet.setWeightsCallbackForLayer(0, FillWeights);
    _testNet.setBiasesCallbackForLayer(0, FillBiases);
    _genDataCallback = loadConstData;
    ASSERT_TRUE(generateNetAndInfer( NetworkInitParams().layoutPreference(layoutPreference) ));
    auto outputBlob = _outputMap.begin()->second;
    const uint16_t *res_ptr = outputBlob->buffer().as<const uint16_t*>();
    size_t res_size = outputBlob->size();

    size_t N = outputBlob->getTensorDesc().getDims()[0];
    size_t C = outputBlob->getTensorDesc().getDims()[1];
    size_t H = outputBlob->getTensorDesc().getDims()[2];
    size_t W = outputBlob->getTensorDesc().getDims()[3];

    for (size_t n = 0; n < N; n++) {
        for (size_t c = 0; c < C; c++) {
            auto ref_offs = outputBlob->getTensorDesc().getLayout() == NCHW ?
                            1 + 1*W + c*W*H + n*W*H*C : c + 1*C + 1*C*W + n*W*H*C;
            float ref_val = PrecisionUtils::f16tof32(res_ptr[ref_offs]);
            for (size_t h = 1; h < H - 1; h++) {
                for (size_t w = 1; w < W - 1; w++) {
                    size_t actualIdx = outputBlob->getTensorDesc().getLayout() == NCHW ?
                                       w + h*W + c*W*H + n*W*H*C : c + w*C + h*C*W + n*W*H*C;
                    float cur_val = PrecisionUtils::f16tof32(res_ptr[actualIdx]);
                    ASSERT_FLOAT_EQ(cur_val, ref_val);
                }
            }
        }
    }
    /* to check max error */
    CompareCommonAbsolute(_outputMap.begin()->second, getReferenceOutput(), 0.02);
}

/* IR version 3 tests, main difference is a changes in padding parameters definitions */
typedef std::tuple<InferenceEngine::SizeVector, param_size, param_size, param_size, param_size, uint32_t, uint32_t> IR3_params;

class myriadLayers_IR3_ConvTests_smoke: public myriadLayersTests_nightly, /*input tensor, kernel, stride, pads_begin, pads_end, out_channel, group */
                                          public testing::WithParamInterface<IR3_params> {
};

TEST_P(myriadLayers_IR3_ConvTests_smoke, Conv) {
    std::map<std::string, std::string> params;
    InferenceEngine::SizeVector output_tensor;
    int32_t IW = 0;
    int32_t IH = 0;
    int32_t IC = 0;
    int32_t I_N = 0;
    size_t  group = 0;


    auto p = ::testing::WithParamInterface<IR3_params>::GetParam();
    auto input_tensor = std::get<0>(p);
    param_size kernel = std::get<1>(p);
    param_size stride = std::get<2>(p);
    param_size pads_begin = std::get<3>(p);
    param_size pads_end = std::get<4>(p);
    size_t out_channels = std::get<5>(p);
    group = std::get<6>(p);
    get_dims(input_tensor, IW, IH, IC, I_N);
    if (I_N > 1)
        _config[VPU_CONFIG_KEY(DETECT_NETWORK_BATCH)] = CONFIG_VALUE(NO);
    else
        _config[VPU_CONFIG_KEY(DETECT_NETWORK_BATCH)] = CONFIG_VALUE(YES);
    size_t out_w = (IW + pads_begin.x + pads_end.x - kernel.x + stride.x) / stride.x;
    size_t out_h = (IH + pads_begin.y + pads_end.y - kernel.y + stride.y) / stride.y;
    gen_dims(output_tensor, input_tensor.size(), out_w, out_h, out_channels, I_N);

    size_t num_weights = kernel.x * kernel.y * (IC / group) * out_channels;
    size_t num_bias    = out_channels;

    std::string padsB = std::to_string(pads_begin.x) + ",";
    padsB += std::to_string(pads_begin.y);
    std::string padsE = std::to_string(pads_end.x) + ",";
    padsE += std::to_string(pads_end.y);
    std::string strides = std::to_string(stride.x) + ",";
    strides += std::to_string(stride.y);
    std::string kern = std::to_string(kernel.x) + ",";
    kern += std::to_string(kernel.y);

    std::map<std::string, std::string> layer_params = {
              {"kernel",     kern}
            , {"strides",    strides}
            , {"pads_begin", padsB}
            , {"pads_end",   padsE}
            , {"output", std::to_string(out_channels)}
            , {"group", std::to_string(group)}
            , {"auto_pad", "same_upper"}
            , {"dilations", "1,1"}
    };
    _testNet.addLayer(LayerInitParams("Convolution")
              .params(layer_params)
              .in({input_tensor})
              .out({output_tensor})
              .weights(num_weights).fillWeights(defaultWeightsRange)
              .biases(num_bias).fillBiases(defaultWeightsRange),
             ref_convolution_wrap);
    ASSERT_TRUE(generateNetAndInfer( NetworkInitParams().useHWOpt( CheckMyriadX() ) ));
    float maxerr = 0.0009f * (IC / group) * kernel.x * kernel.y;
    CompareCommonAbsolute(_outputMap.begin()->second, getReferenceOutput(), maxerr);
}

class myriadLayers_BatchTest_ConvTests_smoke: public myriadLayersTests_nightly, /*input tensor, kernel, stride, pads_begin, pads_end, out_channel, group */
                                                public testing::WithParamInterface<IR3_params> {
};

class myriadLayers_BatchTest2_ConvTests_smoke: public myriadLayersTests_nightly, /*input tensor, kernel, stride, pads_begin, pads_end, out_channel, group */
                                                 public testing::WithParamInterface<IR3_params> {
};

void constWeightsRange(uint16_t* ptr, size_t weightsSize) {
    ASSERT_NE(ptr, nullptr);
    float shft = 0.0011f;
    float val = 0.125f;
    for (size_t count = 0 ; count < weightsSize; ++count) {

        ptr[count] = PrecisionUtils::f32tof16(val);
        val += shft;
        if (val > 1.)
            val = -1.0f;
    }
}

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
                        if (counter > .90f) {
                            counter = -.90f;
                        }
                    }
                }
            }
        }
    } else {
        ASSERT_TRUE(false);
    }
}

TEST_P(myriadLayers_BatchTest_ConvTests_smoke, Conv) {
    std::map<std::string, std::string> params;
    InferenceEngine::SizeVector output_tensor;
    int32_t IW = 0;
    int32_t IH = 0;
    int32_t IC = 0;
    int32_t I_N = 0;
    size_t  group = 0;

    auto p = ::testing::WithParamInterface<IR3_params>::GetParam();
    auto input_tensor = std::get<0>(p);
    param_size kernel = std::get<1>(p);
    param_size stride = std::get<2>(p);
    param_size pads_begin = std::get<3>(p);
    param_size pads_end = std::get<4>(p);
    size_t out_channels = std::get<5>(p);
    group = std::get<6>(p);
    get_dims(input_tensor, IW, IH, IC, I_N);
    if (I_N > 1)
        _config[VPU_CONFIG_KEY(DETECT_NETWORK_BATCH)] = CONFIG_VALUE(NO);
    else
        _config[VPU_CONFIG_KEY(DETECT_NETWORK_BATCH)] = CONFIG_VALUE(YES);
    size_t out_w = (IW + pads_begin.x + pads_end.x - kernel.x + stride.x) / stride.x;
    size_t out_h = (IH + pads_begin.y + pads_end.y - kernel.y + stride.y) / stride.y;
    gen_dims(output_tensor, input_tensor.size(), out_w, out_h, out_channels, I_N);

    size_t num_weights = kernel.x * kernel.y * (IC / group) * out_channels;
    size_t num_bias    = out_channels;

    std::string padsB   = gen_param(pads_begin);
    std::string padsE   = gen_param(pads_end);
    std::string strides = gen_param(stride);
    std::string kern    = gen_param(kernel);

    std::map<std::string, std::string> layer_params = {
              {"kernel",     kern}
            , {"strides",    strides}
            , {"pads_begin", padsB}
            , {"pads_end",   padsE}
            , {"output", std::to_string(out_channels)}
            , {"group", std::to_string(group)}
            , {"auto_pad", "same_upper"}
            , {"dilations", "1,1"}
    };
    _genDataCallback = genTestData;
    _testNet.addLayer( LayerInitParams("Convolution")
           .params(layer_params)
           .in({input_tensor})
           .out({output_tensor})
           .weights(num_weights).fillWeights(defaultWeightsRange)
           .biases(num_bias).fillBiases(defaultWeightsRange),
             ref_convolution_wrap);

    ASSERT_TRUE(generateNetAndInfer( NetworkInitParams().useHWOpt( CheckMyriadX() ).layoutPreference(vpu::LayoutPreference::ChannelMinor) ));

    float maxerr = 0.0009f * (IC / group) * kernel.x * kernel.y;
    CompareCommonAbsolute(_outputMap.begin()->second, getReferenceOutput(), maxerr);
}

static const std::string MODEL_RFCNN = R"V0G0N(
<net name="MODEL_TEST" version="3" batch="10">
    <layers>
        <layer id="0" name="input" precision="FP16" type="Input">
			<output>
				<port id="0">
					<dim>10</dim>
					<dim>576</dim>
					<dim>14</dim>
					<dim>14</dim>
				</port>
			</output>
		</layer>
		<layer id="142" name="MaxPool2D/MaxPool" precision="FP16" type="Pooling">
			<data auto_pad="valid" exclude-pad="true" kernel="2,2" pads_begin="0,0" pads_end="0,0" pool-method="max" strides="2,2"/>
			<input>
				<port id="0">
					<dim>10</dim>
					<dim>576</dim>
					<dim>14</dim>
					<dim>14</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>10</dim>
					<dim>576</dim>
					<dim>7</dim>
					<dim>7</dim>
				</port>
			</output>
		</layer>
		<layer id="143" name="SecondStageFeatureExtractor/InceptionV2/Mixed_5a/Branch_0/Conv2d_0a_1x1/Conv2D" precision="FP16" type="Convolution">
			<data auto_pad="same_upper" dilations="1,1" group="1" kernel="1,1" output="128" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>10</dim>
					<dim>576</dim>
					<dim>7</dim>
					<dim>7</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>10</dim>
					<dim>128</dim>
					<dim>7</dim>
					<dim>7</dim>
				</port>
			</output>
			<blobs>
				<weights offset="0" size="147456"/>
				<biases offset="147456" size="256"/>
			</blobs>
		</layer>
		<layer id="144" name="SecondStageFeatureExtractor/InceptionV2/Mixed_5a/Branch_0/Conv2d_0a_1x1/Relu" precision="FP16" type="ReLU">
			<input>
				<port id="0">
					<dim>10</dim>
					<dim>128</dim>
					<dim>7</dim>
					<dim>7</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>10</dim>
					<dim>128</dim>
					<dim>7</dim>
					<dim>7</dim>
				</port>
			</output>
		</layer>
		<layer id="145" name="SecondStageFeatureExtractor/InceptionV2/Mixed_5a/Branch_0/Conv2d_1a_3x3/Conv2D" precision="FP16" type="Convolution">
			<data auto_pad="same_upper" dilations="1,1" group="1" kernel="3,3" output="192" pads_begin="1,1" pads_end="1,1" strides="2,2"/>
			<input>
				<port id="0">
					<dim>10</dim>
					<dim>128</dim>
					<dim>7</dim>
					<dim>7</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>10</dim>
					<dim>192</dim>
					<dim>4</dim>
					<dim>4</dim>
				</port>
			</output>
			<blobs>
				<weights offset="147712" size="442368"/>
				<biases offset="590080" size="384"/>
			</blobs>
		</layer>
		<layer id="146" name="SecondStageFeatureExtractor/InceptionV2/Mixed_5a/Branch_0/Conv2d_1a_3x3/Relu" precision="FP16" type="ReLU">
			<input>
				<port id="0">
					<dim>10</dim>
					<dim>192</dim>
					<dim>4</dim>
					<dim>4</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>10</dim>
					<dim>192</dim>
					<dim>4</dim>
					<dim>4</dim>
				</port>
			</output>
		</layer>
		<layer id="147" name="SecondStageFeatureExtractor/InceptionV2/Mixed_5a/Branch_1/Conv2d_0a_1x1/Conv2D" precision="FP16" type="Convolution">
			<data auto_pad="same_upper" dilations="1,1" group="1" kernel="1,1" output="192" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>10</dim>
					<dim>576</dim>
					<dim>7</dim>
					<dim>7</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>10</dim>
					<dim>192</dim>
					<dim>7</dim>
					<dim>7</dim>
				</port>
			</output>
			<blobs>
				<weights offset="590464" size="221184"/>
				<biases offset="811648" size="384"/>
			</blobs>
		</layer>
		<layer id="148" name="SecondStageFeatureExtractor/InceptionV2/Mixed_5a/Branch_1/Conv2d_0a_1x1/Relu" precision="FP16" type="ReLU">
			<input>
				<port id="0">
					<dim>10</dim>
					<dim>192</dim>
					<dim>7</dim>
					<dim>7</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>10</dim>
					<dim>192</dim>
					<dim>7</dim>
					<dim>7</dim>
				</port>
			</output>
		</layer>
		<layer id="149" name="SecondStageFeatureExtractor/InceptionV2/Mixed_5a/Branch_1/Conv2d_0b_3x3/Conv2D" precision="FP16" type="Convolution">
			<data auto_pad="same_upper" dilations="1,1" group="1" kernel="3,3" output="256" pads_begin="1,1" pads_end="1,1" strides="1,1"/>
			<input>
				<port id="0">
					<dim>10</dim>
					<dim>192</dim>
					<dim>7</dim>
					<dim>7</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>10</dim>
					<dim>256</dim>
					<dim>7</dim>
					<dim>7</dim>
				</port>
			</output>
			<blobs>
				<weights offset="812032" size="884736"/>
				<biases offset="1696768" size="512"/>
			</blobs>
		</layer>
        </layers>
        <edges>
            <edge from-layer="0" from-port="0" to-layer="142" to-port="0"/>
            <edge from-layer="142" from-port="1" to-layer="143" to-port="0"/>
            <edge from-layer="143" from-port="3" to-layer="144" to-port="0"/>
            <edge from-layer="144" from-port="1" to-layer="145" to-port="0"/>
            <edge from-layer="145" from-port="3" to-layer="146" to-port="0"/>
            <edge from-layer="142" from-port="1" to-layer="147" to-port="0"/>
            <edge from-layer="147" from-port="3" to-layer="148" to-port="0"/>
            <edge from-layer="148" from-port="1" to-layer="149" to-port="0"/>
        </edges>
    </net>
)V0G0N";

TEST_P(myriadLayers_BatchTest2_ConvTests_smoke, Conv) {
    std::map<std::string, std::string> params;
    InferenceEngine::SizeVector output_tensor;
    int32_t IW = 0;
    int32_t IH = 0;
    int32_t IC = 0;
    int32_t I_N = 0;
    size_t  group = 0;

    auto p = ::testing::WithParamInterface<IR3_params>::GetParam();
    auto input_tensor = std::get<0>(p);
    param_size kernel = std::get<1>(p);
    param_size stride = std::get<2>(p);
    param_size pads_begin = std::get<3>(p);
    param_size pads_end = std::get<4>(p);
    size_t out_channels = std::get<5>(p);
    group = std::get<6>(p);
    get_dims(input_tensor, IW, IH, IC, I_N);
    if (I_N > 1)
        _config[VPU_CONFIG_KEY(DETECT_NETWORK_BATCH)] = CONFIG_VALUE(NO);
    else
        _config[VPU_CONFIG_KEY(DETECT_NETWORK_BATCH)] = CONFIG_VALUE(YES);
    size_t out_w = (IW + pads_begin.x + pads_end.x - kernel.x + stride.x) / stride.x;
    size_t out_h = (IH + pads_begin.y + pads_end.y - kernel.y + stride.y) / stride.y;
    gen_dims(output_tensor, input_tensor.size(), out_w, out_h, out_channels, I_N);

    size_t num_weights = kernel.x * kernel.y * (IC / group) * out_channels;
    size_t num_bias    = out_channels;

    std::string padsB   = gen_param(pads_begin);
    std::string padsE   = gen_param(pads_end);
    std::string strides = gen_param(stride);
    std::string kern    = gen_param(kernel);

    std::map<std::string, std::string> layer_params = {
              {"kernel",     kern}
            , {"strides",    strides}
            , {"pads_begin", padsB}
            , {"pads_end",   padsE}
            , {"output", std::to_string(out_channels)}
            , {"group", std::to_string(group)}
            , {"auto_pad", "same_upper"}
            , {"dilations", "1,1"}
    };
    _genDataCallback = genTestData;
    _testNet.addLayer(LayerInitParams("Convolution")
               .params(layer_params)
               .in({input_tensor})
               .out({output_tensor})
               .weights(num_weights).fillWeights(constWeightsRange)
               .biases(num_bias).fillBiases(constWeightsRange),
             ref_convolution_wrap);
    _testNet.addLayer(LayerInitParams("ReLU")
             .in({output_tensor})
             .out({output_tensor}),
             ref_ReLU_wrap);

    std::map<std::string, std::string> conv2_params = {
              {"kernel",     "3,3"}
            , {"strides",    "1,1"}
            , {"pads_begin", "1,1"}
            , {"pads_end",   "1,1"}
            , {"output", "256"}
            , {"group", "1"}
            , {"auto_pad", "same_upper"}
            , {"dilations", "1,1"}
    };
    _testNet.addLayer(LayerInitParams("Convolution")
             .params(conv2_params)
             .in({output_tensor})
             .out({{10, 256, 7, 7}})
             .weights(442368).fillWeights(constWeightsRange)
             .biases(256).fillBiases(constWeightsRange),
             ref_convolution_wrap);
    _testNet.addLayer(LayerInitParams("ReLU")
             .in({{10, 256, 7, 7}})
             .out({{10, 256, 7, 7}}),
             ref_ReLU_wrap);

    ASSERT_TRUE(generateNetAndInfer(NetworkInitParams().useHWOpt( CheckMyriadX()) ));
    // Error is calculated for sum of 2 convolutions
    float maxerr = 0.001f * (IC + 256) * kernel.x * kernel.y * 9;
    CompareCommonAbsolute(_outputMap.begin()->second, getReferenceOutput(), maxerr);
}
