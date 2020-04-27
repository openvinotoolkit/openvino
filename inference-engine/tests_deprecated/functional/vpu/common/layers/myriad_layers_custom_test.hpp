// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "myriad_layers_tests.hpp"

using namespace InferenceEngine;

static void refShuffleChannel(const Blob::Ptr src,
                              Blob::Ptr dst,
                              int group, int isCHW) {
    ASSERT_NE(src, nullptr);
    ASSERT_NE(dst, nullptr);
    const uint16_t *src_data = src->buffer();
          uint16_t *dst_data = dst->buffer();
    ASSERT_NE(src_data, nullptr);
    ASSERT_NE(dst_data, nullptr);
    int32_t IW = 1;
    int32_t IH = 1;
    int32_t IC = 1;
    get_dims(src, IW, IH, IC);

    int G = group;
    int CX = IC / G;
    int CY = G;

    for (int cy = 0; cy < CY; cy++) {
        for (int cx = 0; cx < CX; cx++) {
            for (int h = 0; h < IH; h++) {
                for (int w = 0; w < IW; w++) {
                    if (isCHW) {
                        dst_data[(cx*CY + cy)*IW*IH + h*IW + w] = src_data[(cy*CX + cx)*IW*IH + h*IW + w];
                    } else {
                        dst_data[(cx*CY + cy) + h*IW*IC + w*IC] = src_data[(cy*CX + cx) + h*IW*IC + w*IC];
                    }
                }
            }
        }
    }
}

static void refQuantize(const Blob::Ptr src,
                        const Blob::Ptr input_low,
                        const Blob::Ptr input_high,
                        const Blob::Ptr output_low,
                        const Blob::Ptr output_high,
                        Blob::Ptr dst,
                        int levels, int isCHW) {
    ASSERT_NE(src, nullptr);
    ASSERT_NE(input_low, nullptr);
    ASSERT_NE(input_high, nullptr);
    ASSERT_NE(output_low, nullptr);
    ASSERT_NE(output_high, nullptr);
    ASSERT_NE(dst, nullptr);
    const uint16_t *src_data = src->buffer();
    const uint16_t *input_low_data = input_low->buffer();
    const uint16_t *input_high_data = input_high->buffer();
    const uint16_t *output_low_data = output_low->buffer();
    const uint16_t *output_high_data = output_high->buffer();
    uint16_t *dst_data = dst->buffer();
    ASSERT_NE(src_data, nullptr);
    ASSERT_NE(input_low_data, nullptr);
    ASSERT_NE(input_high_data, nullptr);
    ASSERT_NE(output_low_data, nullptr);
    ASSERT_NE(output_high_data, nullptr);
    ASSERT_NE(dst_data, nullptr);
    int32_t W = 1;
    int32_t H = 1;
    int32_t C = 1;
    get_dims(src, W, H, C);

    for (int c = 0; c < C; c++) {
        float ilow  = PrecisionUtils::f16tof32(input_low->size()   == 1 ? input_low_data[0]   : input_low_data[c]);
        float ihigh = PrecisionUtils::f16tof32(input_high->size()  == 1 ? input_high_data[0]  : input_high_data[c]);
        float olow  = PrecisionUtils::f16tof32(output_low->size()  == 1 ? output_low_data[0]  : output_low_data[c]);
        float ohigh = PrecisionUtils::f16tof32(output_high->size() == 1 ? output_high_data[0] : output_high_data[c]);

        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
                int idx = (isCHW) ? c*W*H + h*W + w : c + h*W*C + w*C;
                float src_val = PrecisionUtils::f16tof32(src_data[idx]);
                float dst_val;

                if (src_val <= ilow) {
                    dst_val = olow;
                } else if (src_val > ihigh) {
                    dst_val = ohigh;
                } else {
                    dst_val = round((src_val - ilow) * ((float)(levels - 1) / (ihigh - ilow))) * ((ohigh - olow) / (float)(levels - 1))+ olow;
                    //dst_val = round((src_val - ilow) / (ihigh - ilow) * (levels - 1)) / (levels - 1) * (ohigh - olow) + olow;
                }

                dst_data[idx] = PrecisionUtils::f32tof16(dst_val);
            }
        }
    }
}

static void ref_QuantizeBinarization(const Blob::Ptr src,
                        const Blob::Ptr input_low,
                        const Blob::Ptr input_high,
                        const Blob::Ptr output_low,
                        const Blob::Ptr output_high,
                        Blob::Ptr dst,
                        int levels) {
    ASSERT_NE(src, nullptr);
    ASSERT_NE(input_low, nullptr);
    ASSERT_NE(input_high, nullptr);
    ASSERT_NE(output_low, nullptr);
    ASSERT_NE(output_high, nullptr);
    ASSERT_NE(dst, nullptr);
    const uint16_t *src_data = src->buffer();
    const uint16_t *input_low_data = input_low->buffer();
    const uint16_t *input_high_data = input_high->buffer();
    const uint16_t *output_low_data = output_low->buffer();
    const uint16_t *output_high_data = output_high->buffer();
    uint16_t *dst_data = dst->buffer();
    ASSERT_NE(src_data, nullptr);
    ASSERT_NE(input_low_data, nullptr);
    ASSERT_NE(input_high_data, nullptr);
    ASSERT_NE(output_low_data, nullptr);
    ASSERT_NE(output_high_data, nullptr);
    ASSERT_NE(dst_data, nullptr);
    int32_t W = 1;
    int32_t H = 1;
    int32_t C = 1;
    get_dims(src, W, H, C);

    for (int c = 0; c < C; c++) {
        float ilow  = PrecisionUtils::f16tof32(input_low->size()   == 1 ? input_low_data[0]   : input_low_data[c]);
        float ihigh = PrecisionUtils::f16tof32(input_high->size()  == 1 ? input_high_data[0]  : input_high_data[c]);
        float olow  = PrecisionUtils::f16tof32(output_low->size()  == 1 ? output_low_data[0]  : output_low_data[c]);
        float ohigh = PrecisionUtils::f16tof32(output_high->size() == 1 ? output_high_data[0] : output_high_data[c]);

        // emulate half math to be close to half float SHAVE implementation
        float hTof_ilow = PrecisionUtils::f16tof32(PrecisionUtils::f32tof16(ilow));
        float hTof_ihigh = PrecisionUtils::f16tof32(PrecisionUtils::f32tof16(ihigh));
        float a = (0.01 > (hTof_ihigh - hTof_ilow)) ? 0.0f : PrecisionUtils::f16tof32(PrecisionUtils::f32tof16((float)(levels - 1) / (hTof_ihigh - hTof_ilow)));
        float b = !(levels - 1) ? 0.0f : PrecisionUtils::f16tof32(PrecisionUtils::f32tof16((ohigh - olow) / (float)(levels - 1)));

        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
                int idx = c*W*H + h*W + w;
                float src_val = PrecisionUtils::f16tof32(src_data[idx]);
                float dst_val;

                if (src_val <= ilow) {
                    dst_val = olow;
                } else if (src_val > ihigh) {
                    dst_val = ohigh;
                } else {
                    if(!(ihigh - ilow) || !(levels - 1))
                        dst_val = olow;
                    else
                    {
                        // quantization pass
                        float quantized = PrecisionUtils::f16tof32(PrecisionUtils::f32tof16((src_val - ilow) * a));
                        // de-quantization pass
                        dst_val = PrecisionUtils::f16tof32(PrecisionUtils::f32tof16(roundf( quantized ) * b)) + olow;
                    }
                }

                dst_data[idx] = PrecisionUtils::f32tof16(dst_val);
            }
        }
    }
}

static void refBinaryConvolution(const Blob::Ptr src, const Blob::Ptr weights, Blob::Ptr dst,
                                 int dilations, int group, param_size kernel, int strides,
                                 int isCHW) {
    ASSERT_NE(src, nullptr);
    ASSERT_NE(dst, nullptr);
    const uint16_t* src_data = src->buffer();
    const uint8_t*  weights_data = weights->buffer();
          uint16_t* dst_data = dst->buffer();
    ASSERT_NE(src_data, nullptr);
    ASSERT_NE(weights_data, nullptr);
    ASSERT_NE(dst_data, nullptr);
    int32_t IW = 1;
    int32_t IH = 1;
    int32_t IC = 1;
    get_dims(src, IW, IH, IC);
    int32_t OW = 1;
    int32_t OH = 1;
    int32_t OC = 1;
    get_dims(dst, OW, OH, OC);

    int KW = kernel.x;
    int KH = kernel.y;
    int KD = 1;

    int SW = strides;
    int SH = strides;
    int SD = 0;

    int DW = dilations;
    int DH = dilations;
    int DD = 0;

    int PW = kernel.x/2;
    int PH = kernel.y/2;
    int PD = 0;

    int GC = group;

    int ID = 1;
    int OD = 1;

    int pad_value = 0;

    int nbits = 8;

    auto extract_weights = [](uint8_t val, uint8_t bit) -> int {
        return (uint8_t)((val >> bit) & 1);
    };

    for (uint32_t g = 0; g < GC; g++) {
        for (uint32_t oc = 0; oc < OC / GC; oc++) {
            for (uint32_t od = 0; od < OD; od++) {
                for (uint32_t oh = 0; oh < OH; oh++) {
                    for (uint32_t ow = 0; ow < OW; ow++) {
                        int oidx = (isCHW) ? g  * OC / GC * OD * OH * OW +
                                             oc *           OD * OH * OW +
                                             od *           OH * OW +
                                             oh *           OW +
                                             ow
                                           : g  * OC / GC * OD +
                                             oc * OD +
                                             od +
                                             oh * OW * OC +
                                             ow * OC;

                        int dst_val = 0;

                        for (int ic = 0; ic < IC / GC; ic++) {
                            for (int kd = 0; kd < KD; kd++) {
                                for (int kh = 0; kh < KH; kh++) {
                                    for (int kw = 0; kw < KW; kw++) {
                                        int widx = g  * OC / GC * IC / GC * KD * KH * KW +
                                                   oc * IC / GC * KD * KH * KW +
                                                   ic * KD * KH * KW +
                                                   kd * KH * KW +
                                                   kh * KW +
                                                   kw;
                                        int w = extract_weights(weights_data[widx/nbits], (uint8_t)(widx % nbits));

                                        int s;

                                        int iw = ow * SW - PW + kw * DW;
                                        int ih = oh * SH - PH + kh * DH;
                                        int id = od * SD - PD + kd * DD;
                                        if (iw < 0 || iw >= (int) IW ||
                                            ih < 0 || ih >= (int) IH ||
                                            id < 0 || id >= (int) ID) {
                                            s = pad_value;
                                        } else {
                                            int iidx = (isCHW) ? g  * IC / GC * ID * IH * IW +
                                                                 ic * ID * IH * IW +
                                                                 id * IH * IW +
                                                                 ih * IW +
                                                                 iw
                                                               : g  * IC / GC * ID +
                                                                 ic * ID +
                                                                 id +
                                                                 ih * IW * IC +
                                                                 iw * IC;
                                            s = ((PrecisionUtils::f16tof32(src_data[iidx]) > 0.f) ? 1 : 0);
                                        }

                                        dst_val += s ^ w;
                                    }
                                }
                            }
                        }

                        dst_data[oidx] = PrecisionUtils::f32tof16((float)(IC/GC*KD*KH*KW - 2*dst_val));
                    }
                }
            }
        }
    }
}

static void refExperimentalDetectronPriorGridGenerator(
        std::vector<Blob::Ptr> &inputs, std::vector<Blob::Ptr> &outputs,
        int grid_h, int grid_w, int stride_h, int stride_w) {
    int num_priors = inputs[0]->getTensorDesc().getDims()[0];

    uint16_t *src_data = inputs[0]->buffer();
    uint16_t *dst_data = outputs[0]->buffer();

    using namespace PrecisionUtils;

    for (int h = 0; h < grid_h; ++h) {
        for (int w = 0; w < grid_w; ++w) {
            for (int s = 0; s < 3; ++s) {
                dst_data[0] = f32tof16(
                        f16tof32(src_data[4 * s + 0]) + stride_w * (w + 0.5f));
                dst_data[1] = f32tof16(
                        f16tof32(src_data[4 * s + 1]) + stride_h * (h + 0.5f));
                dst_data[2] = f32tof16(
                        f16tof32(src_data[4 * s + 2]) + stride_w * (w + 0.5f));
                dst_data[3] = f32tof16(
                        f16tof32(src_data[4 * s + 3]) + stride_h * (h + 0.5f));
                dst_data += 4;
            }
        }
    }
}
static std::vector<std::string> s_CustomConfig = {
#ifdef VPU_HAS_CUSTOM_KERNELS
    getIELibraryPath() + "/vpu_custom_kernels/customLayerBindings.xml"
#endif
};

PRETTY_PARAM(Group, int)
PRETTY_PARAM(Levels, int)
PRETTY_PARAM(SwitchOut, int)
PRETTY_PARAM(Dilations, int)
PRETTY_PARAM(Kernel, param_size)
PRETTY_PARAM(Strides, int)

typedef myriadLayerTestBaseWithParam<std::tuple<Dims, Group, std::string>> myriadLayersTestsShuffleChannel_nightly;
typedef myriadLayerTestBaseWithParam<std::tuple<Dims, Levels, std::string>> myriadLayersTestsQuantize_nightly;
typedef myriadLayerTestBaseWithParam<std::tuple<Dims, Levels, SwitchOut, std::string>> myriadLayersTestsQuantizeBinarize_nightly;
typedef myriadLayerTestBaseWithParam<std::tuple<Dims, Dilations, Group, Kernel, Strides, std::string>> myriadLayersTestsBinaryConvolution_nightly;
typedef myriadLayerTestBaseWithParam<std::tuple<std::vector<size_t>, std::string>>
myriadLayersTestsExperimentalDetectronPriorGridGenerator_nightly;

TEST_P(myriadLayersTestsShuffleChannel_nightly, ShuffleChannel) {
    tensor_test_params dims  = std::get<0>(GetParam());
    int group                = std::get<1>(GetParam());
    std::string customConfig = std::get<2>(GetParam());

    if(!customConfig.empty() && !CheckMyriadX()) {
        GTEST_SKIP()<<"Custom layers for MYRIAD2 not supported";
    }
    _config[VPU_CONFIG_KEY(CUSTOM_LAYERS)] = customConfig;

    SetInputTensor(dims);
    SetOutputTensor(dims);

    std::map<std::string, std::string> params;
    params["group"] = std::to_string(group);

    ASSERT_NO_FATAL_FAILURE(makeSingleLayerNetwork(LayerInitParams("ShuffleChannel").params(params)));

    ASSERT_TRUE(Infer());

    ASSERT_NO_FATAL_FAILURE(refShuffleChannel(_inputMap.begin()->second, _refBlob, group, false));

    CompareCommonAbsolute(_outputMap.begin()->second, _refBlob, 0);
}

static std::vector<Dims> s_ShuffleChannelTensors = {
    {{1,  48, 28, 28}},
    {{1,  96, 14, 14}},
    {{1, 192,  7,  7}},
};

static std::vector<Group> s_ShuffleChannelGroup = {
    2
};

TEST_P(myriadLayersTestsQuantize_nightly, Quantize) {
    tensor_test_params dims  = std::get<0>(GetParam());
    int levels               = std::get<1>(GetParam());
    std::string customConfig = std::get<2>(GetParam());

    if(!customConfig.empty() && !CheckMyriadX()) {
        GTEST_SKIP()<<"Custom layers for MYRIAD2 not supported";
    }
    _config[VPU_CONFIG_KEY(CUSTOM_LAYERS)] = customConfig;

    IN_OUT_desc inpt(5);
    for (int i = 0; i < inpt.size(); ++i) {
        inpt[i].resize(4);
        inpt[i][0] = dims.n;
        inpt[i][1] = 1;
        inpt[i][2] = 1;
        inpt[i][3] = 1;
    }
    inpt[0][1] = dims.c;
    inpt[0][2] = dims.h;
    inpt[0][3] = dims.w;
    for (int i = 1; i < inpt.size(); ++i) {
        if (rand()%2 > 0) {
            inpt[i][1] = dims.c;
        }
    }

    SetInputTensors(inpt);
    SetOutputTensor(dims);

    std::map<std::string, std::string> params;
    params["levels"] = std::to_string(levels);

    ASSERT_NO_FATAL_FAILURE(makeSingleLayerNetwork(LayerInitParams("FakeQuantize").params(params)));

    ASSERT_TRUE(Infer());

    std::vector<Blob::Ptr> inputBlobs(inpt.size());
    auto inptIter = _inputMap.begin();
    for (int i = 0; i < inpt.size(); i++) {
        inputBlobs[i] = inptIter->second;
        inptIter++;
    }

    ASSERT_NO_FATAL_FAILURE(refQuantize(inputBlobs[0],
                                        inputBlobs[1],
                                        inputBlobs[2],
                                        inputBlobs[3],
                                        inputBlobs[4],
                                        _refBlob,
                                        levels, false));

    CompareCommonAbsolute(_outputMap.begin()->second, _refBlob, 0.01f);
}

TEST_P(myriadLayersTestsQuantizeBinarize_nightly, Quantize_Binarization) {
    std::string model = R"V0G0N(
       <net name="Quantize_Binarization" version="2" batch="1">
           <layers>
            <layer id="0" name="data" precision="FP16" type="Input">
                <output>
                    <port id="0">
                        <dim>@IB@</dim>
                        <dim>@IC@</dim>
                        <dim>@IH@</dim>
                        <dim>@IW@</dim>
                    </port>
                </output>
            </layer>
            <layer id="1" name="input_low" precision="FP16" type="Input">
                <output>
                    <port id="0">
                        <dim>1</dim>
                        <dim>@input_low_size@</dim>
                        <dim>1</dim>
                        <dim>1</dim>
                    </port>
                </output>
            </layer>
            <layer id="2" name="input_high" precision="FP16" type="Input">
                <output>
                    <port id="0">
                        <dim>1</dim>
                        <dim>@input_high_size@</dim>
                        <dim>1</dim>
                        <dim>1</dim>
                    </port>
                </output>
            </layer>
            <layer id="3" name="output_low" precision="FP16" type="Input">
                <output>
                    <port id="0">
                        <dim>1</dim>
                        <dim>@output_low_size@</dim>
                        <dim>1</dim>
                        <dim>1</dim>
                    </port>
                </output>
            </layer>
            <layer id="4" name="output_high" precision="FP16" type="Input">
                <output>
                    <port id="0">
                        <dim>1</dim>
                        <dim>@output_high_size@</dim>
                        <dim>1</dim>
                        <dim>1</dim>
                    </port>
                </output>
            </layer>
            <layer id="5" name="Quantize" precision="FP16" type="QuantizeTemporaryType">
                <data levels="@levels@" input_low_size="@input_low_size@" input_high_size="@input_high_size@" output_low_size="@output_low_size@" output_high_size="@output_high_size@" switch_out="@switch_out@"/>
                <input>
                    <port id="0">
                        <dim>@IB@</dim>
                        <dim>@IC@</dim>
                        <dim>@IH@</dim>
                        <dim>@IW@</dim>
                    </port>
                    <port id="1">
                        <dim>1</dim>
                        <dim>@input_low_size@</dim>
                        <dim>1</dim>
                        <dim>1</dim>
                    </port>
                    <port id="2">
                        <dim>1</dim>
                        <dim>@input_high_size@</dim>
                        <dim>1</dim>
                        <dim>1</dim>
                    </port>
                    <port id="3">
                        <dim>1</dim>
                        <dim>@output_low_size@</dim>
                        <dim>1</dim>
                        <dim>1</dim>
                    </port>
                    <port id="4">
                        <dim>1</dim>
                        <dim>@output_high_size@</dim>
                        <dim>1</dim>
                        <dim>1</dim>
                    </port>
                </input>
                <output>
                    <port id="0">
                        <dim>@OB@</dim>
                        <dim>@OC@</dim>
                        <dim>@OH@</dim>
                        <dim>@OW@</dim>
                    </port>
                </output>
            </layer>
           </layers>
           <edges>
               <edge from-layer="0" from-port="0" to-layer="5" to-port="0"/>
               <edge from-layer="1" from-port="0" to-layer="5" to-port="1"/>
               <edge from-layer="2" from-port="0" to-layer="5" to-port="2"/>
               <edge from-layer="3" from-port="0" to-layer="5" to-port="3"/>
               <edge from-layer="4" from-port="0" to-layer="5" to-port="4"/>
           </edges>
       </net>
   )V0G0N";

    SetSeed(DEFAULT_SEED_VALUE + 6);

    tensor_test_params dims  = std::get<0>(GetParam());
    int levels               = std::get<1>(GetParam());
    int switch_out           = std::get<2>(GetParam());
    std::string customConfig = std::get<3>(GetParam());

    if(!customConfig.empty() && !CheckMyriadX()) {
        GTEST_SKIP()<<"Custom layers for MYRIAD2 not supported";
    }
    _config[VPU_CONFIG_KEY(CUSTOM_LAYERS)] = customConfig;

    int IB = dims.n;
    int IC = dims.c;
    int IH = dims.h;
    int IW = dims.w;

    int OB = dims.n;
    int OC = dims.c;
    int OH = dims.h;
    int OW = dims.w;

    int input_low_size = (rand()%2>0) ? dims.c : 1; 
    int input_high_size = (levels == 2) ? input_low_size : ((rand()%2>0) ? dims.c : 1); 
    int output_low_size = (rand()%2>0) ? dims.c : 1; 
    int output_high_size = (levels == 2) ? output_low_size : ((rand()%2>0) ? dims.c : 1); 

    model.replace( model.find("@IB@"), sizeof("@IB@") -1, std::to_string(IB));
    model.replace( model.find("@IB@"), sizeof("@IB@") -1, std::to_string(IB));
    model.replace( model.find("@IC@"), sizeof("@IC@") -1, std::to_string(IC));
    model.replace( model.find("@IC@"), sizeof("@IC@") -1, std::to_string(IC));
    model.replace( model.find("@IH@"), sizeof("@IH@") -1, std::to_string(IH));
    model.replace( model.find("@IH@"), sizeof("@IH@") -1, std::to_string(IH));
    model.replace( model.find("@IW@"), sizeof("@IW@") -1, std::to_string(IW));
    model.replace( model.find("@IW@"), sizeof("@IW@") -1, std::to_string(IW));

    model.replace( model.find("@OB@"), sizeof("@OB@") -1, std::to_string(OB));
    model.replace( model.find("@OC@"), sizeof("@OC@") -1, std::to_string(OC));
    model.replace( model.find("@OH@"), sizeof("@OH@") -1, std::to_string(OH));
    model.replace( model.find("@OW@"), sizeof("@OW@") -1, std::to_string(OW));

    model.replace( model.find("@levels@"), sizeof("@levels@") -1, std::to_string(levels));
    model.replace( model.find("@switch_out@"), sizeof("@switch_out@") -1, std::to_string(switch_out));
    model.replace( model.find("@input_low_size@"), sizeof("@input_low_size@") -1, std::to_string(input_low_size));
    model.replace( model.find("@input_high_size@"), sizeof("@input_high_size@") -1, std::to_string(input_high_size));
    model.replace( model.find("@output_low_size@"), sizeof("@output_low_size@") -1, std::to_string(output_low_size));
    model.replace( model.find("@output_high_size@"), sizeof("@output_high_size@") -1, std::to_string(output_high_size));
    model.replace( model.find("@input_low_size@"), sizeof("@input_low_size@") -1, std::to_string(input_low_size));
    model.replace( model.find("@input_high_size@"), sizeof("@input_high_size@") -1, std::to_string(input_high_size));
    model.replace( model.find("@output_low_size@"), sizeof("@output_low_size@") -1, std::to_string(output_low_size));
    model.replace( model.find("@output_high_size@"), sizeof("@output_high_size@") -1, std::to_string(output_high_size));
    model.replace( model.find("@input_low_size@"), sizeof("@input_low_size@") -1, std::to_string(input_low_size));
    model.replace( model.find("@input_high_size@"), sizeof("@input_high_size@") -1, std::to_string(input_high_size));
    model.replace( model.find("@output_low_size@"), sizeof("@output_low_size@") -1, std::to_string(output_low_size));
    model.replace( model.find("@output_high_size@"), sizeof("@output_high_size@") -1, std::to_string(output_high_size));

    StatusCode st;

    InferenceEngine::Core ie;
    auto network = ie.ReadNetwork(model, InferenceEngine::Blob::CPtr());

    _inputsInfo  = network.getInputsInfo();
    _outputsInfo = network.getOutputsInfo();

    _inputsInfo["data"]->setPrecision(Precision::FP16);
    _inputsInfo["input_low"]->setPrecision(Precision::FP16);
    _inputsInfo["input_high"]->setPrecision(Precision::FP16);
    _inputsInfo["output_low"]->setPrecision(Precision::FP16);
    _inputsInfo["output_high"]->setPrecision(Precision::FP16);
    _outputsInfo["Quantize"]->setPrecision(Precision::FP16);

    _inputsInfo["data"]->setLayout(NCHW);
    _inputsInfo["input_low"]->setLayout(NCHW);
    _inputsInfo["input_high"]->setLayout(NCHW);
    _inputsInfo["output_low"]->setLayout(NCHW);
    _inputsInfo["output_high"]->setLayout(NCHW);
    _outputsInfo["Quantize"]->setLayout(NCHW);

    ASSERT_NO_THROW(st = _vpuPluginPtr->LoadNetwork(_exeNetwork, network,
                                                    {{VPU_CONFIG_KEY(CUSTOM_LAYERS), customConfig }}, &_resp));
    ASSERT_EQ(StatusCode::OK, st) << _resp.msg;
    ASSERT_NE(_exeNetwork, nullptr) << _resp.msg;

    ASSERT_NO_THROW(st = _exeNetwork->CreateInferRequest(_inferRequest, &_resp));
    ASSERT_EQ(StatusCode::OK, st) << _resp.msg;

    Blob::Ptr data;
    ASSERT_NO_THROW(st = _inferRequest->GetBlob("data", data, &_resp));
    ASSERT_EQ(StatusCode::OK, st) << _resp.msg;
    GenRandomData(data);

    Blob::Ptr input_low;
    ASSERT_NO_THROW(st = _inferRequest->GetBlob("input_low", input_low, &_resp));
    ASSERT_EQ(StatusCode::OK, st) << _resp.msg;
    GenRandomData(input_low);

    Blob::Ptr input_high;
    ASSERT_NO_THROW(st = _inferRequest->GetBlob("input_high", input_high, &_resp));
    ASSERT_EQ(StatusCode::OK, st) << _resp.msg;
    Blob::Ptr output_low;
    ASSERT_NO_THROW(st = _inferRequest->GetBlob("output_low", output_low, &_resp));
    ASSERT_EQ(StatusCode::OK, st) << _resp.msg;
    Blob::Ptr output_high;
    ASSERT_NO_THROW(st = _inferRequest->GetBlob("output_high", output_high, &_resp));
    ASSERT_EQ(StatusCode::OK, st) << _resp.msg;

    if(levels == 2){
        memcpy((uint8_t*)input_high->buffer(), (uint8_t*)input_low->buffer(), input_high->byteSize());
        for(int i = 0; i < (output_low->byteSize() / output_low->element_size()); ++i){
            *((ie_fp16*)output_low->buffer() + i) = switch_out ? PrecisionUtils::f32tof16(1.0f) : PrecisionUtils::f32tof16(-1.0f);
            *((ie_fp16*)output_high->buffer() + i) = switch_out ? PrecisionUtils::f32tof16(-1.0f) : PrecisionUtils::f32tof16(1.0f);
        }
    }
    else{
        GenRandomData(input_high);
        GenRandomData(output_low);
        GenRandomData(output_high);
    }

    ASSERT_NO_THROW(st = _inferRequest->Infer(&_resp));
    ASSERT_EQ(StatusCode::OK, st) << _resp.msg;

{
    std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> perfMap;
    _inferRequest->GetPerformanceCounts(perfMap, nullptr);
    std::vector <std::pair<std::string, InferenceEngine::InferenceEngineProfileInfo>> perfVec(perfMap.begin(), perfMap.end());
    std::sort(perfVec.begin(), perfVec.end(),
              [=](const std::pair<std::string, InferenceEngine::InferenceEngineProfileInfo> &pair1,
                  const std::pair<std::string, InferenceEngine::InferenceEngineProfileInfo> &pair2) -> bool {
                  return pair1.second.execution_index < pair2.second.execution_index;
              });

    unsigned currentIndex = 0;
    for (auto it = perfVec.begin(); it != perfVec.end(); ++it) {
        std::string layerName = it->first;
        InferenceEngine::InferenceEngineProfileInfo info = it->second;
        if (info.status == InferenceEngine::InferenceEngineProfileInfo::EXECUTED) {
            printf("\x1B[32m[----------]\x1B[0m Myriad time = '%s' layer with '%s' type is %f ms.\n", layerName.c_str(), info.exec_type, info.realTime_uSec / 1000.f);
        }
    }
}

    Blob::Ptr outputBlob;
    ASSERT_NO_THROW(_inferRequest->GetBlob("Quantize", outputBlob, &_resp));
    ASSERT_EQ(StatusCode::OK, st) << _resp.msg;

    _refBlob = make_shared_blob<ie_fp16>(TensorDesc(Precision::FP16, outputBlob->getTensorDesc().getDims(), NCHW));
    _refBlob->allocate();

    ASSERT_NO_FATAL_FAILURE(ref_QuantizeBinarization(data,
                                                    input_low,
                                                    input_high,
                                                    output_low,
                                                    output_high,
                                                    _refBlob,
                                                    levels));

    CompareCommonAbsolute(outputBlob, _refBlob, 0.1);
}

static std::vector<Dims> s_QuantizeTensors = {
    {{1,  64, 56, 56}},
    {{1, 256, 28, 28}},
    {{1, 512,  7,  7}},
    {{1,  64, 56, 57}},
    {{1, 256, 28, 31}},
    {{1, 512,  8,  9}},
    {{1,  64, 56, 56}},
    {{1, 256, 56, 56}},
    {{1, 128, 56, 56}},
    {{1, 128, 28, 28}},
    {{1, 512, 28, 28}},
    {{1, 256, 28, 28}},
    {{1, 256, 14, 14}},
    {{1, 1024,14, 14}},
    {{1, 512, 14, 14}},
    {{1, 512,  7,  7}},
    {{1, 2048, 7,  7}},
    {{1, 512,  7,  7}}
};

static std::vector<Levels> s_QuantizeLevels = {
    2,
    256
};

static std::vector<SwitchOut> s_QuantizeSwitchOut = {
    0,
    1
};

TEST_P(myriadLayersTestsBinaryConvolution_nightly, BinaryConvolution) {
    tensor_test_params dims  = std::get<0>(GetParam());
    int dilations            = std::get<1>(GetParam());
    int group                = std::get<2>(GetParam());
    param_size kernel        = std::get<3>(GetParam());
    int strides              = std::get<4>(GetParam());
    std::string customConfig = std::get<5>(GetParam());

    if(!customConfig.empty() && !CheckMyriadX()) {
        GTEST_SKIP()<<"Custom layers for MYRIAD2 not supported";
    }
    _config[VPU_CONFIG_KEY(CUSTOM_LAYERS)] = customConfig;

    SetInputTensor(dims);
    auto dimsOutput = dims;
    dimsOutput.h = (dims.h) / strides;
    dimsOutput.w = (dims.w) / strides;
    SetOutputTensor(dimsOutput);
    size_t numWeights = kernel.x * kernel.y * dims.c * dims.c;
    size_t numBiases = 0;
    InferenceEngine::TBlob<uint8_t>::Ptr weights_ptr = InferenceEngine::TBlob<uint8_t>::Ptr(GenWeights(numWeights));

    std::map<std::string, std::string> params;
    params["mode"] = "xnor-popcount";
    params["pad_value"] = "-1.0";
    params["pads_begin"] = std::to_string(kernel.x/2) + "," + std::to_string(kernel.y/2);
    params["pads_end"] = std::to_string(kernel.x/2) + "," + std::to_string(kernel.y/2);
    params["input"] = std::to_string(dims.c);
    params["output"] = std::to_string(dims.c);
    params["dilations"] = std::to_string(dilations) + "," + std::to_string(dilations);
    params["group"] = std::to_string(group);
    params["kernel"] = std::to_string(kernel.x) + "," + std::to_string(kernel.y);
    params["strides"] = std::to_string(strides) + "," + std::to_string(strides);

    ASSERT_NO_FATAL_FAILURE(makeSingleLayerNetwork(LayerInitParams("BinaryConvolution")
                                        .params(params)
                                        .weights(numWeights)
                                        .biases(numBiases),
                                        {},
                                        weights_ptr));

    ASSERT_TRUE(Infer());

    ASSERT_NO_FATAL_FAILURE(refBinaryConvolution(_inputMap.begin()->second, weights_ptr, _refBlob,
                                                 dilations, group, kernel, strides,
                                                 false));

    CompareCommonAbsolute(_outputMap.begin()->second, _refBlob, 0);
}

static std::vector<Dims> s_BinaryConvolutionTensors = {
    {{1, 64, 112, 112}},
    {{1, 128, 56, 56}},
    {{1, 256, 28, 28}},
    {{1, 256, 14, 14}},
    {{1, 16, 16, 16}},
    {{1,  2,  2,  2}},
};

static std::vector<Dilations> s_BinaryConvolutionDilations = {
    1, 2
};
static std::vector<Group> s_BinaryConvolutionGroup = {
    1, 2
};
static std::vector<Kernel> s_BinaryConvolutionKernel = {
    {{1, 1}},
    {{1, 3}},
    {{3, 3}},
};
static std::vector<Strides> s_BinaryConvolutionStrides = {
    1, 2
};

TEST_P(myriadLayersTestsExperimentalDetectronPriorGridGenerator_nightly,
       ExperimentalDetectronPriorGridGenerator) {

    // Setup parameters and configuration.
    std::vector<size_t> image_dims = std::get<0>(GetParam());
    std::string customConfig = std::get<1>(GetParam());
    if(!customConfig.empty() && !CheckMyriadX()) {
        GTEST_SKIP() << "Custom layers for MYRIAD2 not supported";
    }
    _config[VPU_CONFIG_KEY(CUSTOM_LAYERS)] = customConfig;

    IN_OUT_desc inputTensors = {{1, 1, 3, 4}, image_dims, {1, 3, 480, 480}};
    IN_OUT_desc outputTensors = {{1, 1,
         inputTensors[0][2] *
         inputTensors[1][2] *
         inputTensors[1][3],
         inputTensors[0][3]}};
    SetInputTensors(inputTensors);
    SetOutputTensors(outputTensors);

    // Calculate strides. The stride dimensions are calculated by the equation
    // (image feature map dimension) / (input feature map dimension).
    float stride_h = static_cast<float>(inputTensors[2][2]) /
                     inputTensors[1][2];
    float stride_w = static_cast<float>(inputTensors[2][3]) /
                     inputTensors[1][3];

    std::map<std::string, std::string> params = {
        {"stride_h", std::to_string(stride_h)},
        {"stride_w", std::to_string(stride_w)}
    };
    // Run inference on OpenCL kernel.
    ASSERT_NO_FATAL_FAILURE(makeSingleLayerNetwork(
                LayerInitParams("ExperimentalDetectronPriorGridGenerator").params(params)));
    ASSERT_TRUE(Infer());

    // Setup of reference input and reference output blobs.
    std::vector<Blob::Ptr> reference_input_blobs(inputTensors.size());
    std::vector<Blob::Ptr> reference_output_blobs(outputTensors.size());
    int k = 0;
    for (auto& p : _inputMap) {
        reference_input_blobs[k++] = p.second;
    }
    reference_output_blobs[0] = _refBlob;

    // Run inference on reference implementation.
    refExperimentalDetectronPriorGridGenerator(
            reference_input_blobs, reference_output_blobs,
            inputTensors[1][2], inputTensors[1][3], stride_h, stride_w);

    CompareCommonAbsolute(_outputMap.begin()->second, reference_output_blobs[0], 0.01f);
}

static std::vector<std::vector<size_t>>
s_ExperimentalDetectronPriorGridGeneratorImageDims = {
    {1, 128, 240, 240},
    {1, 128, 120, 120},
    {1, 128, 60, 60},
    {1, 128, 30, 30}
};

