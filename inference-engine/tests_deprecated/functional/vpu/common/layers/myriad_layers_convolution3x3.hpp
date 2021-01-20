// // Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cmath>
#include "myriad_layers_tests.hpp"

using namespace InferenceEngine;

#define ERROR_BOUND 0.5f

typedef struct {
    SizeVector  src_dims;
    SizeVector  dst_dims;
    int         stride_xy;
    std::string custom_config;
} dims_config_con3x3;

PRETTY_PARAM(hwAcceleration, std::string);
PRETTY_PARAM(dimsConfig, dims_config_con3x3);

typedef myriadLayerTestBaseWithParam<std::tuple<std::string, dims_config_con3x3>> myriadConvolution3x3LayerTests_smoke;

void refConvolution3x3(const Blob::Ptr src, InferenceEngine::TBlob<uint8_t>::Ptr weights, Blob::Ptr dst, int stride_x, int stride_y, int pad_x, int pad_y, int dilation_x, int dilation_y) {
    
          ie_fp16 *in = static_cast<ie_fp16*>(src->buffer());
    const ie_fp16 *w = weights->readOnly().as<const ie_fp16 *>();
          ie_fp16 *out = static_cast<ie_fp16*>(dst->buffer());

    ASSERT_NE(in, nullptr);
    ASSERT_NE(w, nullptr);
    ASSERT_NE(out, nullptr);

    const auto& in_dims = src->getTensorDesc().getDims();
    size_t in_width      = in_dims[in_dims.size() - 1];
    size_t in_height     = in_dims[in_dims.size() - 2];
    size_t in_channels   = in_dims[in_dims.size() - 3];

    size_t IW = in_width;
    size_t IH = in_height;
    size_t IC = in_channels;
    
    const auto& out_dims = dst->getTensorDesc().getDims();
    size_t out_width      = out_dims[out_dims.size() - 1];
    size_t out_height     = out_dims[out_dims.size() - 2];
    size_t out_channels   = out_dims[out_dims.size() - 3];

    size_t OW = out_width;
    size_t OH = out_height;
    size_t OC = out_channels;

    size_t group = 1;

    size_t src_channels = IC / group;
    size_t dst_channels = OC / group;

    size_t KW = 3;
    size_t KH = 3;

    //the start address after 1 line/column padding(requested by convolution operation with 3x3 kernel )
    in = in + 1 + 1 * IW; 

    int cnt = 0;

    for (size_t g = 0; g < group; ++g)
    {
        for (size_t oc = 0; oc < dst_channels; ++oc)
        {
            size_t dst_channel = (g * dst_channels + oc);
            for (size_t oh = 0; oh < OH; oh++)
            {
                for (size_t ow = 0; ow < OW; ow++)
                {
                    size_t oidx = dst_channel + ow * OC + oh * OC * OW;
                    float val = 0.0f;
                    ie_fp16 hval = PrecisionUtils::f32tof16(val);
                    float fval = 0.0f;

                    for (size_t ic = 0; ic < src_channels; ++ic)
                    {
                        size_t src_channel = (g * src_channels + ic);

                        for (int ky = 0; ky < KH; ++ky)
                        {
                            for (int kx = 0; kx < KW; ++kx)
                            {
                                int32_t iw = ow * stride_x - pad_x + kx * dilation_x;
                                int32_t ih = oh * stride_y - pad_y + ky * dilation_y;

                                float v = PrecisionUtils::f16tof32(in[iw + ih * IW + src_channel * IW * IH])
                                        * 
                                        PrecisionUtils::f16tof32(w[oc*IC*KW*KH + ic*KW*KH + ky*KW + kx]);
                                val += v;
                            }
                        }
                    }

                    out[oc*OH*OW + oh*OW + ow] = PrecisionUtils::f32tof16(val);
                }
            }
        }
    }
}

TEST_P(myriadConvolution3x3LayerTests_smoke, Convolution3x3) {
    std::string model = R"V0G0N(
       <net name="Convolution3x3" version="2" batch="1">
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
            <layer id="2" name="conv3x3" precision="FP16" type="Convolution">
                <data stride="@stride-x@,@stride-y@" pads_begin="1,1" pads_end="1,1" dilation="1,1" output="1" kernel="3,3"/>
                <input>
                    <port id="0">
                        <dim>@IB@</dim>
                        <dim>@IC@</dim>
                        <dim>@IH@</dim>
                        <dim>@IW@</dim>
                    </port>
                </input>
                <output>
                    <port id="2">
                        <dim>@OB@</dim>
                        <dim>@OC@</dim>
                        <dim>@OH@</dim>
                        <dim>@OW@</dim>
                    </port>
                </output>
                <weights offset="0" size="@size_weights@"/>
            </layer>
           </layers>
           <edges>
               <edge from-layer="0" from-port="0" to-layer="2" to-port="0"/>
           </edges>
       </net>
   )V0G0N";
    SetSeed(DEFAULT_SEED_VALUE + 6);
    std::string HWConfigValue = std::get<0>(GetParam());
    dims_config_con3x3 customConfig  = std::get<1>(GetParam());
     
    int stride_xy = customConfig.stride_xy;//(int)std::get<1>(GetParam());

    if(!customConfig.custom_config.empty() && !CheckMyriadX()) {
        GTEST_SKIP()<<"Custom layers for MYRIAD2 not supported";
    }
    _config[InferenceEngine::MYRIAD_ENABLE_HW_ACCELERATION] = HWConfigValue;
    _config[InferenceEngine::MYRIAD_CUSTOM_LAYERS] = customConfig.custom_config;

    int IB = customConfig.src_dims[0];
    int IC = customConfig.src_dims[1];
    int IH = customConfig.src_dims[2];
    int IW = customConfig.src_dims[3];

    int OB = customConfig.dst_dims[0];
    int OC = customConfig.dst_dims[1];
    int OH = customConfig.dst_dims[2];
    int OW = customConfig.dst_dims[3];

    // 3 * 3 = 3x3 kernel size
    size_t num_weights = 3 * 3 * IC * OC;

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

    model.replace( model.find("@stride-x@"), sizeof("@stride-x@") -1, std::to_string(stride_xy));
    model.replace( model.find("@stride-y@"), sizeof("@stride-y@") -1, std::to_string(stride_xy));
    model.replace( model.find("@size_weights@"), sizeof("@size_weights@") -1, std::to_string(num_weights * sizeof(ie_fp16)));

    InferenceEngine::TBlob<uint8_t>::Ptr weights_ptr = InferenceEngine::TBlob<uint8_t>::Ptr(GenWeights(num_weights));

    InferenceEngine::Core ie;
    auto network = ie.ReadNetwork(model, weights_ptr);

    _inputsInfo = network.getInputsInfo();
    _inputsInfo["data"]->setPrecision(Precision::FP16);
    _inputsInfo["data"]->setLayout(NCHW);

    _outputsInfo = network.getOutputsInfo();
    _outputsInfo["conv3x3"]->setPrecision(Precision::FP16);
    _outputsInfo["conv3x3"]->setLayout(NCHW);

    ASSERT_NO_THROW(_exeNetwork = _vpuPluginPtr->LoadNetwork(network,
            {{InferenceEngine::MYRIAD_CUSTOM_LAYERS, customConfig.custom_config},
             {InferenceEngine::MYRIAD_ENABLE_HW_ACCELERATION, HWConfigValue}}));

    ASSERT_NO_THROW(_inferRequest = _exeNetwork.CreateInferRequest());
    
    Blob::Ptr data;
    ASSERT_NO_THROW(data = _inferRequest.GetBlob("data"));
    GenRandomData(data);

    //padding with zeros 1 row(top/bottom), 1 column(left/right) input tensor
    for(int ic = 0; ic < IC; ++ic){
        for(int iw = 0; iw < IW; ++iw){
            int indx_l0 = iw + 0 * IW + ic * IW * IH;
            int indx_ln = iw + (IH - 1) * IW + ic * IW * IH;
            *((ie_fp16*)data->buffer() + indx_l0) = 0.0f;
            *((ie_fp16*)data->buffer() + indx_ln) = 0.0f;
        }
    }
    for(int ic = 0; ic < IC; ++ic){
        for(int ih = 0; ih < IH; ++ih){
            int indx_c0 = 0 + ih * IW + ic * IW * IH;
            int indx_cn = (IW - 1) + ih * IW + ic * IW * IH;
            *((ie_fp16*)data->buffer() + indx_c0) = 0.0f;
            *((ie_fp16*)data->buffer() + indx_cn) = 0.0f;
        }
    }

    ASSERT_NO_THROW(_inferRequest.Infer());

    // TODO: fix CVS-47174
    if (0)
    {
        auto perfMap = _inferRequest.GetPerformanceCounts();
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
                printf("[----------] Myriad time = '%s' layer with '%s' type is %f ms.\n", layerName.c_str(), info.exec_type, info.realTime_uSec / 1000.f);
            }
        }
        printf("[----------] input dim: [%d %d %d %d]; output dim: [%d %d %d %d]; stride: %d.\n", IB, IC, IH, IW, OB, OC, OH, OW, stride_xy);
        printf("[----------] isHardware: %s.\n", HWConfigValue.c_str());
    }

    Blob::Ptr outputBlob;
    ASSERT_NO_THROW(outputBlob = _inferRequest.GetBlob("conv3x3"));

    _refBlob = make_shared_blob<ie_fp16>(TensorDesc(Precision::FP16, outputBlob->getTensorDesc().getDims(), NCHW));
    _refBlob->allocate();

    refConvolution3x3(data, weights_ptr, _refBlob, stride_xy, stride_xy, 1, 1, 1, 1);

    CompareCommonAbsolute(outputBlob, _refBlob, ERROR_BOUND);
}

static std::vector<dims_config_con3x3> s_DimsConfig = {
#ifdef VPU_HAS_CUSTOM_KERNELS
    {{1,   64, 58, 58}, {1,  64, 56, 56}, 1, getIELibraryPath() + "/vpu_custom_kernels/customLayerBindings.xml"},
    {{1,  128, 58, 58}, {1, 128, 56, 56}, 1, getIELibraryPath() + "/vpu_custom_kernels/customLayerBindings.xml"},
    {{1,  128, 30, 30}, {1, 128, 28, 28}, 1, getIELibraryPath() + "/vpu_custom_kernels/customLayerBindings.xml"},
    // {{1,  256, 30, 30}, {1, 256, 28, 28}, 1, getIELibraryPath() + "/vpu_custom_kernels/customLayerBindings.xml"},
    {{1,  256, 16, 16}, {1, 256, 14, 14}, 1, getIELibraryPath() + "/vpu_custom_kernels/customLayerBindings.xml"},
    // {{1,  512, 16, 16}, {1, 512, 14, 14}, 1, getIELibraryPath() + "/vpu_custom_kernels/customLayerBindings.xml"},
    {{1,  512,  9,  9}, {1, 512,  7,  7}, 1, getIELibraryPath() + "/vpu_custom_kernels/customLayerBindings.xml"},
    
    {{1,  128, 58, 58}, {1, 128, 28, 28}, 2, getIELibraryPath() + "/vpu_custom_kernels/customLayerBindings.xml"},
    {{1,  256, 30, 30}, {1, 256, 14, 14}, 2, getIELibraryPath() + "/vpu_custom_kernels/customLayerBindings.xml"},
    {{1,  256, 16, 16}, {1, 384,  7,  7}, 2, getIELibraryPath() + "/vpu_custom_kernels/customLayerBindings.xml"},
#endif
};
