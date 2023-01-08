// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cmath>
#include "myriad_layers_tests.hpp"

using namespace InferenceEngine;

#define ERROR_BOUND 0.5f

typedef struct {
    SizeVector src_dims;
    SizeVector weights_dims;
    SizeVector dst_dims;
    std::string custom_config;
} dims_config;

PRETTY_PARAM(hwAcceleration, std::string);
PRETTY_PARAM(dimsConfig, dims_config);
PRETTY_PARAM(isHWC, int);

typedef myriadLayerTestBaseWithParam<std::tuple<std::string, isHWC, dims_config>> myriadConvolution1x1LayerTests_smoke;

void refConvolution1x1(const Blob::Ptr src, InferenceEngine::TBlob<uint8_t>::Ptr weights, Blob::Ptr dst, int isHWC) {
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

    for (int oc = 0; oc < OC; ++oc)
    {
        for (int oh = 0; oh < OH; oh++)
        {
            for (int ow = 0; ow < OW; ow++)
            {
                float valYXZ = 0.0f;
                ie_fp16 valZYX = 0.0f;
                for (int ic = 0; ic < IC; ++ic)
                {
                    int iw = ow;
                    int ih = oh;

                    if (iw < 0 || iw >= (int)IW || ih < 0 || ih >= (int)IH)
                    {
                        continue;
                    }
                    uint32_t indx;
                    if (isHWC == 1) {
                        indx = ic + iw * IC + ih * IC * IW;
                        valYXZ = (valYXZ) + (PrecisionUtils::f16tof32(in[indx]) * PrecisionUtils::f16tof32(w[oc*IC + ic]));
                    }
                    else
                    {
                        indx = iw + ih * IW + ic * IW * IH;
                        valZYX = PrecisionUtils::f32tof16(PrecisionUtils::f16tof32(valZYX) + PrecisionUtils::f16tof32(PrecisionUtils::f32tof16(PrecisionUtils::f16tof32(in[indx]) * PrecisionUtils::f16tof32(w[oc*IC + ic]))));
                    }
                }
                if (isHWC == 1) {
                    out[oc*OH*OW + oh*OW + ow] = PrecisionUtils::f32tof16(valYXZ);
                }
                else {
                    out[oc*OH*OW + oh*OW + ow] = (valZYX);
                }
            }
        }
    }
}

TEST_P(myriadConvolution1x1LayerTests_smoke, Convolution1x1) {
    std::string model = R"V0G0N(
       <net name="Convolution1x1" version="2" batch="1">
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
            <layer id="2" name="conv1x1" precision="FP16" type="Convolution">
                <data stride="1,1" pad="0,0" kernel="1,1" dilation="1,1" output="48" group="1"/>
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
    int isHWC                 = std::get<1>(GetParam());
    dims_config customConfig  = std::get<2>(GetParam());
    const auto layout = isHWC ? Layout::NHWC : Layout::NCHW;

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

    size_t num_weights = IC * OC;

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

    model.replace( model.find("@size_weights@"), sizeof("@size_weights@") -1, std::to_string(num_weights * sizeof(ie_fp16)));

    InferenceEngine::TBlob<uint8_t>::Ptr weights_ptr = InferenceEngine::TBlob<uint8_t>::Ptr(GenWeights(num_weights));

    InferenceEngine::Core ie;
    auto network = ie.ReadNetwork(model, weights_ptr);

    _inputsInfo = network.getInputsInfo();
    _inputsInfo["data"]->setPrecision(Precision::FP16);
    _inputsInfo["data"]->setLayout(layout);

    _outputsInfo = network.getOutputsInfo();
    _outputsInfo["conv1x1"]->setPrecision(Precision::FP16);
    _outputsInfo["conv1x1"]->setLayout(layout);

    ASSERT_NO_THROW(_exeNetwork = _vpuPluginPtr->LoadNetwork(network,
            {{InferenceEngine::MYRIAD_CUSTOM_LAYERS, customConfig.custom_config},
             {InferenceEngine::MYRIAD_ENABLE_HW_ACCELERATION, HWConfigValue}}));

    ASSERT_NO_THROW(_inferRequest = _exeNetwork.CreateInferRequest());
    
    Blob::Ptr data;
    ASSERT_NO_THROW(data = _inferRequest.GetBlob("data"));
    GenRandomData(data);

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

        for (auto it = perfVec.begin(); it != perfVec.end(); ++it) {
            std::string layerName = it->first;
            InferenceEngine::InferenceEngineProfileInfo info = it->second;
            if (info.status == InferenceEngine::InferenceEngineProfileInfo::EXECUTED) {
                printf("[----------] Myriad time = '%s' layer with '%s' type is %f ms.\n", layerName.c_str(), info.exec_type, info.realTime_uSec / 1000.f);
            }
        }
        printf("[----------] input dim: [%d %d %d %d]; output dim: [%d %d %d %d].\n", IB, IC, IH, IW, OB, OC, OH, OW);
        printf("[----------] isHardware: %s; isHWC: %d.\n", HWConfigValue.c_str(), isHWC);
    }

    Blob::Ptr outputBlob;
    ASSERT_NO_THROW(outputBlob = _inferRequest.GetBlob("conv1x1"));
    
    _refBlob = make_shared_blob<ie_fp16>(TensorDesc(Precision::FP16, outputBlob->getTensorDesc().getDims(), NCHW));
    _refBlob->allocate();

    refConvolution1x1(data, weights_ptr, _refBlob, isHWC);

    CompareCommonAbsolute(outputBlob, _refBlob, ERROR_BOUND);
}
static std::vector<dims_config> s_DimsConfig = {
#ifdef VPU_HAS_CUSTOM_KERNELS
    {{1,   64, 56, 56}, {1, 1, 1,   64 *   64}, {1,   64, 56, 56}, getIELibraryPath() + "/vpu_custom_kernels/customLayerBindings.xml"},
    {{1,   64, 56, 56}, {1, 1, 1,   64 *  256}, {1,  256, 56, 56}, getIELibraryPath() + "/vpu_custom_kernels/customLayerBindings.xml"},
    {{1,  256, 56, 56}, {1, 1, 1,  256 *  256}, {1,  256, 56, 56}, getIELibraryPath() + "/vpu_custom_kernels/customLayerBindings.xml"},
    {{1,  256, 56, 56}, {1, 1, 1,  256 *  128}, {1,  128, 56, 56}, getIELibraryPath() + "/vpu_custom_kernels/customLayerBindings.xml"},
    {{1,  128, 28, 28}, {1, 1, 1,  128 *  512}, {1,  512, 28, 28}, getIELibraryPath() + "/vpu_custom_kernels/customLayerBindings.xml"},
    {{1,  512, 28, 28}, {1, 1, 1,  512 *  128}, {1,  128, 28, 28}, getIELibraryPath() + "/vpu_custom_kernels/customLayerBindings.xml"},
    {{1,  512, 28, 28}, {1, 1, 1,  512 *  256}, {1,  256, 28, 28}, getIELibraryPath() + "/vpu_custom_kernels/customLayerBindings.xml"},
    {{1,  256, 14, 14}, {1, 1, 1,  256 * 1024}, {1, 1024, 14, 14}, getIELibraryPath() + "/vpu_custom_kernels/customLayerBindings.xml"},
    {{1, 1024, 14, 14}, {1, 1, 1, 1024 *  256}, {1,  256, 14, 14}, getIELibraryPath() + "/vpu_custom_kernels/customLayerBindings.xml"},
    {{1, 1024, 14, 14}, {1, 1, 1, 1024 *  512}, {1,  512, 14, 14}, getIELibraryPath() + "/vpu_custom_kernels/customLayerBindings.xml"},
    {{1,  512,  7,  7}, {1, 1, 1,  512 * 2048}, {1, 2048,  7,  7}, getIELibraryPath() + "/vpu_custom_kernels/customLayerBindings.xml"},
    {{1, 2048,  7,  7}, {1, 1, 1, 2048 *  512}, {1,  512,  7,  7}, getIELibraryPath() + "/vpu_custom_kernels/customLayerBindings.xml"},
#endif
};

static std::vector<isHWC> s_isHWC = {
#ifdef VPU_HAS_CUSTOM_KERNELS
   {0, 1}
#endif
};
