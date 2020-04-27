// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cmath>
#include "myriad_layers_tests.hpp"
// #include <iostream>

using namespace InferenceEngine;

#define ERROR_BOUND 1e-3

static inline float triangleCoeff(float x)
{
    return (1.0f - fabsf(x));
}
void refResample(const Blob::Ptr src, Blob::Ptr dst, int antialias) {
    ie_fp16 *src_data = static_cast<ie_fp16*>(src->buffer());
    ie_fp16 *output_sequences = static_cast<ie_fp16*>(dst->buffer());
    ASSERT_NE(src_data, nullptr);
    ASSERT_NE(output_sequences, nullptr);

    const auto& src_dims = src->getTensorDesc().getDims();
    const auto& dst_dims = dst->getTensorDesc().getDims();
    int OH = dst_dims[2];
    int OW = dst_dims[3];

    int C  = src_dims[1];
    int IH = src_dims[2];
    int IW = src_dims[3];

    if (IH == OH && IW == OW)
    {
        int b = 0;
        for (int c = 0; c < C; c++)
            for (int h = 0; h < IH; h++)
                for (int w = 0; w < IW; w++){
                int dst_index = w + IW * h + IW * IH * c;
                int src_index = dst_index;
                output_sequences[dst_index] = src_data[src_index];
                }
        return;
    }

    const float fy = static_cast<float>(IH) / static_cast<float>(OH);
    const float fx = static_cast<float>(IW) / static_cast<float>(OW);

    float ax = 1.0f / fx;
    float ay = 1.0f / fy;

    int rx = (fx < 1.0f) ? 2 : ceil((1.0f)/ax);
    int ry = (fy < 1.0f) ? 2 : ceil((1.0f)/ay);

    for (int c = 0; c < C; c++)
    {
        const ie_fp16* in_ptr = src_data + IW*IH*c;
        ie_fp16* out_ptr = output_sequences + OW*OH*c;

        for (int oy = 0; oy < OH; oy++)
        {
            for (int ox = 0; ox < OW; ox++)
            {
                float ix = ox*fx + fx / 2.0f - 0.5f;
                float iy = oy*fy + fy / 2.0f - 0.5f;

                int ix_r = (int)(round(ix));
                int iy_r = (int)(round(iy));

                float sum=0;
                float wsum=0;

                if(antialias){
                    for (int y = iy_r - ry; y <= iy_r + ry; y++)
                    {
                        for (int x = ix_r - rx; x <= ix_r + rx; x++)
                        {
                            if (y < 0 || x < 0) continue;
                            if (y >= (int)IH || x >= (int)IW) continue;

                            float dx = ix - x;
                            float dy = iy - y;

                            float w = ax*triangleCoeff(ax*dx) * ay*triangleCoeff(ay*dy);

                            sum += w * PrecisionUtils::f16tof32(in_ptr[y*IW + x]);
                            wsum += w;
                        }
                    }
                    out_ptr[oy * OW + ox] = PrecisionUtils::f32tof16((!wsum) ? 0.0f : (sum / wsum));
                }
                else{
                    out_ptr[oy * OW + ox] = in_ptr[iy_r * IW + ix_r];
                }
            }
        }
    }
}

PRETTY_PARAM(hwAcceleration, std::string);
PRETTY_PARAM(customConfig, std::string);
PRETTY_PARAM(Antialias, int)

typedef myriadLayerTestBaseWithParam<std::tuple<std::string, std::string, Antialias>> myriadResampleLayerTests_nightly;

TEST_P(myriadResampleLayerTests_nightly, Resample) {
    std::string model = R"V0G0N(
       <net name="Resample" version="2" batch="1">
           <layers>
            <layer id="0" name="data" precision="FP16" type="Input">
                <output>
                    <port id="0">
                        <dim>1</dim>
                        <dim>128</dim>
                        <dim>26</dim>
                        <dim>26</dim>
                    </port>
                </output>
            </layer>
               <layer id="1" name="detector/yolo-v3/ResizeNearestNeighbor" precision="FP16" type="Resample">
                  <data antialias="@TEST@" factor="2.0" type="caffe.ResampleParameter.NEAREST" fx="0.5" fy="0.5"/>
                <input>
                    <port id="1">
                        <dim>1</dim>
                        <dim>128</dim>
                        <dim>26</dim>
                        <dim>26</dim>
                    </port>
                </input>
                <output>
                    <port id="2">
                        <dim>1</dim>
                        <dim>128</dim>
                        <dim>52</dim>
                        <dim>52</dim>
                    </port>
                </output>
            </layer>
           </layers>
           <edges>
               <edge from-layer="0" from-port="0" to-layer="1" to-port="1"/>
           </edges>
       </net>
   )V0G0N";

    SetSeed(DEFAULT_SEED_VALUE + 6);

    std::string HWConfigValue = std::get<0>(GetParam());
    std::string customConfig = std::get<1>(GetParam());
    int antialias = std::get<2>(GetParam());

    model.replace( model.find("@TEST@"), sizeof("@TEST@") -1, std::to_string(antialias));
    if((customConfig != "") || (antialias != 1)){
        if(!customConfig.empty() && !CheckMyriadX()) {
            GTEST_SKIP()<<"Custom layers for MYRIAD2 not supported";
        }
        _config[VPU_CONFIG_KEY(HW_STAGES_OPTIMIZATION)] = HWConfigValue;
        _config[VPU_CONFIG_KEY(CUSTOM_LAYERS)] = customConfig;
        StatusCode st;

        ASSERT_NO_THROW(readNetwork(model));

        const auto& network = _cnnNetwork;

        _inputsInfo = network.getInputsInfo();
        _inputsInfo["data"]->setPrecision(Precision::FP16);
        _inputsInfo["data"]->setLayout(NCHW);

        _outputsInfo = network.getOutputsInfo();
        _outputsInfo["detector/yolo-v3/ResizeNearestNeighbor"]->setPrecision(Precision::FP16);

        ASSERT_NO_THROW(st = _vpuPluginPtr->LoadNetwork(_exeNetwork, network,
                                                        {{VPU_CONFIG_KEY(CUSTOM_LAYERS), customConfig}, {VPU_CONFIG_KEY(HW_STAGES_OPTIMIZATION), HWConfigValue}}, &_resp));
        ASSERT_EQ(StatusCode::OK, st) << _resp.msg;
        ASSERT_NE(_exeNetwork, nullptr) << _resp.msg;

        ASSERT_NO_THROW(st = _exeNetwork->CreateInferRequest(_inferRequest, &_resp));
        ASSERT_EQ(StatusCode::OK, st) << _resp.msg;

        Blob::Ptr data;
        ASSERT_NO_THROW(st = _inferRequest->GetBlob("data", data, &_resp));
        ASSERT_EQ(StatusCode::OK, st) << _resp.msg;

        GenRandomData(data);

        ASSERT_NO_THROW(st = _inferRequest->Infer(&_resp));
        ASSERT_EQ(StatusCode::OK, st) << _resp.msg;

        Blob::Ptr outputBlob;
        ASSERT_NO_THROW(_inferRequest->GetBlob("detector/yolo-v3/ResizeNearestNeighbor", outputBlob, &_resp));
        ASSERT_EQ(StatusCode::OK, st) << _resp.msg;

        _refBlob = make_shared_blob<ie_fp16>(TensorDesc(Precision::FP16, outputBlob->getTensorDesc().getDims(), NCHW));
        _refBlob->allocate();

        refResample(data, _refBlob, antialias);

        CompareCommonAbsolute(outputBlob, _refBlob, ERROR_BOUND);
    }
}

static std::vector<std::string> s_ResampleCustomConfig = {
    "",
#ifdef VPU_HAS_CUSTOM_KERNELS
   getIELibraryPath() + "/vpu_custom_kernels/customLayerBindings.xml"
#endif
};

static std::vector<Antialias> s_ResampleAntialias = {
        {0, 1}
};
