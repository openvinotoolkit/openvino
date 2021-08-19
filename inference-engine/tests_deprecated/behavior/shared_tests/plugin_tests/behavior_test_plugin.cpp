// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior_test_plugin.h"
#include "common_test_utils/xml_net_builder/xml_filler.hpp"
#include "common_layers_params.hpp"

std::string getRawConvReluNormPoolFcModel() {
    return (R"V0G0N(
<net name="_NAME_" version="_VER_" batch="1">
    <layers>
        <layer name="data" type="Input" precision="_PRC_" id="0">
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>227</dim>
                    <dim>227</dim>
                </port>
            </output>
        </layer>
        <layer name="conv1" type="Convolution" precision="_PRC_" id="1">
            <convolution_data stride-x="4" stride-y="4" pad-x="0" pad-y="0" kernel-x="11" kernel-y="11" output="16" group="1"/>
            <input>
                <port id="1">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>227</dim>
                    <dim>227</dim>
                </port>
            </input>
            <output>
                <port id="2">
                    <dim>1</dim>
                    <dim>16</dim>
                    <dim>55</dim>
                    <dim>55</dim>
                </port>
            </output>
            <weights offset="0" size="_CONV_WS_"/>
            <biases offset="_CONV_WS_" size="_CONV_BS_"/>
        </layer>
        <layer name="relu1" type="ReLU" precision="_PRC_" id="2">
            <input>
                <port id="3">
                    <dim>1</dim>
                    <dim>16</dim>
                    <dim>55</dim>
                    <dim>55</dim>
                </port>
            </input>
            <output>
                <port id="4">
                    <dim>1</dim>
                    <dim>16</dim>
                    <dim>55</dim>
                    <dim>55</dim>
                </port>
            </output>
        </layer>
        <layer name="norm1" type="Norm" precision="_PRC_" id="3">
            <norm_data alpha="9.9999997e-05" beta="0.75" local-size="5" region="across"/>
            <input>
                <port id="5">
                    <dim>1</dim>
                    <dim>16</dim>
                    <dim>55</dim>
                    <dim>55</dim>
                </port>
            </input>
            <output>
                <port id="6">
                    <dim>1</dim>
                    <dim>16</dim>
                    <dim>55</dim>
                    <dim>55</dim>
                </port>
            </output>
        </layer>
        <layer name="pool1" type="Pooling" precision="_PRC_" id="4">
            <pooling_data kernel-x="3" kernel-y="3" pad-x="0" pad-y="0" stride-x="2" stride-y="2" rounding-type="ceil" pool-method="max"/>
            <input>
                <port id="7">
                    <dim>1</dim>
                    <dim>16</dim>
                    <dim>55</dim>
                    <dim>55</dim>
                </port>
            </input>
            <output>
                <port id="8">
                    <dim>1</dim>
                    <dim>16</dim>
                    <dim>27</dim>
                    <dim>27</dim>
                </port>
            </output>
        </layer>
        <layer name="fc6" type="FullyConnected" precision="_PRC_" id="5">
            <fc_data out-size="10"/>
            <input>
                <port id="9">
                    <dim>1</dim>
                    <dim>16</dim>
                    <dim>27</dim>
                    <dim>27</dim>
                </port>
            </input>
            <output>
                <port id="10">
                    <dim>1</dim>
                    <dim>10</dim>
                </port>
            </output>
            <weights offset="_FC_W_OFFS_" size="_FC_WS_"/>
            <biases offset="_FC_B_OFFS_" size="_FC_BS_"/>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="1" to-port="1"/>
        <edge from-layer="1" from-port="2" to-layer="2" to-port="3"/>
        <edge from-layer="2" from-port="4" to-layer="3" to-port="5"/>
        <edge from-layer="3" from-port="6" to-layer="4" to-port="7"/>
        <edge from-layer="4" from-port="8" to-layer="5" to-port="9"/>
    </edges>
</net>
)V0G0N");
}

TestModel getConvReluNormPoolFcModel(InferenceEngine::Precision netPrc) {
    std::string model_str = getRawConvReluNormPoolFcModel();
    /* Default values for FP16 are used */
    size_t convWeigthsLen = 5808;  // kernel_x * kernel_y * in_channels * out_channels
    size_t convWeigthsSize = convWeigthsLen * 2;  // 2 (bytes in FP16)
    size_t convBiasesLen = 16;  // out_channels
    size_t convBiasesSize = convBiasesLen * 2;
    size_t fcWeigthsLen = 116640;  // fc_in_channels * fc_h * fc_w * fc_out_channels
    size_t fcWeigthsSize = fcWeigthsLen * 2;
    size_t fcBiasesLen = 10;  // fc_out_channels
    size_t fcBiasesSize = fcBiasesLen * 2;
    switch (netPrc) {
        case InferenceEngine::Precision::FP16:
        case InferenceEngine::Precision::Q78:
            break;
        case InferenceEngine::Precision::FP32:
            convWeigthsSize *= 2;  // 4 bytes in FP32
            convBiasesSize *= 2;
            fcWeigthsSize *= 2;
            fcBiasesSize *= 2;
            break;
        default:
            std::string err = "ConvReluNormPoolFcModel can not be constructed with precision ";
            err += netPrc.name();
            throw std::runtime_error(err);
    }
    std::string irName = std::string("ConvReluNormPoolFcModel") + netPrc.name();
    REPLACE_WITH_STR(model_str, "_NAME_", irName);
    REPLACE_WITH_NUM(model_str, "_VER_", 2);
    REPLACE_WITH_STR(model_str, "_PRC_", netPrc.name());
    REPLACE_WITH_NUM(model_str, "_CONV_WS_", convWeigthsSize);
    REPLACE_WITH_NUM(model_str, "_CONV_BS_", convBiasesSize);
    REPLACE_WITH_NUM(model_str, "_FC_W_OFFS_", convWeigthsSize + convBiasesSize);
    REPLACE_WITH_NUM(model_str, "_FC_WS_", fcWeigthsSize);
    REPLACE_WITH_NUM(model_str, "_FC_B_OFFS_", convWeigthsSize + convBiasesSize + fcWeigthsSize);
    REPLACE_WITH_NUM(model_str, "_FC_BS_", fcBiasesSize);
    return TestModel(model_str, CommonTestUtils::getWeightsBlob(
            convWeigthsSize + convBiasesSize + fcWeigthsSize + fcBiasesSize));
}
