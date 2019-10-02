// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_reader_tests.hpp"

TEST_F(NGraphReaderTests, DISABLED_ReadDetectionOutputNetwork) {
    std::string model = R"V0G0N(
<net name="Activation" version="10">
    <layers>
        <layer name="in1" type="Parameter" id="0">
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>38360</dim>
                </port>
            </output>
        </layer>
        <layer name="in2" type="Parameter" id="1">
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>19180</dim>
                </port>
            </output>
        </layer>
        <layer name="in3" type="Parameter" id="4">
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>2</dim>
                    <dim>38360</dim>
                </port>
            </output>
        </layer>
        <layer name="detectionOut" id="2" type="DetectionOutput" precision="FP32">
            <data num_classes="2" share_location="1" background_label_id="0" nms_threshold="0.450000" top_k="400" input_height="1" input_width="1" code_type="caffe.PriorBoxParameter.CENTER_SIZE" variance_encoded_in_target="0" keep_top_k="200" confidence_threshold="0.010000"/>
            <input>
                <port id="1">
                    <dim>1</dim>
                    <dim>38360</dim>
                </port>
                <port id="2">
                    <dim>1</dim>
                    <dim>19180</dim>
                </port>
                <port id="3">
                    <dim>1</dim>
                    <dim>2</dim>
                    <dim>38360</dim>
                </port>
                <port id="4">
                    <dim>1</dim>
                    <dim>38360</dim>
                </port>
                <port id="5">
                    <dim>1</dim>
                    <dim>19180</dim>
                </port>
            </input>
            <output>
                <port id="6" precision="FP32">
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>200</dim>
                    <dim>7</dim>
                </port>
            </output>
        </layer>
        <layer name="output" type="Result" id="3">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>200</dim>
                    <dim>7</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="2" to-port="1"/>
        <edge from-layer="1" from-port="0" to-layer="2" to-port="2"/>
        <edge from-layer="4" from-port="0" to-layer="2" to-port="3"/>
        <edge from-layer="2" from-port="6" to-layer="3" to-port="0"/>
    </edges>
</net>
)V0G0N";
    std::string modelV5 = R"V0G0N(
<net name="Activation" version="5" precision="FP32" batch="1">
    <layers>
        <layer name="in1" type="Input" precision="FP32" id="0">
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>38360</dim>
                </port>
            </output>
        </layer>
        <layer name="in2" type="Input" precision="FP32" id="1">
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>19180</dim>
                </port>
            </output>
        </layer>
        <layer name="in3" type="Input" precision="FP32" id="4">
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>2</dim>
                    <dim>38360</dim>
                </port>
            </output>
        </layer>
        <layer name="detectionOut" id="2" type="DetectionOutput" precision="FP32">
            <data num_classes="2" clip_after_nms="0" clip_before_nms="0" decrease_label_id="0" normalized="0" objectness_score="0.000000" share_location="1" background_label_id="0" nms_threshold="0.450000" top_k="400" input_height="1" input_width="1" code_type="caffe.PriorBoxParameter.CENTER_SIZE" variance_encoded_in_target="0" keep_top_k="200" confidence_threshold="0.010000"/>
            <input>
                <port id="1">
                    <dim>1</dim>
                    <dim>38360</dim>
                </port>
                <port id="2">
                    <dim>1</dim>
                    <dim>19180</dim>
                </port>
                <port id="3">
                    <dim>1</dim>
                    <dim>2</dim>
                    <dim>38360</dim>
                </port>
            </input>
            <output>
                <port id="6" precision="FP32">
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>200</dim>
                    <dim>7</dim>
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="2" to-port="1"/>
        <edge from-layer="1" from-port="0" to-layer="2" to-port="2"/>
        <edge from-layer="4" from-port="0" to-layer="2" to-port="3"/>
    </edges>
</net>
)V0G0N";

    IRReader reader;

    Blob::CPtr blob;
    auto nGraph = reader.read(model, blob);
    ICNNNetwork::Ptr network = convertFunctionToICNNNetwork(nGraph);

    InferenceEngine::CNNNetReader net_reader;
    net_reader.ReadNetwork(modelV5.data(), modelV5.length());

    compareICNNNetworks(network, net_reader.getNetwork());
}