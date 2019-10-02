// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_reader_tests.hpp"

TEST_F(NGraphReaderTests, ReadSplitNetwork) {
    std::string model = R"V0G0N(
<net name="Activation" version="10">
    <layers>
        <layer name="in1" type="Parameter" id="0">
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>6</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
        <layer name="split" id="2" type="Split">
            <data axis="1"/>
            <input>
                <port id="1" precision="FP32">
                    <dim>1</dim>
                    <dim>6</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </input>
            <output>
                <port id="2" precision="FP32">
                    <dim>1</dim>
                    <dim>2</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
                <port id="3" precision="FP32">
                    <dim>1</dim>
                    <dim>4</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
        <layer name="output1" type="Result" id="1">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>4</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </input>
        </layer>
        <layer name="output" type="Result" id="3">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>2</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="2" to-port="1"/>
        <edge from-layer="2" from-port="2" to-layer="3" to-port="0"/>
        <edge from-layer="2" from-port="3" to-layer="1" to-port="0"/>
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
                    <dim>6</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
        <layer name="split" id="2" type="Split" precision="FP32">
            <data axis="1"/>
            <input>
                <port id="1">
                    <dim>1</dim>
                    <dim>6</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </input>
            <output>
                <port id="2">
                    <dim>1</dim>
                    <dim>2</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
                <port id="3">
                    <dim>1</dim>
                    <dim>4</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="2" to-port="1"/>
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