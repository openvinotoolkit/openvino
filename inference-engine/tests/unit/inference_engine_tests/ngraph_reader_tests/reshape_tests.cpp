// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_reader_tests.hpp"

TEST_F(NGraphReaderTests, ReadReshapeNetwork) {
    std::string model = R"V0G0N(
<net name="Reshape" version="10">
    <layers>
        <layer id="0" name="data" type="Parameter">
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>2048</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="const1" type="Const">
            <data offset="0" size="16"/>
            <output>
                <port id="1" precision="I64">
                    <dim>2</dim>
                </port>
            </output>
        </layer>
        <layer id="3" name="reshape1" type="Reshape">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>2048</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                </port>
                <port id="1" precision="I64">
                    <dim>2</dim>
                </port>
            </input>
            <output>
                <port id="3" precision="FP32">
                    <dim>1</dim>
                    <dim>2048</dim>
                </port>
            </output>
        </layer>
        <layer name="output" type="Result" id="2">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>2048</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="3" to-port="0"/>
        <edge from-layer="1" from-port="1" to-layer="3" to-port="1"/>
        <edge from-layer="3" from-port="3" to-layer="2" to-port="0"/>
    </edges>
</net>
)V0G0N";
    std::string modelV5 = R"V0G0N(
<net name="Convolution" version="5" precision="FP32" batch="1">
    <layers>
        <layer id="0" name="data" precision="FP32" type="Input">
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>2048</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="const1" precision="I64" type="Const">
            <output>
                <port id="0">
                    <dim>2</dim>
                </port>
            </output>
            <blobs>
                <custom offset="0" size="16"/>
            </blobs>
        </layer>
        <layer id="3" name="reshape1" precision="FP32" type="Reshape">
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>2048</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                </port>
                <port id="1">
                    <dim>2</dim>
                </port>
            </input>
            <output>
                <port id="3">
                    <dim>1</dim>
                    <dim>2048</dim>
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="3" to-port="0"/>
        <edge from-layer="1" from-port="0" to-layer="3" to-port="1"/>
    </edges>
</net>
)V0G0N";
    IRReader reader;

    Blob::Ptr weights = make_shared_blob<uint8_t>(TensorDesc(Precision::U8, {16}, Layout::C));
    weights->allocate();
    auto *data = weights->buffer().as<int64_t *>();
    data[0] = 1;
    data[1] = 2048;

    auto tWeights = std::dynamic_pointer_cast<TBlob<uint8_t>>(weights);

    auto nGraph = reader.read(model, weights);
    ICNNNetwork::Ptr network = convertFunctionToICNNNetwork(nGraph);

    InferenceEngine::CNNNetReader net_reader;
    net_reader.ReadNetwork(modelV5.data(), modelV5.length());
    net_reader.SetWeights(tWeights);

    compareICNNNetworks(network, net_reader.getNetwork());
}