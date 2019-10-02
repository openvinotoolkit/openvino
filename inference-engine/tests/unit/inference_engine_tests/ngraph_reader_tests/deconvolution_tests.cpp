// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_reader_tests.hpp"

TEST_F(NGraphReaderTests, ReadDeconvolution3DNetwork) {
    std::string model = R"V0G0N(
<net name="Convolution" version="10">
    <layers>
        <layer id="0" name="data" type="Parameter">
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>512</dim>
                    <dim>4</dim>
                    <dim>4</dim>
                    <dim>4</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="embedded_input__const" type="Const">
            <data offset="0" size="33554432"/>
            <output>
                <port id="1" precision="FP32">
                    <dim>512</dim>
                    <dim>256</dim>
                    <dim>4</dim>
                    <dim>4</dim>
                    <dim>4</dim>
                </port>
            </output>
        </layer>
        <layer id="3" name="deconv1" precision="FP32" type="Deconvolution">
            <data auto_pad="same_upper" kernel="4,4,4" output="256" pads_begin="1,1,1" pads_end="1,1,1" strides="2,2,2"/>
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>512</dim>
                    <dim>4</dim>
                    <dim>4</dim>
                    <dim>4</dim>
                </port>
                <port id="1">
                    <dim>512</dim>
                    <dim>256</dim>
                    <dim>4</dim>
                    <dim>4</dim>
                    <dim>4</dim>
                </port>
            </input>
            <output>
                <port id="2">
                    <dim>1</dim>
                    <dim>256</dim>
                    <dim>8</dim>
                    <dim>8</dim>
                    <dim>8</dim>
                </port>
            </output>
        </layer>
        <layer name="output" type="Result" id="2">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>256</dim>
                    <dim>8</dim>
                    <dim>8</dim>
                    <dim>8</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="3" to-port="0"/>
        <edge from-layer="1" from-port="1" to-layer="3" to-port="1"/>
        <edge from-layer="3" from-port="2" to-layer="2" to-port="0"/>
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
                    <dim>512</dim>
                    <dim>4</dim>
                    <dim>4</dim>
                    <dim>4</dim>
                </port>
            </output>
        </layer>
        <layer id="3" name="deconv1" precision="FP32" type="Deconvolution">
            <data dilations="1,1,1" auto_pad="same_upper" kernel="4,4,4" output="256" pads_begin="1,1,1" pads_end="1,1,1" strides="2,2,2"/>
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>512</dim>
                    <dim>4</dim>
                    <dim>4</dim>
                    <dim>4</dim>
                </port>
            </input>
            <output>
                <port id="3">
                    <dim>1</dim>
                    <dim>256</dim>
                    <dim>8</dim>
                    <dim>8</dim>
                    <dim>8</dim>
                </port>
            </output>
            <weights offset="0" size="33554432" />
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="3" to-port="0"/>
    </edges>
</net>
)V0G0N";
    IRReader reader;

    Blob::Ptr weights = make_shared_blob<uint8_t>(TensorDesc(Precision::U8, {33554432}, Layout::C));
    weights->allocate();
    fill_data((float *) weights->buffer(), weights->size() / sizeof(float));

    auto tWeights = std::dynamic_pointer_cast<TBlob<uint8_t>>(weights);

    auto nGraph = reader.read(model, weights);
    ICNNNetwork::Ptr network = convertFunctionToICNNNetwork(nGraph);

    InferenceEngine::CNNNetReader net_reader;
    net_reader.ReadNetwork(modelV5.data(), modelV5.length());
    net_reader.SetWeights(tWeights);

    compareICNNNetworks(network, net_reader.getNetwork());
}

TEST_F(NGraphReaderTests, ReadDeconvolution2DNetwork) {
    std::string model = R"V0G0N(
<net name="Convolution" version="10">
    <layers>
        <layer id="0" name="data" type="Parameter">
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>512</dim>
                    <dim>4</dim>
                    <dim>4</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="embedded_input__const" type="Const">
            <data offset="0" size="8388608"/>
            <output>
                <port id="1" precision="FP32">
                    <dim>512</dim>
                    <dim>256</dim>
                    <dim>4</dim>
                    <dim>4</dim>
                </port>
            </output>
        </layer>
        <layer id="3" name="deconv1" precision="FP32" type="Deconvolution">
            <data auto_pad="same_upper" kernel="4,4" output="256" pads_begin="1,1" pads_end="1,1" strides="2,2"/>
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>512</dim>
                    <dim>4</dim>
                    <dim>4</dim>
                </port>
                <port id="1">
                    <dim>512</dim>
                    <dim>256</dim>
                    <dim>4</dim>
                    <dim>4</dim>
                </port>
            </input>
            <output>
                <port id="2">
                    <dim>1</dim>
                    <dim>256</dim>
                    <dim>8</dim>
                    <dim>8</dim>
                </port>
            </output>
        </layer>
        <layer name="output" type="Result" id="2">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>256</dim>
                    <dim>8</dim>
                    <dim>8</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="3" to-port="0"/>
        <edge from-layer="1" from-port="1" to-layer="3" to-port="1"/>
        <edge from-layer="3" from-port="2" to-layer="2" to-port="0"/>
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
                    <dim>512</dim>
                    <dim>4</dim>
                    <dim>4</dim>
                </port>
            </output>
        </layer>
        <layer id="3" name="deconv1" precision="FP32" type="Deconvolution">
            <data dilations="1,1" auto_pad="same_upper" kernel="4,4" output="256" pads_begin="1,1" pads_end="1,1" strides="2,2"/>
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>512</dim>
                    <dim>4</dim>
                    <dim>4</dim>
                </port>
            </input>
            <output>
                <port id="3">
                    <dim>1</dim>
                    <dim>256</dim>
                    <dim>8</dim>
                    <dim>8</dim>
                </port>
            </output>
            <weights offset="0" size="8388608" />
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="3" to-port="0"/>
    </edges>
</net>
)V0G0N";
    IRReader reader;

    Blob::Ptr weights = make_shared_blob<uint8_t>(TensorDesc(Precision::U8, {33554432}, Layout::C));
    weights->allocate();
    fill_data((float *) weights->buffer(), weights->size() / sizeof(float));

    auto tWeights = std::dynamic_pointer_cast<TBlob<uint8_t>>(weights);

    auto nGraph = reader.read(model, weights);
    ICNNNetwork::Ptr network = convertFunctionToICNNNetwork(nGraph);

    InferenceEngine::CNNNetReader net_reader;
    net_reader.ReadNetwork(modelV5.data(), modelV5.length());
    net_reader.SetWeights(tWeights);

    compareICNNNetworks(network, net_reader.getNetwork());
}