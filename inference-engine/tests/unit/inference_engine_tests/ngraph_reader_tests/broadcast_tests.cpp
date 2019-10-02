// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_reader_tests.hpp"

TEST_F(NGraphReaderTests, ReadBroadcast3Network) {
    std::string model = R"V0G0N(
<net name="Activation" version="10">
    <layers>
        <layer id="0" name="data_add_5451_const" type="Const">
            <data offset="0" size="256"/>
            <output>
                <port id="1" precision="FP32">
                    <dim>64</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="Add1_/Fused_Add_/Broadcast/Shape7264_const" type="Const">
            <data offset="256" size="32"/>
            <output>
                <port id="1" precision="I64">
                    <dim>4</dim>
                </port>
            </output>
        </layer>
        <layer id="2" name="Add1_/Fused_Add_/Broadcast/Axis7265_const" type="Const">
            <data offset="288" size="24"/>
            <output>
                <port id="1" precision="I64">
                    <dim>3</dim>
                </port>
            </output>
        </layer>
        <layer id="3" name="Add1_/Fused_Add_/Broadcast/" type="Broadcast">
            <input>
                <port id="0" precision="FP32">
                    <dim>64</dim>
                </port>
                <port id="1" precision="I64">
                    <dim>4</dim>
                </port>
                <port id="2" precision="I64">
                    <dim>3</dim>
                </port>
            </input>
            <output>
                <port id="3" precision="FP32">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </output>
        </layer>
        <layer name="output" type="Result" id="4">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="1" to-layer="3" to-port="0"/>
        <edge from-layer="1" from-port="1" to-layer="3" to-port="1"/>
        <edge from-layer="2" from-port="1" to-layer="3" to-port="2"/>
        <edge from-layer="3" from-port="3" to-layer="4" to-port="0"/>
    </edges>
</net>
)V0G0N";

    IRReader reader;
    Blob::Ptr weights = make_shared_blob<uint8_t>(TensorDesc(Precision::U8, {320}, Layout::C));
    weights->allocate();
    fill_data((float *) weights->buffer(), weights->size() / sizeof(float));

    int64_t * broadcast_shape = (int64_t *)((int8_t *) weights->buffer() + 256);
    broadcast_shape[0] = 1;
    broadcast_shape[1] = 64;
    broadcast_shape[2] = 112;
    broadcast_shape[3] = 112;

    int64_t * broadcast_axis = (int64_t *)((int8_t *) weights->buffer() + 288);
    broadcast_axis[0] = 0;
    broadcast_axis[1] = 2;
    broadcast_axis[2] = 3;

    auto nGraph = reader.read(model, weights);
}

TEST_F(NGraphReaderTests, ReadBroadcast2Network) {
    std::string model = R"V0G0N(
<net name="Activation" version="10">
    <layers>
        <layer id="0" name="data_add_5451_const" type="Const">
            <data offset="0" size="256"/>
            <output>
                <port id="1" precision="FP32">
                    <dim>64</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="Add1_/Fused_Add_/Broadcast/Shape7264_const" type="Const">
            <data offset="256" size="32"/>
            <output>
                <port id="1" precision="I64">
                    <dim>4</dim>
                </port>
            </output>
        </layer>
        <layer id="3" name="Add1_/Fused_Add_/Broadcast/" type="Broadcast">
            <input>
                <port id="0" precision="FP32">
                    <dim>64</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                </port>
                <port id="1" precision="I64">
                    <dim>4</dim>
                </port>
            </input>
            <output>
                <port id="3" precision="FP32">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </output>
        </layer>
        <layer name="output" type="Result" id="4">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="1" to-layer="3" to-port="0"/>
        <edge from-layer="1" from-port="1" to-layer="3" to-port="1"/>
        <edge from-layer="3" from-port="3" to-layer="4" to-port="0"/>
    </edges>
</net>
)V0G0N";

    IRReader reader;
    Blob::Ptr weights = make_shared_blob<uint8_t>(TensorDesc(Precision::U8, {320}, Layout::C));
    weights->allocate();
    fill_data((float *) weights->buffer(), weights->size() / sizeof(float));

    int64_t * broadcast_shape = (int64_t *)((int8_t *) weights->buffer() + 256);
    broadcast_shape[0] = 1;
    broadcast_shape[1] = 64;
    broadcast_shape[2] = 112;
    broadcast_shape[3] = 112;
    try {
        auto nGraph = reader.read(model, weights);
    } catch (InferenceEngine::details::InferenceEngineException & e) {
        return;
    }
    FAIL();
}

TEST_F(NGraphReaderTests, ConvertBroadcastToTiles1) {
    std::string model = R"V0G0N(
<net name="Multiply" version="10">
    <layers>
        <layer id="14" name="data" type="Parameter">
            <output>
                <port id="1" precision="FP32">
                    <dim>112</dim>
                    <dim>1</dim>
                </port>
            </output>
        </layer>
        <layer id="15" name="broadcast1_shape" type="Const">
            <data offset="256" size="32"/>
            <output>
                <port id="1" precision="I64">
                    <dim>4</dim>
                </port>
            </output>
        </layer>
        <layer id="16" name="broadcast1_axis" type="Const">
            <data offset="288" size="32"/>
            <output>
                <port id="1" precision="I64">
                    <dim>4</dim>
                </port>
            </output>
        </layer>
        <layer id="17" name="broadcast_1" type="Broadcast">
            <input>
                <port id="0" precision="FP32">
                    <dim>112</dim>
                    <dim>1</dim>
                </port>
                <port id="1" precision="I64">
                    <dim>4</dim>
                </port>
                <port id="2" precision="I64">
                    <dim>4</dim>
                </port>
            </input>
            <output>
                <port id="3" precision="FP32">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </output>
        </layer>
        <layer id="6" name="output" type="Result">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="14" from-port="1" to-layer="17" to-port="0"/>
        <edge from-layer="15" from-port="1" to-layer="17" to-port="1"/>
        <edge from-layer="16" from-port="1" to-layer="17" to-port="2"/>
        <edge from-layer="17" from-port="3" to-layer="6" to-port="0"/>
    </edges>
</net>
)V0G0N";
    std::string modelV5 = R"V0G0N(
<net name="Convolution" version="5" precision="FP32" batch="1">
    <layers>
        <layer id="0" name="data" precision="FP32" type="Input">
            <output>
                <port id="0">
                    <dim>112</dim>
                    <dim>1</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="Constant_107" precision="I64" type="Const">
            <output>
                <port id="0">
                    <dim>4</dim>
                </port>
            </output>
            <blobs>
                <custom offset="0" size="32"/>
            </blobs>
        </layer>
        <layer id="2" name="DynReshape_108" precision="FP32" type="Reshape">
            <input>
                <port id="0">
                    <dim>112</dim>
                    <dim>1</dim>
                </port>
                <port id="1">
                    <dim>4</dim>
                </port>
            </input>
            <output>
                <port id="3">
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>112</dim>
                    <dim>1</dim>
                </port>
            </output>
        </layer>
        <layer id="3" name="broadcast_1" precision="FP32" type="Tile">
            <data axis="3" tiles="112"/>
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>112</dim>
                    <dim>1</dim>
                </port>
            </input>
            <output>
                <port id="3">
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </output>
        </layer>
        <layer id="4" name="broadcast_1_3" precision="FP32" type="Tile">
            <data axis="1" tiles="64"/>
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </input>
            <output>
                <port id="3">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="2" to-port="0"/>
        <edge from-layer="1" from-port="0" to-layer="2" to-port="1"/>
        <edge from-layer="2" from-port="3" to-layer="3" to-port="0"/>
        <edge from-layer="3" from-port="3" to-layer="4" to-port="0"/>
    </edges>
</net>
)V0G0N";
    IRReader reader;

    Blob::Ptr weights = make_shared_blob<uint8_t>(TensorDesc(Precision::U8, {6422528}, Layout::C));
    weights->allocate();
    fill_data((float *) weights->buffer(), weights->size() / sizeof(float));

    // Set shape constant for broadcast1
    int64_t * broadcast1_shape = (int64_t *)((int8_t *) weights->buffer() + 256);
    broadcast1_shape[0] = 1;
    broadcast1_shape[1] = 64;
    broadcast1_shape[2] = 112;
    broadcast1_shape[3] = 112;

    auto tWeights = std::dynamic_pointer_cast<TBlob<uint8_t>>(weights);

    auto nGraph = reader.read(model, weights);

    ICNNNetwork::Ptr network = convertFunctionToICNNNetwork(nGraph);

//    std::vector<std::shared_ptr<ngraph::Function>> g2{nGraph};
//    ngraph::pass::VisualizeTree("after.png").run_on_module(g2);
//
    InferenceEngine::CNNNetReader net_reader;
    net_reader.ReadNetwork(modelV5.data(), modelV5.length());
    net_reader.SetWeights(tWeights);

    compareICNNNetworks(network, net_reader.getNetwork());
}

TEST_F(NGraphReaderTests, ConvertBroadcastToTiles2) {
    std::string model = R"V0G0N(
<net name="Multiply" version="10">
    <layers>
        <layer id="14" name="data" type="Parameter">
            <output>
                <port id="1" precision="FP32">
                    <dim>1</dim>
                </port>
            </output>
        </layer>
        <layer id="15" name="broadcast1_shape" type="Const">
            <data offset="256" size="32"/>
            <output>
                <port id="1" precision="I64">
                    <dim>4</dim>
                </port>
            </output>
        </layer>
        <layer id="16" name="broadcast1_axis" type="Const">
            <data offset="288" size="32"/>
            <output>
                <port id="1" precision="I64">
                    <dim>4</dim>
                </port>
            </output>
        </layer>
        <layer id="17" name="broadcast_1" type="Broadcast">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                </port>
                <port id="1" precision="I64">
                    <dim>4</dim>
                </port>
                <port id="2" precision="I64">
                    <dim>4</dim>
                </port>
            </input>
            <output>
                <port id="3" precision="FP32">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </output>
        </layer>
        <layer id="6" name="output" type="Result">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="14" from-port="1" to-layer="17" to-port="0"/>
        <edge from-layer="15" from-port="1" to-layer="17" to-port="1"/>
        <edge from-layer="16" from-port="1" to-layer="17" to-port="2"/>
        <edge from-layer="17" from-port="3" to-layer="6" to-port="0"/>
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
                </port>
            </output>
        </layer>
        <layer id="1" name="Constant_107" precision="I64" type="Const">
            <output>
                <port id="0">
                    <dim>4</dim>
                </port>
            </output>
            <blobs>
                <custom offset="0" size="32"/>
            </blobs>
        </layer>
        <layer id="2" name="DynReshape_108" precision="FP32" type="Reshape">
            <input>
                <port id="0">
                    <dim>1</dim>
                </port>
                <port id="1">
                    <dim>4</dim>
                </port>
            </input>
            <output>
                <port id="3">
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                </port>
            </output>
        </layer>
        <layer id="3" name="broadcast_1" precision="FP32" type="Tile">
            <data axis="3" tiles="112"/>
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                </port>
            </input>
            <output>
                <port id="3">
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>112</dim>
                </port>
            </output>
        </layer>
        <layer id="4" name="broadcast_1_3" precision="FP32" type="Tile">
            <data axis="2" tiles="112"/>
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>112</dim>
                </port>
            </input>
            <output>
                <port id="3">
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </output>
        </layer>
        <layer id="5" name="broadcast_1_3_2" precision="FP32" type="Tile">
            <data axis="1" tiles="64"/>
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </input>
            <output>
                <port id="3">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="2" to-port="0"/>
        <edge from-layer="1" from-port="0" to-layer="2" to-port="1"/>
        <edge from-layer="2" from-port="3" to-layer="3" to-port="0"/>
        <edge from-layer="3" from-port="3" to-layer="4" to-port="0"/>
        <edge from-layer="4" from-port="3" to-layer="5" to-port="0"/>
    </edges>
</net>
)V0G0N";
    IRReader reader;

    Blob::Ptr weights = make_shared_blob<uint8_t>(TensorDesc(Precision::U8, {6422528}, Layout::C));
    weights->allocate();
    fill_data((float *) weights->buffer(), weights->size() / sizeof(float));

    // Set shape constant for broadcast1
    int64_t * broadcast1_shape = (int64_t *)((int8_t *) weights->buffer() + 256);
    broadcast1_shape[0] = 1;
    broadcast1_shape[1] = 64;
    broadcast1_shape[2] = 112;
    broadcast1_shape[3] = 112;

    auto tWeights = std::dynamic_pointer_cast<TBlob<uint8_t>>(weights);

    auto nGraph = reader.read(model, weights);

    ICNNNetwork::Ptr network = convertFunctionToICNNNetwork(nGraph);

    InferenceEngine::CNNNetReader net_reader;
    net_reader.ReadNetwork(modelV5.data(), modelV5.length());
    net_reader.SetWeights(tWeights);

    compareICNNNetworks(network, net_reader.getNetwork());
}

TEST_F(NGraphReaderTests, ConvertBroadcastToTiles3) {
    std::string model = R"V0G0N(
<net name="Multiply" version="10">
    <layers>
        <layer id="14" name="data" type="Parameter">
            <output>
                <port id="1" precision="FP32">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>1</dim>
                    <dim>112</dim>
                </port>
            </output>
        </layer>
        <layer id="15" name="broadcast1_shape" type="Const">
            <data offset="256" size="32"/>
            <output>
                <port id="1" precision="I64">
                    <dim>4</dim>
                </port>
            </output>
        </layer>
        <layer id="16" name="broadcast1_axis" type="Const">
            <data offset="288" size="32"/>
            <output>
                <port id="1" precision="I64">
                    <dim>4</dim>
                </port>
            </output>
        </layer>
        <layer id="17" name="broadcast_1" type="Broadcast">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>1</dim>
                    <dim>112</dim>
                </port>
                <port id="1" precision="I64">
                    <dim>4</dim>
                </port>
                <port id="2" precision="I64">
                    <dim>4</dim>
                </port>
            </input>
            <output>
                <port id="3" precision="FP32">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </output>
        </layer>
        <layer id="6" name="output" type="Result">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="14" from-port="1" to-layer="17" to-port="0"/>
        <edge from-layer="15" from-port="1" to-layer="17" to-port="1"/>
        <edge from-layer="16" from-port="1" to-layer="17" to-port="2"/>
        <edge from-layer="17" from-port="3" to-layer="6" to-port="0"/>
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
                    <dim>64</dim>
                    <dim>1</dim>
                    <dim>112</dim>
                </port>
            </output>
        </layer>
        <layer id="3" name="broadcast_1" precision="FP32" type="Tile">
            <data axis="2" tiles="112"/>
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>1</dim>
                    <dim>112</dim>
                </port>
            </input>
            <output>
                <port id="3">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="3" to-port="0"/>
    </edges>
</net>
)V0G0N";
    IRReader reader;

    Blob::Ptr weights = make_shared_blob<uint8_t>(TensorDesc(Precision::U8, {6422528}, Layout::C));
    weights->allocate();
    fill_data((float *) weights->buffer(), weights->size() / sizeof(float));

    // Set shape constant for broadcast1
    int64_t * broadcast1_shape = (int64_t *)((int8_t *) weights->buffer() + 256);
    broadcast1_shape[0] = 1;
    broadcast1_shape[1] = 64;
    broadcast1_shape[2] = 112;
    broadcast1_shape[3] = 112;

    auto tWeights = std::dynamic_pointer_cast<TBlob<uint8_t>>(weights);

    auto nGraph = reader.read(model, weights);

    ICNNNetwork::Ptr network = convertFunctionToICNNNetwork(nGraph);

//    std::vector<std::shared_ptr<ngraph::Function>> g2{nGraph};
//    ngraph::pass::VisualizeTree("after.png").run_on_module(g2);

    InferenceEngine::CNNNetReader net_reader;
    net_reader.ReadNetwork(modelV5.data(), modelV5.length());
    net_reader.SetWeights(tWeights);

    compareICNNNetworks(network, net_reader.getNetwork());
}