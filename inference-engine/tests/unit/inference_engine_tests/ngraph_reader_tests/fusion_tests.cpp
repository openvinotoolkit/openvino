// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_reader_tests.hpp"

TEST_F(NGraphReaderTests, ConvBiasFusion) {
    std::string model = R"V0G0N(
<net name="ConvBias" version="10">
    <layers>
        <layer id="0" name="data" type="Parameter">
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>227</dim>
                    <dim>227</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="conv_weights" type="Const">
            <data offset="0" size="139392"/>
            <output>
                <port id="1" precision="FP32">
                    <dim>96</dim>
                    <dim>3</dim>
                    <dim>11</dim>
                    <dim>11</dim>
                </port>
            </output>
        </layer>
        <layer id="2" name="conv" type="Convolution">
            <data dilations="1,1" group="1" kernel="11,11" output="96" pads_begin="0,0" pads_end="0,0" strides="4,4"/>
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>227</dim>
                    <dim>227</dim>
                </port>
                <port id="1" precision="FP32">
                    <dim>96</dim>
                    <dim>3</dim>
                    <dim>11</dim>
                    <dim>11</dim>
                </port>
            </input>
            <output>
                <port id="3" precision="FP32">
                    <dim>1</dim>
                    <dim>96</dim>
                    <dim>55</dim>
                    <dim>55</dim>
                </port>
            </output>
        </layer>
        <layer id="4" name="data_add_5451_const" type="Const">
            <data offset="139392" size="384"/>
            <output>
                <port id="1" precision="FP32">
                    <dim>96</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                </port>
            </output>
        </layer>
        <layer id="5" name="Add1_/Fused_Add_/Broadcast/Shape7264_const" type="Const">
            <data offset="139776" size="32"/>
            <output>
                <port id="1" precision="I64">
                    <dim>4</dim>
                </port>
            </output>
        </layer>
        <layer id="6" name="Add1_/Fused_Add_/Broadcast/Axis7265_const" type="Const">
            <data offset="139808" size="32"/>
            <output>
                <port id="1" precision="I64">
                    <dim>4</dim>
                </port>
            </output>
        </layer>
        <layer id="7" name="Add1_/Fused_Add_/Broadcast/" type="Broadcast">
            <input>
                <port id="0" precision="FP32">
                    <dim>96</dim>
                    <dim>1</dim>
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
                    <dim>96</dim>
                    <dim>55</dim>
                    <dim>55</dim>
                </port>
            </output>
        </layer>
        <layer id="3" name="add" type="Add">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>96</dim>
                    <dim>55</dim>
                    <dim>55</dim>
                </port>
                <port id="1" precision="FP32">
                    <dim>1</dim>
                    <dim>96</dim>
                    <dim>55</dim>
                    <dim>55</dim>
                </port>
            </input>
            <output>
                <port id="3" precision="FP32">
                    <dim>1</dim>
                    <dim>96</dim>
                    <dim>55</dim>
                    <dim>55</dim>
                </port>
            </output>
        </layer>
        <layer  id="8" name="output" type="Result">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>96</dim>
                    <dim>55</dim>
                    <dim>55</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="2" to-port="0"/>
        <edge from-layer="1" from-port="1" to-layer="2" to-port="1"/>
        <edge from-layer="4" from-port="1" to-layer="7" to-port="0"/>
        <edge from-layer="5" from-port="1" to-layer="7" to-port="1"/>
        <edge from-layer="6" from-port="1" to-layer="7" to-port="2"/>
        <edge from-layer="7" from-port="3" to-layer="3" to-port="1"/>
        <edge from-layer="2" from-port="3" to-layer="3" to-port="0"/>
        <edge from-layer="3" from-port="3" to-layer="8" to-port="0"/>
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
                    <dim>3</dim>
                    <dim>227</dim>
                    <dim>227</dim>
                </port>
            </output>
        </layer>
        <layer id="3" name="add" precision="FP32" type="Convolution">
            <data dilations="1,1" group="1" kernel="11,11" output="96" pads_begin="0,0" pads_end="0,0" strides="4,4"/>
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>227</dim>
                    <dim>227</dim>
                </port>
            </input>
            <output>
                <port id="3">
                    <dim>1</dim>
                    <dim>96</dim>
                    <dim>55</dim>
                    <dim>55</dim>
                </port>
            </output>
            <weights offset="0" size="139392" />
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="3" to-port="0"/>
    </edges>
</net>
)V0G0N";

    IRReader reader;
    Blob::Ptr weights = make_shared_blob<uint8_t>(TensorDesc(Precision::U8, {139840}, Layout::C));
    weights->allocate();
    fill_data((float *) weights->buffer(), weights->size() / sizeof(float));

    int64_t * broadcast_shape = (int64_t *)((int8_t *) weights->buffer() + 139776);
    broadcast_shape[0] = 1;
    broadcast_shape[1] = 96;
    broadcast_shape[2] = 55;
    broadcast_shape[3] = 55;

    auto nGraph = reader.read(model, weights);
    auto tWeights = std::dynamic_pointer_cast<TBlob<uint8_t>>(weights);

    ICNNNetwork::Ptr network = convertFunctionToICNNNetwork(nGraph);

    InferenceEngine::CNNNetReader net_reader;
    net_reader.ReadNetwork(modelV5.data(), modelV5.length());
    net_reader.SetWeights(tWeights);

    compareICNNNetworks(network, net_reader.getNetwork());
}

TEST_F(NGraphReaderTests, ConvBiasFusionFP16) {
    std::string model = R"V0G0N(
<net name="ConvBias" version="10">
    <layers>
        <layer id="0" name="data" type="Parameter">
            <output>
                <port id="0" precision="FP16">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>227</dim>
                    <dim>227</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="conv_weights" type="Const">
            <data offset="0" size="69696"/>
            <output>
                <port id="1" precision="FP16">
                    <dim>96</dim>
                    <dim>3</dim>
                    <dim>11</dim>
                    <dim>11</dim>
                </port>
            </output>
        </layer>
        <layer id="2" name="conv" type="Convolution">
            <data dilations="1,1" group="1" kernel="11,11" output="96" pads_begin="0,0" pads_end="0,0" strides="4,4"/>
            <input>
                <port id="0" precision="FP16">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>227</dim>
                    <dim>227</dim>
                </port>
                <port id="1" precision="FP16">
                    <dim>96</dim>
                    <dim>3</dim>
                    <dim>11</dim>
                    <dim>11</dim>
                </port>
            </input>
            <output>
                <port id="3" precision="FP16">
                    <dim>1</dim>
                    <dim>96</dim>
                    <dim>55</dim>
                    <dim>55</dim>
                </port>
            </output>
        </layer>
        <layer id="4" name="data_add_5451_const" type="Const">
            <data offset="139392" size="384"/>
            <output>
                <port id="1" precision="FP16">
                    <dim>96</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                </port>
            </output>
        </layer>
        <layer id="5" name="Add1_/Fused_Add_/Broadcast/Shape7264_const" type="Const">
            <data offset="139776" size="32"/>
            <output>
                <port id="1" precision="I64">
                    <dim>4</dim>
                </port>
            </output>
        </layer>
        <layer id="6" name="Add1_/Fused_Add_/Broadcast/Axis7265_const" type="Const">
            <data offset="139808" size="32"/>
            <output>
                <port id="1" precision="I64">
                    <dim>4</dim>
                </port>
            </output>
        </layer>
        <layer id="7" name="Add1_/Fused_Add_/Broadcast/" type="Broadcast">
            <input>
                <port id="0" precision="FP16">
                    <dim>96</dim>
                    <dim>1</dim>
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
                <port id="3" precision="FP16">
                    <dim>1</dim>
                    <dim>96</dim>
                    <dim>55</dim>
                    <dim>55</dim>
                </port>
            </output>
        </layer>
        <layer id="3" name="add" type="Add">
            <input>
                <port id="0" precision="FP16">
                    <dim>1</dim>
                    <dim>96</dim>
                    <dim>55</dim>
                    <dim>55</dim>
                </port>
                <port id="1" precision="FP16">
                    <dim>1</dim>
                    <dim>96</dim>
                    <dim>55</dim>
                    <dim>55</dim>
                </port>
            </input>
            <output>
                <port id="3" precision="FP16">
                    <dim>1</dim>
                    <dim>96</dim>
                    <dim>55</dim>
                    <dim>55</dim>
                </port>
            </output>
        </layer>
        <layer  id="8" name="output" type="Result">
            <input>
                <port id="0" precision="FP16">
                    <dim>1</dim>
                    <dim>96</dim>
                    <dim>55</dim>
                    <dim>55</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="2" to-port="0"/>
        <edge from-layer="1" from-port="1" to-layer="2" to-port="1"/>
        <edge from-layer="4" from-port="1" to-layer="7" to-port="0"/>
        <edge from-layer="5" from-port="1" to-layer="7" to-port="1"/>
        <edge from-layer="6" from-port="1" to-layer="7" to-port="2"/>
        <edge from-layer="7" from-port="3" to-layer="3" to-port="1"/>
        <edge from-layer="2" from-port="3" to-layer="3" to-port="0"/>
        <edge from-layer="3" from-port="3" to-layer="8" to-port="0"/>
    </edges>
</net>
)V0G0N";
    std::string modelV5 = R"V0G0N(
<net name="Convolution" version="5" precision="FP16" batch="1">
    <layers>
        <layer id="0" name="data" precision="FP16" type="Input">
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>227</dim>
                    <dim>227</dim>
                </port>
            </output>
        </layer>
        <layer id="3" name="add" precision="FP16" type="Convolution">
            <data dilations="1,1" group="1" kernel="11,11" output="96" pads_begin="0,0" pads_end="0,0" strides="4,4"/>
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>227</dim>
                    <dim>227</dim>
                </port>
            </input>
            <output>
                <port id="3">
                    <dim>1</dim>
                    <dim>96</dim>
                    <dim>55</dim>
                    <dim>55</dim>
                </port>
            </output>
            <weights offset="0" size="69696" />
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="3" to-port="0"/>
    </edges>
</net>
)V0G0N";

    IRReader reader;
    Blob::Ptr weights = make_shared_blob<uint8_t>(TensorDesc(Precision::U8, {139840}, Layout::C));
    weights->allocate();
    fill_data((float *) weights->buffer(), weights->size() / sizeof(float));

    int64_t * broadcast_shape = (int64_t *)((int8_t *) weights->buffer() + 139776);
    broadcast_shape[0] = 1;
    broadcast_shape[1] = 96;
    broadcast_shape[2] = 55;
    broadcast_shape[3] = 55;

    auto nGraph = reader.read(model, weights);
    auto tWeights = std::dynamic_pointer_cast<TBlob<uint8_t>>(weights);

    ICNNNetwork::Ptr network = convertFunctionToICNNNetwork(nGraph);

    InferenceEngine::CNNNetReader net_reader;
    net_reader.ReadNetwork(modelV5.data(), modelV5.length());
    net_reader.SetWeights(tWeights);

    compareICNNNetworks(network, net_reader.getNetwork());
}

TEST_F(NGraphReaderTests, MatMulBiasFusionNoBroadcast) {
    std::string model = R"V0G0N(
<net name="MatMulBias" version="10">
    <layers>
        <layer id="0" name="data" type="Parameter">
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>2048</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="weights" type="Const">
            <data offset="0" size="8192000"/>
            <output>
                <port id="1" precision="FP32">
                    <dim>2048</dim>
                    <dim>1000</dim>
                </port>
            </output>
        </layer>
        <layer id="2" name="fc" type="MatMul">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>2048</dim>
                </port>
                <port id="1" precision="FP32">
                    <dim>2048</dim>
                    <dim>1000</dim>
                </port>
            </input>
            <output>
                <port id="2" precision="FP32">
                    <dim>1</dim>
                    <dim>1000</dim>
                </port>
            </output>
        </layer>
        <layer id="3" name="b_input" type="Const">
            <data offset="8192000" size="4000"/>
            <output>
                <port id="1" precision="FP32">
                    <dim>1</dim>
                    <dim>1000</dim>
                </port>
            </output>
        </layer>
        <layer id="7" name="add" type="Add">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>1000</dim>
                </port>
                <port id="1" precision="FP32">
                    <dim>1</dim>
                    <dim>1000</dim>
                </port>
            </input>
            <output>
                <port id="2" precision="FP32">
                    <dim>1</dim>
                    <dim>1000</dim>
                </port>
            </output>
        </layer>
        <layer  id="8" name="output" type="Result">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>1000</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="2" to-port="0"/>
        <edge from-layer="1" from-port="1" to-layer="2" to-port="1"/>
        <edge from-layer="3" from-port="1" to-layer="7" to-port="1"/>
        <edge from-layer="2" from-port="2" to-layer="7" to-port="0"/>
        <edge from-layer="7" from-port="2" to-layer="8" to-port="0"/>
    </edges>
</net>
)V0G0N";
    std::string modelV5 = R"V0G0N(
<net name="MatMulBias" version="5" precision="FP32" batch="1">
    <layers>
        <layer id="0" name="data" precision="FP32" type="Input">
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>2048</dim>
                </port>
            </output>
        </layer>
        <layer id="3" name="add" precision="FP32" type="FullyConnected">
            <data alpha="0" beta="0" out-size="1000"/>
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>2048</dim>
                </port>
            </input>
            <output>
                <port id="3">
                    <dim>1</dim>
                    <dim>1000</dim>
                </port>
            </output>
            <weights offset="0" size="8192000"/>
            <biases offset="8192000" size="4000"/>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="3" to-port="0"/>
    </edges>
</net>
)V0G0N";

    IRReader reader;
    Blob::Ptr weights = make_shared_blob<uint8_t>(TensorDesc(Precision::U8, {8196024}, Layout::C));
    weights->allocate();
    fill_data((float *) weights->buffer(), weights->size() / sizeof(float));
    auto nGraph = reader.read(model, weights);
    auto tWeights = std::dynamic_pointer_cast<TBlob<uint8_t>>(weights);

    ICNNNetwork::Ptr network = convertFunctionToICNNNetwork(nGraph);

    InferenceEngine::CNNNetReader net_reader;
    net_reader.ReadNetwork(modelV5.data(), modelV5.length());
    net_reader.SetWeights(tWeights);

    compareICNNNetworks(network, net_reader.getNetwork());
}

TEST_F(NGraphReaderTests, DISABLED_MatMulBiasFusion) {
    std::string model = R"V0G0N(
<net name="MatMulBias" version="10">
    <layers>
        <layer id="0" name="data" type="Parameter">
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>2048</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="weights" type="Const">
            <data offset="0" size="8192000"/>
            <output>
                <port id="1" precision="FP32">
                    <dim>2048</dim>
                    <dim>1000</dim>
                </port>
            </output>
        </layer>
        <layer id="2" name="fc" type="MatMul">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>2048</dim>
                </port>
                <port id="1" precision="FP32">
                    <dim>2048</dim>
                    <dim>1000</dim>
                </port>
            </input>
            <output>
                <port id="2" precision="FP32">
                    <dim>1</dim>
                    <dim>1000</dim>
                </port>
            </output>
        </layer>
        <layer id="3" name="b_input" type="Const">
            <data offset="8192000" size="4000"/>
            <output>
                <port id="1" precision="FP32">
                    <dim>1000</dim>
                </port>
            </output>
        </layer>
        <layer id="4" name="b_shape" type="Const">
            <data offset="8196000" size="8"/>
            <output>
                <port id="1" precision="I64">
                    <dim>2</dim>
                </port>
            </output>
        </layer>
        <layer id="5" name="b_axis" type="Const">
            <data offset="8196008" size="16"/>
            <output>
                <port id="1" precision="I64">
                    <dim>4</dim>
                </port>
            </output>
        </layer>
        <layer id="6" name="broadcast" type="Broadcast">
            <input>
                <port id="0" precision="FP32">
                    <dim>1000</dim>
                </port>
                <port id="1" precision="I64">
                    <dim>2</dim>
                </port>
                <port id="2" precision="I64">
                    <dim>4</dim>
                </port>
            </input>
            <output>
                <port id="3" precision="FP32">
                    <dim>1</dim>
                    <dim>1000</dim>
                </port>
            </output>
        </layer>
        <layer id="7" name="add" type="Add">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>1000</dim>
                </port>
                <port id="1" precision="FP32">
                    <dim>1</dim>
                    <dim>1000</dim>
                </port>
            </input>
            <output>
                <port id="2" precision="FP32">
                    <dim>1</dim>
                    <dim>1000</dim>
                </port>
            </output>
        </layer>
        <layer  id="8" name="output" type="Result">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>1000</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="2" to-port="0"/>
        <edge from-layer="1" from-port="1" to-layer="2" to-port="1"/>
        <edge from-layer="3" from-port="1" to-layer="6" to-port="0"/>
        <edge from-layer="4" from-port="1" to-layer="6" to-port="1"/>
        <edge from-layer="5" from-port="1" to-layer="6" to-port="2"/>
        <edge from-layer="6" from-port="3" to-layer="7" to-port="1"/>
        <edge from-layer="2" from-port="2" to-layer="7" to-port="0"/>
        <edge from-layer="7" from-port="2" to-layer="8" to-port="0"/>
    </edges>
</net>
)V0G0N";
    std::string modelV5 = R"V0G0N(
<net name="MatMulBias" version="5" precision="FP32" batch="1">
    <layers>
        <layer id="0" name="data" precision="FP32" type="Input">
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>2048</dim>
                </port>
            </output>
        </layer>
        <layer id="3" name="fc" precision="FP32" type="FullyConnected">
            <data alpha="0" beta="0" out-size="1000"/>
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>2048</dim>
                </port>
            </input>
            <output>
                <port id="3">
                    <dim>1</dim>
                    <dim>1000</dim>
                </port>
            </output>
            <weights offset="0" size="8192000"/>
            <biases offset="8192000" size="4000"/>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="3" to-port="0"/>
    </edges>
</net>
)V0G0N";

    IRReader reader;
    Blob::Ptr weights = make_shared_blob<uint8_t>(TensorDesc(Precision::U8, {8196024}, Layout::C));
    weights->allocate();
    fill_data((float *) weights->buffer(), weights->size() / sizeof(float));
    auto nGraph = reader.read(model, weights);
    auto tWeights = std::dynamic_pointer_cast<TBlob<uint8_t>>(weights);

    ICNNNetwork::Ptr network = convertFunctionToICNNNetwork(nGraph);

    InferenceEngine::CNNNetReader net_reader;
    net_reader.ReadNetwork(modelV5.data(), modelV5.length());
    net_reader.SetWeights(tWeights);

    compareICNNNetworks(network, net_reader.getNetwork());
}