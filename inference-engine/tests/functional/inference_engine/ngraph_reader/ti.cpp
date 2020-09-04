// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include "ngraph_reader_tests.hpp"
TEST_F(NGraphReaderTests, ReadTensorIteratorNetwork_opset1) {
    std::string model_v10 = R"V0G0N(
    <net name="Transpose" version="10">
        <layers>
            <layer id="0" name="data1" type="Parameter" version="opset1">
                <data element_type="f32" shape="1,25,512"/>
                <output>
                    <port id="0" precision="FP32">
                        <dim>1</dim>
                        <dim>25</dim>
                        <dim>512</dim>
                    </port>
                </output>
            </layer>
            <layer id="1" name="data2" type="Parameter" version="opset1">
                <data element_type="f32" shape="1,256"/>
                <output>
                    <port id="0" precision="FP32">
                        <dim>1</dim>
                        <dim>256</dim>
                    </port>
                </output>
            </layer>
            <layer id="2" name="data3" type="Parameter" version="opset1">
                <data element_type="f32" shape="1,256"/>
                <output>
                    <port id="0" precision="FP32">
                        <dim>1</dim>
                        <dim>256</dim>
                    </port>
                </output>
            </layer>
            <layer id="3" name="TensorIterator" type="TensorIterator" version="opset1">
                <input>
                    <port id="0">
                        <dim>1</dim>
                        <dim>25</dim>
                        <dim>512</dim>
                    </port>
                    <port id="1">
                        <dim>1</dim>
                        <dim>256</dim>
                    </port>
                    <port id="2">
                        <dim>1</dim>
                        <dim>256</dim>
                    </port>
                </input>
                <output>
                    <port id="3" precision="FP32">
                        <dim>1</dim>
                        <dim>25</dim>
                        <dim>256</dim>
                    </port>
                </output>
                <port_map>
                    <input axis="1" external_port_id="0" internal_layer_id="0" start="0"/>
                    <input external_port_id="1" internal_layer_id="3"/>
                    <input external_port_id="2" internal_layer_id="4"/>
                    <output axis="1" external_port_id="3" internal_layer_id="13"/>
                </port_map>
                <back_edges>
                    <edge from-layer="9" to-layer="4"/>
                    <edge from-layer="10" to-layer="3"/>
                </back_edges>
                <body>
                    <layers>
                        <layer id="0" name="32" type="Parameter" version="opset1">
                            <data element_type="f32" shape="1,1,512"/>
                            <output>
                                <port id="0" precision="FP32">
                                    <dim>1</dim>
                                    <dim>1</dim>
                                    <dim>512</dim>
                                </port>
                            </output>
                        </layer>
                        <layer id="1" name="25_const" type="Const" version="opset1">
                            <data offset="0" size="16"/>
                            <output>
                                <port id="1" precision="I64">
                                    <dim>2</dim>
                                </port>
                            </output>
                        </layer>
                        <layer id="2" name="Data_/InputSqueeze" type="Reshape" version="opset1">
                            <data special_zero="True"/>
                            <input>
                                <port id="0">
                                    <dim>1</dim>
                                    <dim>1</dim>
                                    <dim>512</dim>
                                </port>
                                <port id="1">
                                    <dim>2</dim>
                                </port>
                            </input>
                            <output>
                                <port id="2" precision="FP32">
                                    <dim>1</dim>
                                    <dim>512</dim>
                                </port>
                            </output>
                        </layer>
                        <layer id="3" name="34" type="Parameter" version="opset1">
                            <data element_type="f32" shape="1,256"/>
                            <output>
                                <port id="0" precision="FP32">
                                    <dim>1</dim>
                                    <dim>256</dim>
                                </port>
                            </output>
                        </layer>
                        <layer id="4" name="36" type="Parameter" version="opset1">
                            <data element_type="f32" shape="1,256"/>
                            <output>
                                <port id="0" precision="FP32">
                                    <dim>1</dim>
                                    <dim>256</dim>
                                </port>
                            </output>
                        </layer>
                        <layer id="5" name="concat/LSTMCell/Split256_const" type="Const" version="opset1">
                            <data offset="16" size="2097152"/>
                            <output>
                                <port id="1" precision="FP32">
                                    <dim>1024</dim>
                                    <dim>512</dim>
                                </port>
                            </output>
                        </layer>
                        <layer id="6" name="LSTMCell/Split257_const" type="Const" version="opset1">
                            <data offset="2097168" size="1048576"/>
                            <output>
                                <port id="1" precision="FP32">
                                    <dim>1024</dim>
                                    <dim>256</dim>
                                </port>
                            </output>
                        </layer>
                        <layer id="7" name="Output_0/Data__const" type="Const" version="opset1">
                            <data offset="3145744" size="4096"/>
                            <output>
                                <port id="1" precision="FP32">
                                    <dim>1024</dim>
                                </port>
                            </output>
                        </layer>
                        <layer id="8" name="concat/LSTMCell" type="LSTMCell" version="opset1">
                            <data hidden_size="256"/>
                            <input>
                                <port id="0">
                                    <dim>1</dim>
                                    <dim>512</dim>
                                </port>
                                <port id="1">
                                    <dim>1</dim>
                                    <dim>256</dim>
                                </port>
                                <port id="2">
                                    <dim>1</dim>
                                    <dim>256</dim>
                                </port>
                                <port id="3">
                                    <dim>1024</dim>
                                    <dim>512</dim>
                                </port>
                                <port id="4">
                                    <dim>1024</dim>
                                    <dim>256</dim>
                                </port>
                                <port id="5">
                                    <dim>1024</dim>
                                </port>
                            </input>
                            <output>
                                <port id="6" precision="FP32">
                                    <dim>1</dim>
                                    <dim>256</dim>
                                </port>
                                <port id="7" precision="FP32">
                                    <dim>1</dim>
                                    <dim>256</dim>
                                </port>
                            </output>
                        </layer>
                        <layer id="9" name="sink_port_0" type="Result" version="opset1">
                            <input>
                                <port id="0">
                                    <dim>1</dim>
                                    <dim>256</dim>
                                </port>
                            </input>
                        </layer>
                        <layer id="10" name="sink_port_1" type="Result" version="opset1">
                            <input>
                                <port id="0">
                                    <dim>1</dim>
                                    <dim>256</dim>
                                </port>
                            </input>
                        </layer>
                        <layer id="11" name="28_const" type="Const" version="opset1">
                            <data offset="3149840" size="24"/>
                            <output>
                                <port id="1" precision="I64">
                                    <dim>3</dim>
                                </port>
                            </output>
                        </layer>
                        <layer id="12" name="OutputUnsqueeze" type="Reshape" version="opset1">
                            <data special_zero="True"/>
                            <input>
                                <port id="0">
                                    <dim>1</dim>
                                    <dim>256</dim>
                                </port>
                                <port id="1">
                                    <dim>3</dim>
                                </port>
                            </input>
                            <output>
                                <port id="2" precision="FP32">
                                    <dim>1</dim>
                                    <dim>1</dim>
                                    <dim>256</dim>
                                </port>
                            </output>
                        </layer>
                        <layer id="13" name="30/sink_port_0" type="Result" version="opset1">
                            <input>
                                <port id="0">
                                    <dim>1</dim>
                                    <dim>1</dim>
                                    <dim>256</dim>
                                </port>
                            </input>
                        </layer>
                    </layers>
                    <edges>
                        <edge from-layer="0" from-port="0" to-layer="2" to-port="0"/>
                        <edge from-layer="1" from-port="1" to-layer="2" to-port="1"/>
                        <edge from-layer="2" from-port="2" to-layer="8" to-port="0"/>
                        <edge from-layer="3" from-port="0" to-layer="8" to-port="1"/>
                        <edge from-layer="4" from-port="0" to-layer="8" to-port="2"/>
                        <edge from-layer="5" from-port="1" to-layer="8" to-port="3"/>
                        <edge from-layer="6" from-port="1" to-layer="8" to-port="4"/>
                        <edge from-layer="7" from-port="1" to-layer="8" to-port="5"/>
                        <edge from-layer="8" from-port="7" to-layer="9" to-port="0"/>
                        <edge from-layer="8" from-port="6" to-layer="10" to-port="0"/>
                        <edge from-layer="8" from-port="6" to-layer="12" to-port="0"/>
                        <edge from-layer="11" from-port="1" to-layer="12" to-port="1"/>
                        <edge from-layer="12" from-port="2" to-layer="13" to-port="0"/>
                    </edges>
                </body>
            </layer>
            <layer id="4" name="result" type="Result" version="opset1">
                <input>
                    <port id="0">
                        <dim>1</dim>
                        <dim>25</dim>
                        <dim>256</dim>
                    </port>
                </input>
            </layer>
        </layers>
        <edges>
            <edge from-layer="0" from-port="0" to-layer="3" to-port="0"/>
            <edge from-layer="1" from-port="0" to-layer="3" to-port="1"/>
            <edge from-layer="2" from-port="0" to-layer="3" to-port="2"/>
            <edge from-layer="3" from-port="3" to-layer="4" to-port="0"/>
        </edges>
    </net>
    )V0G0N";
    std::string model_v6 = R"VOGON(
    <net name="Transpose" version="6">
        <layers>
            <layer id="0" precision="FP32" name="data1" type="Input">
                <output>
                    <port id="0" precision="FP32">
                        <dim>1</dim>
                        <dim>25</dim>
                        <dim>512</dim>
                    </port>
                </output>
            </layer>
            <layer id="1" precision="FP32" name="data2" type="Input">
                <output>
                    <port id="0" precision="FP32">
                        <dim>1</dim>
                        <dim>256</dim>
                    </port>
                </output>
            </layer>
            <layer id="2" precision="FP32" name="data3" type="Input">
                <output>
                    <port id="0" precision="FP32">
                        <dim>1</dim>
                        <dim>256</dim>
                    </port>
                </output>
            </layer>
            <layer id="3" precision="FP32" name="TensorIterator" type="TensorIterator">
                <input>
                    <port id="0">
                        <dim>1</dim>
                        <dim>25</dim>
                        <dim>512</dim>
                    </port>
                    <port id="1">
                        <dim>1</dim>
                        <dim>256</dim>
                    </port>
                    <port id="2">
                        <dim>1</dim>
                        <dim>256</dim>
                    </port>
                </input>
                <output>
                    <port id="3" precision="FP32">
                        <dim>1</dim>
                        <dim>25</dim>
                        <dim>256</dim>
                    </port>
                </output>
                <port_map>
                    <input axis="1" external_port_id="0" internal_layer_id="1" internal_port_id="0" start="0"/>
                    <input external_port_id="1" internal_layer_id="2" internal_port_id="1"/>
                    <input external_port_id="2" internal_layer_id="2" internal_port_id="2"/>
                    <output axis="1" external_port_id="3" internal_layer_id="4" internal_port_id="2"/>
                </port_map>
                <back_edges>
                    <edge from-layer="2" from-port="5" to-layer="2" to-port="1"/>
                    <edge from-layer="2" from-port="6" to-layer="2" to-port="2"/>
                </back_edges>
                <body>
                    <layers>
                        <layer id="0" name="25_const" type="Const">
                            <output>
                                <port id="1" precision="I64">
                                    <dim>2</dim>
                                </port>
                            </output>
                            <blobs>
                                <custom offset="0" precision="I64" size="8"/>
                            </blobs>
                        </layer>
                        <layer id="1" name="Data_/InputSqueeze" type="Reshape">
                            <input>
                                <port id="0">
                                    <dim>1</dim>
                                    <dim>1</dim>
                                    <dim>512</dim>
                                </port>
                                <port id="1">
                                    <dim>2</dim>
                                </port>
                            </input>
                            <output>
                                <port id="2" precision="FP32">
                                    <dim>1</dim>
                                    <dim>512</dim>
                                </port>
                            </output>
                        </layer>
                        <layer id="2" name="concat/LSTMCell" type="LSTMCell">
                            <data hidden_size="256" activations="sigmoid,tanh,tanh" activations_alpha="" activations_beta="" clip="0"/>
                            <input>
                                <port id="0">
                                    <dim>1</dim>
                                    <dim>512</dim>
                                </port>
                                <port id="1">
                                    <dim>1</dim>
                                    <dim>256</dim>
                                </port>
                                <port id="2">
                                    <dim>1</dim>
                                    <dim>256</dim>
                                </port>
                            </input>
                            <output>
                                <port id="5" precision="FP32">
                                    <dim>1</dim>
                                    <dim>256</dim>
                                </port>
                                <port id="6" precision="FP32">
                                    <dim>1</dim>
                                    <dim>256</dim>
                                </port>
                            </output>
                            <blobs>
                                <weights offset="8" precision="FP32" size="3145728"/>
                                <biases offset="3145736" precision="FP32" size="4096"/>
                            </blobs>
                        </layer>
                        <layer id="3" name="28_const" type="Const">
                            <output>
                                <port id="1" precision="I64">
                                    <dim>3</dim>
                                </port>
                            </output>
                            <blobs>
                                <custom offset="3149832" precision="I64" size="12"/>
                            </blobs>
                        </layer>
                        <layer id="4" name="OutputUnsqueeze" type="Reshape">
                            <input>
                                <port id="0">
                                    <dim>1</dim>
                                    <dim>256</dim>
                                </port>
                                <port id="1">
                                    <dim>3</dim>
                                </port>
                            </input>
                            <output>
                                <port id="2" precision="FP32">
                                    <dim>1</dim>
                                    <dim>1</dim>
                                    <dim>256</dim>
                                </port>
                            </output>
                        </layer>
                    </layers>
                    <edges>
                        <edge from-layer="0" from-port="1" to-layer="1" to-port="1"/>
                        <edge from-layer="1" from-port="2" to-layer="2" to-port="0"/>
                        <edge from-layer="2" from-port="5" to-layer="4" to-port="0"/>
                        <edge from-layer="3" from-port="1" to-layer="4" to-port="1"/>
                    </edges>
                </body>
            </layer>
        </layers>
        <edges>
            <edge from-layer="0" from-port="0" to-layer="3" to-port="0"/>
            <edge from-layer="1" from-port="0" to-layer="3" to-port="1"/>
            <edge from-layer="2" from-port="0" to-layer="3" to-port="2"/>
        </edges>
    </net>
    )VOGON";

    compareIRs(model_v10, model_v6, 3149864, [](Blob::Ptr& weights) {
        auto *data = weights->buffer().as<int64_t *>();
        data[0] = 1;
        data[1] = 512;

        data[393730] = 1;
        data[393731] = 1;
        data[393732] = 256;
    });
}

TEST_F(NGraphReaderTests, ReadTensorIteratorNetwork_resnet_opset1) {
    std::string model_v10 = R"V0G0N(
    <net name="Resnet" version="10">
        <layers>
            <layer id="0" name="data1" type="Parameter" version="opset1">
                <data element_type="f32" shape="16,1,512"/>
                <output>
                    <port id="0" precision="FP32">
                        <dim>16</dim>
                        <dim>1</dim>
                        <dim>512</dim>
                    </port>
                </output>
            </layer>
            <layer id="1" name="data2" type="Parameter" version="opset1">
                <data element_type="f32" shape="1,512"/>
                <output>
                    <port id="0" precision="FP32">
                        <dim>1</dim>
                        <dim>512</dim>
                    </port>
                </output>
            </layer>
            <layer id="2" name="data3" type="Parameter" version="opset1">
                <data element_type="f32" shape="1,512"/>
                <output>
                    <port id="0" precision="FP32">
                        <dim>1</dim>
                        <dim>512</dim>
                    </port>
                </output>
            </layer>
            <layer id="3" name="471/TensorIterator" type="TensorIterator" version="opset1">
                <input>
                    <port id="0">
                        <dim>16</dim>
                        <dim>1</dim>
                        <dim>512</dim>
                    </port>
                    <port id="1">
                        <dim>1</dim>
                        <dim>512</dim>
                    </port>
                    <port id="2">
                        <dim>1</dim>
                        <dim>512</dim>
                    </port>
                </input>
                <output>
                    <port id="3" precision="FP32">
                        <dim>16</dim>
                        <dim>1</dim>
                        <dim>512</dim>
                    </port>
                    <port id="4" precision="FP32">
                        <dim>1</dim>
                        <dim>512</dim>
                    </port>
                    <port id="5" precision="FP32">
                        <dim>1</dim>
                        <dim>512</dim>
                    </port>
                </output>
                <port_map>
                    <input axis="0" external_port_id="0" internal_layer_id="0" part_size="1" stride="1"/>
                    <input external_port_id="1" internal_layer_id="3"/>
                    <input external_port_id="2" internal_layer_id="4"/>
                    <output axis="0" external_port_id="3" internal_layer_id="13" part_size="1" stride="1"/>
                    <output external_port_id="4" internal_layer_id="9"/>
                    <output external_port_id="5" internal_layer_id="10"/>
                </port_map>
                <back_edges>
                    <edge from-layer="9" to-layer="3"/>
                    <edge from-layer="10" to-layer="4"/>
                </back_edges>
                <body>
                    <layers>
                        <layer id="0" name="20" type="Parameter" version="opset1">
                            <data element_type="f32" shape="1,1,512"/>
                            <output>
                                <port id="0" precision="FP32">
                                    <dim>1</dim>
                                    <dim>1</dim>
                                    <dim>512</dim>
                                </port>
                            </output>
                        </layer>
                        <layer id="1" name="7_const" type="Const" version="opset1">
                            <data offset="0" size="16"/>
                            <output>
                                <port id="1" precision="I64">
                                    <dim>2</dim>
                                </port>
                            </output>
                        </layer>
                        <layer id="2" name="471/input_squeeze" type="Reshape" version="opset1">
                            <data special_zero="True"/>
                            <input>
                                <port id="0">
                                    <dim>1</dim>
                                    <dim>1</dim>
                                    <dim>512</dim>
                                </port>
                                <port id="1">
                                    <dim>2</dim>
                                </port>
                            </input>
                            <output>
                                <port id="2" precision="FP32">
                                    <dim>1</dim>
                                    <dim>512</dim>
                                </port>
                            </output>
                        </layer>
                        <layer id="3" name="22" type="Parameter" version="opset1">
                            <data element_type="f32" shape="1,512"/>
                            <output>
                                <port id="0" precision="FP32">
                                    <dim>1</dim>
                                    <dim>512</dim>
                                </port>
                            </output>
                        </layer>
                        <layer id="4" name="24" type="Parameter" version="opset1">
                            <data element_type="f32" shape="1,512"/>
                            <output>
                                <port id="0" precision="FP32">
                                    <dim>1</dim>
                                    <dim>512</dim>
                                </port>
                            </output>
                        </layer>
                        <layer id="5" name="471/LSTMCell/Split149_const" type="Const" version="opset1">
                            <data offset="16" size="4194304"/>
                            <output>
                                <port id="1" precision="FP32">
                                    <dim>2048</dim>
                                    <dim>512</dim>
                                </port>
                            </output>
                        </layer>
                        <layer id="6" name="471/LSTMCell/Split150_const" type="Const" version="opset1">
                            <data offset="4194320" size="4194304"/>
                            <output>
                                <port id="1" precision="FP32">
                                    <dim>2048</dim>
                                    <dim>512</dim>
                                </port>
                            </output>
                        </layer>
                        <layer id="7" name="471/inport/2_const" type="Const" version="opset1">
                            <data offset="8388624" size="8192"/>
                            <output>
                                <port id="1" precision="FP32">
                                    <dim>2048</dim>
                                </port>
                            </output>
                        </layer>
                        <layer id="8" name="471/LSTMCell" type="LSTMCell" version="opset1">
                            <data hidden_size="512"/>
                            <input>
                                <port id="0">
                                    <dim>1</dim>
                                    <dim>512</dim>
                                </port>
                                <port id="1">
                                    <dim>1</dim>
                                    <dim>512</dim>
                                </port>
                                <port id="2">
                                    <dim>1</dim>
                                    <dim>512</dim>
                                </port>
                                <port id="3">
                                    <dim>2048</dim>
                                    <dim>512</dim>
                                </port>
                                <port id="4">
                                    <dim>2048</dim>
                                    <dim>512</dim>
                                </port>
                                <port id="5">
                                    <dim>2048</dim>
                                </port>
                            </input>
                            <output>
                                <port id="6" precision="FP32">
                                    <dim>1</dim>
                                    <dim>512</dim>
                                </port>
                                <port id="7" precision="FP32">
                                    <dim>1</dim>
                                    <dim>512</dim>
                                </port>
                            </output>
                        </layer>
                        <layer id="9" name="471/outport/0/sink_port_0" type="Result" version="opset1">
                            <input>
                                <port id="0">
                                    <dim>1</dim>
                                    <dim>512</dim>
                                </port>
                            </input>
                        </layer>
                        <layer id="10" name="471/outport/1/sink_port_0" type="Result" version="opset1">
                            <input>
                                <port id="0">
                                    <dim>1</dim>
                                    <dim>512</dim>
                                </port>
                            </input>
                        </layer>
                        <layer id="11" name="15_const" type="Const" version="opset1">
                            <data offset="8396816" size="24"/>
                            <output>
                                <port id="1" precision="I64">
                                    <dim>3</dim>
                                </port>
                            </output>
                        </layer>
                        <layer id="12" name="471output_unsqueeze" type="Reshape" version="opset1">
                            <data special_zero="True"/>
                            <input>
                                <port id="0">
                                    <dim>1</dim>
                                    <dim>512</dim>
                                </port>
                                <port id="1">
                                    <dim>3</dim>
                                </port>
                            </input>
                            <output>
                                <port id="2" precision="FP32">
                                    <dim>1</dim>
                                    <dim>1</dim>
                                    <dim>512</dim>
                                </port>
                            </output>
                        </layer>
                        <layer id="13" name="18/sink_port_0" type="Result" version="opset1">
                            <input>
                                <port id="0">
                                    <dim>1</dim>
                                    <dim>1</dim>
                                    <dim>512</dim>
                                </port>
                            </input>
                        </layer>
                    </layers>
                    <edges>
                        <edge from-layer="0" from-port="0" to-layer="2" to-port="0"/>
                        <edge from-layer="1" from-port="1" to-layer="2" to-port="1"/>
                        <edge from-layer="2" from-port="2" to-layer="8" to-port="0"/>
                        <edge from-layer="3" from-port="0" to-layer="8" to-port="1"/>
                        <edge from-layer="4" from-port="0" to-layer="8" to-port="2"/>
                        <edge from-layer="5" from-port="1" to-layer="8" to-port="3"/>
                        <edge from-layer="6" from-port="1" to-layer="8" to-port="4"/>
                        <edge from-layer="7" from-port="1" to-layer="8" to-port="5"/>
                        <edge from-layer="8" from-port="6" to-layer="9" to-port="0"/>
                        <edge from-layer="8" from-port="7" to-layer="10" to-port="0"/>
                        <edge from-layer="8" from-port="6" to-layer="12" to-port="0"/>
                        <edge from-layer="11" from-port="1" to-layer="12" to-port="1"/>
                        <edge from-layer="12" from-port="2" to-layer="13" to-port="0"/>
                    </edges>
                </body>
            </layer>
            <layer id="4" name="result" type="Result" version="opset1">
                <input>
                    <port id="0">
                        <dim>16</dim>
                        <dim>1</dim>
                        <dim>512</dim>
                    </port>
                </input>
            </layer>
            <layer id="5" name="result_2" type="Result" version="opset1">
                <input>
                    <port id="0">
                        <dim>1</dim>
                        <dim>512</dim>
                    </port>
                </input>
            </layer>
            <layer id="6" name="result_3" type="Result" version="opset1">
                <input>
                    <port id="0">
                        <dim>1</dim>
                        <dim>512</dim>
                    </port>
                </input>
            </layer>
        </layers>
        <edges>
            <edge from-layer="0" from-port="0" to-layer="3" to-port="0"/>
            <edge from-layer="1" from-port="0" to-layer="3" to-port="1"/>
            <edge from-layer="2" from-port="0" to-layer="3" to-port="2"/>
            <edge from-layer="3" from-port="3" to-layer="4" to-port="0"/>
            <edge from-layer="3" from-port="4" to-layer="5" to-port="0"/>
            <edge from-layer="3" from-port="5" to-layer="6" to-port="0"/>
        </edges>
    </net>
    )V0G0N";
    std::string model_v6 = R"V0G0N(
    <net name="Resnet" version="7">
        <layers>
            <layer id="0" name="data1" precision="FP32" type="Input">
                <output>
                    <port id="0" precision="FP32">
                        <dim>16</dim>
                        <dim>1</dim>
                        <dim>512</dim>
                    </port>
                </output>
            </layer>
            <layer id="1" name="data2" precision="FP32" type="Input">
                <output>
                    <port id="0" precision="FP32">
                        <dim>1</dim>
                        <dim>512</dim>
                    </port>
                </output>
            </layer>
            <layer id="2" name="data3" precision="FP32" type="Input">
                <output>
                    <port id="0" precision="FP32">
                        <dim>1</dim>
                        <dim>512</dim>
                    </port>
                </output>
            </layer>
            <layer id="3" name="471/TensorIterator" precision="FP32" type="TensorIterator">
                <input>
                    <port id="0">
                        <dim>16</dim>
                        <dim>1</dim>
                        <dim>512</dim>
                    </port>
                    <port id="1">
                        <dim>1</dim>
                        <dim>512</dim>
                    </port>
                    <port id="2">
                        <dim>1</dim>
                        <dim>512</dim>
                    </port>
                </input>
                <output>
                    <port id="3" precision="FP32">
                        <dim>16</dim>
                        <dim>1</dim>
                        <dim>512</dim>
                    </port>
                    <port id="4" precision="FP32">
                        <dim>1</dim>
                        <dim>512</dim>
                    </port>
                    <port id="5" precision="FP32">
                        <dim>1</dim>
                        <dim>512</dim>
                    </port>
                </output>
                <port_map>
                    <input axis="0" external_port_id="0" internal_layer_id="1" internal_port_id="0" part_size="1" stride="1"/>
                    <input external_port_id="1" internal_layer_id="2" internal_port_id="1"/>
                    <input external_port_id="2" internal_layer_id="2" internal_port_id="2"/>
                    <output axis="0" external_port_id="3" internal_layer_id="4" internal_port_id="2" part_size="1" stride="1"/>
                    <output external_port_id="4" internal_layer_id="2" internal_port_id="5"/>
                    <output external_port_id="5" internal_layer_id="2" internal_port_id="6"/>
                </port_map>
                <back_edges>
                    <edge from-layer="2" from-port="5" to-layer="2" to-port="1"/>
                    <edge from-layer="2" from-port="6" to-layer="2" to-port="2"/>
                </back_edges>
                <body>
                    <layers>
                        <layer id="0" name="7_const" type="Const">
                            <output>
                                <port id="1" precision="I64">
                                    <dim>2</dim>
                                </port>
                            </output>
                            <blobs>
                                <custom offset="0" precision="I64" size="16"/>
                            </blobs>
                        </layer>
                        <layer id="1" name="471/input_squeeze" precision="FP32" type="Reshape">
                            <input>
                                <port id="0">
                                    <dim>1</dim>
                                    <dim>1</dim>
                                    <dim>512</dim>
                                </port>
                                <port id="1">
                                    <dim>2</dim>
                                </port>
                            </input>
                            <output>
                                <port id="2" precision="FP32">
                                    <dim>1</dim>
                                    <dim>512</dim>
                                </port>
                            </output>
                        </layer>
                        <layer id="2" name="471/LSTMCell" precision="FP32" type="LSTMCell">
                            <data hidden_size="512" activations="sigmoid,tanh,tanh" activations_alpha="" activations_beta="" clip="0"/>
                            <input>
                                <port id="0">
                                    <dim>1</dim>
                                    <dim>512</dim>
                                </port>
                                <port id="1">
                                    <dim>1</dim>
                                    <dim>512</dim>
                                </port>
                                <port id="2">
                                    <dim>1</dim>
                                    <dim>512</dim>
                                </port>
                            </input>
                            <output>
                                <port id="5" precision="FP32">
                                    <dim>1</dim>
                                    <dim>512</dim>
                                </port>
                                <port id="6" precision="FP32">
                                    <dim>1</dim>
                                    <dim>512</dim>
                                </port>
                            </output>
                            <blobs>
                                <weights offset="16" precision="FP32" size="8388608"/>
                                <biases offset="8388620" precision="FP32" size="8192"/>
                            </blobs>
                        </layer>
                        <layer id="3" name="15_const" type="Const">
                            <output>
                                <port id="1" precision="I64">
                                    <dim>3</dim>
                                </port>
                            </output>
                            <blobs>
                                <custom offset="8396812" precision="I64" size="24"/>
                            </blobs>
                        </layer>
                        <layer id="4" name="471output_unsqueeze" precision="FP32" type="Reshape">
                            <input>
                                <port id="0">
                                    <dim>1</dim>
                                    <dim>512</dim>
                                </port>
                                <port id="1">
                                    <dim>3</dim>
                                </port>
                            </input>
                            <output>
                                <port id="2" precision="FP32">
                                    <dim>1</dim>
                                    <dim>1</dim>
                                    <dim>512</dim>
                                </port>
                            </output>
                        </layer>
                    </layers>
                    <edges>
                        <edge from-layer="0" from-port="1" to-layer="1" to-port="1"/>
                        <edge from-layer="1" from-port="2" to-layer="2" to-port="0"/>
                        <edge from-layer="2" from-port="5" to-layer="4" to-port="0"/>
                        <edge from-layer="3" from-port="1" to-layer="4" to-port="1"/>
                    </edges>
                </body>
            </layer>
        </layers>
        <edges>
            <edge from-layer="0" from-port="0" to-layer="3" to-port="0"/>
            <edge from-layer="1" from-port="0" to-layer="3" to-port="1"/>
            <edge from-layer="2" from-port="0" to-layer="3" to-port="2"/>
        </edges>
    </net>
    )V0G0N";

    compareIRs(model_v10, model_v6, 8396840, [](Blob::Ptr& weights) {
        auto *data = weights->buffer().as<int64_t *>();
        data[0] = 1;
        data[1] = 512;

        data[1049602] = 1;
        data[1049603] = 1;
        data[1049604] = 512;
    });
}

TEST_F(NGraphReaderTests, ReadTensorIteratorNetwork_negative_stride_opset1) {
    std::string model_v10 = R"V0G0N(
    <net name="Transpose" version="10">
        <layers>
            <layer id="0" name="data1" type="Parameter" version="opset1">
                <data element_type="f32" shape="1,25,512"/>
                <output>
                    <port id="0" precision="FP32">
                        <dim>1</dim>
                        <dim>25</dim>
                        <dim>512</dim>
                    </port>
                </output>
            </layer>
            <layer id="1" name="data2" type="Parameter" version="opset1">
                <data element_type="f32" shape="1,256"/>
                <output>
                    <port id="0" precision="FP32">
                        <dim>1</dim>
                        <dim>256</dim>
                    </port>
                </output>
            </layer>
            <layer id="2" name="data3" type="Parameter" version="opset1">
                <data element_type="f32" shape="1,256"/>
                <output>
                    <port id="0" precision="FP32">
                        <dim>1</dim>
                        <dim>256</dim>
                    </port>
                </output>
            </layer>
            <layer id="3" name="TensorIterator" type="TensorIterator" version="opset1">
                <input>
                    <port id="0">
                        <dim>1</dim>
                        <dim>25</dim>
                        <dim>512</dim>
                    </port>
                    <port id="1">
                        <dim>1</dim>
                        <dim>256</dim>
                    </port>
                    <port id="2">
                        <dim>1</dim>
                        <dim>256</dim>
                    </port>
                </input>
                <output>
                    <port id="3" precision="FP32">
                        <dim>1</dim>
                        <dim>25</dim>
                        <dim>256</dim>
                    </port>
                </output>
                <port_map>
                    <input axis="1" end="0" external_port_id="0" internal_layer_id="0" start="-1" stride="-1"/>
                    <input external_port_id="1" internal_layer_id="3"/>
                    <input external_port_id="2" internal_layer_id="4"/>
                    <output axis="1" end="0" external_port_id="3" internal_layer_id="13" start="-1" stride="-1"/>
                </port_map>
                <back_edges>
                    <edge from-layer="10" to-layer="3"/>
                    <edge from-layer="9" to-layer="4"/>
                </back_edges>
                <body>
                    <layers>
                        <layer id="0" name="32" type="Parameter" version="opset1">
                            <data element_type="f32" shape="1,1,512"/>
                            <output>
                                <port id="0" precision="FP32">
                                    <dim>1</dim>
                                    <dim>1</dim>
                                    <dim>512</dim>
                                </port>
                            </output>
                        </layer>
                        <layer id="1" name="25_const" type="Const" version="opset1">
                            <data offset="0" size="16"/>
                            <output>
                                <port id="1" precision="I64">
                                    <dim>2</dim>
                                </port>
                            </output>
                        </layer>
                        <layer id="2" name="shadow/LSTMLayers/stack_bidirectional_rnn/cell_1/bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Output_0/Data_/InputSqueeze" type="Reshape" version="opset1">
                            <data special_zero="True"/>
                            <input>
                                <port id="0">
                                    <dim>1</dim>
                                    <dim>1</dim>
                                    <dim>512</dim>
                                </port>
                                <port id="1">
                                    <dim>2</dim>
                                </port>
                            </input>
                            <output>
                                <port id="2" precision="FP32">
                                    <dim>1</dim>
                                    <dim>512</dim>
                                </port>
                            </output>
                        </layer>
                        <layer id="3" name="34" type="Parameter" version="opset1">
                            <data element_type="f32" shape="1,256"/>
                            <output>
                                <port id="0" precision="FP32">
                                    <dim>1</dim>
                                    <dim>256</dim>
                                </port>
                            </output>
                        </layer>
                        <layer id="4" name="36" type="Parameter" version="opset1">
                            <data element_type="f32" shape="1,256"/>
                            <output>
                                <port id="0" precision="FP32">
                                    <dim>1</dim>
                                    <dim>256</dim>
                                </port>
                            </output>
                        </layer>
                        <layer id="5" name="shadow/LSTMLayers/stack_bidirectional_rnn/cell_1/bidirectional_rnn/bw/bw/while/bw/basic_lstm_cell/concat/LSTMCell/Split269_const" type="Const" version="opset1">
                            <data offset="16" size="2097152"/>
                            <output>
                                <port id="1" precision="FP32">
                                    <dim>1024</dim>
                                    <dim>512</dim>
                                </port>
                            </output>
                        </layer>
                        <layer id="6" name="shadow/LSTMLayers/stack_bidirectional_rnn/cell_1/bidirectional_rnn/bw/bw/while/bw/basic_lstm_cell/concat/LSTMCell/Split270_const" type="Const" version="opset1">
                            <data offset="2097168" size="1048576"/>
                            <output>
                                <port id="1" precision="FP32">
                                    <dim>1024</dim>
                                    <dim>256</dim>
                                </port>
                            </output>
                        </layer>
                        <layer id="7" name="shadow/LSTMLayers/stack_bidirectional_rnn/cell_1/bidirectional_rnn/bw/bw/while/bw/basic_lstm_cell/BiasAdd/Enter/Output_0/Data__const" type="Const" version="opset1">
                            <data offset="3145744" size="4096"/>
                            <output>
                                <port id="1" precision="FP32">
                                    <dim>1024</dim>
                                </port>
                            </output>
                        </layer>
                        <layer id="8" name="shadow/LSTMLayers/stack_bidirectional_rnn/cell_1/bidirectional_rnn/bw/bw/while/bw/basic_lstm_cell/concat/LSTMCell" type="LSTMCell" version="opset1">
                            <data hidden_size="256"/>
                            <input>
                                <port id="0">
                                    <dim>1</dim>
                                    <dim>512</dim>
                                </port>
                                <port id="1">
                                    <dim>1</dim>
                                    <dim>256</dim>
                                </port>
                                <port id="2">
                                    <dim>1</dim>
                                    <dim>256</dim>
                                </port>
                                <port id="3">
                                    <dim>1024</dim>
                                    <dim>512</dim>
                                </port>
                                <port id="4">
                                    <dim>1024</dim>
                                    <dim>256</dim>
                                </port>
                                <port id="5">
                                    <dim>1024</dim>
                                </port>
                            </input>
                            <output>
                                <port id="6" precision="FP32">
                                    <dim>1</dim>
                                    <dim>256</dim>
                                </port>
                                <port id="7" precision="FP32">
                                    <dim>1</dim>
                                    <dim>256</dim>
                                </port>
                            </output>
                        </layer>
                        <layer id="9" name="shadow/LSTMLayers/stack_bidirectional_rnn/cell_1/bidirectional_rnn/bw/bw/while/bw/basic_lstm_cell/concat/LSTMCell/Output_1/Data_/sink_port_0" type="Result" version="opset1">
                            <input>
                                <port id="0">
                                    <dim>1</dim>
                                    <dim>256</dim>
                                </port>
                            </input>
                        </layer>
                        <layer id="10" name="shadow/LSTMLayers/stack_bidirectional_rnn/cell_1/bidirectional_rnn/bw/bw/while/bw/basic_lstm_cell/concat/LSTMCell/Output_0/Data_/sink_port_0" type="Result" version="opset1">
                            <input>
                                <port id="0">
                                    <dim>1</dim>
                                    <dim>256</dim>
                                </port>
                            </input>
                        </layer>
                        <layer id="11" name="28_const" type="Const" version="opset1">
                            <data offset="3149840" size="24"/>
                            <output>
                                <port id="1" precision="I64">
                                    <dim>3</dim>
                                </port>
                            </output>
                        </layer>
                        <layer id="12" name="shadow/LSTMLayers/stack_bidirectional_rnn/cell_1/bidirectional_rnn/bw/bw/while/bw/basic_lstm_cell/concat/LSTMCell/Output_0/Data_/OutputUnsqueeze" type="Reshape" version="opset1">
                            <data special_zero="True"/>
                            <input>
                                <port id="0">
                                    <dim>1</dim>
                                    <dim>256</dim>
                                </port>
                                <port id="1">
                                    <dim>3</dim>
                                </port>
                            </input>
                            <output>
                                <port id="2" precision="FP32">
                                    <dim>1</dim>
                                    <dim>1</dim>
                                    <dim>256</dim>
                                </port>
                            </output>
                        </layer>
                        <layer id="13" name="30/sink_port_0" type="Result" version="opset1">
                            <input>
                                <port id="0">
                                    <dim>1</dim>
                                    <dim>1</dim>
                                    <dim>256</dim>
                                </port>
                            </input>
                        </layer>
                    </layers>
                    <edges>
                        <edge from-layer="0" from-port="0" to-layer="2" to-port="0"/>
                        <edge from-layer="1" from-port="1" to-layer="2" to-port="1"/>
                        <edge from-layer="2" from-port="2" to-layer="8" to-port="0"/>
                        <edge from-layer="3" from-port="0" to-layer="8" to-port="1"/>
                        <edge from-layer="4" from-port="0" to-layer="8" to-port="2"/>
                        <edge from-layer="5" from-port="1" to-layer="8" to-port="3"/>
                        <edge from-layer="6" from-port="1" to-layer="8" to-port="4"/>
                        <edge from-layer="7" from-port="1" to-layer="8" to-port="5"/>
                        <edge from-layer="8" from-port="7" to-layer="9" to-port="0"/>
                        <edge from-layer="8" from-port="6" to-layer="10" to-port="0"/>
                        <edge from-layer="8" from-port="6" to-layer="12" to-port="0"/>
                        <edge from-layer="11" from-port="1" to-layer="12" to-port="1"/>
                        <edge from-layer="12" from-port="2" to-layer="13" to-port="0"/>
                    </edges>
                </body>
            </layer>
            <layer id="4" name="result" type="Result" version="opset1">
                <input>
                    <port id="0">
                        <dim>1</dim>
                        <dim>25</dim>
                        <dim>256</dim>
                    </port>
                </input>
            </layer>
        </layers>
        <edges>
            <edge from-layer="0" from-port="0" to-layer="3" to-port="0"/>
            <edge from-layer="1" from-port="0" to-layer="3" to-port="1"/>
            <edge from-layer="2" from-port="0" to-layer="3" to-port="2"/>
            <edge from-layer="3" from-port="3" to-layer="4" to-port="0"/>
        </edges>
    </net>
    )V0G0N";
    std::string model_v6 = R"VOGON(
    <net name="Transpose" version="6">
        <layers>
            <layer id="0" precision="FP32" name="data1" type="Input">
                <output>
                    <port id="0" precision="FP32">
                        <dim>1</dim>
                        <dim>25</dim>
                        <dim>512</dim>
                    </port>
                </output>
            </layer>
            <layer id="1" precision="FP32" name="data2" type="Input">
                <output>
                    <port id="0" precision="FP32">
                        <dim>1</dim>
                        <dim>256</dim>
                    </port>
                </output>
            </layer>
            <layer id="2" precision="FP32" name="data3" type="Input">
                <output>
                    <port id="0" precision="FP32">
                        <dim>1</dim>
                        <dim>256</dim>
                    </port>
                </output>
            </layer>
		    <layer id="3" name="TensorIterator" precision="FP32" type="TensorIterator">
                <input>
                    <port id="0">
                        <dim>1</dim>
                        <dim>25</dim>
                        <dim>512</dim>
                    </port>
                    <port id="1">
                        <dim>1</dim>
                        <dim>256</dim>
                    </port>
                    <port id="2">
                        <dim>1</dim>
                        <dim>256</dim>
                    </port>
                </input>
                <output>
                    <port id="3" precision="FP32">
                        <dim>1</dim>
                        <dim>25</dim>
                        <dim>256</dim>
                    </port>
                </output>
                <port_map>
                    <input axis="1" end="0" external_port_id="0" internal_layer_id="1" internal_port_id="0" start="-1" stride="-1"/>
                    <input external_port_id="1" internal_layer_id="2" internal_port_id="1"/>
                    <input external_port_id="2" internal_layer_id="2" internal_port_id="2"/>
                    <output axis="1" end="0" external_port_id="3" internal_layer_id="4" internal_port_id="2" start="-1" stride="-1"/>
                </port_map>
                <back_edges>
                    <edge from-layer="2" from-port="5" to-layer="2" to-port="1"/>
                    <edge from-layer="2" from-port="6" to-layer="2" to-port="2"/>
                </back_edges>
                <body>
                    <layers>
                        <layer id="0" name="25_const" type="Const">
                            <output>
                                <port id="1" precision="I64">
                                    <dim>2</dim>
                                </port>
                            </output>
                            <blobs>
                                <custom offset="0" precision="I64" size="16"/>
                            </blobs>
                        </layer>
                        <layer id="1" precision="FP32" name="shadow/LSTMLayers/stack_bidirectional_rnn/cell_1/bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Output_0/Data_/InputSqueeze" type="Reshape">
                            <input>
                                <port id="0">
                                    <dim>1</dim>
                                    <dim>1</dim>
                                    <dim>512</dim>
                                </port>
                                <port id="1">
                                    <dim>2</dim>
                                </port>
                            </input>
                            <output>
                                <port id="2" precision="FP32">
                                    <dim>1</dim>
                                    <dim>512</dim>
                                </port>
                            </output>
                        </layer>
                        <layer id="2" precision="FP32" name="shadow/LSTMLayers/stack_bidirectional_rnn/cell_1/bidirectional_rnn/bw/bw/while/bw/basic_lstm_cell/concat/LSTMCell" type="LSTMCell">
                            <data hidden_size="256" activations="sigmoid,tanh,tanh" activations_alpha="" activations_beta="" clip="0"/>
                            <input>
                                <port id="0">
                                    <dim>1</dim>
                                    <dim>512</dim>
                                </port>
                                <port id="1">
                                    <dim>1</dim>
                                    <dim>256</dim>
                                </port>
                                <port id="2">
                                    <dim>1</dim>
                                    <dim>256</dim>
                                </port>
                            </input>
                            <output>
                                <port id="5" precision="FP32">
                                    <dim>1</dim>
                                    <dim>256</dim>
                                </port>
                                <port id="6" precision="FP32">
                                    <dim>1</dim>
                                    <dim>256</dim>
                                </port>
                            </output>
                            <blobs>
                                <weights offset="16" precision="FP32" size="3145728"/>
                                <biases offset="3145744" precision="FP32" size="4096"/>
                            </blobs>
                        </layer>
                        <layer id="3" name="28_const" type="Const">
                            <output>
                                <port id="1" precision="I64">
                                    <dim>3</dim>
                                </port>
                            </output>
                            <blobs>
                                <custom offset="3149840" precision="I64" size="24"/>
                            </blobs>
                        </layer>
                        <layer id="4" precision="FP32" name="shadow/LSTMLayers/stack_bidirectional_rnn/cell_1/bidirectional_rnn/bw/bw/while/bw/basic_lstm_cell/concat/LSTMCell/Output_0/Data_/OutputUnsqueeze" type="Reshape">
                            <input>
                                <port id="0">
                                    <dim>1</dim>
                                    <dim>256</dim>
                                </port>
                                <port id="1">
                                    <dim>3</dim>
                                </port>
                            </input>
                            <output>
                                <port id="2" precision="FP32">
                                    <dim>1</dim>
                                    <dim>1</dim>
                                    <dim>256</dim>
                                </port>
                            </output>
                        </layer>
                    </layers>
                    <edges>
                        <edge from-layer="0" from-port="1" to-layer="1" to-port="1"/>
                        <edge from-layer="1" from-port="2" to-layer="2" to-port="0"/>
                        <edge from-layer="2" from-port="5" to-layer="4" to-port="0"/>
                        <edge from-layer="3" from-port="1" to-layer="4" to-port="1"/>
                    </edges>
                </body>
            </layer>
        </layers>
        <edges>
            <edge from-layer="0" from-port="0" to-layer="3" to-port="0"/>
            <edge from-layer="1" from-port="0" to-layer="3" to-port="1"/>
            <edge from-layer="2" from-port="0" to-layer="3" to-port="2"/>
        </edges>
    </net>
    )VOGON";

    compareIRs(model_v10, model_v6, 3149864, [](Blob::Ptr& weights) {
        auto *data = weights->buffer().as<int64_t *>();
        data[0] = 1;
        data[1] = 512;

        data[393730] = 1;
        data[393731] = 1;
        data[393732] = 256;
    });
}

TEST_F(NGraphReaderTests, ReadTensorIteratorNetwork_opset4) {
    std::string model_v10 = R"V0G0N(
    <net name="Transpose" version="10">
        <layers>
            <layer id="0" name="data1" type="Parameter" version="opset1">
                <data element_type="f32" shape="1,25,512"/>
                <output>
                    <port id="0" precision="FP32">
                        <dim>1</dim>
                        <dim>25</dim>
                        <dim>512</dim>
                    </port>
                </output>
            </layer>
            <layer id="1" name="data2" type="Parameter" version="opset1">
                <data element_type="f32" shape="1,256"/>
                <output>
                    <port id="0" precision="FP32">
                        <dim>1</dim>
                        <dim>256</dim>
                    </port>
                </output>
            </layer>
            <layer id="2" name="data3" type="Parameter" version="opset1">
                <data element_type="f32" shape="1,256"/>
                <output>
                    <port id="0" precision="FP32">
                        <dim>1</dim>
                        <dim>256</dim>
                    </port>
                </output>
            </layer>
            <layer id="3" name="TensorIterator" type="TensorIterator" version="opset1">
                <input>
                    <port id="0">
                        <dim>1</dim>
                        <dim>25</dim>
                        <dim>512</dim>
                    </port>
                    <port id="1">
                        <dim>1</dim>
                        <dim>256</dim>
                    </port>
                    <port id="2">
                        <dim>1</dim>
                        <dim>256</dim>
                    </port>
                </input>
                <output>
                    <port id="3" precision="FP32">
                        <dim>1</dim>
                        <dim>25</dim>
                        <dim>256</dim>
                    </port>
                </output>
                <port_map>
                    <input axis="1" external_port_id="0" internal_layer_id="0" start="0"/>
                    <input external_port_id="1" internal_layer_id="3"/>
                    <input external_port_id="2" internal_layer_id="4"/>
                    <output axis="1" external_port_id="3" internal_layer_id="13"/>
                </port_map>
                <back_edges>
                    <edge from-layer="9" to-layer="4"/>
                    <edge from-layer="10" to-layer="3"/>
                </back_edges>
                <body>
                    <layers>
                        <layer id="0" name="32" type="Parameter" version="opset1">
                            <data element_type="f32" shape="1,1,512"/>
                            <output>
                                <port id="0" precision="FP32">
                                    <dim>1</dim>
                                    <dim>1</dim>
                                    <dim>512</dim>
                                </port>
                            </output>
                        </layer>
                        <layer id="1" name="25_const" type="Const" version="opset1">
                            <data offset="0" size="16"/>
                            <output>
                                <port id="1" precision="I64">
                                    <dim>2</dim>
                                </port>
                            </output>
                        </layer>
                        <layer id="2" name="Data_/InputSqueeze" type="Reshape" version="opset1">
                            <data special_zero="True"/>
                            <input>
                                <port id="0">
                                    <dim>1</dim>
                                    <dim>1</dim>
                                    <dim>512</dim>
                                </port>
                                <port id="1">
                                    <dim>2</dim>
                                </port>
                            </input>
                            <output>
                                <port id="2" precision="FP32">
                                    <dim>1</dim>
                                    <dim>512</dim>
                                </port>
                            </output>
                        </layer>
                        <layer id="3" name="34" type="Parameter" version="opset1">
                            <data element_type="f32" shape="1,256"/>
                            <output>
                                <port id="0" precision="FP32">
                                    <dim>1</dim>
                                    <dim>256</dim>
                                </port>
                            </output>
                        </layer>
                        <layer id="4" name="36" type="Parameter" version="opset1">
                            <data element_type="f32" shape="1,256"/>
                            <output>
                                <port id="0" precision="FP32">
                                    <dim>1</dim>
                                    <dim>256</dim>
                                </port>
                            </output>
                        </layer>
                        <layer id="5" name="concat/LSTMCell/Split256_const" type="Const" version="opset1">
                            <data offset="16" size="2097152"/>
                            <output>
                                <port id="1" precision="FP32">
                                    <dim>1024</dim>
                                    <dim>512</dim>
                                </port>
                            </output>
                        </layer>
                        <layer id="6" name="LSTMCell/Split257_const" type="Const" version="opset1">
                            <data offset="2097168" size="1048576"/>
                            <output>
                                <port id="1" precision="FP32">
                                    <dim>1024</dim>
                                    <dim>256</dim>
                                </port>
                            </output>
                        </layer>
                        <layer id="7" name="Output_0/Data__const" type="Const" version="opset1">
                            <data offset="3145744" size="4096"/>
                            <output>
                                <port id="1" precision="FP32">
                                    <dim>1024</dim>
                                </port>
                            </output>
                        </layer>
                        <layer id="8" name="concat/LSTMCell" type="LSTMCell" version="opset4">
                            <data hidden_size="256"/>
                            <input>
                                <port id="0">
                                    <dim>1</dim>
                                    <dim>512</dim>
                                </port>
                                <port id="1">
                                    <dim>1</dim>
                                    <dim>256</dim>
                                </port>
                                <port id="2">
                                    <dim>1</dim>
                                    <dim>256</dim>
                                </port>
                                <port id="3">
                                    <dim>1024</dim>
                                    <dim>512</dim>
                                </port>
                                <port id="4">
                                    <dim>1024</dim>
                                    <dim>256</dim>
                                </port>
                                <port id="5">
                                    <dim>1024</dim>
                                </port>
                            </input>
                            <output>
                                <port id="6" precision="FP32">
                                    <dim>1</dim>
                                    <dim>256</dim>
                                </port>
                                <port id="7" precision="FP32">
                                    <dim>1</dim>
                                    <dim>256</dim>
                                </port>
                            </output>
                        </layer>
                        <layer id="9" name="sink_port_0" type="Result" version="opset1">
                            <input>
                                <port id="0">
                                    <dim>1</dim>
                                    <dim>256</dim>
                                </port>
                            </input>
                        </layer>
                        <layer id="10" name="sink_port_1" type="Result" version="opset1">
                            <input>
                                <port id="0">
                                    <dim>1</dim>
                                    <dim>256</dim>
                                </port>
                            </input>
                        </layer>
                        <layer id="11" name="28_const" type="Const" version="opset1">
                            <data offset="3149840" size="24"/>
                            <output>
                                <port id="1" precision="I64">
                                    <dim>3</dim>
                                </port>
                            </output>
                        </layer>
                        <layer id="12" name="OutputUnsqueeze" type="Reshape" version="opset1">
                            <data special_zero="True"/>
                            <input>
                                <port id="0">
                                    <dim>1</dim>
                                    <dim>256</dim>
                                </port>
                                <port id="1">
                                    <dim>3</dim>
                                </port>
                            </input>
                            <output>
                                <port id="2" precision="FP32">
                                    <dim>1</dim>
                                    <dim>1</dim>
                                    <dim>256</dim>
                                </port>
                            </output>
                        </layer>
                        <layer id="13" name="30/sink_port_0" type="Result" version="opset1">
                            <input>
                                <port id="0">
                                    <dim>1</dim>
                                    <dim>1</dim>
                                    <dim>256</dim>
                                </port>
                            </input>
                        </layer>
                    </layers>
                    <edges>
                        <edge from-layer="0" from-port="0" to-layer="2" to-port="0"/>
                        <edge from-layer="1" from-port="1" to-layer="2" to-port="1"/>
                        <edge from-layer="2" from-port="2" to-layer="8" to-port="0"/>
                        <edge from-layer="3" from-port="0" to-layer="8" to-port="1"/>
                        <edge from-layer="4" from-port="0" to-layer="8" to-port="2"/>
                        <edge from-layer="5" from-port="1" to-layer="8" to-port="3"/>
                        <edge from-layer="6" from-port="1" to-layer="8" to-port="4"/>
                        <edge from-layer="7" from-port="1" to-layer="8" to-port="5"/>
                        <edge from-layer="8" from-port="7" to-layer="9" to-port="0"/>
                        <edge from-layer="8" from-port="6" to-layer="10" to-port="0"/>
                        <edge from-layer="8" from-port="6" to-layer="12" to-port="0"/>
                        <edge from-layer="11" from-port="1" to-layer="12" to-port="1"/>
                        <edge from-layer="12" from-port="2" to-layer="13" to-port="0"/>
                    </edges>
                </body>
            </layer>
            <layer id="4" name="result" type="Result" version="opset1">
                <input>
                    <port id="0">
                        <dim>1</dim>
                        <dim>25</dim>
                        <dim>256</dim>
                    </port>
                </input>
            </layer>
        </layers>
        <edges>
            <edge from-layer="0" from-port="0" to-layer="3" to-port="0"/>
            <edge from-layer="1" from-port="0" to-layer="3" to-port="1"/>
            <edge from-layer="2" from-port="0" to-layer="3" to-port="2"/>
            <edge from-layer="3" from-port="3" to-layer="4" to-port="0"/>
        </edges>
    </net>
    )V0G0N";
    std::string model_v6 = R"VOGON(
    <net name="Transpose" version="6">
        <layers>
            <layer id="0" precision="FP32" name="data1" type="Input">
                <output>
                    <port id="0" precision="FP32">
                        <dim>1</dim>
                        <dim>25</dim>
                        <dim>512</dim>
                    </port>
                </output>
            </layer>
            <layer id="1" precision="FP32" name="data2" type="Input">
                <output>
                    <port id="0" precision="FP32">
                        <dim>1</dim>
                        <dim>256</dim>
                    </port>
                </output>
            </layer>
            <layer id="2" precision="FP32" name="data3" type="Input">
                <output>
                    <port id="0" precision="FP32">
                        <dim>1</dim>
                        <dim>256</dim>
                    </port>
                </output>
            </layer>
            <layer id="3" precision="FP32" name="TensorIterator" type="TensorIterator">
                <input>
                    <port id="0">
                        <dim>1</dim>
                        <dim>25</dim>
                        <dim>512</dim>
                    </port>
                    <port id="1">
                        <dim>1</dim>
                        <dim>256</dim>
                    </port>
                    <port id="2">
                        <dim>1</dim>
                        <dim>256</dim>
                    </port>
                </input>
                <output>
                    <port id="3" precision="FP32">
                        <dim>1</dim>
                        <dim>25</dim>
                        <dim>256</dim>
                    </port>
                </output>
                <port_map>
                    <input axis="1" external_port_id="0" internal_layer_id="1" internal_port_id="0" start="0"/>
                    <input external_port_id="1" internal_layer_id="2" internal_port_id="1"/>
                    <input external_port_id="2" internal_layer_id="2" internal_port_id="2"/>
                    <output axis="1" external_port_id="3" internal_layer_id="4" internal_port_id="2"/>
                </port_map>
                <back_edges>
                    <edge from-layer="2" from-port="5" to-layer="2" to-port="1"/>
                    <edge from-layer="2" from-port="6" to-layer="2" to-port="2"/>
                </back_edges>
                <body>
                    <layers>
                        <layer id="0" name="25_const" type="Const">
                            <output>
                                <port id="1" precision="I64">
                                    <dim>2</dim>
                                </port>
                            </output>
                            <blobs>
                                <custom offset="0" precision="I64" size="8"/>
                            </blobs>
                        </layer>
                        <layer id="1" name="Data_/InputSqueeze" type="Reshape">
                            <input>
                                <port id="0">
                                    <dim>1</dim>
                                    <dim>1</dim>
                                    <dim>512</dim>
                                </port>
                                <port id="1">
                                    <dim>2</dim>
                                </port>
                            </input>
                            <output>
                                <port id="2" precision="FP32">
                                    <dim>1</dim>
                                    <dim>512</dim>
                                </port>
                            </output>
                        </layer>
                        <layer id="2" name="concat/LSTMCell" type="LSTMCell">
                            <data hidden_size="256" activations="sigmoid,tanh,tanh" activations_alpha="" activations_beta="" clip="0"/>
                            <input>
                                <port id="0">
                                    <dim>1</dim>
                                    <dim>512</dim>
                                </port>
                                <port id="1">
                                    <dim>1</dim>
                                    <dim>256</dim>
                                </port>
                                <port id="2">
                                    <dim>1</dim>
                                    <dim>256</dim>
                                </port>
                            </input>
                            <output>
                                <port id="5" precision="FP32">
                                    <dim>1</dim>
                                    <dim>256</dim>
                                </port>
                                <port id="6" precision="FP32">
                                    <dim>1</dim>
                                    <dim>256</dim>
                                </port>
                            </output>
                            <blobs>
                                <weights offset="8" precision="FP32" size="3145728"/>
                                <biases offset="3145736" precision="FP32" size="4096"/>
                            </blobs>
                        </layer>
                        <layer id="3" name="28_const" type="Const">
                            <output>
                                <port id="1" precision="I64">
                                    <dim>3</dim>
                                </port>
                            </output>
                            <blobs>
                                <custom offset="3149832" precision="I64" size="12"/>
                            </blobs>
                        </layer>
                        <layer id="4" name="OutputUnsqueeze" type="Reshape">
                            <input>
                                <port id="0">
                                    <dim>1</dim>
                                    <dim>256</dim>
                                </port>
                                <port id="1">
                                    <dim>3</dim>
                                </port>
                            </input>
                            <output>
                                <port id="2" precision="FP32">
                                    <dim>1</dim>
                                    <dim>1</dim>
                                    <dim>256</dim>
                                </port>
                            </output>
                        </layer>
                    </layers>
                    <edges>
                        <edge from-layer="0" from-port="1" to-layer="1" to-port="1"/>
                        <edge from-layer="1" from-port="2" to-layer="2" to-port="0"/>
                        <edge from-layer="2" from-port="5" to-layer="4" to-port="0"/>
                        <edge from-layer="3" from-port="1" to-layer="4" to-port="1"/>
                    </edges>
                </body>
            </layer>
        </layers>
        <edges>
            <edge from-layer="0" from-port="0" to-layer="3" to-port="0"/>
            <edge from-layer="1" from-port="0" to-layer="3" to-port="1"/>
            <edge from-layer="2" from-port="0" to-layer="3" to-port="2"/>
        </edges>
    </net>
    )VOGON";

    compareIRs(model_v10, model_v6, 3149864, [](Blob::Ptr& weights) {
        auto *data = weights->buffer().as<int64_t *>();
        data[0] = 1;
        data[1] = 512;

        data[393730] = 1;
        data[393731] = 1;
        data[393732] = 256;
    });
}

TEST_F(NGraphReaderTests, ReadTensorIteratorNetwork_resnet_opset4) {
    std::string model_v10 = R"V0G0N(
    <net name="Resnet" version="10">
        <layers>
            <layer id="0" name="data1" type="Parameter" version="opset1">
                <data element_type="f32" shape="16,1,512"/>
                <output>
                    <port id="0" precision="FP32">
                        <dim>16</dim>
                        <dim>1</dim>
                        <dim>512</dim>
                    </port>
                </output>
            </layer>
            <layer id="1" name="data2" type="Parameter" version="opset1">
                <data element_type="f32" shape="1,512"/>
                <output>
                    <port id="0" precision="FP32">
                        <dim>1</dim>
                        <dim>512</dim>
                    </port>
                </output>
            </layer>
            <layer id="2" name="data3" type="Parameter" version="opset1">
                <data element_type="f32" shape="1,512"/>
                <output>
                    <port id="0" precision="FP32">
                        <dim>1</dim>
                        <dim>512</dim>
                    </port>
                </output>
            </layer>
            <layer id="3" name="471/TensorIterator" type="TensorIterator" version="opset1">
                <input>
                    <port id="0">
                        <dim>16</dim>
                        <dim>1</dim>
                        <dim>512</dim>
                    </port>
                    <port id="1">
                        <dim>1</dim>
                        <dim>512</dim>
                    </port>
                    <port id="2">
                        <dim>1</dim>
                        <dim>512</dim>
                    </port>
                </input>
                <output>
                    <port id="3" precision="FP32">
                        <dim>16</dim>
                        <dim>1</dim>
                        <dim>512</dim>
                    </port>
                    <port id="4" precision="FP32">
                        <dim>1</dim>
                        <dim>512</dim>
                    </port>
                    <port id="5" precision="FP32">
                        <dim>1</dim>
                        <dim>512</dim>
                    </port>
                </output>
                <port_map>
                    <input axis="0" external_port_id="0" internal_layer_id="0" part_size="1" stride="1"/>
                    <input external_port_id="1" internal_layer_id="3"/>
                    <input external_port_id="2" internal_layer_id="4"/>
                    <output axis="0" external_port_id="3" internal_layer_id="13" part_size="1" stride="1"/>
                    <output external_port_id="4" internal_layer_id="9"/>
                    <output external_port_id="5" internal_layer_id="10"/>
                </port_map>
                <back_edges>
                    <edge from-layer="9" to-layer="3"/>
                    <edge from-layer="10" to-layer="4"/>
                </back_edges>
                <body>
                    <layers>
                        <layer id="0" name="20" type="Parameter" version="opset1">
                            <data element_type="f32" shape="1,1,512"/>
                            <output>
                                <port id="0" precision="FP32">
                                    <dim>1</dim>
                                    <dim>1</dim>
                                    <dim>512</dim>
                                </port>
                            </output>
                        </layer>
                        <layer id="1" name="7_const" type="Const" version="opset1">
                            <data offset="0" size="16"/>
                            <output>
                                <port id="1" precision="I64">
                                    <dim>2</dim>
                                </port>
                            </output>
                        </layer>
                        <layer id="2" name="471/input_squeeze" type="Reshape" version="opset1">
                            <data special_zero="True"/>
                            <input>
                                <port id="0">
                                    <dim>1</dim>
                                    <dim>1</dim>
                                    <dim>512</dim>
                                </port>
                                <port id="1">
                                    <dim>2</dim>
                                </port>
                            </input>
                            <output>
                                <port id="2" precision="FP32">
                                    <dim>1</dim>
                                    <dim>512</dim>
                                </port>
                            </output>
                        </layer>
                        <layer id="3" name="22" type="Parameter" version="opset1">
                            <data element_type="f32" shape="1,512"/>
                            <output>
                                <port id="0" precision="FP32">
                                    <dim>1</dim>
                                    <dim>512</dim>
                                </port>
                            </output>
                        </layer>
                        <layer id="4" name="24" type="Parameter" version="opset1">
                            <data element_type="f32" shape="1,512"/>
                            <output>
                                <port id="0" precision="FP32">
                                    <dim>1</dim>
                                    <dim>512</dim>
                                </port>
                            </output>
                        </layer>
                        <layer id="5" name="471/LSTMCell/Split149_const" type="Const" version="opset1">
                            <data offset="16" size="4194304"/>
                            <output>
                                <port id="1" precision="FP32">
                                    <dim>2048</dim>
                                    <dim>512</dim>
                                </port>
                            </output>
                        </layer>
                        <layer id="6" name="471/LSTMCell/Split150_const" type="Const" version="opset1">
                            <data offset="4194320" size="4194304"/>
                            <output>
                                <port id="1" precision="FP32">
                                    <dim>2048</dim>
                                    <dim>512</dim>
                                </port>
                            </output>
                        </layer>
                        <layer id="7" name="471/inport/2_const" type="Const" version="opset1">
                            <data offset="8388624" size="8192"/>
                            <output>
                                <port id="1" precision="FP32">
                                    <dim>2048</dim>
                                </port>
                            </output>
                        </layer>
                        <layer id="8" name="471/LSTMCell" type="LSTMCell" version="opset4">
                            <data hidden_size="512"/>
                            <input>
                                <port id="0">
                                    <dim>1</dim>
                                    <dim>512</dim>
                                </port>
                                <port id="1">
                                    <dim>1</dim>
                                    <dim>512</dim>
                                </port>
                                <port id="2">
                                    <dim>1</dim>
                                    <dim>512</dim>
                                </port>
                                <port id="3">
                                    <dim>2048</dim>
                                    <dim>512</dim>
                                </port>
                                <port id="4">
                                    <dim>2048</dim>
                                    <dim>512</dim>
                                </port>
                                <port id="5">
                                    <dim>2048</dim>
                                </port>
                            </input>
                            <output>
                                <port id="6" precision="FP32">
                                    <dim>1</dim>
                                    <dim>512</dim>
                                </port>
                                <port id="7" precision="FP32">
                                    <dim>1</dim>
                                    <dim>512</dim>
                                </port>
                            </output>
                        </layer>
                        <layer id="9" name="471/outport/0/sink_port_0" type="Result" version="opset1">
                            <input>
                                <port id="0">
                                    <dim>1</dim>
                                    <dim>512</dim>
                                </port>
                            </input>
                        </layer>
                        <layer id="10" name="471/outport/1/sink_port_0" type="Result" version="opset1">
                            <input>
                                <port id="0">
                                    <dim>1</dim>
                                    <dim>512</dim>
                                </port>
                            </input>
                        </layer>
                        <layer id="11" name="15_const" type="Const" version="opset1">
                            <data offset="8396816" size="24"/>
                            <output>
                                <port id="1" precision="I64">
                                    <dim>3</dim>
                                </port>
                            </output>
                        </layer>
                        <layer id="12" name="471output_unsqueeze" type="Reshape" version="opset1">
                            <data special_zero="True"/>
                            <input>
                                <port id="0">
                                    <dim>1</dim>
                                    <dim>512</dim>
                                </port>
                                <port id="1">
                                    <dim>3</dim>
                                </port>
                            </input>
                            <output>
                                <port id="2" precision="FP32">
                                    <dim>1</dim>
                                    <dim>1</dim>
                                    <dim>512</dim>
                                </port>
                            </output>
                        </layer>
                        <layer id="13" name="18/sink_port_0" type="Result" version="opset1">
                            <input>
                                <port id="0">
                                    <dim>1</dim>
                                    <dim>1</dim>
                                    <dim>512</dim>
                                </port>
                            </input>
                        </layer>
                    </layers>
                    <edges>
                        <edge from-layer="0" from-port="0" to-layer="2" to-port="0"/>
                        <edge from-layer="1" from-port="1" to-layer="2" to-port="1"/>
                        <edge from-layer="2" from-port="2" to-layer="8" to-port="0"/>
                        <edge from-layer="3" from-port="0" to-layer="8" to-port="1"/>
                        <edge from-layer="4" from-port="0" to-layer="8" to-port="2"/>
                        <edge from-layer="5" from-port="1" to-layer="8" to-port="3"/>
                        <edge from-layer="6" from-port="1" to-layer="8" to-port="4"/>
                        <edge from-layer="7" from-port="1" to-layer="8" to-port="5"/>
                        <edge from-layer="8" from-port="6" to-layer="9" to-port="0"/>
                        <edge from-layer="8" from-port="7" to-layer="10" to-port="0"/>
                        <edge from-layer="8" from-port="6" to-layer="12" to-port="0"/>
                        <edge from-layer="11" from-port="1" to-layer="12" to-port="1"/>
                        <edge from-layer="12" from-port="2" to-layer="13" to-port="0"/>
                    </edges>
                </body>
            </layer>
            <layer id="4" name="result" type="Result" version="opset1">
                <input>
                    <port id="0">
                        <dim>16</dim>
                        <dim>1</dim>
                        <dim>512</dim>
                    </port>
                </input>
            </layer>
            <layer id="5" name="result_2" type="Result" version="opset1">
                <input>
                    <port id="0">
                        <dim>1</dim>
                        <dim>512</dim>
                    </port>
                </input>
            </layer>
            <layer id="6" name="result_3" type="Result" version="opset1">
                <input>
                    <port id="0">
                        <dim>1</dim>
                        <dim>512</dim>
                    </port>
                </input>
            </layer>
        </layers>
        <edges>
            <edge from-layer="0" from-port="0" to-layer="3" to-port="0"/>
            <edge from-layer="1" from-port="0" to-layer="3" to-port="1"/>
            <edge from-layer="2" from-port="0" to-layer="3" to-port="2"/>
            <edge from-layer="3" from-port="3" to-layer="4" to-port="0"/>
            <edge from-layer="3" from-port="4" to-layer="5" to-port="0"/>
            <edge from-layer="3" from-port="5" to-layer="6" to-port="0"/>
        </edges>
    </net>
    )V0G0N";
    std::string model_v6 = R"V0G0N(
    <net name="Resnet" version="7">
        <layers>
            <layer id="0" name="data1" precision="FP32" type="Input">
                <output>
                    <port id="0" precision="FP32">
                        <dim>16</dim>
                        <dim>1</dim>
                        <dim>512</dim>
                    </port>
                </output>
            </layer>
            <layer id="1" name="data2" precision="FP32" type="Input">
                <output>
                    <port id="0" precision="FP32">
                        <dim>1</dim>
                        <dim>512</dim>
                    </port>
                </output>
            </layer>
            <layer id="2" name="data3" precision="FP32" type="Input">
                <output>
                    <port id="0" precision="FP32">
                        <dim>1</dim>
                        <dim>512</dim>
                    </port>
                </output>
            </layer>
            <layer id="3" name="471/TensorIterator" precision="FP32" type="TensorIterator">
                <input>
                    <port id="0">
                        <dim>16</dim>
                        <dim>1</dim>
                        <dim>512</dim>
                    </port>
                    <port id="1">
                        <dim>1</dim>
                        <dim>512</dim>
                    </port>
                    <port id="2">
                        <dim>1</dim>
                        <dim>512</dim>
                    </port>
                </input>
                <output>
                    <port id="3" precision="FP32">
                        <dim>16</dim>
                        <dim>1</dim>
                        <dim>512</dim>
                    </port>
                    <port id="4" precision="FP32">
                        <dim>1</dim>
                        <dim>512</dim>
                    </port>
                    <port id="5" precision="FP32">
                        <dim>1</dim>
                        <dim>512</dim>
                    </port>
                </output>
                <port_map>
                    <input axis="0" external_port_id="0" internal_layer_id="1" internal_port_id="0" part_size="1" stride="1"/>
                    <input external_port_id="1" internal_layer_id="2" internal_port_id="1"/>
                    <input external_port_id="2" internal_layer_id="2" internal_port_id="2"/>
                    <output axis="0" external_port_id="3" internal_layer_id="4" internal_port_id="2" part_size="1" stride="1"/>
                    <output external_port_id="4" internal_layer_id="2" internal_port_id="5"/>
                    <output external_port_id="5" internal_layer_id="2" internal_port_id="6"/>
                </port_map>
                <back_edges>
                    <edge from-layer="2" from-port="5" to-layer="2" to-port="1"/>
                    <edge from-layer="2" from-port="6" to-layer="2" to-port="2"/>
                </back_edges>
                <body>
                    <layers>
                        <layer id="0" name="7_const" type="Const">
                            <output>
                                <port id="1" precision="I64">
                                    <dim>2</dim>
                                </port>
                            </output>
                            <blobs>
                                <custom offset="0" precision="I64" size="16"/>
                            </blobs>
                        </layer>
                        <layer id="1" name="471/input_squeeze" precision="FP32" type="Reshape">
                            <input>
                                <port id="0">
                                    <dim>1</dim>
                                    <dim>1</dim>
                                    <dim>512</dim>
                                </port>
                                <port id="1">
                                    <dim>2</dim>
                                </port>
                            </input>
                            <output>
                                <port id="2" precision="FP32">
                                    <dim>1</dim>
                                    <dim>512</dim>
                                </port>
                            </output>
                        </layer>
                        <layer id="2" name="471/LSTMCell" precision="FP32" type="LSTMCell">
                            <data hidden_size="512" activations="sigmoid,tanh,tanh" activations_alpha="" activations_beta="" clip="0"/>
                            <input>
                                <port id="0">
                                    <dim>1</dim>
                                    <dim>512</dim>
                                </port>
                                <port id="1">
                                    <dim>1</dim>
                                    <dim>512</dim>
                                </port>
                                <port id="2">
                                    <dim>1</dim>
                                    <dim>512</dim>
                                </port>
                            </input>
                            <output>
                                <port id="5" precision="FP32">
                                    <dim>1</dim>
                                    <dim>512</dim>
                                </port>
                                <port id="6" precision="FP32">
                                    <dim>1</dim>
                                    <dim>512</dim>
                                </port>
                            </output>
                            <blobs>
                                <weights offset="16" precision="FP32" size="8388608"/>
                                <biases offset="8388620" precision="FP32" size="8192"/>
                            </blobs>
                        </layer>
                        <layer id="3" name="15_const" type="Const">
                            <output>
                                <port id="1" precision="I64">
                                    <dim>3</dim>
                                </port>
                            </output>
                            <blobs>
                                <custom offset="8396812" precision="I64" size="24"/>
                            </blobs>
                        </layer>
                        <layer id="4" name="471output_unsqueeze" precision="FP32" type="Reshape">
                            <input>
                                <port id="0">
                                    <dim>1</dim>
                                    <dim>512</dim>
                                </port>
                                <port id="1">
                                    <dim>3</dim>
                                </port>
                            </input>
                            <output>
                                <port id="2" precision="FP32">
                                    <dim>1</dim>
                                    <dim>1</dim>
                                    <dim>512</dim>
                                </port>
                            </output>
                        </layer>
                    </layers>
                    <edges>
                        <edge from-layer="0" from-port="1" to-layer="1" to-port="1"/>
                        <edge from-layer="1" from-port="2" to-layer="2" to-port="0"/>
                        <edge from-layer="2" from-port="5" to-layer="4" to-port="0"/>
                        <edge from-layer="3" from-port="1" to-layer="4" to-port="1"/>
                    </edges>
                </body>
            </layer>
        </layers>
        <edges>
            <edge from-layer="0" from-port="0" to-layer="3" to-port="0"/>
            <edge from-layer="1" from-port="0" to-layer="3" to-port="1"/>
            <edge from-layer="2" from-port="0" to-layer="3" to-port="2"/>
        </edges>
    </net>
    )V0G0N";

    compareIRs(model_v10, model_v6, 8396840, [](Blob::Ptr& weights) {
        auto *data = weights->buffer().as<int64_t *>();
        data[0] = 1;
        data[1] = 512;

        data[1049602] = 1;
        data[1049603] = 1;
        data[1049604] = 512;
    });
}

TEST_F(NGraphReaderTests, ReadTensorIteratorNetwork_negative_stride_opset4) {
    std::string model_v10 = R"V0G0N(
    <net name="Transpose" version="10">
        <layers>
            <layer id="0" name="data1" type="Parameter" version="opset1">
                <data element_type="f32" shape="1,25,512"/>
                <output>
                    <port id="0" precision="FP32">
                        <dim>1</dim>
                        <dim>25</dim>
                        <dim>512</dim>
                    </port>
                </output>
            </layer>
            <layer id="1" name="data2" type="Parameter" version="opset1">
                <data element_type="f32" shape="1,256"/>
                <output>
                    <port id="0" precision="FP32">
                        <dim>1</dim>
                        <dim>256</dim>
                    </port>
                </output>
            </layer>
            <layer id="2" name="data3" type="Parameter" version="opset1">
                <data element_type="f32" shape="1,256"/>
                <output>
                    <port id="0" precision="FP32">
                        <dim>1</dim>
                        <dim>256</dim>
                    </port>
                </output>
            </layer>
            <layer id="3" name="TensorIterator" type="TensorIterator" version="opset1">
                <input>
                    <port id="0">
                        <dim>1</dim>
                        <dim>25</dim>
                        <dim>512</dim>
                    </port>
                    <port id="1">
                        <dim>1</dim>
                        <dim>256</dim>
                    </port>
                    <port id="2">
                        <dim>1</dim>
                        <dim>256</dim>
                    </port>
                </input>
                <output>
                    <port id="3" precision="FP32">
                        <dim>1</dim>
                        <dim>25</dim>
                        <dim>256</dim>
                    </port>
                </output>
                <port_map>
                    <input axis="1" end="0" external_port_id="0" internal_layer_id="0" start="-1" stride="-1"/>
                    <input external_port_id="1" internal_layer_id="3"/>
                    <input external_port_id="2" internal_layer_id="4"/>
                    <output axis="1" end="0" external_port_id="3" internal_layer_id="13" start="-1" stride="-1"/>
                </port_map>
                <back_edges>
                    <edge from-layer="10" to-layer="3"/>
                    <edge from-layer="9" to-layer="4"/>
                </back_edges>
                <body>
                    <layers>
                        <layer id="0" name="32" type="Parameter" version="opset1">
                            <data element_type="f32" shape="1,1,512"/>
                            <output>
                                <port id="0" precision="FP32">
                                    <dim>1</dim>
                                    <dim>1</dim>
                                    <dim>512</dim>
                                </port>
                            </output>
                        </layer>
                        <layer id="1" name="25_const" type="Const" version="opset1">
                            <data offset="0" size="16"/>
                            <output>
                                <port id="1" precision="I64">
                                    <dim>2</dim>
                                </port>
                            </output>
                        </layer>
                        <layer id="2" name="shadow/LSTMLayers/stack_bidirectional_rnn/cell_1/bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Output_0/Data_/InputSqueeze" type="Reshape" version="opset1">
                            <data special_zero="True"/>
                            <input>
                                <port id="0">
                                    <dim>1</dim>
                                    <dim>1</dim>
                                    <dim>512</dim>
                                </port>
                                <port id="1">
                                    <dim>2</dim>
                                </port>
                            </input>
                            <output>
                                <port id="2" precision="FP32">
                                    <dim>1</dim>
                                    <dim>512</dim>
                                </port>
                            </output>
                        </layer>
                        <layer id="3" name="34" type="Parameter" version="opset1">
                            <data element_type="f32" shape="1,256"/>
                            <output>
                                <port id="0" precision="FP32">
                                    <dim>1</dim>
                                    <dim>256</dim>
                                </port>
                            </output>
                        </layer>
                        <layer id="4" name="36" type="Parameter" version="opset1">
                            <data element_type="f32" shape="1,256"/>
                            <output>
                                <port id="0" precision="FP32">
                                    <dim>1</dim>
                                    <dim>256</dim>
                                </port>
                            </output>
                        </layer>
                        <layer id="5" name="shadow/LSTMLayers/stack_bidirectional_rnn/cell_1/bidirectional_rnn/bw/bw/while/bw/basic_lstm_cell/concat/LSTMCell/Split269_const" type="Const" version="opset1">
                            <data offset="16" size="2097152"/>
                            <output>
                                <port id="1" precision="FP32">
                                    <dim>1024</dim>
                                    <dim>512</dim>
                                </port>
                            </output>
                        </layer>
                        <layer id="6" name="shadow/LSTMLayers/stack_bidirectional_rnn/cell_1/bidirectional_rnn/bw/bw/while/bw/basic_lstm_cell/concat/LSTMCell/Split270_const" type="Const" version="opset1">
                            <data offset="2097168" size="1048576"/>
                            <output>
                                <port id="1" precision="FP32">
                                    <dim>1024</dim>
                                    <dim>256</dim>
                                </port>
                            </output>
                        </layer>
                        <layer id="7" name="shadow/LSTMLayers/stack_bidirectional_rnn/cell_1/bidirectional_rnn/bw/bw/while/bw/basic_lstm_cell/BiasAdd/Enter/Output_0/Data__const" type="Const" version="opset1">
                            <data offset="3145744" size="4096"/>
                            <output>
                                <port id="1" precision="FP32">
                                    <dim>1024</dim>
                                </port>
                            </output>
                        </layer>
                        <layer id="8" name="shadow/LSTMLayers/stack_bidirectional_rnn/cell_1/bidirectional_rnn/bw/bw/while/bw/basic_lstm_cell/concat/LSTMCell" type="LSTMCell" version="opset4">
                            <data hidden_size="256"/>
                            <input>
                                <port id="0">
                                    <dim>1</dim>
                                    <dim>512</dim>
                                </port>
                                <port id="1">
                                    <dim>1</dim>
                                    <dim>256</dim>
                                </port>
                                <port id="2">
                                    <dim>1</dim>
                                    <dim>256</dim>
                                </port>
                                <port id="3">
                                    <dim>1024</dim>
                                    <dim>512</dim>
                                </port>
                                <port id="4">
                                    <dim>1024</dim>
                                    <dim>256</dim>
                                </port>
                                <port id="5">
                                    <dim>1024</dim>
                                </port>
                            </input>
                            <output>
                                <port id="6" precision="FP32">
                                    <dim>1</dim>
                                    <dim>256</dim>
                                </port>
                                <port id="7" precision="FP32">
                                    <dim>1</dim>
                                    <dim>256</dim>
                                </port>
                            </output>
                        </layer>
                        <layer id="9" name="shadow/LSTMLayers/stack_bidirectional_rnn/cell_1/bidirectional_rnn/bw/bw/while/bw/basic_lstm_cell/concat/LSTMCell/Output_1/Data_/sink_port_0" type="Result" version="opset1">
                            <input>
                                <port id="0">
                                    <dim>1</dim>
                                    <dim>256</dim>
                                </port>
                            </input>
                        </layer>
                        <layer id="10" name="shadow/LSTMLayers/stack_bidirectional_rnn/cell_1/bidirectional_rnn/bw/bw/while/bw/basic_lstm_cell/concat/LSTMCell/Output_0/Data_/sink_port_0" type="Result" version="opset1">
                            <input>
                                <port id="0">
                                    <dim>1</dim>
                                    <dim>256</dim>
                                </port>
                            </input>
                        </layer>
                        <layer id="11" name="28_const" type="Const" version="opset1">
                            <data offset="3149840" size="24"/>
                            <output>
                                <port id="1" precision="I64">
                                    <dim>3</dim>
                                </port>
                            </output>
                        </layer>
                        <layer id="12" name="shadow/LSTMLayers/stack_bidirectional_rnn/cell_1/bidirectional_rnn/bw/bw/while/bw/basic_lstm_cell/concat/LSTMCell/Output_0/Data_/OutputUnsqueeze" type="Reshape" version="opset1">
                            <data special_zero="True"/>
                            <input>
                                <port id="0">
                                    <dim>1</dim>
                                    <dim>256</dim>
                                </port>
                                <port id="1">
                                    <dim>3</dim>
                                </port>
                            </input>
                            <output>
                                <port id="2" precision="FP32">
                                    <dim>1</dim>
                                    <dim>1</dim>
                                    <dim>256</dim>
                                </port>
                            </output>
                        </layer>
                        <layer id="13" name="30/sink_port_0" type="Result" version="opset1">
                            <input>
                                <port id="0">
                                    <dim>1</dim>
                                    <dim>1</dim>
                                    <dim>256</dim>
                                </port>
                            </input>
                        </layer>
                    </layers>
                    <edges>
                        <edge from-layer="0" from-port="0" to-layer="2" to-port="0"/>
                        <edge from-layer="1" from-port="1" to-layer="2" to-port="1"/>
                        <edge from-layer="2" from-port="2" to-layer="8" to-port="0"/>
                        <edge from-layer="3" from-port="0" to-layer="8" to-port="1"/>
                        <edge from-layer="4" from-port="0" to-layer="8" to-port="2"/>
                        <edge from-layer="5" from-port="1" to-layer="8" to-port="3"/>
                        <edge from-layer="6" from-port="1" to-layer="8" to-port="4"/>
                        <edge from-layer="7" from-port="1" to-layer="8" to-port="5"/>
                        <edge from-layer="8" from-port="7" to-layer="9" to-port="0"/>
                        <edge from-layer="8" from-port="6" to-layer="10" to-port="0"/>
                        <edge from-layer="8" from-port="6" to-layer="12" to-port="0"/>
                        <edge from-layer="11" from-port="1" to-layer="12" to-port="1"/>
                        <edge from-layer="12" from-port="2" to-layer="13" to-port="0"/>
                    </edges>
                </body>
            </layer>
            <layer id="4" name="result" type="Result" version="opset1">
                <input>
                    <port id="0">
                        <dim>1</dim>
                        <dim>25</dim>
                        <dim>256</dim>
                    </port>
                </input>
            </layer>
        </layers>
        <edges>
            <edge from-layer="0" from-port="0" to-layer="3" to-port="0"/>
            <edge from-layer="1" from-port="0" to-layer="3" to-port="1"/>
            <edge from-layer="2" from-port="0" to-layer="3" to-port="2"/>
            <edge from-layer="3" from-port="3" to-layer="4" to-port="0"/>
        </edges>
    </net>
    )V0G0N";
    std::string model_v6 = R"VOGON(
    <net name="Transpose" version="6">
        <layers>
            <layer id="0" precision="FP32" name="data1" type="Input">
                <output>
                    <port id="0" precision="FP32">
                        <dim>1</dim>
                        <dim>25</dim>
                        <dim>512</dim>
                    </port>
                </output>
            </layer>
            <layer id="1" precision="FP32" name="data2" type="Input">
                <output>
                    <port id="0" precision="FP32">
                        <dim>1</dim>
                        <dim>256</dim>
                    </port>
                </output>
            </layer>
            <layer id="2" precision="FP32" name="data3" type="Input">
                <output>
                    <port id="0" precision="FP32">
                        <dim>1</dim>
                        <dim>256</dim>
                    </port>
                </output>
            </layer>
		    <layer id="3" name="TensorIterator" precision="FP32" type="TensorIterator">
                <input>
                    <port id="0">
                        <dim>1</dim>
                        <dim>25</dim>
                        <dim>512</dim>
                    </port>
                    <port id="1">
                        <dim>1</dim>
                        <dim>256</dim>
                    </port>
                    <port id="2">
                        <dim>1</dim>
                        <dim>256</dim>
                    </port>
                </input>
                <output>
                    <port id="3" precision="FP32">
                        <dim>1</dim>
                        <dim>25</dim>
                        <dim>256</dim>
                    </port>
                </output>
                <port_map>
                    <input axis="1" end="0" external_port_id="0" internal_layer_id="1" internal_port_id="0" start="-1" stride="-1"/>
                    <input external_port_id="1" internal_layer_id="2" internal_port_id="1"/>
                    <input external_port_id="2" internal_layer_id="2" internal_port_id="2"/>
                    <output axis="1" end="0" external_port_id="3" internal_layer_id="4" internal_port_id="2" start="-1" stride="-1"/>
                </port_map>
                <back_edges>
                    <edge from-layer="2" from-port="5" to-layer="2" to-port="1"/>
                    <edge from-layer="2" from-port="6" to-layer="2" to-port="2"/>
                </back_edges>
                <body>
                    <layers>
                        <layer id="0" name="25_const" type="Const">
                            <output>
                                <port id="1" precision="I64">
                                    <dim>2</dim>
                                </port>
                            </output>
                            <blobs>
                                <custom offset="0" precision="I64" size="16"/>
                            </blobs>
                        </layer>
                        <layer id="1" precision="FP32" name="shadow/LSTMLayers/stack_bidirectional_rnn/cell_1/bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Output_0/Data_/InputSqueeze" type="Reshape">
                            <input>
                                <port id="0">
                                    <dim>1</dim>
                                    <dim>1</dim>
                                    <dim>512</dim>
                                </port>
                                <port id="1">
                                    <dim>2</dim>
                                </port>
                            </input>
                            <output>
                                <port id="2" precision="FP32">
                                    <dim>1</dim>
                                    <dim>512</dim>
                                </port>
                            </output>
                        </layer>
                        <layer id="2" precision="FP32" name="shadow/LSTMLayers/stack_bidirectional_rnn/cell_1/bidirectional_rnn/bw/bw/while/bw/basic_lstm_cell/concat/LSTMCell" type="LSTMCell">
                            <data hidden_size="256" activations="sigmoid,tanh,tanh" activations_alpha="" activations_beta="" clip="0"/>
                            <input>
                                <port id="0">
                                    <dim>1</dim>
                                    <dim>512</dim>
                                </port>
                                <port id="1">
                                    <dim>1</dim>
                                    <dim>256</dim>
                                </port>
                                <port id="2">
                                    <dim>1</dim>
                                    <dim>256</dim>
                                </port>
                            </input>
                            <output>
                                <port id="5" precision="FP32">
                                    <dim>1</dim>
                                    <dim>256</dim>
                                </port>
                                <port id="6" precision="FP32">
                                    <dim>1</dim>
                                    <dim>256</dim>
                                </port>
                            </output>
                            <blobs>
                                <weights offset="16" precision="FP32" size="3145728"/>
                                <biases offset="3145744" precision="FP32" size="4096"/>
                            </blobs>
                        </layer>
                        <layer id="3" name="28_const" type="Const">
                            <output>
                                <port id="1" precision="I64">
                                    <dim>3</dim>
                                </port>
                            </output>
                            <blobs>
                                <custom offset="3149840" precision="I64" size="24"/>
                            </blobs>
                        </layer>
                        <layer id="4" precision="FP32" name="shadow/LSTMLayers/stack_bidirectional_rnn/cell_1/bidirectional_rnn/bw/bw/while/bw/basic_lstm_cell/concat/LSTMCell/Output_0/Data_/OutputUnsqueeze" type="Reshape">
                            <input>
                                <port id="0">
                                    <dim>1</dim>
                                    <dim>256</dim>
                                </port>
                                <port id="1">
                                    <dim>3</dim>
                                </port>
                            </input>
                            <output>
                                <port id="2" precision="FP32">
                                    <dim>1</dim>
                                    <dim>1</dim>
                                    <dim>256</dim>
                                </port>
                            </output>
                        </layer>
                    </layers>
                    <edges>
                        <edge from-layer="0" from-port="1" to-layer="1" to-port="1"/>
                        <edge from-layer="1" from-port="2" to-layer="2" to-port="0"/>
                        <edge from-layer="2" from-port="5" to-layer="4" to-port="0"/>
                        <edge from-layer="3" from-port="1" to-layer="4" to-port="1"/>
                    </edges>
                </body>
            </layer>
        </layers>
        <edges>
            <edge from-layer="0" from-port="0" to-layer="3" to-port="0"/>
            <edge from-layer="1" from-port="0" to-layer="3" to-port="1"/>
            <edge from-layer="2" from-port="0" to-layer="3" to-port="2"/>
        </edges>
    </net>
    )VOGON";

    compareIRs(model_v10, model_v6, 3149864, [](Blob::Ptr& weights) {
        auto *data = weights->buffer().as<int64_t *>();
        data[0] = 1;
        data[1] = 512;

        data[393730] = 1;
        data[393731] = 1;
        data[393732] = 256;
    });
}
