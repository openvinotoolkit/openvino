// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include "ngraph_reader_tests.hpp"

TEST_F(NGraphReaderTests, ReadLSTMCellNetwork) {
    std::string model = R"V0G0N(
<net name="LSTMCell" version="10">
    <layers>
        <layer id="0" name="in0" type="Parameter" version="opset1">
            <data shape="1,512" element_type="f32"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>512</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="in1" type="Parameter" version="opset1">
            <data shape="1,256" element_type="f32"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>256</dim>
                </port>
            </output>
        </layer>
        <layer id="2" name="in2" type="Parameter" version="opset1">
            <data shape="1,256" element_type="f32"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>256</dim>
                </port>
            </output>
        </layer>
        <layer id="3" name="in3" type="Const" version="opset1">
            <data offset="22223012" size="2097152" shape="1024,512" element_type="f32"/>
            <output>
                <port id="1" precision="FP32">
                    <dim>1024</dim>
                    <dim>512</dim>
                </port>
            </output>
        </layer>
        <layer id="4" name="in4" type="Const" version="opset1">
            <data offset="24320164" size="1048576" shape="1024,256" element_type="f32"/>
            <output>
                <port id="1" precision="FP32">
                    <dim>1024</dim>
                    <dim>256</dim>
                </port>
            </output>
        </layer>
        <layer id="5" name="in5" type="Const" version="opset1">
            <data offset="25368740" size="4096" shape="1024" element_type="f32"/>
            <output>
                <port id="1" precision="FP32">
                    <dim>1024</dim>
                </port>
            </output>
        </layer>
        <layer id="6" name="LSTMCell" type="LSTMCell" version="opset1" precision="FP32"> 
            <data hidden_size="256" element_type="f32"/>
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>512</dim>
                </port>
                <port id="1" precision="FP32">
                    <dim>1</dim>
                    <dim>256</dim>
                </port>
                <port id="2" precision="FP32">
                    <dim>1</dim>
                    <dim>256</dim>
                </port>
                <port id="3" precision="FP32">
                    <dim>1024</dim>
                    <dim>512</dim>
                </port>
                <port id="4" precision="FP32">
                    <dim>1024</dim>
                    <dim>256</dim>
                </port>
                <port id="5" precision="FP32">
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
		<layer id="7" name="485/sink_port_0" type="Result" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>256</dim>
				</port>
			</input>
		</layer>
		<layer id="8" name="485/sink_port_1" type="Result" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>256</dim>
				</port>
			</input>
		</layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="6" to-port="0"/>
        <edge from-layer="1" from-port="0" to-layer="6" to-port="1"/>
        <edge from-layer="2" from-port="0" to-layer="6" to-port="2"/>
        <edge from-layer="3" from-port="1" to-layer="6" to-port="3"/>
        <edge from-layer="4" from-port="1" to-layer="6" to-port="4"/>
        <edge from-layer="5" from-port="1" to-layer="6" to-port="5"/>
        <edge from-layer="6" from-port="6" to-layer="7" to-port="0"/>
        <edge from-layer="6" from-port="7" to-layer="8" to-port="0"/>
    </edges>
</net>
)V0G0N";
    std::string modelV7 = R"V0G0N(
<net name="LSTMCell" version="7">
    <layers>
        <layer id="0" name="in0" type="Input" version="opset1">
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>512</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="in1" type="Input" version="opset1">
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>256</dim>
                </port>
            </output>
        </layer>
        <layer id="2" name="in2" type="Input" version="opset1">
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>256</dim>
                </port>
            </output>
        </layer>
        <layer id="3" name="LSTMCell" type="LSTMCell" version="opset1" precision="FP32">
            <data hidden_size="256" element_type="f32" activations="sigmoid,tanh,tanh" activations_alpha="" activations_beta="" clip="0.00000"/>
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>512</dim>
                </port>
                <port id="1" precision="FP32">
                    <dim>1</dim>
                    <dim>256</dim>
                </port>
                <port id="2" precision="FP32">
                    <dim>1</dim>
                    <dim>256</dim>
                </port>
            </input>
            <output>
                <port id="3">
                    <dim>1</dim>
                    <dim>256</dim>
                </port>
                <port id="4">
                    <dim>1</dim>
                    <dim>256</dim>
                </port>
            </output>
            <blobs>
                <weights offset="22222928" size="3145728" precision="FP32"/>
                <biases offset="25368656" size="4096" precision="FP32"/>
            </blobs>
        </layer>
    </layers>
    <edges>
		<edge from-layer="0" from-port="0" to-layer="3" to-port="0"/>
		<edge from-layer="1" from-port="0" to-layer="3" to-port="1"/>
		<edge from-layer="2" from-port="0" to-layer="3" to-port="2"/>
    </edges>
</net>
)V0G0N";
    compareIRs(model, modelV7, 26000000);
}
