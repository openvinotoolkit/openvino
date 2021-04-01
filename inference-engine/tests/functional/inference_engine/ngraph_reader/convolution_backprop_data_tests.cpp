// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include "ngraph_reader_tests.hpp"

TEST_F(NGraphReaderTests, ReadConvolutionBackpropDataNetwork) {
    std::string model = R"V0G0N(
<net name="ConvBackpropData" version="10">
	<layers>
		<layer id="0" name="170/placeholder_port_0" type="Parameter" version="opset1">
			<data shape="1,1024,23,30" element_type="f32"/>
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>1024</dim>
					<dim>23</dim>
					<dim>30</dim>
				</port>
			</output>
		</layer>
		<layer id="1" name="onnx_initializer_node_up_path.0.up.weight/Output_0/Data__const" type="Const" version="opset1">
			<data offset="0" size="8388608" shape="1024,512,2,2" element_type="f32"/>
			<output>
				<port id="0" precision="FP32">
					<dim>1024</dim>
					<dim>512</dim>
					<dim>2</dim>
					<dim>2</dim>
				</port>
			</output>
		</layer>
		<layer id="2" name="171/WithoutBiases" type="ConvolutionBackpropData" version="opset1">
			<data strides="2,2" dilations="1,1" pads_begin="0,0" pads_end="0,0" output_padding="0,0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>1024</dim>
					<dim>23</dim>
					<dim>30</dim>
				</port>
				<port id="1">
					<dim>1024</dim>
					<dim>512</dim>
					<dim>2</dim>
					<dim>2</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>512</dim>
					<dim>46</dim>
					<dim>60</dim>
				</port>
			</output>
		</layer>
        <layer id="3" name="171/Dims386/copy_const" type="Const" version="opset1">
			<data offset="8388608" size="2048" shape="1,512,1,1" element_type="f32"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>512</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="4" name="171" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>512</dim>
					<dim>46</dim>
					<dim>60</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>512</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>512</dim>
					<dim>46</dim>
					<dim>60</dim>
				</port>
			</output>
		</layer>
		<layer id="5" name="171/sink_port_0" type="Result" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>512</dim>
					<dim>46</dim>
					<dim>60</dim>
				</port>
			</input>
		</layer>
	</layers>
	<edges>
		<edge from-layer="0" from-port="0" to-layer="2" to-port="0"/>
		<edge from-layer="1" from-port="0" to-layer="2" to-port="1"/>
		<edge from-layer="2" from-port="2" to-layer="4" to-port="0"/>
		<edge from-layer="3" from-port="1" to-layer="4" to-port="1"/>
		<edge from-layer="4" from-port="2" to-layer="5" to-port="0"/>
	</edges>
</net>
)V0G0N";
    std::string modelV7 = R"V0G0N(
<net name="ConvBackpropData" version="7">
	<layers>
		<layer id="0" name="170/placeholder_port_0" type="Input" version="opset1">
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>1024</dim>
					<dim>23</dim>
					<dim>30</dim>
				</port>
			</output>
		</layer>
		<layer id="1" name="171" type="Deconvolution" version="opset1">
			<data group="1" strides="2,2" dilations="1,1" kernel="2,2" pads_begin="0,0" pads_end="0,0" output="512"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>1024</dim>
					<dim>23</dim>
					<dim>30</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>512</dim>
					<dim>46</dim>
					<dim>60</dim>
				</port>
			</output>
			<blobs>
				<weights offset="0" size="8388608" precision="FP32"/>
				<biases offset="8388608" size="2048" precision="FP32"/>
			</blobs>
		</layer>
	</layers>
	<edges>
		<edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>
	</edges>
</net>
)V0G0N";
    compareIRs(model, modelV7, 8390656);
}