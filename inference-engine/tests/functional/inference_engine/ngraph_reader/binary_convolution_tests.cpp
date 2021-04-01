// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include "ngraph_reader_tests.hpp"

TEST_F(NGraphReaderTests, ReadBinaryConvolutionNetwork) {
    std::string model = R"V0G0N(
<net name="model_bin" version="10">
	<layers>
		<layer id="0" name="612/placeholder_port_0" type="Parameter" version="opset1">
			<data shape="1,64,28,28" element_type="f32"/>
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>64</dim>
					<dim>28</dim>
					<dim>28</dim>
				</port>
			</output>
		</layer>
		<layer id="1" name="626/Output_0/Data__const" type="Const" version="opset1">
			<data offset="264" size="512" shape="64,64,1,1" element_type="u1"/>
			<output>
				<port id="0" precision="U1">
					<dim>64</dim>
					<dim>64</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="2" name="627" type="BinaryConvolution" version="opset1">
			<data strides="1,1" dilations="1,1" pads_begin="0,0" pads_end="0,0" output_padding="0,0" pad_value="-1.0" mode="xnor-popcount"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>28</dim>
					<dim>28</dim>
				</port>
				<port id="1">
					<dim>64</dim>
					<dim>64</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>64</dim>
					<dim>28</dim>
					<dim>28</dim>
				</port>
			</output>
		</layer>
		<layer id="3" name="627/sink_port_0" type="Result" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>28</dim>
					<dim>28</dim>
				</port>
			</input>
		</layer>
	</layers>
	<edges>
		<edge from-layer="0" from-port="0" to-layer="2" to-port="0"/>
		<edge from-layer="1" from-port="0" to-layer="2" to-port="1"/>
		<edge from-layer="2" from-port="2" to-layer="3" to-port="0"/>
	</edges>
</net>
)V0G0N";
    std::string modelV7 = R"V0G0N(
<net name="model_bin" version="7">
	<layers>
		<layer id="0" name="612/placeholder_port_0" type="Input" version="opset1">
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>64</dim>
					<dim>28</dim>
					<dim>28</dim>
				</port>
			</output>
		</layer>
		<layer id="1" name="627" type="BinaryConvolution" version="opset1">
			<data group="1" strides="1,1" dilations="1,1" kernel="1,1" pads_begin="0,0" pads_end="0,0" output="64" pad_value="-1.0" mode="xnor-popcount" input="64"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>28</dim>
					<dim>28</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>64</dim>
					<dim>28</dim>
					<dim>28</dim>
				</port>
			</output>
			<blobs>
				<weights offset="264" size="512" precision="U8"/>
			</blobs>
		</layer>
	</layers>
	<edges>
		<edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>

	</edges>
</net>
)V0G0N";
    compareIRs(model, modelV7, 1288);
}
