// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include "ngraph_reader_tests.hpp"

TEST_F(NGraphReaderTests, ReadGroupConvolutionNetwork) {
    std::string model = R"V0G0N(
<net name="GroupConvolution" version="10">
	<layers>
		<layer id="0" name="in1" type="Parameter" version="opset1">
			<data shape="1,64,65,51" element_type="f32"/>
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>64</dim>
					<dim>65</dim>
					<dim>51</dim>
				</port>
			</output>
		</layer>
		<layer id="1" name="const_val" type="Const" version="opset1">
			<data offset="0" size="2304" shape="64,1,1,3,3" element_type="f32"/>
			<output>
				<port id="1" precision="FP32">
					<dim>64</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="2" name="GroupConvolutionOp" type="GroupConvolution" version="opset1">
			<data strides="1,1" dilations="1,1" pads_begin="1,1" pads_end="1,1" output_padding="0,0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>65</dim>
					<dim>51</dim>
				</port>
				<port id="1">
					<dim>64</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>64</dim>
					<dim>65</dim>
					<dim>51</dim>
				</port>
			</output>
		</layer>
		<layer id="3" name="140/sink_port_0" type="Result" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>65</dim>
					<dim>51</dim>
				</port>
			</input>
		</layer>
	</layers>
	<edges>
		<edge from-layer="0" from-port="0" to-layer="2" to-port="0"/>
		<edge from-layer="1" from-port="1" to-layer="2" to-port="1"/>
		<edge from-layer="2" from-port="2" to-layer="3" to-port="0"/>
	</edges>
</net>
)V0G0N";
    std::string modelV7 = R"V0G0N(
<net name="GroupConvolution" version="7">
	<layers>
		<layer id="0" name="in1" type="Input" version="opset1">
			<data shape="1,64,65,51" element_type="f32"/>
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>64</dim>
					<dim>65</dim>
					<dim>51</dim>
				</port>
			</output>
		</layer>
		<layer id="1" name="GroupConvolutionOp" type="Convolution" version="opset1">
			<data group="64" strides="1,1" dilations="1,1" kernel="3,3" pads_begin="1,1" pads_end="1,1" output="64"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>65</dim>
					<dim>51</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>64</dim>
					<dim>65</dim>
					<dim>51</dim>
				</port>
			</output>
			<blobs>
				<weights offset="0" size="2304" precision="FP32"/>
			</blobs>
		</layer>
	</layers>
	<edges>
		<edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>
	</edges>
</net>
)V0G0N";
    compareIRs(model, modelV7, 2304);
}
