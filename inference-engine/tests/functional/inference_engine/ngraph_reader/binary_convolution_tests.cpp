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
			<data shape="1,64,56,56" element_type="f32"/>
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>64</dim>
					<dim>56</dim>
					<dim>56</dim>
				</port>
			</output>
		</layer>
		<layer id="1" name="612" type="MaxPool" version="opset1">
			<data strides="2,2" kernel="3,3" pads_begin="1,1" pads_end="1,1" rounding_type="floor"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>56</dim>
					<dim>56</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>64</dim>
					<dim>28</dim>
					<dim>28</dim>
				</port>
			</output>
		</layer>
		<layer id="2" name="Copy_618/Unsqueeze/Output_0/Data__const" type="Const" version="opset1">
			<data offset="0" size="256" shape="1,64,1,1" element_type="f32"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>64</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="3" name="Copy_618/Unsqueeze/Output_0/Data_2155_const" type="Const" version="opset1">
			<data offset="0" size="256" shape="1,64,1,1" element_type="f32"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>64</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="4" name="2123_const" type="Const" version="opset1">
			<data offset="256" size="4" shape="1,1,1,1" element_type="f32"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="5" name="2125_const" type="Const" version="opset1">
			<data offset="260" size="4" shape="1,1,1,1" element_type="f32"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="6" name="620" type="FakeQuantize" version="opset1">
			<data levels="2"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>28</dim>
					<dim>28</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>64</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
				<port id="2">
					<dim>1</dim>
					<dim>64</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
				<port id="3">
					<dim>1</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
				<port id="4">
					<dim>1</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="5" precision="FP32">
					<dim>1</dim>
					<dim>64</dim>
					<dim>28</dim>
					<dim>28</dim>
				</port>
			</output>
		</layer>
		<layer id="7" name="626/Output_0/Data__const" type="Const" version="opset1">
			<data offset="264" size="512" shape="64,64,1,1" element_type="u1"/>
			<output>
				<port id="1" precision="U1">
					<dim>64</dim>
					<dim>64</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="8" name="627" type="BinaryConvolution" version="opset1">
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
		<layer id="9" name="data_mul_2354/copy_const" type="Const" version="opset1">
			<data offset="776" size="4" shape="1,1,1,1" element_type="f32"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="10" name="2265/Fused_Mul_" type="Multiply" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>28</dim>
					<dim>28</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>1</dim>
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
		<layer id="11" name="data_add_2356/copy_const" type="Const" version="opset1">
			<data offset="780" size="256" shape="1,64,1,1" element_type="f32"/>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>64</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="12" name="output" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>28</dim>
					<dim>28</dim>
				</port>
				<port id="1">
					<dim>1</dim>
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
		<layer id="13" name="627/sink_port_0" type="Result" version="opset1">
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
		<edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>
		<edge from-layer="1" from-port="1" to-layer="6" to-port="0"/>
		<edge from-layer="2" from-port="1" to-layer="6" to-port="1"/>
		<edge from-layer="3" from-port="1" to-layer="6" to-port="2"/>
		<edge from-layer="4" from-port="1" to-layer="6" to-port="3"/>
		<edge from-layer="5" from-port="1" to-layer="6" to-port="4"/>
		<edge from-layer="6" from-port="5" to-layer="8" to-port="0"/>
		<edge from-layer="7" from-port="1" to-layer="8" to-port="1"/>
		<edge from-layer="8" from-port="2" to-layer="10" to-port="0"/>
		<edge from-layer="9" from-port="1" to-layer="10" to-port="1"/>
		<edge from-layer="10" from-port="2" to-layer="12" to-port="0"/>
		<edge from-layer="11" from-port="1" to-layer="12" to-port="1"/>
		<edge from-layer="12" from-port="2" to-layer="13" to-port="0"/>
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
					<dim>56</dim>
					<dim>56</dim>
				</port>
			</output>
		</layer>
		<layer id="1" name="612" type="Pooling" version="opset1">
			<data strides="2,2" kernel="3,3" pads_begin="1,1" pads_end="1,1" pool-method="max" exclude-pad="true" rounding_type="floor"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>56</dim>
					<dim>56</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>64</dim>
					<dim>28</dim>
					<dim>28</dim>
				</port>
			</output>
		</layer>
		<layer id="2" name="Copy_618/Unsqueeze/Output_0/Data__const" type="Const" version="opset1">
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>64</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
			<blobs>
				<custom offset="0" size="256" precision="FP32"/>
			</blobs>
		</layer>
		<layer id="3" name="Copy_618/Unsqueeze/Output_0/Data_2155_const" type="Const" version="opset1">
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>64</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
			<blobs>
				<custom offset="0" size="256" precision="FP32"/>
			</blobs>
		</layer>
		<layer id="4" name="2137_const" type="Const" version="opset1">
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
			<blobs>
				<custom offset="256" size="4" precision="FP32"/>
			</blobs>
		</layer>
		<layer id="5" name="2139_const" type="Const" version="opset1">
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
			<blobs>
				<custom offset="260" size="4" precision="FP32"/>
			</blobs>
		</layer>
		<layer id="6" name="620" type="FakeQuantize" version="opset1">
			<data levels="2"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>28</dim>
					<dim>28</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>64</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
				<port id="2">
					<dim>1</dim>
					<dim>64</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
				<port id="3">
					<dim>1</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
				<port id="4">
					<dim>1</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="5" precision="FP32">
					<dim>1</dim>
					<dim>64</dim>
					<dim>28</dim>
					<dim>28</dim>
				</port>
			</output>
		</layer>
		<layer id="7" name="627" type="BinaryConvolution" version="opset1">
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
		<layer id="8" name="output" type="ScaleShift">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>28</dim>
					<dim>28</dim>
				</port>
			</input>
			<output>
				<port id="3" precision="FP32">
					<dim>1</dim>
					<dim>64</dim>
					<dim>28</dim>
					<dim>28</dim>
				</port>
			</output>
			<blobs>
				<weights offset="776" size="256" precision="FP32"/>
				<biases offset="1032" size="256" precision="FP32"/>
			</blobs>
		</layer>
	</layers>
	<edges>
		<edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>
		<edge from-layer="1" from-port="1" to-layer="6" to-port="0"/>
		<edge from-layer="2" from-port="1" to-layer="6" to-port="1"/>
		<edge from-layer="3" from-port="1" to-layer="6" to-port="2"/>
		<edge from-layer="4" from-port="1" to-layer="6" to-port="3"/>
		<edge from-layer="5" from-port="1" to-layer="6" to-port="4"/>
		<edge from-layer="6" from-port="5" to-layer="7" to-port="0"/>
		<edge from-layer="7" from-port="2" to-layer="8" to-port="0"/>
	</edges>
</net>
)V0G0N";
    compareIRs(model, modelV7, 1288);
}
