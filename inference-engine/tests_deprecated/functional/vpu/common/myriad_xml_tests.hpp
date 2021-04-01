// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "myriad_layers_tests.hpp"

std::string full_model = R"V0G0N(
<net batch="1" name="model" version="2">
	<layers>
		<layer id="6" name="data" precision="FP16" type="Input">
			<output>
				<port id="0">
					<dim>1</dim>
					<dim>3</dim>
					<dim>64</dim>
					<dim>64</dim>
				</port>
			</output>
		</layer>
		<layer id="5" name="conv1_7x7_s2" precision="FP16" type="Convolution">
			<data dilation-x="1" dilation-y="1" group="1" kernel-x="7" kernel-y="7" output="64" pad-x="3" pad-y="3" stride="1,1,2,2" stride-x="2" stride-y="2"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>3</dim>
					<dim>64</dim>
					<dim>64</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>64</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</output>
			<blobs>
				<weights offset="0" size="18816"/>
				<biases offset="18816" size="128"/>
			</blobs>
		</layer>
		<layer id="10" name="conv1_relu_7x7" precision="FP16" type="ReLU">
			<data engine="caffe.ReLUParameter.DEFAULT" negative_slope="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>64</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</output>
		</layer>
		<layer id="4" name="pool1_3x3_s2" precision="FP16" type="Pooling">
			<data exclude-pad="false" kernel-x="3" kernel-y="3" pad-x="0" pad-y="0" pool-method="max" rounding_type="ceil" stride="1,1,2,2" stride-x="2" stride-y="2"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>64</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
		</layer>
		<layer id="16" name="loss3_classifier" precision="FP16" type="FullyConnected">
			<data out-size="1000"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>1000</dim>
				</port>
			</output>
			<blobs>
				<weights offset="18944" size="32768000"/>
				<biases offset="32786944" size="2000"/>
			</blobs>
		</layer>
		<layer id="3" name="ReluFC" precision="FP16" type="ReLU">
			<data engine="caffe.ReLUParameter.DEFAULT" negative_slope="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>1000</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>1000</dim>
				</port>
			</output>
		</layer>
		<layer id="8" name="prob" precision="FP16" type="SoftMax">
			<data axis="1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>1000</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>1000</dim>
				</port>
			</output>
		</layer>
	</layers>
	<edges>
		<edge from-layer="6" from-port="0" to-layer="5" to-port="0"/>
		<edge from-layer="5" from-port="3" to-layer="10" to-port="0"/>
		<edge from-layer="10" from-port="1" to-layer="4" to-port="0"/>
		<edge from-layer="4" from-port="1" to-layer="16" to-port="0"/>
		<edge from-layer="16" from-port="3" to-layer="3" to-port="0"/>
		<edge from-layer="3" from-port="1" to-layer="8" to-port="0"/>
	</edges>
</net>
    )V0G0N";

std::string fcModel = R"V0G0N(
    <net batch="1" name="model" version="2">
	<layers>
		<layer id="10" name="data" precision="FP16" type="Input">
			<output>
				<port id="0">
					<dim>1</dim>
					<dim>3</dim>
					<dim>64</dim>
					<dim>64</dim>
				</port>
			</output>
		</layer>
		<layer id="2" name="conv1_7x7_s2" precision="FP16" type="Convolution">
			<data dilation-x="1" dilation-y="1" group="1" kernel-x="7" kernel-y="7" output="64" pad-x="3" pad-y="3" stride="1,1,2,2" stride-x="2" stride-y="2"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>3</dim>
					<dim>64</dim>
					<dim>64</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>64</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</output>
			<blobs>
				<weights offset="0" size="18816"/>
				<biases offset="18816" size="128"/>
			</blobs>
		</layer>
		<layer id="3" name="conv1_relu_7x7" precision="FP16" type="ReLU">
			<data engine="caffe.ReLUParameter.DEFAULT" negative_slope="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>64</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</output>
		</layer>
		<layer id="4" name="pool1_3x3_s2" precision="FP16" type="Pooling">
			<data exclude-pad="false" kernel-x="3" kernel-y="3" pad-x="0" pad-y="0" pool-method="max" rounding_type="ceil" stride="1,1,2,2" stride-x="2" stride-y="2"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>64</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
		</layer>
		<layer id="6" name="loss3_classifier" precision="FP16" type="FullyConnected">
			<data out-size="1000"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>1000</dim>
				</port>
			</output>
			<blobs>
				<weights offset="18944" size="32768000"/>
				<biases offset="32786944" size="2000"/>
			</blobs>
		</layer>
	</layers>
	<edges>
		<edge from-layer="10" from-port="0" to-layer="2" to-port="0"/>
		<edge from-layer="2" from-port="3" to-layer="3" to-port="0"/>
		<edge from-layer="3" from-port="1" to-layer="4" to-port="0"/>
		<edge from-layer="4" from-port="1" to-layer="6" to-port="0"/>
	</edges>
</net>
        )V0G0N";


std::string reluFcModel = R"V0G0N(
    <net batch="1" name="model" version="2">
	<layers>
		<layer id="7" name="data" precision="FP16" type="Input">
			<output>
				<port id="0">
					<dim>1</dim>
					<dim>3</dim>
					<dim>64</dim>
					<dim>64</dim>
				</port>
			</output>
		</layer>
		<layer id="11" name="conv1_7x7_s2" precision="FP16" type="Convolution">
			<data dilation-x="1" dilation-y="1" group="1" kernel-x="7" kernel-y="7" output="64" pad-x="3" pad-y="3" stride="1,1,2,2" stride-x="2" stride-y="2"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>3</dim>
					<dim>64</dim>
					<dim>64</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>64</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</output>
			<blobs>
				<weights offset="0" size="18816"/>
				<biases offset="18816" size="128"/>
			</blobs>
		</layer>
		<layer id="15" name="conv1_relu_7x7" precision="FP16" type="ReLU">
			<data engine="caffe.ReLUParameter.DEFAULT" negative_slope="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>64</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</output>
		</layer>
		<layer id="6" name="pool1_3x3_s2" precision="FP16" type="Pooling">
			<data exclude-pad="false" kernel-x="3" kernel-y="3" pad-x="0" pad-y="0" pool-method="max" rounding_type="ceil" stride="1,1,2,2" stride-x="2" stride-y="2"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>64</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
		</layer>
		<layer id="4" name="loss3_classifier" precision="FP16" type="FullyConnected">
			<data out-size="1000"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>1000</dim>
				</port>
			</output>
			<blobs>
				<weights offset="18944" size="32768000"/>
				<biases offset="32786944" size="2000"/>
			</blobs>
		</layer>
		<layer id="2" name="ReluFC" precision="FP16" type="ReLU">
			<data engine="caffe.ReLUParameter.DEFAULT" negative_slope="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>1000</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>1000</dim>
				</port>
			</output>
		</layer>
	</layers>
	<edges>
		<edge from-layer="7" from-port="0" to-layer="11" to-port="0"/>
		<edge from-layer="11" from-port="3" to-layer="15" to-port="0"/>
		<edge from-layer="15" from-port="1" to-layer="6" to-port="0"/>
		<edge from-layer="6" from-port="1" to-layer="4" to-port="0"/>
		<edge from-layer="4" from-port="3" to-layer="2" to-port="0"/>
	</edges>
</net>
    )V0G0N";

std::string poolModel = R"V0G0N(
    <net batch="1" name="model" version="2">
	<layers>
		<layer id="5" name="data" precision="FP16" type="Input">
			<output>
				<port id="0">
					<dim>1</dim>
					<dim>3</dim>
					<dim>64</dim>
					<dim>64</dim>
				</port>
			</output>
		</layer>
		<layer id="1" name="conv1_7x7_s2" precision="FP16" type="Convolution">
			<data dilation-x="1" dilation-y="1" group="1" kernel-x="7" kernel-y="7" output="64" pad-x="3" pad-y="3" stride="1,1,2,2" stride-x="2" stride-y="2"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>3</dim>
					<dim>64</dim>
					<dim>64</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>64</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</output>
			<blobs>
				<weights offset="0" size="18816"/>
				<biases offset="18816" size="128"/>
			</blobs>
		</layer>
		<layer id="0" name="conv1_relu_7x7" precision="FP16" type="ReLU">
			<data engine="caffe.ReLUParameter.DEFAULT" negative_slope="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>64</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</output>
		</layer>
		<layer id="7" name="pool1_3x3_s2" precision="FP16" type="Pooling">
			<data exclude-pad="false" kernel-x="3" kernel-y="3" pad-x="0" pad-y="0" pool-method="max" rounding_type="ceil" stride="1,1,2,2" stride-x="2" stride-y="2"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>64</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
		</layer>
	</layers>
	<edges>
		<edge from-layer="5" from-port="0" to-layer="1" to-port="0"/>
		<edge from-layer="1" from-port="3" to-layer="0" to-port="0"/>
		<edge from-layer="0" from-port="1" to-layer="7" to-port="0"/>
	</edges>
</net>
        )V0G0N";

std::string reluConvModel = R"V0G0N(
    <net batch="1" name="model" version="2">
	<layers>
		<layer id="6" name="data" precision="FP16" type="Input">
			<output>
				<port id="0">
					<dim>1</dim>
					<dim>3</dim>
					<dim>64</dim>
					<dim>64</dim>
				</port>
			</output>
		</layer>
		<layer id="7" name="conv1_7x7_s2" precision="FP16" type="Convolution">
			<data dilation-x="1" dilation-y="1" group="1" kernel-x="7" kernel-y="7" output="64" pad-x="3" pad-y="3" stride="1,1,2,2" stride-x="2" stride-y="2"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>3</dim>
					<dim>64</dim>
					<dim>64</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>64</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</output>
			<blobs>
				<weights offset="0" size="18816"/>
				<biases offset="18816" size="128"/>
			</blobs>
		</layer>
		<layer id="2" name="conv1_relu_7x7" precision="FP16" type="ReLU">
			<data engine="caffe.ReLUParameter.DEFAULT" negative_slope="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>64</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</output>
		</layer>
	</layers>
	<edges>
		<edge from-layer="6" from-port="0" to-layer="7" to-port="0"/>
		<edge from-layer="7" from-port="3" to-layer="2" to-port="0"/>
	</edges>
</net>
        )V0G0N";

std::string convModel = R"V0G0N(
    <net batch="1" name="model" version="2">
	<layers>
		<layer id="5" name="data" precision="FP16" type="Input">
			<output>
				<port id="0">
					<dim>1</dim>
					<dim>3</dim>
					<dim>64</dim>
					<dim>64</dim>
				</port>
			</output>
		</layer>
		<layer id="1" name="conv1_7x7_s2" precision="FP16" type="Convolution">
			<data dilation-x="1" dilation-y="1" group="1" kernel-x="7" kernel-y="7" output="64" pad-x="3" pad-y="3" stride="1,1,2,2" stride-x="2" stride-y="2"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>3</dim>
					<dim>64</dim>
					<dim>64</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>64</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</output>
			<blobs>
				<weights offset="0" size="18816"/>
				<biases offset="18816" size="128"/>
			</blobs>
		</layer>
	</layers>
	<edges>
		<edge from-layer="5" from-port="0" to-layer="1" to-port="0"/>
	</edges>
</net>
        )V0G0N";


std::string concatModel = R"V0G0N(
	<net batch="1" name="model" version="2">
	<layers>
		<layer id="8" name="data" precision="FP16" type="Input">
			<output>
				<port id="0">
					<dim>1</dim>
					<dim>3</dim>
					<dim>64</dim>
					<dim>64</dim>
				</port>
			</output>
		</layer>
		<layer id="6" name="inPower" precision="FP16" type="Power">
			<data power="1.0" scale="1.0" shift="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>3</dim>
					<dim>64</dim>
					<dim>64</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>3</dim>
					<dim>64</dim>
					<dim>64</dim>
				</port>
			</output>
		</layer>
		<layer id="9" name="conv1_2" precision="FP16" type="Convolution">
			<data dilation-x="1" dilation-y="1" group="1" kernel-x="7" kernel-y="7" output="16" pad-x="3" pad-y="3" stride="1,1,2,2" stride-x="2" stride-y="2"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>3</dim>
					<dim>64</dim>
					<dim>64</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>16</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</output>
			<blobs>
				<weights offset="64" size="4704"/>
				<biases offset="0" size="32"/>
			</blobs>
		</layer>
		<layer id="4" name="conv1_2_Relu" precision="FP16" type="ReLU">
			<data engine="caffe.ReLUParameter.DEFAULT" negative_slope="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>16</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>16</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</output>
		</layer>
		<layer id="7" name="conv1_1" precision="FP16" type="Convolution">
			<data dilation-x="1" dilation-y="1" group="1" kernel-x="7" kernel-y="7" output="16" pad-x="3" pad-y="3" stride="1,1,2,2" stride-x="2" stride-y="2"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>3</dim>
					<dim>64</dim>
					<dim>64</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>16</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</output>
			<blobs>
				<weights offset="4768" size="4704"/>
				<biases offset="32" size="32"/>
			</blobs>
		</layer>
		<layer id="17" name="conv1_1_Relu" precision="FP16" type="ReLU">
			<data engine="caffe.ReLUParameter.DEFAULT" negative_slope="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>16</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>16</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</output>
		</layer>
		<layer id="13" name="Concat" precision="FP16" type="Concat">
			<data axis="1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>16</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>16</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</input>
			<output>
				<port id="2">
					<dim>1</dim>
					<dim>32</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</output>
		</layer>
		<layer id="14" name="outPower" precision="FP16" type="Power">
			<data power="1.0" scale="1.0" shift="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>32</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>32</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</output>
		</layer>
	</layers>
	<edges>
		<edge from-layer="8" from-port="0" to-layer="6" to-port="0"/>
		<edge from-layer="6" from-port="1" to-layer="9" to-port="0"/>
		<edge from-layer="9" from-port="3" to-layer="4" to-port="0"/>
		<edge from-layer="6" from-port="1" to-layer="7" to-port="0"/>
		<edge from-layer="7" from-port="3" to-layer="17" to-port="0"/>
		<edge from-layer="17" from-port="1" to-layer="13" to-port="0"/>
		<edge from-layer="4" from-port="1" to-layer="13" to-port="1"/>
		<edge from-layer="13" from-port="2" to-layer="14" to-port="0"/>
	</edges>
</net>
	        )V0G0N";


std::string concatModelConv = R"V0G0N(
	<net batch="1" name="model" version="2">
	<layers>
		<layer id="2" name="data" precision="FP16" type="Input">
			<output>
				<port id="0">
					<dim>1</dim>
					<dim>3</dim>
					<dim>64</dim>
					<dim>64</dim>
				</port>
			</output>
		</layer>
		<layer id="3" name="inPower" precision="FP16" type="Power">
			<data power="1.0" scale="1.0" shift="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>3</dim>
					<dim>64</dim>
					<dim>64</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>3</dim>
					<dim>64</dim>
					<dim>64</dim>
				</port>
			</output>
		</layer>
		<layer id="5" name="conv1_2" precision="FP16" type="Convolution">
			<data dilation-x="1" dilation-y="1" group="1" kernel-x="7" kernel-y="7" output="16" pad-x="3" pad-y="3" stride="1,1,2,2" stride-x="2" stride-y="2"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>3</dim>
					<dim>64</dim>
					<dim>64</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>16</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</output>
			<blobs>
				<weights offset="64" size="4704"/>
				<biases offset="0" size="32"/>
			</blobs>
		</layer>
	</layers>
	<edges>
		<edge from-layer="2" from-port="0" to-layer="3" to-port="0"/>
		<edge from-layer="3" from-port="1" to-layer="5" to-port="0"/>
	</edges>
</net>
	)V0G0N";