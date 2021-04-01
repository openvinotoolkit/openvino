// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include "ngraph_reader_tests.hpp"

TEST_F(NGraphReaderTests, ReadDeformableConvolutionNetwork) {
    std::string model = R"V0G0N(
<net name="deformable_convolution" version="10">
	<layers>
		<layer id="0" name="in1" type="Parameter" version="opset1">
			<data shape="1,256,200,336" element_type="f32"/>
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>256</dim>
					<dim>200</dim>
					<dim>336</dim>
				</port>
			</output>
		</layer>
		<layer id="1" name="in2" type="Parameter" version="opset1">
			<data shape="1,18,100,168" element_type="f32"/>
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>18</dim>
					<dim>100</dim>
					<dim>168</dim>
				</port>
			</output>
		</layer>
		<layer id="2" name="ConstVal" type="Const" version="opset1">
			<data offset="0" size="2359296" shape="256,256,3,3" element_type="f32"/>
			<output>
				<port id="0" precision="FP32">
					<dim>256</dim>
					<dim>256</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
	    <layer id="3" name="1090" type="DeformableConvolution" version="opset1">
            <data deformable_group="1" dilations="1,1" group="1" pads_begin="1,1" pads_end="1,1" strides="2,2"/>
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>256</dim>
                    <dim>200</dim>
                    <dim>336</dim>
                </port>
                <port id="1">
                    <dim>1</dim>
                    <dim>18</dim>
                    <dim>100</dim>
                    <dim>168</dim>
                </port>
                <port id="2">
                    <dim>256</dim>
                    <dim>256</dim>
                    <dim>3</dim>
                    <dim>3</dim>
                </port>
            </input>
            <output>
                <port id="3" precision="FP32">
                    <dim>1</dim>
                    <dim>256</dim>
                    <dim>100</dim>
                    <dim>168</dim>
                </port>
            </output>
		</layer>
		<layer id="4" name="output" type="Result" version="opset1">
			<input>
				<port id="0">
                    <dim>1</dim>
                    <dim>256</dim>
                    <dim>100</dim>
                    <dim>168</dim>
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
    std::string modelV7 = R"V0G0N(
<net name="deformable_convolution" version="7">
	<layers>
		<layer id="0" name="in1" type="Input" version="opset1">
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>256</dim>
					<dim>200</dim>
					<dim>336</dim>
				</port>
			</output>
		</layer>
		<layer id="1" name="in2" type="Input" version="opset1">
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>18</dim>
					<dim>100</dim>
					<dim>168</dim>
				</port>
			</output>
		</layer>
		<layer id="3" name="1090" type="DeformableConvolution" version="opset1">
            <data deformable_group="1" dilations="1,1" kernel="3,3" output="256" group="1" pads_begin="1,1" pads_end="1,1" strides="2,2"/>
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>256</dim>
                    <dim>200</dim>
                    <dim>336</dim>
                </port>
                <port id="1">
                    <dim>1</dim>
                    <dim>18</dim>
                    <dim>100</dim>
                    <dim>168</dim>
                </port>
            </input>
            <output>
                <port id="3" precision="FP32">
                    <dim>1</dim>
                    <dim>256</dim>
                    <dim>100</dim>
                    <dim>168</dim>
                </port>
            </output>
			<blobs>
				<custom offset="0" size="2359296" precision="FP32"/>
			</blobs>
        </layer>
	</layers>
	<edges>
		<edge from-layer="0" from-port="0" to-layer="3" to-port="0"/>
		<edge from-layer="1" from-port="0" to-layer="3" to-port="1"/>
	</edges>
</net>
)V0G0N";
    compareIRs(model, modelV7, 2359296);
}