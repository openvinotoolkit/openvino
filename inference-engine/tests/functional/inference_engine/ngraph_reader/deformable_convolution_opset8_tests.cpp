// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include "ngraph_reader_tests.hpp"

TEST_F(NGraphReaderTests, ReadDeformableConvolution8Network) {
    std::string model = R"V0G0N(
<net name="deformable_convolution" version="10">
	<layers>
		<layer id="0" name="in1" type="Parameter" version="opset8">
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
		<layer id="1" name="in2" type="Parameter" version="opset8">
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
		<layer id="2" name="in3" type="Parameter" version="opset8">
			<data shape="256,256,3,3" element_type="f32"/>
			<output>
				<port id="0" precision="FP32">
					<dim>256</dim>
					<dim>256</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
	    <layer id="3" name="1090" type="DeformableConvolution" version="opset8">
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
		<layer id="4" name="output" type="Result" version="opset8">
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

    Core reader;
    Blob::Ptr weights;
    weights = make_shared_blob<uint8_t>(TensorDesc(Precision::U8, {8}, Layout::C));
    weights->allocate();
    EXPECT_NO_THROW(reader.ReadNetwork(model, weights));
}

TEST_F(NGraphReaderTests, ReadDeformableConvolution8WithMaskNetwork) {
    std::string model = R"V0G0N(
<net name="deformable_convolution" version="10">
	<layers>
		<layer id="0" name="in1" type="Parameter" version="opset8">
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
		<layer id="1" name="in2" type="Parameter" version="opset8">
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
		<layer id="2" name="in3" type="Parameter" version="opset8">
			<data shape="256,256,3,3" element_type="f32"/>
			<output>
				<port id="0" precision="FP32">
					<dim>256</dim>
					<dim>256</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="5" name="in4" type="Parameter" version="opset8">
			<data shape="1,9,100,168" element_type="f32"/>
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>9</dim>
					<dim>100</dim>
					<dim>168</dim>
				</port>
			</output>
		</layer>
	    <layer id="3" name="1090" type="DeformableConvolution" version="opset8">
            <data deformable_group="1" dilations="1,1" group="1" pads_begin="1,1" pads_end="1,1" strides="2,2" bilinear_interpolation_pad="true"/>
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
                <port id="4">
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
		<layer id="4" name="output" type="Result" version="opset8">
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
		<edge from-layer="5" from-port="0" to-layer="3" to-port="4"/>
	</edges>
</net>
)V0G0N";

    Core reader;
    Blob::Ptr weights;
    weights = make_shared_blob<uint8_t>(TensorDesc(Precision::U8, {8}, Layout::C));
    weights->allocate();
    EXPECT_NO_THROW(reader.ReadNetwork(model, weights));
}