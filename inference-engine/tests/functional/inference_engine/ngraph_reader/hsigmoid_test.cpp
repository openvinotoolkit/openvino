// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>

#include "ngraph_reader_tests.hpp"

TEST_F(NGraphReaderTests, ReadHSigmoidNetwork) {
    std::string model = R"V0G0N(
<net name="Network" version="10">
    <layers>
        <layer name="in1" type="Parameter" id="0" version="opset1">
            <data element_type="f32" shape="1,3,22,22"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
        <layer name="activation" id="1" type="HSigmoid" version="opset5">
            <input>
                <port id="1" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </input>
            <output>
                <port id="2" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
        <layer name="output" type="Result" id="2" version="opset1">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="1" to-port="1"/>
        <edge from-layer="1" from-port="2" to-layer="2" to-port="0"/>
    </edges>
</net>
)V0G0N";
    std::string modelV5 = R"V0G0N(
<net name="Network" version="5" precision="FP32" batch="1">
    <layers>
		<layer name="in1" type="Input" precision="FP32" id="0">
			<data originalLayersNames="in1" />
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>3</dim>
					<dim>22</dim>
					<dim>22</dim>
				</port>
			</output>
		</layer>
		<layer name="Add_735" type="Power" precision="FP32" id="1">
			<data originalLayersNames="activation" power="1" scale="1" shift="3" />
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>3</dim>
					<dim>22</dim>
					<dim>22</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>3</dim>
					<dim>22</dim>
					<dim>22</dim>
				</port>
			</output>
		</layer>
		<layer name="Relu_736" type="ReLU" precision="FP32" id="2">
			<data originalLayersNames="activation" />
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>3</dim>
					<dim>22</dim>
					<dim>22</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>3</dim>
					<dim>22</dim>
					<dim>22</dim>
				</port>
			</output>
		</layer>
		<layer name="Multiply_742" type="Power" precision="FP32" id="3">
			<data originalLayersNames="activation" power="1" scale="-1" shift="0" />
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>3</dim>
					<dim>22</dim>
					<dim>22</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>3</dim>
					<dim>22</dim>
					<dim>22</dim>
				</port>
			</output>
		</layer>
		<layer name="Multiply_744" type="Const" precision="FP32" id="4">
			<output>
				<port id="0" precision="FP32">
				</port>
			</output>
			<blobs>
				<custom offset="0" size="4" precision="FP32" />
			</blobs>
		</layer>
		<layer name="Maximum_745" type="Eltwise" precision="FP32" id="5">
			<data auto_broadcast="numpy" operation="max" originalLayersNames="activation" />
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>3</dim>
					<dim>22</dim>
					<dim>22</dim>
				</port>
				<port id="1">
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>3</dim>
					<dim>22</dim>
					<dim>22</dim>
				</port>
			</output>
		</layer>
		<layer name="activation" type="Power" precision="FP32" id="6">
			<data originalLayersNames="activation" power="1" scale="-0.166667" shift="0" />
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>3</dim>
					<dim>22</dim>
					<dim>22</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>3</dim>
					<dim>22</dim>
					<dim>22</dim>
				</port>
			</output>
		</layer>
	</layers>
	<edges>
		<edge from-layer="0" from-port="0" to-layer="1" to-port="0" />
		<edge from-layer="1" from-port="1" to-layer="2" to-port="0" />
		<edge from-layer="2" from-port="1" to-layer="3" to-port="0" />
		<edge from-layer="3" from-port="1" to-layer="5" to-port="0" />
		<edge from-layer="4" from-port="0" to-layer="5" to-port="1" />
		<edge from-layer="5" from-port="2" to-layer="6" to-port="0" />
	</edges>
</net>
)V0G0N";

    compareIRs(model, modelV5, 40);
}
