// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include "ngraph_reader_tests.hpp"
TEST_F(NGraphReaderTests, ReadSubtractNetwork) {
    std::string model = R"V0G0N(
<net name="model" version="10">
	<layers>
		<layer id="0" name="x" type="Parameter" version="opset1">
			<data element_type="f32" shape="3,4,5"/>
			<output>
				<port id="0" precision="FP32">
					<dim>3</dim>
					<dim>4</dim>
					<dim>5</dim>
				</port>
			</output>
		</layer>
		<layer id="1" name="y" type="Parameter" version="opset1">
			<data element_type="f32" shape="3,4,5"/>
			<output>
				<port id="0" precision="FP32">
					<dim>3</dim>
					<dim>4</dim>
					<dim>5</dim>
				</port>
			</output>
		</layer>
		<layer id="2" name="z/sub" type="Subtract" version="opset1">
			<input>
				<port id="0">
					<dim>3</dim>
					<dim>4</dim>
					<dim>5</dim>
				</port>
				<port id="1">
					<dim>3</dim>
					<dim>4</dim>
					<dim>5</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>3</dim>
					<dim>4</dim>
					<dim>5</dim>
				</port>
			</output>
		</layer>
		<layer id="3" name="z/sink_port_0" type="Result" version="opset1">
			<input>
				<port id="0">
					<dim>3</dim>
					<dim>4</dim>
					<dim>5</dim>
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
<net name="model" version="7">
	<layers>
		<layer id="0" name="x" type="Input" version="opset1">
			<output>
				<port id="0" precision="FP32">
					<dim>3</dim>
					<dim>4</dim>
					<dim>5</dim>
				</port>
			</output>
		</layer>
		<layer id="1" name="y" type="Input" version="opset1">
			<output>
				<port id="0" precision="FP32">
					<dim>3</dim>
					<dim>4</dim>
					<dim>5</dim>
				</port>
			</output>
		</layer>
		<layer id="2" name="z/neg_" type="Power" version="opset1">
			<data power="1" scale="-1.0" shift="0"/>
			<input>
				<port id="0">
					<dim>3</dim>
					<dim>4</dim>
					<dim>5</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>3</dim>
					<dim>4</dim>
					<dim>5</dim>
				</port>
			</output>
		</layer>
		<layer id="3" name="z/sub" type="Eltwise" version="opset1">
			<data operation="sum"/>
			<input>
				<port id="0">
					<dim>3</dim>
					<dim>4</dim>
					<dim>5</dim>
				</port>
				<port id="1">
					<dim>3</dim>
					<dim>4</dim>
					<dim>5</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>3</dim>
					<dim>4</dim>
					<dim>5</dim>
				</port>
			</output>
		</layer>
	</layers>
	<edges>
		<edge from-layer="1" from-port="0" to-layer="2" to-port="0"/>
		<edge from-layer="0" from-port="0" to-layer="3" to-port="0"/>
		<edge from-layer="2" from-port="1" to-layer="3" to-port="1"/>
	</edges>
</net>
)V0G0N";
    compareIRs(model, modelV7, 0);
}
