// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include "ngraph_reader_tests.hpp"
TEST_F(NGraphReaderTests, ReadLogicalNotNetwork) {
    std::string model = R"V0G0N(
<net name="LogicalNot" version="10">
	<layers>
		<layer id="0" name="input_a" type="Parameter" version="opset1">
			<data shape="256,56" element_type="boolean"/>
			<output>
				<port id="0" precision="BOOL">
					<dim>256</dim>
					<dim>56</dim>
				</port>
			</output>
		</layer>
		<layer id="1" name="LogicalNot" type="LogicalNot" version="opset1">
			<input>
				<port id="0">
					<dim>256</dim>
					<dim>56</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="BOOL">
					<dim>256</dim>
					<dim>56</dim>
				</port>
			</output>
		</layer>
		<layer id="2" name="Identity/sink_port_0" type="Result" version="opset1">
			<input>
				<port id="0">
					<dim>256</dim>
					<dim>56</dim>
				</port>
			</input>
		</layer>
	</layers>
	<edges>
		<edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>
		<edge from-layer="1" from-port="1" to-layer="2" to-port="0"/>
	</edges>
</net>
)V0G0N";
    std::string modelV7 = R"V0G0N(
<net name="LogicalNot" version="7">
	<layers>
		<layer id="0" name="input_a" type="Input" version="opset1">
			<output>
				<port id="0" precision="BOOL">
					<dim>256</dim>
					<dim>56</dim>
				</port>
			</output>
		</layer>
		<layer id="1" name="LogicalNot" type="Activation" version="opset1">
			<data type="not"/>
			<input>
				<port id="0">
					<dim>256</dim>
					<dim>56</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="BOOL">
					<dim>256</dim>
					<dim>56</dim>
				</port>
			</output>
		</layer>
	</layers>
	<edges>
		<edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>
	</edges>
</net>
)V0G0N";

    compareIRs(model, modelV7);
}
