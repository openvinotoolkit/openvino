// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include "ngraph_reader_tests.hpp"
TEST_F(NGraphReaderTests, ReadMinimumNetwork) {
    std::string model = R"V0G0N(
<net name="saved_model" version="10">
	<layers>
		<layer id="0" name="input_a" type="Parameter" version="opset1">
			<data shape="1,1,27,27" element_type="f32"/>
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>1</dim>
					<dim>27</dim>
					<dim>27</dim>
				</port>
			</output>
		</layer>
		<layer id="1" name="input_b" type="Parameter" version="opset1">
			<data shape="1,1,27,27" element_type="f32"/>
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>1</dim>
					<dim>27</dim>
					<dim>27</dim>
				</port>
			</output>
		</layer>
		<layer id="2" name="PartitionedCall/functional_1/output/Minimum" type="Minimum" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>1</dim>
					<dim>27</dim>
					<dim>27</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>1</dim>
					<dim>27</dim>
					<dim>27</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>1</dim>
					<dim>27</dim>
					<dim>27</dim>
				</port>
			</output>
		</layer>
		<layer id="3" name="Identity/sink_port_0" type="Result" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>1</dim>
					<dim>27</dim>
					<dim>27</dim>
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
<net name="saved_model" version="7">
	<layers>
		<layer id="0" name="input_a" type="Input" version="opset1">
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>1</dim>
					<dim>27</dim>
					<dim>27</dim>
				</port>
			</output>
		</layer>
		<layer id="1" name="PartitionedCall/functional_1/output/Minimum/negate1_" type="Power" version="opset1">
			<data power="1" scale="-1" shift="0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>1</dim>
					<dim>27</dim>
					<dim>27</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>1</dim>
					<dim>27</dim>
					<dim>27</dim>
				</port>
			</output>
		</layer>
		<layer id="2" name="input_b" type="Input" version="opset1">
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>1</dim>
					<dim>27</dim>
					<dim>27</dim>
				</port>
			</output>
		</layer>
		<layer id="3" name="PartitionedCall/functional_1/output/Minimum/negate2_" type="Power" version="opset1">
			<data power="1" scale="-1" shift="0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>1</dim>
					<dim>27</dim>
					<dim>27</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>1</dim>
					<dim>27</dim>
					<dim>27</dim>
				</port>
			</output>
		</layer>
		<layer id="4" name="PartitionedCall/functional_1/output/Minimum/Max_" type="Eltwise" version="opset1">
			<data operation="max"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>1</dim>
					<dim>27</dim>
					<dim>27</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>1</dim>
					<dim>27</dim>
					<dim>27</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>1</dim>
					<dim>27</dim>
					<dim>27</dim>
				</port>
			</output>
		</layer>
		<layer id="5" name="PartitionedCall/functional_1/output/Minimum" type="Power" version="opset1">
			<data power="1" scale="-1" shift="0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>1</dim>
					<dim>27</dim>
					<dim>27</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>1</dim>
					<dim>27</dim>
					<dim>27</dim>
				</port>
			</output>
		</layer>
	</layers>
	<edges>
		<edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>
		<edge from-layer="2" from-port="0" to-layer="3" to-port="0"/>
		<edge from-layer="1" from-port="1" to-layer="4" to-port="0"/>
		<edge from-layer="3" from-port="1" to-layer="4" to-port="1"/>
		<edge from-layer="4" from-port="2" to-layer="5" to-port="0"/>
	</edges>
</net>
)V0G0N";
    compareIRs(model, modelV7, 0);
}