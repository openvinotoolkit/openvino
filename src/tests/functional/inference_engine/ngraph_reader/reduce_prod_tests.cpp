// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include "ngraph_reader_tests.hpp"
TEST_F(NGraphReaderTests, ReadReduceProdNetwork) {
    std::string model = R"V0G0N(
<net name="saved_model" version="10">
	<layers>
		<layer id="0" name="input_a" type="Parameter" version="opset1">
			<data shape="1,1,4" element_type="f32"/>
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>1</dim>
					<dim>4</dim>
				</port>
			</output>
		</layer>
		<layer id="1" name="PartitionedCall/functional_1/tf_op_layer_output/output/Cast_186_const" type="Const" version="opset1">
			<data offset="0" size="8" shape="1" element_type="i64"/>
			<output>
				<port id="1" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="2" name="PartitionedCall/functional_1/tf_op_layer_output/output" type="ReduceProd" version="opset1">
			<data keep_dims="False"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>1</dim>
					<dim>4</dim>
				</port>
				<port id="1">
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>4</dim>
				</port>
			</output>
		</layer>
		<layer id="3" name="Identity/sink_port_0" type="Result" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>4</dim>
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
<net name="saved_model" version="7">
	<layers>
		<layer id="0" name="input_a" type="Input" version="opset1">
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>1</dim>
					<dim>4</dim>
				</port>
			</output>
		</layer>
		<layer id="1" name="PartitionedCall/functional_1/tf_op_layer_output/output/Cast_195_const" type="Const" version="opset1">
			<output>
				<port id="1" precision="I64">
					<dim>1</dim>
				</port>
			</output>
			<blobs>
				<custom offset="0" size="4" precision="I32"/>
			</blobs>
		</layer>
		<layer id="2" name="PartitionedCall/functional_1/tf_op_layer_output/output" type="ReduceProd" version="opset1">
			<data keep_dims="False"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>1</dim>
					<dim>4</dim>
				</port>
				<port id="1">
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>4</dim>
				</port>
			</output>
		</layer>
	</layers>
	<edges>
		<edge from-layer="0" from-port="0" to-layer="2" to-port="0"/>
		<edge from-layer="1" from-port="1" to-layer="2" to-port="1"/>
	</edges>
</net>
)V0G0N";
    compareIRs(model, modelV7, 16, [](Blob::Ptr& weights) {
                auto *buffer = weights->buffer().as<int64_t *>();
                buffer[0] = 1;
            });
}
