// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <string>
#include "ngraph_reader_tests.hpp"
TEST_F(NGraphReaderTests, ReadFloorModNetwork) {
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
		<layer id="1" name="input_b" type="Parameter" version="opset1">
			<data shape="1" element_type="f32"/>
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="2" name="EltwiseReshapeNormalization/Cast_163_const" type="Const" version="opset1">
			<data offset="0" size="24" shape="3" element_type="i64"/>
			<output>
				<port id="1" precision="I64">
					<dim>3</dim>
				</port>
			</output>
		</layer>
		<layer id="3" name="EltwiseReshapeNormalization" type="Reshape" version="opset1">
			<data special_zero="True"/>
			<input>
				<port id="0">
					<dim>1</dim>
				</port>
				<port id="1">
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="4" name="PartitionedCall/functional_1/tf_op_layer_output/output" type="FloorMod" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>1</dim>
					<dim>4</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>1</dim>
					<dim>4</dim>
				</port>
			</output>
		</layer>
		<layer id="5" name="Identity/sink_port_0" type="Result" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>1</dim>
					<dim>4</dim>
				</port>
			</input>
		</layer>
	</layers>
	<edges>
		<edge from-layer="1" from-port="0" to-layer="3" to-port="0"/>
		<edge from-layer="2" from-port="1" to-layer="3" to-port="1"/>
		<edge from-layer="0" from-port="0" to-layer="4" to-port="0"/>
		<edge from-layer="3" from-port="2" to-layer="4" to-port="1"/>
		<edge from-layer="4" from-port="2" to-layer="5" to-port="0"/>
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
		<layer id="1" name="input_b" type="Input" version="opset1">
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="2" name="EltwiseReshapeNormalization/Cast_175_const" type="Const" version="opset1">
			<output>
				<port id="1" precision="I64">
					<dim>3</dim>
				</port>
			</output>
			<blobs>
				<custom offset="0" size="12" precision="I32"/>
			</blobs>
		</layer>
		<layer id="3" name="EltwiseReshapeNormalization" type="Reshape" version="opset1">
			<data special_zero="True"/>
			<input>
				<port id="0">
					<dim>1</dim>
				</port>
				<port id="1">
					<dim>3</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="4" name="PartitionedCall/functional_1/tf_op_layer_output/output" type="Eltwise" version="opset1">
			<data operation="floor_mod"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>1</dim>
					<dim>4</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>1</dim>
					<dim>4</dim>
				</port>
			</output>
		</layer>
	</layers>
	<edges>
		<edge from-layer="1" from-port="0" to-layer="3" to-port="0"/>
		<edge from-layer="2" from-port="1" to-layer="3" to-port="1"/>
		<edge from-layer="0" from-port="0" to-layer="4" to-port="0"/>
		<edge from-layer="3" from-port="2" to-layer="4" to-port="1"/>
	</edges>
</net>
)V0G0N";
    // compareIRs(model, modelV7, 0);
    compareIRs(model, modelV7, 40, [](Blob::Ptr& weights) {
                auto* buffer = weights->buffer().as<int64_t*>();
                buffer[0] = 1;
                buffer[1] = 1;
                buffer[2] = 1;
            });
}
