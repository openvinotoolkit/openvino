// Copyright (C) 2019-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <string>
#include "ngraph_reader_tests.hpp"
TEST_F(NGraphReaderTests, ReadDepthToSpaceNetwork) {
    std::string model = R"V0G0N(
<net name="saved_model" version="10">
	<layers>
		<layer id="0" name="input_a" type="Parameter" version="opset1">
			<data shape="5,4,28,2" element_type="f32"/>
			<output>
				<port id="0" precision="FP32">
					<dim>5</dim>
					<dim>4</dim>
					<dim>28</dim>
					<dim>2</dim>
				</port>
			</output>
		</layer>
		<layer id="1" name="PartitionedCall/functional_1/tf_op_layer_output/output" type="DepthToSpace" version="opset1">
			<data mode="blocks_first" block_size="2"/>
			<input>
				<port id="0">
					<dim>5</dim>
					<dim>4</dim>
					<dim>28</dim>
					<dim>2</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>5</dim>
					<dim>1</dim>
					<dim>56</dim>
					<dim>4</dim>
				</port>
			</output>
		</layer>
		<layer id="2" name="Identity/sink_port_0" type="Result" version="opset1">
			<input>
				<port id="0">
					<dim>5</dim>
					<dim>1</dim>
					<dim>56</dim>
					<dim>4</dim>
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
<net name="saved_model" version="7">
	<layers>
		<layer id="0" name="input_a" type="Input" version="opset1">
			<output>
				<port id="0" precision="FP32">
					<dim>5</dim>
					<dim>4</dim>
					<dim>28</dim>
					<dim>2</dim>
				</port>
			</output>
		</layer>
		<layer id="1" name="PartitionedCall/functional_1/tf_op_layer_output/output/Reshape_to_6D/Cast_1204_const" type="Const" version="opset1">
			<output>
				<port id="1" precision="I64">
					<dim>6</dim>
				</port>
			</output>
			<blobs>
				<custom offset="0" size="24" precision="I64"/>
			</blobs>
		</layer>
		<layer id="2" name="PartitionedCall/functional_1/tf_op_layer_output/output/Reshape_to_6D" type="Reshape" version="opset1">
			<data special_zero="True"/>
			<input>
				<port id="0">
					<dim>5</dim>
					<dim>4</dim>
					<dim>28</dim>
					<dim>2</dim>
				</port>
				<port id="1">
					<dim>6</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>5</dim>
					<dim>2</dim>
					<dim>2</dim>
					<dim>1</dim>
					<dim>28</dim>
					<dim>2</dim>
				</port>
			</output>
		</layer>
		<layer id="3" name="PartitionedCall/functional_1/tf_op_layer_output/output/Transpose" type="Permute" version="opset1">
			<data order="0,3,4,1,5,2"/>
			<input>
				<port id="0">
					<dim>5</dim>
					<dim>2</dim>
					<dim>2</dim>
					<dim>1</dim>
					<dim>28</dim>
					<dim>2</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>5</dim>
					<dim>1</dim>
					<dim>28</dim>
					<dim>2</dim>
					<dim>2</dim>
					<dim>2</dim>
				</port>
			</output>
		</layer>
		<layer id="4" name="PartitionedCall/functional_1/tf_op_layer_output/output/Reshape_to_4D/Cast_1202_const" type="Const" version="opset1">
			<output>
				<port id="1" precision="I64">
					<dim>4</dim>
				</port>
			</output>
			<blobs>
				<custom offset="24" size="16" precision="I64"/>
			</blobs>
		</layer>
		<layer id="5" name="PartitionedCall/functional_1/tf_op_layer_output/output" type="Reshape" version="opset1">
			<data special_zero="True"/>
			<input>
				<port id="0">
					<dim>5</dim>
					<dim>1</dim>
					<dim>28</dim>
					<dim>2</dim>
					<dim>2</dim>
					<dim>2</dim>
				</port>
				<port id="1">
					<dim>4</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>5</dim>
					<dim>1</dim>
					<dim>56</dim>
					<dim>4</dim>
				</port>
			</output>
		</layer>
	</layers>
	<edges>
		<edge from-layer="0" from-port="0" to-layer="2" to-port="0"/>
		<edge from-layer="1" from-port="1" to-layer="2" to-port="1"/>
		<edge from-layer="2" from-port="2" to-layer="3" to-port="0"/>
		<edge from-layer="3" from-port="1" to-layer="5" to-port="0"/>
		<edge from-layer="4" from-port="1" to-layer="5" to-port="1"/>
	</edges>
</net>
)V0G0N";
    compareIRs(model, modelV7, 80, [](Blob::Ptr& weights) {
                auto* buffer = weights->buffer().as<int64_t*>();
                buffer[0] = 0;
                buffer[1] = 2;
                buffer[2] = 2;
                buffer[3] = 1;
                buffer[4] = 28;
                buffer[5] = 2;
                buffer[7] = 0;
                buffer[7] = 1;
                buffer[8] = 56;
                buffer[9] = 4;
            });
}
