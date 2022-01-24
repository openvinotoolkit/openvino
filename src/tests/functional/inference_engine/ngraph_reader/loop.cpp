// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include "ngraph_reader_tests.hpp"
#include "common_test_utils/data_utils.hpp"

TEST_F(NGraphReaderTests, ReadLoopNetwork) {
    std::string model = R"V0G0N(
<net name="yolov3-10" version="10">
	<layers>
		<layer id="0" name="TFNodes/yolo_evaluation_layer_1/arange_4__77_trip_cnt/placeholder_port_0" type="Parameter" version="opset1">
			<data element_type="f32" shape=""/>
			<output>
				<port id="0" precision="FP32"/>
			</output>
		</layer>
		<layer id="1" name="TFNodes/yolo_evaluation_layer_1/arange_4__77_trip_cnt" type="Convert" version="opset1">
			<data destination_type="i64"/>
			<input>
				<port id="0"/>
			</input>
			<output>
				<port id="1" precision="I64"/>
			</output>
		</layer>
		<layer id="2" name="TFNodes/yolo_evaluation_layer_1/arange__271_cond/Output_0/Data__const" type="Parameter" version="opset1">
			<data element_type="boolean" offset="0" shape="" size="1"/>
			<output>
				<port id="1" precision="BOOL"/>
			</output>
		</layer>
		<layer id="3" name="TFNodes/yolo_evaluation_layer_1/arange_5/start:0/Output_0/Data__const" type="Parameter" version="opset1">
			<data element_type="i32" offset="1" shape="" size="4"/>
			<output>
				<port id="1" precision="I32"/>
			</output>
		</layer>
		<layer id="4" name="TFNodes/yolo_evaluation_layer_1/arange_5/delta:0/Output_0/Data__const" type="Parameter" version="opset1">
			<data element_type="i32" offset="5" shape="" size="4"/>
			<output>
				<port id="1" precision="I32"/>
			</output>
		</layer>
		<layer id="5" name="TFNodes/yolo_evaluation_layer_1/arange_4__77_loop" type="Loop" version="opset5">
			<input>
				<port id="0"/>
				<port id="1"/>
				<port id="2"/>
				<port id="3"/>
			</input>
			<output>
				<port id="4" precision="I32">
					<dim>1</dim>
				</port>
			</output>
			<port_map>
				<input external_port_id="3" internal_layer_id="6"/>
				<input external_port_id="1" internal_layer_id="0"/>
				<input external_port_id="2" internal_layer_id="2"/>
				<output external_port_id="-1" internal_layer_id="1" purpose="execution_condition"/>
				<output axis="0" external_port_id="4" internal_layer_id="5"/>
			</port_map>
			<back_edges>
				<edge from-layer="1" from-port="0" to-layer="0" to-port="0"/>
				<edge from-layer="8" from-port="0" to-layer="2" to-port="0"/>
			</back_edges>
			<body>
				<layers>
					<layer id="0" name="cond" type="Parameter" version="opset1">
						<data element_type="boolean" shape=""/>
						<output>
							<port id="0" precision="BOOL"/>
						</output>
					</layer>
					<layer id="1" name="Identity__82/sink_port_0" type="Result" version="opset1">
						<data order="0"/>
						<input>
							<port id="0"/>
						</input>
					</layer>
					<layer id="2" name="prev" type="Parameter" version="opset1">
						<data element_type="i32" shape=""/>
						<output>
							<port id="0" precision="I32"/>
						</output>
					</layer>
					<layer id="3" name="11_input_port_1/value/Output_0/Data__const" type="Const" version="opset1">
						<data element_type="i64" offset="0" shape="1" size="8"/>
						<output>
							<port id="1" precision="I64">
								<dim>1</dim>
							</port>
						</output>
					</layer>
					<layer id="4" name="11" type="Unsqueeze" version="opset1">
						<input>
							<port id="0"/>
							<port id="1">
								<dim>1</dim>
							</port>
						</input>
						<output>
							<port id="2" precision="I32">
								<dim>1</dim>
							</port>
						</output>
					</layer>
					<layer id="5" name="Identity__84/sink_port_0" type="Result" version="opset1">
						<data order="2"/>
						<input>
							<port id="0">
								<dim>1</dim>
							</port>
						</input>
					</layer>
					<layer id="6" name="TFNodes/yolo_evaluation_layer_1/arange_5/delta:0" type="Parameter" version="opset1">
						<data element_type="i32" shape=""/>
						<output>
							<port id="0" precision="I32"/>
						</output>
					</layer>
					<layer id="7" name="add__83" type="Add" version="opset1">
						<input>
							<port id="0"/>
							<port id="1"/>
						</input>
						<output>
							<port id="2" precision="I32"/>
						</output>
					</layer>
					<layer id="8" name="add__83/sink_port_0" type="Result" version="opset1">
						<data order="1"/>
						<input>
							<port id="0"/>
						</input>
					</layer>
				</layers>
				<edges>
					<edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>
					<edge from-layer="2" from-port="0" to-layer="4" to-port="0"/>
					<edge from-layer="3" from-port="1" to-layer="4" to-port="1"/>
					<edge from-layer="4" from-port="2" to-layer="5" to-port="0"/>
					<edge from-layer="2" from-port="0" to-layer="7" to-port="0"/>
					<edge from-layer="6" from-port="0" to-layer="7" to-port="1"/>
					<edge from-layer="7" from-port="2" to-layer="8" to-port="0"/>
				</edges>
			</body>
		</layer>
		<layer id="8" name="TFNodes/yolo_evaluation_layer_1/Reshape_13/sink_port_0" type="Result" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
		</layer>
	</layers>
	<edges>
		<edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>
		<edge from-layer="1" from-port="1" to-layer="5" to-port="0"/>
		<edge from-layer="2" from-port="1" to-layer="5" to-port="1"/>
		<edge from-layer="3" from-port="1" to-layer="5" to-port="2"/>
		<edge from-layer="4" from-port="1" to-layer="5" to-port="3"/>
		<edge from-layer="5" from-port="4" to-layer="8" to-port="0"/>
	</edges>
</net>

)V0G0N";

    Core reader;
    Blob::Ptr weights;
    weights = make_shared_blob<uint8_t>(TensorDesc(Precision::U8, {8}, Layout::C));
    weights->allocate();
    weights->buffer().as<int64_t*>()[0] = 0;
    EXPECT_NO_THROW(reader.ReadNetwork(model, weights));
}

TEST_F(NGraphReaderTests, ReadLoopNetwork_2) {
    std::string model = R"V0G0N(
<?xml version="1.0" ?>
<net name="loop_2d_add" version="10">
	<layers>
		<layer id="0" name="trip_count/Output_0/Data__const" type="Parameter" version="opset1">
			<data element_type="i64" offset="0" shape="1" size="8"/>
			<output>
				<port id="1" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="1" name="a_final5/execution_cond/Output_0/Data__const" type="Parameter" version="opset1">
			<data element_type="boolean" offset="8" shape="" size="1"/>
			<output>
				<port id="1" precision="BOOL"/>
			</output>
		</layer>
		<layer id="2" name="a_init" type="Parameter" version="opset1">
			<data element_type="f32" shape="1,2"/>
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>2</dim>
				</port>
			</output>
		</layer>
		<layer id="3" name="a_final5" type="Loop" version="opset5">
			<input>
				<port id="0">
					<dim>1</dim>
				</port>
				<port id="1"/>
				<port id="2">
					<dim>1</dim>
					<dim>2</dim>
				</port>
			</input>
			<output>
				<port id="3" precision="FP32">
					<dim>1</dim>
					<dim>2</dim>
				</port>
				<port id="4" precision="FP32">
					<dim>3</dim>
					<dim>1</dim>
					<dim>2</dim>
				</port>
			</output>
			<port_map>
				<input external_port_id="1" internal_layer_id="0"/>
				<input external_port_id="2" internal_layer_id="2"/>
				<output external_port_id="-1" internal_layer_id="1" purpose="execution_condition"/>
				<output external_port_id="3" internal_layer_id="5"/>
				<output axis="0" external_port_id="4" internal_layer_id="8"/>
			</port_map>
			<back_edges>
				<edge from-layer="1" from-port="0" to-layer="0" to-port="0"/>
				<edge from-layer="5" from-port="0" to-layer="2" to-port="0"/>
			</back_edges>
			<body>
				<layers>
					<layer id="0" name="cond_in" type="Parameter" version="opset1">
						<data element_type="boolean" shape=""/>
						<output>
							<port id="0" precision="BOOL"/>
						</output>
					</layer>
					<layer id="1" name="cond_identity/sink_port_0" type="Result" version="opset1">
						<input>
							<port id="0"/>
						</input>
					</layer>
					<layer id="2" name="a_in" type="Parameter" version="opset1">
						<data element_type="f32" shape="1,2"/>
						<output>
							<port id="0" precision="FP32">
								<dim>1</dim>
								<dim>2</dim>
							</port>
						</output>
					</layer>
					<layer id="3" name="b/Output_0/Data__const" type="Const" version="opset1">
						<data element_type="f32" offset="0" shape="1,2" size="8"/>
						<output>
							<port id="1" precision="FP32">
								<dim>1</dim>
								<dim>2</dim>
							</port>
						</output>
					</layer>
					<layer id="4" name="loop_body_add" type="Add" version="opset1">
						<input>
							<port id="0">
								<dim>1</dim>
								<dim>2</dim>
							</port>
							<port id="1">
								<dim>1</dim>
								<dim>2</dim>
							</port>
						</input>
						<output>
							<port id="2" precision="FP32">
								<dim>1</dim>
								<dim>2</dim>
							</port>
						</output>
					</layer>
					<layer id="5" name="loop_body_add/sink_port_0" type="Result" version="opset1">
						<input>
							<port id="0">
								<dim>1</dim>
								<dim>2</dim>
							</port>
						</input>
					</layer>
					<layer id="6" name="11_input_port_1/value/Output_0/Data__const" type="Const" version="opset1">
						<data element_type="i64" offset="0" shape="1" size="8"/>
						<output>
							<port id="1" precision="I64">
								<dim>1</dim>
							</port>
						</output>
					</layer>
					<layer id="7" name="11" type="Unsqueeze" version="opset1">
						<input>
							<port id="0">
								<dim>1</dim>
								<dim>2</dim>
							</port>
							<port id="1">
								<dim>1</dim>
							</port>
						</input>
						<output>
							<port id="2" precision="FP32">
								<dim>3</dim>
								<dim>1</dim>
								<dim>2</dim>
							</port>
						</output>
					</layer>
					<layer id="8" name="output_accumulator/sink_port_0" type="Result" version="opset1">
						<input>
							<port id="0">
								<dim>3</dim>
								<dim>1</dim>
								<dim>2</dim>
							</port>
						</input>
					</layer>
				</layers>
				<edges>
					<edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>
					<edge from-layer="2" from-port="0" to-layer="4" to-port="0"/>
					<edge from-layer="3" from-port="1" to-layer="4" to-port="1"/>
					<edge from-layer="4" from-port="2" to-layer="5" to-port="0"/>
					<edge from-layer="4" from-port="2" to-layer="7" to-port="0"/>
					<edge from-layer="6" from-port="1" to-layer="7" to-port="1"/>
					<edge from-layer="7" from-port="2" to-layer="8" to-port="0"/>
				</edges>
			</body>
		</layer>
		<layer id="6" name="a_final/sink_port_0" type="Result" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>2</dim>
				</port>
			</input>
		</layer>
		<layer id="9" name="a_values/sink_port_0" type="Result" version="opset1">
			<input>
				<port id="0">
					<dim>3</dim>
					<dim>1</dim>
					<dim>2</dim>
				</port>
			</input>
		</layer>
	</layers>
	<edges>
		<edge from-layer="0" from-port="1" to-layer="3" to-port="0"/>
		<edge from-layer="1" from-port="1" to-layer="3" to-port="1"/>
		<edge from-layer="2" from-port="0" to-layer="3" to-port="2"/>
		<edge from-layer="3" from-port="3" to-layer="6" to-port="0"/>
		<edge from-layer="3" from-port="4" to-layer="9" to-port="0"/>
	</edges>
</net>

)V0G0N";

    Core reader;
    Blob::Ptr weights;
    weights = make_shared_blob<uint8_t>(TensorDesc(Precision::U8, {8}, Layout::C));
    weights->allocate();
    weights->buffer().as<float*>()[0] = 0;
    weights->buffer().as<float*>()[1] = 0;
    EXPECT_NO_THROW(reader.ReadNetwork(model, weights));
}

TEST_F(NGraphReaderTests, ReadLoopNetwork_ExternalPort1IsNotConnected) {
    std::string model = R"V0G0N(
<?xml version="1.0" ?>
<net name="loop_2d_add" version="10">
	<layers>
		<layer id="0" name="trip_count/Output_0/Data__const" type="Parameter" version="opset1">
			<data element_type="i64" offset="0" shape="1" size="8"/>
			<output>
				<port id="1" precision="I64">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="1" name="a_final5/execution_cond/Output_0/Data__const" type="Parameter" version="opset1">
			<data element_type="boolean" offset="8" shape="" size="1"/>
			<output>
				<port id="1" precision="BOOL"/>
			</output>
		</layer>
		<layer id="2" name="a_init" type="Parameter" version="opset1">
			<data element_type="f32" shape="1,2"/>
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>2</dim>
				</port>
			</output>
		</layer>
		<layer id="3" name="a_final5" type="Loop" version="opset5">
			<input>
				<port id="0">
					<dim>1</dim>
				</port>
				<port id="1"/>
				<port id="2">
					<dim>1</dim>
					<dim>2</dim>
				</port>
			</input>
			<output>
				<port id="3" precision="FP32">
					<dim>1</dim>
					<dim>2</dim>
				</port>
				<port id="4" precision="FP32">
					<dim>3</dim>
					<dim>1</dim>
					<dim>2</dim>
				</port>
			</output>
			<port_map>
				<input external_port_id="0" internal_layer_id="0"/>
				<input external_port_id="2" internal_layer_id="2"/>
				<output external_port_id="-1" internal_layer_id="1" purpose="execution_condition"/>
				<output external_port_id="3" internal_layer_id="5"/>
				<output axis="0" external_port_id="4" internal_layer_id="8"/>
			</port_map>
			<back_edges>
				<edge from-layer="1" from-port="0" to-layer="0" to-port="0"/>
				<edge from-layer="5" from-port="0" to-layer="2" to-port="0"/>
			</back_edges>
			<body>
				<layers>
					<layer id="0" name="cond_in" type="Parameter" version="opset1">
						<data element_type="boolean" shape=""/>
						<output>
							<port id="0" precision="BOOL"/>
						</output>
					</layer>
					<layer id="1" name="cond_identity/sink_port_0" type="Result" version="opset1">
						<input>
							<port id="0"/>
						</input>
					</layer>
					<layer id="2" name="a_in" type="Parameter" version="opset1">
						<data element_type="f32" shape="1,2"/>
						<output>
							<port id="0" precision="FP32">
								<dim>1</dim>
								<dim>2</dim>
							</port>
						</output>
					</layer>
					<layer id="3" name="b/Output_0/Data__const" type="Const" version="opset1">
						<data element_type="f32" offset="0" shape="1,2" size="8"/>
						<output>
							<port id="1" precision="FP32">
								<dim>1</dim>
								<dim>2</dim>
							</port>
						</output>
					</layer>
					<layer id="4" name="loop_body_add" type="Add" version="opset1">
						<input>
							<port id="0">
								<dim>1</dim>
								<dim>2</dim>
							</port>
							<port id="1">
								<dim>1</dim>
								<dim>2</dim>
							</port>
						</input>
						<output>
							<port id="2" precision="FP32">
								<dim>1</dim>
								<dim>2</dim>
							</port>
						</output>
					</layer>
					<layer id="5" name="loop_body_add/sink_port_0" type="Result" version="opset1">
						<input>
							<port id="0">
								<dim>1</dim>
								<dim>2</dim>
							</port>
						</input>
					</layer>
					<layer id="6" name="11_input_port_1/value/Output_0/Data__const" type="Const" version="opset1">
						<data element_type="i64" offset="0" shape="1" size="8"/>
						<output>
							<port id="1" precision="I64">
								<dim>1</dim>
							</port>
						</output>
					</layer>
					<layer id="7" name="11" type="Unsqueeze" version="opset1">
						<input>
							<port id="0">
								<dim>1</dim>
								<dim>2</dim>
							</port>
							<port id="1">
								<dim>1</dim>
							</port>
						</input>
						<output>
							<port id="2" precision="FP32">
								<dim>3</dim>
								<dim>1</dim>
								<dim>2</dim>
							</port>
						</output>
					</layer>
					<layer id="8" name="output_accumulator/sink_port_0" type="Result" version="opset1">
						<input>
							<port id="0">
								<dim>3</dim>
								<dim>1</dim>
								<dim>2</dim>
							</port>
						</input>
					</layer>
				</layers>
				<edges>
					<edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>
					<edge from-layer="2" from-port="0" to-layer="4" to-port="0"/>
					<edge from-layer="3" from-port="1" to-layer="4" to-port="1"/>
					<edge from-layer="4" from-port="2" to-layer="5" to-port="0"/>
					<edge from-layer="4" from-port="2" to-layer="7" to-port="0"/>
					<edge from-layer="6" from-port="1" to-layer="7" to-port="1"/>
					<edge from-layer="7" from-port="2" to-layer="8" to-port="0"/>
				</edges>
			</body>
		</layer>
		<layer id="6" name="a_final/sink_port_0" type="Result" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>2</dim>
				</port>
			</input>
		</layer>
		<layer id="9" name="a_values/sink_port_0" type="Result" version="opset1">
			<input>
				<port id="0">
					<dim>3</dim>
					<dim>1</dim>
					<dim>2</dim>
				</port>
			</input>
		</layer>
	</layers>
	<edges>
		<edge from-layer="0" from-port="1" to-layer="3" to-port="0"/>
		<edge from-layer="1" from-port="1" to-layer="3" to-port="1"/>
		<edge from-layer="2" from-port="0" to-layer="3" to-port="2"/>
		<edge from-layer="3" from-port="3" to-layer="6" to-port="0"/>
		<edge from-layer="3" from-port="4" to-layer="9" to-port="0"/>
	</edges>
</net>

)V0G0N";

    Core reader;
    Blob::Ptr weights;
    weights = make_shared_blob<uint8_t>(TensorDesc(Precision::U8, {8}, Layout::C));
    weights->allocate();
    weights->buffer().as<float*>()[0] = 0;
    weights->buffer().as<float*>()[1] = 0;
    EXPECT_NO_THROW(reader.ReadNetwork(model, weights));
}

