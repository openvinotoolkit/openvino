// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "frontend_test.hpp"
#include "openvino/opsets/opset1.hpp"
#include "openvino/opsets/opset8.hpp"

class IRFrontendTestsTensorIterator : public ::testing::Test, public IRFrontendTestsImpl {
protected:
    void SetUp() override {}

    void TearDown() override {
        RemoveTemporalFiles();
    }
};

TEST_F(IRFrontendTestsTensorIterator, tensor_iterator_merged_input) {
    std::string testModel = R"V0G0N(
<net name="Network" version="11">
    <layers>
        <layer id="0" name="Parameter1" type="Parameter" version="opset1">
            <data element_type="f32" shape="1,2,3"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>2</dim>
                    <dim>3</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="TensorIterator" type="TensorIterator" version="opset1">
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>2</dim>
                    <dim>3</dim>
                </port>
            </input>
            <output>
                <port id="1" precision="FP32">
                    <dim>1</dim>
                    <dim>2</dim>
                    <dim>3</dim>
                </port>
            </output>
            <port_map>
                <input external_port_id="0" internal_layer_id="0"/>
                <output external_port_id="1" internal_layer_id="1"/>
            </port_map>
            <back_edges>
                <edge from-layer="1" to-layer="0"/>
            </back_edges>
            <body>
                <layers>
                    <layer id="0" name="internalParameter1" type="Parameter" version="opset1">
                        <data element_type="f32" shape="1,2,3"/>
                        <output>
                            <port id="0" precision="FP32">
                                <dim>1</dim>
                                <dim>2</dim>
                                <dim>3</dim>
                            </port>
                        </output>
                    </layer>
                    <layer id="1" name="internalResult1" type="Result" version="opset1">
                        <input>
                            <port id="0">
                                <dim>1</dim>
                                <dim>2</dim>
                                <dim>3</dim>
                            </port>
                        </input>
                    </layer>
                </layers>
                <edges>
                    <edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>
                </edges>
            </body>
        </layer>
        <layer id="2" name="Result1" type="Result" version="opset1">
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>2</dim>
                    <dim>3</dim>
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

    std::shared_ptr<ov::Model> model;

    OV_ASSERT_NO_THROW(model = core.read_model(testModel, ov::Tensor()));
    ASSERT_TRUE(!!model);

    std::shared_ptr<ov::Model> modelRef;
    {
        auto parameter = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{1, 2, 3});
        parameter->set_friendly_name("Parameter1");
        auto tensor_iterator = std::make_shared<ov::opset8::TensorIterator>();

        std::shared_ptr<ov::Model> body;
        auto internalParameter = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{1, 2, 3});
        internalParameter->set_friendly_name("internalParameter1");
        auto result1 = std::make_shared<ov::opset1::Result>(internalParameter);
        result1->set_friendly_name("internalResult1");
        body = std::make_shared<ov::Model>(ov::NodeVector{result1}, ov::ParameterVector{internalParameter});
        tensor_iterator->set_body(body);
        tensor_iterator->set_friendly_name("TensorIterator");
        tensor_iterator->set_merged_input(internalParameter, parameter, result1);
        auto out0 = tensor_iterator->get_iter_value(result1, -1);

        auto result = std::make_shared<ov::opset1::Result>(tensor_iterator->output(0));
        result->set_friendly_name("Result1");

        modelRef = std::make_shared<ov::Model>(ov::NodeVector{result}, ov::ParameterVector{parameter});
    }

    const auto fc = FunctionsComparator::with_default()
                        .enable(FunctionsComparator::ATTRIBUTES)
                        .enable(FunctionsComparator::PRECISIONS)
                        .enable(FunctionsComparator::RUNTIME_KEYS)
                        .enable(FunctionsComparator::NAMES)
                        .enable(FunctionsComparator::CONST_VALUES);
    const auto res = fc.compare(model, modelRef);
    EXPECT_TRUE(res.valid) << res.message;
}

TEST_F(IRFrontendTestsTensorIterator, tensor_iterator_slised_input) {
    std::string testModel = R"V0G0N(
<net name="Network" version="11">
    <layers>
        <layer id="0" name="Parameter1" type="Parameter" version="opset1">
            <data element_type="f32" shape="1,2,3"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>2</dim>
                    <dim>3</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="TensorIterator" type="TensorIterator" version="opset1">
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>2</dim>
                    <dim>3</dim>
                </port>
            </input>
            <output>
                <port id="1" precision="FP32">
                    <dim>1</dim>
                    <dim>2</dim>
                    <dim>3</dim>
                </port>
            </output>
            <port_map>
                <input axis="2" external_port_id="0" internal_layer_id="0" part_size="1" stride="1"/>
                <output axis="2" external_port_id="1" internal_layer_id="1" part_size="1" stride="1"/>
            </port_map>
            <back_edges>
                <edge from-layer="1" to-layer="0"/>
            </back_edges>
            <body>
                <layers>
                    <layer id="0" name="internalParameter1" type="Parameter" version="opset1">
                        <data element_type="f32" shape="1,2,1"/>
                        <output>
                            <port id="0" precision="FP32">
                                <dim>1</dim>
                                <dim>2</dim>
                                <dim>1</dim>
                            </port>
                        </output>
                    </layer>
                    <layer id="1" name="internalResult1" type="Result" version="opset1">
                        <input>
                            <port id="0">
                                <dim>1</dim>
                                <dim>2</dim>
                                <dim>1</dim>
                            </port>
                        </input>
                    </layer>
                </layers>
                <edges>
                    <edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>
                </edges>
            </body>
        </layer>
        <layer id="2" name="Result1" type="Result" version="opset1">
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>2</dim>
                    <dim>3</dim>
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

    std::shared_ptr<ov::Model> model;

    OV_ASSERT_NO_THROW(model = core.read_model(testModel, ov::Tensor()));
    ASSERT_TRUE(!!model);

    std::shared_ptr<ov::Model> modelRef;
    {
        auto parameter = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{1, 2, 3});
        parameter->set_friendly_name("Parameter1");
        auto tensor_iterator = std::make_shared<ov::opset8::TensorIterator>();

        std::shared_ptr<ov::Model> body;
        auto internalParameter = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{1, 2, 3});
        internalParameter->set_friendly_name("internalParameter1");
        auto result1 = std::make_shared<ov::opset1::Result>(internalParameter);
        result1->set_friendly_name("internalResult1");
        body = std::make_shared<ov::Model>(ov::NodeVector{result1}, ov::ParameterVector{internalParameter});
        tensor_iterator->set_body(body);
        tensor_iterator->set_friendly_name("TensorIterator");
        tensor_iterator->set_sliced_input(internalParameter, parameter, 0, 1, 1, -1, 2);
        auto out0 = tensor_iterator->get_concatenated_slices(result1, 0, 1, 1, -1, 2);

        auto result = std::make_shared<ov::opset1::Result>(tensor_iterator->output(0));
        result->set_friendly_name("Result1");

        modelRef = std::make_shared<ov::Model>(ov::NodeVector{result}, ov::ParameterVector{parameter});
    }

    const auto fc = FunctionsComparator::with_default()
                        .enable(FunctionsComparator::ATTRIBUTES)
                        .enable(FunctionsComparator::PRECISIONS)
                        .enable(FunctionsComparator::RUNTIME_KEYS)
                        .enable(FunctionsComparator::NAMES)
                        .enable(FunctionsComparator::CONST_VALUES);
    const auto res = fc.compare(model, modelRef);
    EXPECT_TRUE(res.valid) << res.message;
}

TEST_F(IRFrontendTestsTensorIterator, loop1) {
    std::string xmlModel = R"V0G0N(
<net name="yolov3-10" version="11">
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

    std::vector<unsigned char> buffer(8, 0);
    int64_t* int64Buffer = reinterpret_cast<int64_t*>(buffer.data());
    int64Buffer[0] = 0;

    createTemporalModelFile(xmlModel, buffer);
    std::shared_ptr<ov::Model> model;

    OV_ASSERT_NO_THROW(model = core.read_model(xmlFileName, binFileName));
    ASSERT_TRUE(!!model);
}

TEST_F(IRFrontendTestsTensorIterator, loop2) {
    std::string xmlModel = R"V0G0N(
<net name="loop_2d_add" version="11">
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

    std::vector<unsigned char> buffer(8, 0);
    float* floatBuffer = reinterpret_cast<float*>(buffer.data());
    floatBuffer[0] = 0;
    floatBuffer[1] = 0;

    createTemporalModelFile(xmlModel, buffer);
    std::shared_ptr<ov::Model> model;

    OV_ASSERT_NO_THROW(model = core.read_model(xmlFileName, binFileName));
    ASSERT_TRUE(!!model);
}

TEST_F(IRFrontendTestsTensorIterator, loop_external_port1_is_not_connected) {
    std::string xmlModel = R"V0G0N(
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

    std::vector<unsigned char> buffer(8, 0);
    float* floatBuffer = reinterpret_cast<float*>(buffer.data());
    floatBuffer[0] = 0;
    floatBuffer[1] = 0;

    createTemporalModelFile(xmlModel, buffer);
    std::shared_ptr<ov::Model> model;

    OV_ASSERT_NO_THROW(model = core.read_model(xmlFileName, binFileName));
    ASSERT_TRUE(!!model);
}

TEST_F(IRFrontendTestsTensorIterator, tensor_iterator_resnet_opset4) {
    std::string xmlModel = R"V0G0N(
    <net name="Resnet" version="10">
        <layers>
            <layer id="0" name="data1" type="Parameter" version="opset1">
                <data element_type="f32" shape="16,1,512"/>
                <output>
                    <port id="0" precision="FP32">
                        <dim>16</dim>
                        <dim>1</dim>
                        <dim>512</dim>
                    </port>
                </output>
            </layer>
            <layer id="1" name="data2" type="Parameter" version="opset1">
                <data element_type="f32" shape="1,512"/>
                <output>
                    <port id="0" precision="FP32">
                        <dim>1</dim>
                        <dim>512</dim>
                    </port>
                </output>
            </layer>
            <layer id="2" name="data3" type="Parameter" version="opset1">
                <data element_type="f32" shape="1,512"/>
                <output>
                    <port id="0" precision="FP32">
                        <dim>1</dim>
                        <dim>512</dim>
                    </port>
                </output>
            </layer>
            <layer id="3" name="471/TensorIterator" type="TensorIterator" version="opset1">
                <input>
                    <port id="0">
                        <dim>16</dim>
                        <dim>1</dim>
                        <dim>512</dim>
                    </port>
                    <port id="1">
                        <dim>1</dim>
                        <dim>512</dim>
                    </port>
                    <port id="2">
                        <dim>1</dim>
                        <dim>512</dim>
                    </port>
                </input>
                <output>
                    <port id="3" precision="FP32">
                        <dim>16</dim>
                        <dim>1</dim>
                        <dim>512</dim>
                    </port>
                    <port id="4" precision="FP32">
                        <dim>1</dim>
                        <dim>512</dim>
                    </port>
                    <port id="5" precision="FP32">
                        <dim>1</dim>
                        <dim>512</dim>
                    </port>
                </output>
                <port_map>
                    <input axis="0" external_port_id="0" internal_layer_id="0" part_size="1" stride="1"/>
                    <input external_port_id="1" internal_layer_id="3"/>
                    <input external_port_id="2" internal_layer_id="4"/>
                    <output axis="0" external_port_id="3" internal_layer_id="13" part_size="1" stride="1"/>
                    <output external_port_id="4" internal_layer_id="9"/>
                    <output external_port_id="5" internal_layer_id="10"/>
                </port_map>
                <back_edges>
                    <edge from-layer="9" to-layer="3"/>
                    <edge from-layer="10" to-layer="4"/>
                </back_edges>
                <body>
                    <layers>
                        <layer id="0" name="20" type="Parameter" version="opset1">
                            <data element_type="f32" shape="1,1,512"/>
                            <output>
                                <port id="0" precision="FP32">
                                    <dim>1</dim>
                                    <dim>1</dim>
                                    <dim>512</dim>
                                </port>
                            </output>
                        </layer>
                        <layer id="1" name="7_const" type="Const" version="opset1">
                            <data element_type="i64" offset="0" shape="2" size="16"/>
                            <output>
                                <port id="1" precision="I64">
                                    <dim>2</dim>
                                </port>
                            </output>
                        </layer>
                        <layer id="2" name="471/input_squeeze" type="Reshape" version="opset1">
                            <data special_zero="True"/>
                            <input>
                                <port id="0">
                                    <dim>1</dim>
                                    <dim>1</dim>
                                    <dim>512</dim>
                                </port>
                                <port id="1">
                                    <dim>2</dim>
                                </port>
                            </input>
                            <output>
                                <port id="2" precision="FP32">
                                    <dim>1</dim>
                                    <dim>512</dim>
                                </port>
                            </output>
                        </layer>
                        <layer id="3" name="22" type="Parameter" version="opset1">
                            <data element_type="f32" shape="1,512"/>
                            <output>
                                <port id="0" precision="FP32">
                                    <dim>1</dim>
                                    <dim>512</dim>
                                </port>
                            </output>
                        </layer>
                        <layer id="4" name="24" type="Parameter" version="opset1">
                            <data element_type="f32" shape="1,512"/>
                            <output>
                                <port id="0" precision="FP32">
                                    <dim>1</dim>
                                    <dim>512</dim>
                                </port>
                            </output>
                        </layer>
                        <layer id="5" name="471/LSTMCell/Split149_const" type="Const" version="opset1">
                            <data element_type="f32" offset="16" shape="2048,512" size="4194304"/>
                            <output>
                                <port id="1" precision="FP32">
                                    <dim>2048</dim>
                                    <dim>512</dim>
                                </port>
                            </output>
                        </layer>
                        <layer id="6" name="471/LSTMCell/Split150_const" type="Const" version="opset1">
                            <data element_type="f32" offset="4194320" shape="2048,512" size="4194304"/>
                            <output>
                                <port id="1" precision="FP32">
                                    <dim>2048</dim>
                                    <dim>512</dim>
                                </port>
                            </output>
                        </layer>
                        <layer id="7" name="471/inport/2_const" type="Const" version="opset1">
                            <data element_type="f32" offset="8388624" shape="2048" size="8192"/>
                            <output>
                                <port id="1" precision="FP32">
                                    <dim>2048</dim>
                                </port>
                            </output>
                        </layer>
                        <layer id="8" name="471/LSTMCell" type="LSTMCell" version="opset4">
                            <data hidden_size="512"/>
                            <input>
                                <port id="0">
                                    <dim>1</dim>
                                    <dim>512</dim>
                                </port>
                                <port id="1">
                                    <dim>1</dim>
                                    <dim>512</dim>
                                </port>
                                <port id="2">
                                    <dim>1</dim>
                                    <dim>512</dim>
                                </port>
                                <port id="3">
                                    <dim>2048</dim>
                                    <dim>512</dim>
                                </port>
                                <port id="4">
                                    <dim>2048</dim>
                                    <dim>512</dim>
                                </port>
                                <port id="5">
                                    <dim>2048</dim>
                                </port>
                            </input>
                            <output>
                                <port id="6" precision="FP32">
                                    <dim>1</dim>
                                    <dim>512</dim>
                                </port>
                                <port id="7" precision="FP32">
                                    <dim>1</dim>
                                    <dim>512</dim>
                                </port>
                            </output>
                        </layer>
                        <layer id="9" name="471/outport/0/sink_port_0" type="Result" version="opset1">
                            <input>
                                <port id="0">
                                    <dim>1</dim>
                                    <dim>512</dim>
                                </port>
                            </input>
                        </layer>
                        <layer id="10" name="471/outport/1/sink_port_0" type="Result" version="opset1">
                            <input>
                                <port id="0">
                                    <dim>1</dim>
                                    <dim>512</dim>
                                </port>
                            </input>
                        </layer>
                        <layer id="11" name="15_const" type="Const" version="opset1">
                            <data element_type="i64" offset="8396816" shape="3" size="24"/>
                            <output>
                                <port id="1" precision="I64">
                                    <dim>3</dim>
                                </port>
                            </output>
                        </layer>
                        <layer id="12" name="471output_unsqueeze" type="Reshape" version="opset1">
                            <data special_zero="True"/>
                            <input>
                                <port id="0">
                                    <dim>1</dim>
                                    <dim>512</dim>
                                </port>
                                <port id="1">
                                    <dim>3</dim>
                                </port>
                            </input>
                            <output>
                                <port id="2" precision="FP32">
                                    <dim>1</dim>
                                    <dim>1</dim>
                                    <dim>512</dim>
                                </port>
                            </output>
                        </layer>
                        <layer id="13" name="18/sink_port_0" type="Result" version="opset1">
                            <input>
                                <port id="0">
                                    <dim>1</dim>
                                    <dim>1</dim>
                                    <dim>512</dim>
                                </port>
                            </input>
                        </layer>
                    </layers>
                    <edges>
                        <edge from-layer="0" from-port="0" to-layer="2" to-port="0"/>
                        <edge from-layer="1" from-port="1" to-layer="2" to-port="1"/>
                        <edge from-layer="2" from-port="2" to-layer="8" to-port="0"/>
                        <edge from-layer="3" from-port="0" to-layer="8" to-port="1"/>
                        <edge from-layer="4" from-port="0" to-layer="8" to-port="2"/>
                        <edge from-layer="5" from-port="1" to-layer="8" to-port="3"/>
                        <edge from-layer="6" from-port="1" to-layer="8" to-port="4"/>
                        <edge from-layer="7" from-port="1" to-layer="8" to-port="5"/>
                        <edge from-layer="8" from-port="6" to-layer="9" to-port="0"/>
                        <edge from-layer="8" from-port="7" to-layer="10" to-port="0"/>
                        <edge from-layer="8" from-port="6" to-layer="12" to-port="0"/>
                        <edge from-layer="11" from-port="1" to-layer="12" to-port="1"/>
                        <edge from-layer="12" from-port="2" to-layer="13" to-port="0"/>
                    </edges>
                </body>
            </layer>
            <layer id="4" name="result" type="Result" version="opset1">
                <input>
                    <port id="0">
                        <dim>16</dim>
                        <dim>1</dim>
                        <dim>512</dim>
                    </port>
                </input>
            </layer>
            <layer id="5" name="result_2" type="Result" version="opset1">
                <input>
                    <port id="0">
                        <dim>1</dim>
                        <dim>512</dim>
                    </port>
                </input>
            </layer>
            <layer id="6" name="result_3" type="Result" version="opset1">
                <input>
                    <port id="0">
                        <dim>1</dim>
                        <dim>512</dim>
                    </port>
                </input>
            </layer>
        </layers>
        <edges>
            <edge from-layer="0" from-port="0" to-layer="3" to-port="0"/>
            <edge from-layer="1" from-port="0" to-layer="3" to-port="1"/>
            <edge from-layer="2" from-port="0" to-layer="3" to-port="2"/>
            <edge from-layer="3" from-port="3" to-layer="4" to-port="0"/>
            <edge from-layer="3" from-port="4" to-layer="5" to-port="0"/>
            <edge from-layer="3" from-port="5" to-layer="6" to-port="0"/>
        </edges>
    </net>
    )V0G0N";

    std::vector<unsigned char> buffer(8396840, 0);
    int64_t* int64Buffer = reinterpret_cast<int64_t*>(buffer.data());
    int64Buffer[0] = 1;
    int64Buffer[1] = 512;

    int64Buffer[1049602] = 1;
    int64Buffer[1049603] = 1;
    int64Buffer[1049604] = 512;

    createTemporalModelFile(xmlModel, buffer);
    std::shared_ptr<ov::Model> model;

    OV_ASSERT_NO_THROW(model = core.read_model(xmlFileName, binFileName));
    ASSERT_TRUE(!!model);
}

TEST_F(IRFrontendTestsTensorIterator, tensor_iterator_negative_stride_opset4) {
    std::string xmlModel = R"V0G0N(
    <net name="Transpose" version="10">
        <layers>
            <layer id="0" name="data1" type="Parameter" version="opset1">
                <data element_type="f32" shape="1,25,512"/>
                <output>
                    <port id="0" precision="FP32">
                        <dim>1</dim>
                        <dim>25</dim>
                        <dim>512</dim>
                    </port>
                </output>
            </layer>
            <layer id="1" name="data2" type="Parameter" version="opset1">
                <data element_type="f32" shape="1,256"/>
                <output>
                    <port id="0" precision="FP32">
                        <dim>1</dim>
                        <dim>256</dim>
                    </port>
                </output>
            </layer>
            <layer id="2" name="data3" type="Parameter" version="opset1">
                <data element_type="f32" shape="1,256"/>
                <output>
                    <port id="0" precision="FP32">
                        <dim>1</dim>
                        <dim>256</dim>
                    </port>
                </output>
            </layer>
            <layer id="3" name="TensorIterator" type="TensorIterator" version="opset1">
                <input>
                    <port id="0">
                        <dim>1</dim>
                        <dim>25</dim>
                        <dim>512</dim>
                    </port>
                    <port id="1">
                        <dim>1</dim>
                        <dim>256</dim>
                    </port>
                    <port id="2">
                        <dim>1</dim>
                        <dim>256</dim>
                    </port>
                </input>
                <output>
                    <port id="3" precision="FP32">
                        <dim>1</dim>
                        <dim>25</dim>
                        <dim>256</dim>
                    </port>
                </output>
                <port_map>
                    <input axis="1" end="0" external_port_id="0" internal_layer_id="0" start="-1" stride="-1"/>
                    <input external_port_id="1" internal_layer_id="3"/>
                    <input external_port_id="2" internal_layer_id="4"/>
                    <output axis="1" end="0" external_port_id="3" internal_layer_id="13" start="-1" stride="-1"/>
                </port_map>
                <back_edges>
                    <edge from-layer="10" to-layer="3"/>
                    <edge from-layer="9" to-layer="4"/>
                </back_edges>
                <body>
                    <layers>
                        <layer id="0" name="32" type="Parameter" version="opset1">
                            <data element_type="f32" shape="1,1,512"/>
                            <output>
                                <port id="0" precision="FP32">
                                    <dim>1</dim>
                                    <dim>1</dim>
                                    <dim>512</dim>
                                </port>
                            </output>
                        </layer>
                        <layer id="1" name="25_const" type="Const" version="opset1">
                            <data element_type="i64" offset="0" shape="2" size="16"/>
                            <output>
                                <port id="1" precision="I64">
                                    <dim>2</dim>
                                </port>
                            </output>
                        </layer>
                        <layer id="2" name="shadow/LSTMLayers/stack_bidirectional_rnn/cell_1/bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Output_0/Data_/InputSqueeze" type="Reshape" version="opset1">
                            <data special_zero="True"/>
                            <input>
                                <port id="0">
                                    <dim>1</dim>
                                    <dim>1</dim>
                                    <dim>512</dim>
                                </port>
                                <port id="1">
                                    <dim>2</dim>
                                </port>
                            </input>
                            <output>
                                <port id="2" precision="FP32">
                                    <dim>1</dim>
                                    <dim>512</dim>
                                </port>
                            </output>
                        </layer>
                        <layer id="3" name="34" type="Parameter" version="opset1">
                            <data element_type="f32" shape="1,256"/>
                            <output>
                                <port id="0" precision="FP32">
                                    <dim>1</dim>
                                    <dim>256</dim>
                                </port>
                            </output>
                        </layer>
                        <layer id="4" name="36" type="Parameter" version="opset1">
                            <data element_type="f32" shape="1,256"/>
                            <output>
                                <port id="0" precision="FP32">
                                    <dim>1</dim>
                                    <dim>256</dim>
                                </port>
                            </output>
                        </layer>
                        <layer id="5" name="shadow/LSTMLayers/stack_bidirectional_rnn/cell_1/bidirectional_rnn/bw/bw/while/bw/basic_lstm_cell/concat/LSTMCell/Split269_const" type="Const" version="opset1">
                            <data element_type="f32" offset="16" shape="1024,512" size="2097152"/>
                            <output>
                                <port id="1" precision="FP32">
                                    <dim>1024</dim>
                                    <dim>512</dim>
                                </port>
                            </output>
                        </layer>
                        <layer id="6" name="shadow/LSTMLayers/stack_bidirectional_rnn/cell_1/bidirectional_rnn/bw/bw/while/bw/basic_lstm_cell/concat/LSTMCell/Split270_const" type="Const" version="opset1">
                            <data element_type="f32" offset="2097168" shape="1024,256" size="1048576"/>
                            <output>
                                <port id="1" precision="FP32">
                                    <dim>1024</dim>
                                    <dim>256</dim>
                                </port>
                            </output>
                        </layer>
                        <layer id="7" name="shadow/LSTMLayers/stack_bidirectional_rnn/cell_1/bidirectional_rnn/bw/bw/while/bw/basic_lstm_cell/BiasAdd/Enter/Output_0/Data__const" type="Const" version="opset1">
                            <data element_type="f32" offset="3145744" shape="1024" size="4096"/>
                            <output>
                                <port id="1" precision="FP32">
                                    <dim>1024</dim>
                                </port>
                            </output>
                        </layer>
                        <layer id="8" name="shadow/LSTMLayers/stack_bidirectional_rnn/cell_1/bidirectional_rnn/bw/bw/while/bw/basic_lstm_cell/concat/LSTMCell" type="LSTMCell" version="opset4">
                            <data hidden_size="256"/>
                            <input>
                                <port id="0">
                                    <dim>1</dim>
                                    <dim>512</dim>
                                </port>
                                <port id="1">
                                    <dim>1</dim>
                                    <dim>256</dim>
                                </port>
                                <port id="2">
                                    <dim>1</dim>
                                    <dim>256</dim>
                                </port>
                                <port id="3">
                                    <dim>1024</dim>
                                    <dim>512</dim>
                                </port>
                                <port id="4">
                                    <dim>1024</dim>
                                    <dim>256</dim>
                                </port>
                                <port id="5">
                                    <dim>1024</dim>
                                </port>
                            </input>
                            <output>
                                <port id="6" precision="FP32">
                                    <dim>1</dim>
                                    <dim>256</dim>
                                </port>
                                <port id="7" precision="FP32">
                                    <dim>1</dim>
                                    <dim>256</dim>
                                </port>
                            </output>
                        </layer>
                        <layer id="9" name="shadow/LSTMLayers/stack_bidirectional_rnn/cell_1/bidirectional_rnn/bw/bw/while/bw/basic_lstm_cell/concat/LSTMCell/Output_1/Data_/sink_port_0" type="Result" version="opset1">
                            <input>
                                <port id="0">
                                    <dim>1</dim>
                                    <dim>256</dim>
                                </port>
                            </input>
                        </layer>
                        <layer id="10" name="shadow/LSTMLayers/stack_bidirectional_rnn/cell_1/bidirectional_rnn/bw/bw/while/bw/basic_lstm_cell/concat/LSTMCell/Output_0/Data_/sink_port_0" type="Result" version="opset1">
                            <input>
                                <port id="0">
                                    <dim>1</dim>
                                    <dim>256</dim>
                                </port>
                            </input>
                        </layer>
                        <layer id="11" name="28_const" type="Const" version="opset1">
                            <data element_type="i64" offset="3149840" shape="3" size="24"/>
                            <output>
                                <port id="1" precision="I64">
                                    <dim>3</dim>
                                </port>
                            </output>
                        </layer>
                        <layer id="12" name="shadow/LSTMLayers/stack_bidirectional_rnn/cell_1/bidirectional_rnn/bw/bw/while/bw/basic_lstm_cell/concat/LSTMCell/Output_0/Data_/OutputUnsqueeze" type="Reshape" version="opset1">
                            <data special_zero="True"/>
                            <input>
                                <port id="0">
                                    <dim>1</dim>
                                    <dim>256</dim>
                                </port>
                                <port id="1">
                                    <dim>3</dim>
                                </port>
                            </input>
                            <output>
                                <port id="2" precision="FP32">
                                    <dim>1</dim>
                                    <dim>1</dim>
                                    <dim>256</dim>
                                </port>
                            </output>
                        </layer>
                        <layer id="13" name="30/sink_port_0" type="Result" version="opset1">
                            <input>
                                <port id="0">
                                    <dim>1</dim>
                                    <dim>1</dim>
                                    <dim>256</dim>
                                </port>
                            </input>
                        </layer>
                    </layers>
                    <edges>
                        <edge from-layer="0" from-port="0" to-layer="2" to-port="0"/>
                        <edge from-layer="1" from-port="1" to-layer="2" to-port="1"/>
                        <edge from-layer="2" from-port="2" to-layer="8" to-port="0"/>
                        <edge from-layer="3" from-port="0" to-layer="8" to-port="1"/>
                        <edge from-layer="4" from-port="0" to-layer="8" to-port="2"/>
                        <edge from-layer="5" from-port="1" to-layer="8" to-port="3"/>
                        <edge from-layer="6" from-port="1" to-layer="8" to-port="4"/>
                        <edge from-layer="7" from-port="1" to-layer="8" to-port="5"/>
                        <edge from-layer="8" from-port="7" to-layer="9" to-port="0"/>
                        <edge from-layer="8" from-port="6" to-layer="10" to-port="0"/>
                        <edge from-layer="8" from-port="6" to-layer="12" to-port="0"/>
                        <edge from-layer="11" from-port="1" to-layer="12" to-port="1"/>
                        <edge from-layer="12" from-port="2" to-layer="13" to-port="0"/>
                    </edges>
                </body>
            </layer>
            <layer id="4" name="result" type="Result" version="opset1">
                <input>
                    <port id="0">
                        <dim>1</dim>
                        <dim>25</dim>
                        <dim>256</dim>
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

    std::vector<unsigned char> buffer(3149864, 0);
    int64_t* int64Buffer = reinterpret_cast<int64_t*>(buffer.data());
    int64Buffer[0] = 1;
    int64Buffer[1] = 512;

    int64Buffer[393730] = 1;
    int64Buffer[393731] = 1;
    int64Buffer[393732] = 256;

    createTemporalModelFile(xmlModel, buffer);
    std::shared_ptr<ov::Model> model;

    OV_ASSERT_NO_THROW(model = core.read_model(xmlFileName, binFileName));
    ASSERT_TRUE(!!model);
}

TEST_F(IRFrontendTestsTensorIterator, test1) {
    std::string xmlModel = R"V0G0N(
    <?xml version="1.0"?>
<net name="Model169" version="10">
	<layers>
		<layer id="0" name="Parameter_2683" type="Parameter" version="opset1">
			<data shape="1,128" element_type="f32" />
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>128</dim>
				</port>
			</output>
		</layer>
		<layer id="1" name="Parameter_2682" type="Parameter" version="opset1">
			<data shape="1,2,10" element_type="f32" />
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>2</dim>
					<dim>10</dim>
				</port>
			</output>
		</layer>
		<layer id="2" name="TensorIterator_2680" type="TensorIterator" version="opset1">
			<port_map>
				<input axis="1" external_port_id="0" internal_layer_id="1" start="0" end="-1" stride="1" part_size="1" />
				<input external_port_id="1" internal_layer_id="0" />
				<output axis="1" external_port_id="2" internal_layer_id="9" start="0" end="-1" stride="1" part_size="1" />
				<output external_port_id="3" internal_layer_id="10" />
			</port_map>
			<back_edges>
				<edge from-layer="10" to-layer="0" />
			</back_edges>
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>2</dim>
					<dim>10</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>128</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>2</dim>
					<dim>128</dim>
				</port>
				<port id="3" precision="FP32">
					<dim>1</dim>
					<dim>128</dim>
				</port>
			</output>
			<body>
				<layers>
					<layer id="0" name="Parameter_2685" type="Parameter" version="opset1">
						<data shape="1,128" element_type="f32" />
						<output>
							<port id="0" precision="FP32">
								<dim>1</dim>
								<dim>128</dim>
							</port>
						</output>
					</layer>
					<layer id="1" name="Parameter_2684" type="Parameter" version="opset1">
						<data shape="1,1,10" element_type="f32" />
						<output>
							<port id="0" precision="FP32">
								<dim>1</dim>
								<dim>1</dim>
								<dim>10</dim>
							</port>
						</output>
					</layer>
					<layer id="2" name="Constant_2681" type="Const" version="opset1">
						<data element_type="i64" shape="1" offset="0" size="8" />
						<output>
							<port id="0" precision="I64">
								<dim>1</dim>
							</port>
						</output>
					</layer>
					<layer id="3" name="Squeeze_2686" type="Squeeze" version="opset1">
						<input>
							<port id="0" precision="FP32">
								<dim>1</dim>
								<dim>1</dim>
								<dim>10</dim>
							</port>
							<port id="1" precision="I64">
								<dim>1</dim>
							</port>
						</input>
						<output>
							<port id="2" precision="FP32">
								<dim>1</dim>
								<dim>10</dim>
							</port>
						</output>
					</layer>
					<layer id="4" name="Constant_2687" type="Const" version="opset1">
						<data element_type="f32" shape="384, 10" offset="8" size="15360" />
						<output>
							<port id="0" precision="FP32">
								<dim>384</dim>
								<dim>10</dim>
							</port>
						</output>
					</layer>
					<layer id="5" name="Constant_2688" type="Const" version="opset1">
						<data element_type="f32" shape="384, 128" offset="15368" size="196608" />
						<output>
							<port id="0" precision="FP32">
								<dim>384</dim>
								<dim>128</dim>
							</port>
						</output>
					</layer>
					<layer id="6" name="Constant_2689" type="Const" version="opset1">
						<data element_type="f32" shape="384" offset="211976" size="1536" />
						<output>
							<port id="0" precision="FP32">
								<dim>384</dim>
							</port>
						</output>
					</layer>
					<layer id="7" name="GRUCell_2690" type="GRUCell" version="opset3">
						<data linear_before_reset="false" hidden_size="128" activations="sigmoid, tanh" activations_alpha="" activations_beta="" clip="0" />
						<input>
							<port id="0" precision="FP32">
								<dim>1</dim>
								<dim>10</dim>
							</port>
							<port id="1" precision="FP32">
								<dim>1</dim>
								<dim>128</dim>
							</port>
							<port id="2" precision="FP32">
								<dim>384</dim>
								<dim>10</dim>
							</port>
							<port id="3" precision="FP32">
								<dim>384</dim>
								<dim>128</dim>
							</port>
							<port id="4" precision="FP32">
								<dim>384</dim>
							</port>
						</input>
						<output>
							<port id="5" precision="FP32">
								<dim>1</dim>
								<dim>128</dim>
							</port>
						</output>
					</layer>
					<layer id="8" name="Unsqueeze_2691" type="Unsqueeze" version="opset1">
						<input>
							<port id="0" precision="FP32">
								<dim>1</dim>
								<dim>128</dim>
							</port>
							<port id="1" precision="I64">
								<dim>1</dim>
							</port>
						</input>
						<output>
							<port id="2" precision="FP32">
								<dim>1</dim>
								<dim>1</dim>
								<dim>128</dim>
							</port>
						</output>
					</layer>
					<layer id="9" name="Result_2693" type="Result" version="opset1">
						<input>
							<port id="0" precision="FP32">
								<dim>1</dim>
								<dim>1</dim>
								<dim>128</dim>
							</port>
						</input>
					</layer>
					<layer id="10" name="Result_2692" type="Result" version="opset1">
						<input>
							<port id="0" precision="FP32">
								<dim>1</dim>
								<dim>128</dim>
							</port>
						</input>
					</layer>
				</layers>
				<edges>
					<edge from-layer="0" from-port="0" to-layer="7" to-port="1" />
					<edge from-layer="1" from-port="0" to-layer="3" to-port="0" />
					<edge from-layer="2" from-port="0" to-layer="3" to-port="1" />
					<edge from-layer="2" from-port="0" to-layer="8" to-port="1" />
					<edge from-layer="3" from-port="2" to-layer="7" to-port="0" />
					<edge from-layer="4" from-port="0" to-layer="7" to-port="2" />
					<edge from-layer="5" from-port="0" to-layer="7" to-port="3" />
					<edge from-layer="6" from-port="0" to-layer="7" to-port="4" />
					<edge from-layer="7" from-port="5" to-layer="8" to-port="0" />
					<edge from-layer="7" from-port="5" to-layer="10" to-port="0" />
					<edge from-layer="8" from-port="2" to-layer="9" to-port="0" />
				</edges>
				<rt_info />
			</body>
		</layer>
		<layer id="3" name="Result_2695" type="Result" version="opset1">
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>128</dim>
				</port>
			</input>
		</layer>
		<layer id="4" name="Result_2694" type="Result" version="opset1">
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>2</dim>
					<dim>128</dim>
				</port>
			</input>
		</layer>
	</layers>
	<edges>
		<edge from-layer="0" from-port="0" to-layer="2" to-port="1" />
		<edge from-layer="1" from-port="0" to-layer="2" to-port="0" />
		<edge from-layer="2" from-port="3" to-layer="3" to-port="0" />
		<edge from-layer="2" from-port="2" to-layer="4" to-port="0" />
	</edges>
	<rt_info />
</net>
    )V0G0N";

    std::vector<unsigned char> buffer(213512, 0);
    int64_t* int64Buffer = reinterpret_cast<int64_t*>(buffer.data());
    int64Buffer[0] = 1;

    createTemporalModelFile(xmlModel, buffer);
    std::shared_ptr<ov::Model> model;

    OV_ASSERT_NO_THROW(model = core.read_model(xmlFileName, binFileName));
    ASSERT_TRUE(!!model);
}
