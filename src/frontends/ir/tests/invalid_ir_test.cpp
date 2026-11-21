// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>
#include <sstream>
#include <string>

#include "common_test_utils/test_assertions.hpp"
#include "openvino/frontend/manager.hpp"

/**
 * @brief Generic test fixture for verifying diagnostics and validation during IR XML deserialization.
 */
class InvalidIRTest : public testing::Test {
protected:
    std::shared_ptr<ov::Model> get_with_ir_frontend(const std::string& model) {
        std::istringstream model_string_stream(model);
        std::istream& model_stream = model_string_stream;

        ov::frontend::FrontEnd::Ptr fe;
        ov::frontend::InputModel::Ptr input_model;

        ov::AnyVector params{&model_stream};

        fe = manager.load_by_model(params);
        if (fe)
            input_model = fe->load(params);

        if (input_model)
            return fe->convert(input_model);

        return nullptr;
    }

private:
    ov::frontend::FrontEndManager manager;
};

using DisconnectedPortDeserializationTest = InvalidIRTest;

/**
 * @brief Verifies that an IR model with a missing incoming edge on an input port throws an
 * ov::Exception containing the layer friendly name, layer type, layer ID, and missing port ID.
 */
TEST_F(DisconnectedPortDeserializationTest, missing_input_edge_throws_descriptive_exception) {
    const std::string model = R"V0G0N(
<net name="test_missing_edge" version="10">
	<layers>
		<layer id="0" name="Input_A" type="Parameter" version="opset1">
			<data element_type="f32" shape="1"/>
			<output>
				<port id="0" names="Input_A" precision="FP32">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="1" name="Input_B" type="Parameter" version="opset1">
			<data element_type="f32" shape="1"/>
			<output>
				<port id="0" names="Input_B" precision="FP32">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="2" name="Layer_MissingEdge" type="Add" version="opset1">
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="3" name="Output_Result" type="Result" version="opset1">
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
				</port>
			</input>
		</layer>
	</layers>
	<edges>
		<edge from-layer="0" from-port="0" to-layer="2" to-port="0"/>
		<edge from-layer="2" from-port="2" to-layer="3" to-port="0"/>
	</edges>
</net>
)V0G0N";

    OV_EXPECT_THROW(get_with_ir_frontend(model),
                    ov::Exception,
                    testing::AllOf(testing::HasSubstr("Layer_MissingEdge"),
                                   testing::HasSubstr("Add"),
                                   testing::HasSubstr("id: 2"),
                                   testing::HasSubstr("[1]")));
}

/**
 * @brief Positive test verifying that a valid IR model with correctly connected input ports
 * deserializes without throwing.
 */
TEST_F(DisconnectedPortDeserializationTest, valid_connected_ports_deserializes_successfully) {
    const std::string model = R"V0G0N(
<net name="test_valid_ports" version="10">
	<layers>
		<layer id="0" name="Input_A" type="Parameter" version="opset1">
			<data element_type="f32" shape="1"/>
			<output>
				<port id="0" names="Input_A" precision="FP32">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="1" name="Input_B" type="Parameter" version="opset1">
			<data element_type="f32" shape="1"/>
			<output>
				<port id="0" names="Input_B" precision="FP32">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="2" name="Layer_Valid" type="Add" version="opset1">
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="3" name="Output_Result" type="Result" version="opset1">
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
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

    std::shared_ptr<ov::Model> ov_model;
    EXPECT_NO_THROW(ov_model = get_with_ir_frontend(model));
    ASSERT_NE(ov_model, nullptr);
    EXPECT_EQ(ov_model->get_ops().size(), 4);
}

/**
 * @brief Regression test: verifies that duplicate destination edges targeting the same input port
 * cannot bypass missing-input validation, especially when declared ports use non-contiguous IDs (e.g. 10 and 20).
 */
TEST_F(DisconnectedPortDeserializationTest, duplicate_destination_edges_with_noncontiguous_ports_throws_descriptive_exception) {
    const std::string model = R"V0G0N(
<net name="test_duplicate_edges_noncontiguous" version="10">
	<layers>
		<layer id="0" name="Input_A" type="Parameter" version="opset1">
			<data element_type="f32" shape="1"/>
			<output>
				<port id="0" names="Input_A" precision="FP32">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="1" name="Input_B" type="Parameter" version="opset1">
			<data element_type="f32" shape="1"/>
			<output>
				<port id="0" names="Input_B" precision="FP32">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="2" name="Layer_DuplicateEdges" type="Add" version="opset1">
			<input>
				<port id="10" precision="FP32">
					<dim>1</dim>
				</port>
				<port id="20" precision="FP32">
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="30" precision="FP32">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="3" name="Output_Result" type="Result" version="opset1">
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
				</port>
			</input>
		</layer>
	</layers>
	<edges>
		<edge from-layer="0" from-port="0" to-layer="2" to-port="10"/>
		<edge from-layer="1" from-port="0" to-layer="2" to-port="10"/>
		<edge from-layer="2" from-port="30" to-layer="3" to-port="0"/>
	</edges>
</net>
)V0G0N";

    OV_EXPECT_THROW(get_with_ir_frontend(model),
                    ov::Exception,
                    testing::AllOf(testing::HasSubstr("Layer_DuplicateEdges"),
                                   testing::HasSubstr("Add"),
                                   testing::HasSubstr("id: 2"),
                                   testing::HasSubstr("[20]")));
}
