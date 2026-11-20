// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <vector>

#include "common_test_utils/test_assertions.hpp"
#include "frontend_test.hpp"
#include "openvino/openvino.hpp"

/**
 * @brief Test fixture for verifying descriptive diagnostics during IR XML deserialization
 * when Constant operator tensor buffer sizes or shapes are inconsistent.
 */
class ConstShapeMismatchDeserializationTest : public ::testing::Test, public IRFrontendTestsImpl {
protected:
    void SetUp() override {}

    void TearDown() override {
        RemoveTemporalFiles();
    }
};

/**
 * @brief Verifies that an IR model with an undersized FP32 constant buffer throws an
 * ov::Exception containing the layer friendly name, layer type, layer ID, shape,
 * element type, expected bytes, actual bytes, and byte delta.
 */
TEST_F(ConstShapeMismatchDeserializationTest, undersized_fp32_buffer_throws_descriptive_exception) {
    const std::string xmlModel = R"V0G0N(
<net name="test_undersized_constant" version="10">
	<layers>
		<layer id="0" name="Input_Param" type="Parameter" version="opset1">
			<data element_type="f32" shape="2,2"/>
			<output>
				<port id="0" names="Input_Param" precision="FP32">
					<dim>2</dim>
					<dim>2</dim>
				</port>
			</output>
		</layer>
		<layer id="42" name="weights_layer_bad" type="Const" version="opset1">
			<data element_type="f32" shape="2,2" offset="0" size="8"/>
			<output>
				<port id="0" names="weights_layer_bad" precision="FP32">
					<dim>2</dim>
					<dim>2</dim>
				</port>
			</output>
		</layer>
		<layer id="2" name="Add_Node" type="Add" version="opset1">
			<input>
				<port id="0" precision="FP32">
					<dim>2</dim>
					<dim>2</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>2</dim>
					<dim>2</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>2</dim>
					<dim>2</dim>
				</port>
			</output>
		</layer>
		<layer id="3" name="Output_Result" type="Result" version="opset1">
			<input>
				<port id="0" precision="FP32">
					<dim>2</dim>
					<dim>2</dim>
				</port>
			</input>
		</layer>
	</layers>
	<edges>
		<edge from-layer="0" from-port="0" to-layer="2" to-port="0"/>
		<edge from-layer="42" from-port="0" to-layer="2" to-port="1"/>
		<edge from-layer="2" from-port="2" to-layer="3" to-port="0"/>
	</edges>
</net>
)V0G0N";

    std::vector<unsigned char> binBuffer(16, 0);
    createTemporalModelFile(xmlModel, binBuffer);

    OV_EXPECT_THROW(core.read_model(xmlFileName, binFileName),
                    ov::Exception,
                    testing::AllOf(testing::HasSubstr("weights_layer_bad"),
                                   testing::HasSubstr("Const"),
                                   testing::HasSubstr("id: 42"),
                                   testing::HasSubstr("shape [2, 2]"),
                                   testing::HasSubstr("f32"),
                                   testing::HasSubstr("Expected at least 16 byte(s)"),
                                   testing::HasSubstr("got 8 byte(s)"),
                                   testing::HasSubstr("8 byte(s) missing")));
}

/**
 * @brief Verifies that an IR model with an unnamed Constant layer emits safe fallback "[Unnamed Constant Node]".
 */
TEST_F(ConstShapeMismatchDeserializationTest, unnamed_layer_emits_safe_fallback) {
    const std::string xmlModel = R"V0G0N(
<net name="test_unnamed_constant" version="10">
	<layers>
		<layer id="0" name="Input_Param" type="Parameter" version="opset1">
			<data element_type="f32" shape="2,2"/>
			<output>
				<port id="0" names="Input_Param" precision="FP32">
					<dim>2</dim>
					<dim>2</dim>
				</port>
			</output>
		</layer>
		<layer id="43" name="" type="Const" version="opset1">
			<data element_type="f32" shape="2,2" offset="0" size="4"/>
			<output>
				<port id="0" precision="FP32">
					<dim>2</dim>
					<dim>2</dim>
				</port>
			</output>
		</layer>
		<layer id="2" name="Add_Node" type="Add" version="opset1">
			<input>
				<port id="0" precision="FP32">
					<dim>2</dim>
					<dim>2</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>2</dim>
					<dim>2</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>2</dim>
					<dim>2</dim>
				</port>
			</output>
		</layer>
		<layer id="3" name="Output_Result" type="Result" version="opset1">
			<input>
				<port id="0" precision="FP32">
					<dim>2</dim>
					<dim>2</dim>
				</port>
			</input>
		</layer>
	</layers>
	<edges>
		<edge from-layer="0" from-port="0" to-layer="2" to-port="0"/>
		<edge from-layer="43" from-port="0" to-layer="2" to-port="1"/>
		<edge from-layer="2" from-port="2" to-layer="3" to-port="0"/>
	</edges>
</net>
)V0G0N";

    std::vector<unsigned char> binBuffer(16, 0);
    createTemporalModelFile(xmlModel, binBuffer);

    OV_EXPECT_THROW(core.read_model(xmlFileName, binFileName),
                    ov::Exception,
                    testing::AllOf(testing::HasSubstr("[Unnamed Constant Node]"),
                                   testing::HasSubstr("id: 43"),
                                   testing::HasSubstr("shape [2, 2]"),
                                   testing::HasSubstr("Expected at least 16 byte(s)"),
                                   testing::HasSubstr("got 4 byte(s)"),
                                   testing::HasSubstr("12 byte(s) missing")));
}

/**
 * @brief Verifies that an IR model with an undersized sub-byte U4 quantized constant buffer
 * correctly computes ceiling bitwidth arithmetic and throws with accurate byte delta.
 */
TEST_F(ConstShapeMismatchDeserializationTest, undersized_u4_buffer_computes_exact_bitwidth_delta) {
    const std::string xmlModel = R"V0G0N(
<net name="test_u4_constant" version="10">
	<layers>
		<layer id="0" name="Input_Param" type="Parameter" version="opset1">
			<data element_type="u4" shape="6"/>
			<output>
				<port id="0" names="Input_Param" precision="U4">
					<dim>6</dim>
				</port>
			</output>
		</layer>
		<layer id="99" name="q4_weights_node" type="Const" version="opset1">
			<data element_type="u4" shape="6" offset="0" size="1"/>
			<output>
				<port id="0" names="q4_weights_node" precision="U4">
					<dim>6</dim>
				</port>
			</output>
		</layer>
		<layer id="2" name="Add_Node" type="Add" version="opset1">
			<input>
				<port id="0" precision="U4">
					<dim>6</dim>
				</port>
				<port id="1" precision="U4">
					<dim>6</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="U4">
					<dim>6</dim>
				</port>
			</output>
		</layer>
		<layer id="3" name="Output_Result" type="Result" version="opset1">
			<input>
				<port id="0" precision="U4">
					<dim>6</dim>
				</port>
			</input>
		</layer>
	</layers>
	<edges>
		<edge from-layer="0" from-port="0" to-layer="2" to-port="0"/>
		<edge from-layer="99" from-port="0" to-layer="2" to-port="1"/>
		<edge from-layer="2" from-port="2" to-layer="3" to-port="0"/>
	</edges>
</net>
)V0G0N";

    std::vector<unsigned char> binBuffer(8, 0);
    createTemporalModelFile(xmlModel, binBuffer);

    OV_EXPECT_THROW(core.read_model(xmlFileName, binFileName),
                    ov::Exception,
                    testing::AllOf(testing::HasSubstr("q4_weights_node"),
                                   testing::HasSubstr("id: 99"),
                                   testing::HasSubstr("shape [6]"),
                                   testing::HasSubstr("u4"),
                                   testing::HasSubstr("Expected at least 3 byte(s)"),
                                   testing::HasSubstr("got 1 byte(s)"),
                                   testing::HasSubstr("2 byte(s) missing")));
}

/**
 * @brief Positive test verifying that valid Constant layers deserialize with 0 regressions.
 */
TEST_F(ConstShapeMismatchDeserializationTest, valid_constant_model_deserializes_successfully) {
    const std::string xmlModel = R"V0G0N(
<net name="test_valid_constant" version="10">
	<layers>
		<layer id="0" name="Input_Param" type="Parameter" version="opset1">
			<data element_type="f32" shape="2,2"/>
			<output>
				<port id="0" names="Input_Param" precision="FP32">
					<dim>2</dim>
					<dim>2</dim>
				</port>
			</output>
		</layer>
		<layer id="1" name="valid_weights" type="Const" version="opset1">
			<data element_type="f32" shape="2,2" offset="0" size="16"/>
			<output>
				<port id="0" names="valid_weights" precision="FP32">
					<dim>2</dim>
					<dim>2</dim>
				</port>
			</output>
		</layer>
		<layer id="2" name="Add_Node" type="Add" version="opset1">
			<input>
				<port id="0" precision="FP32">
					<dim>2</dim>
					<dim>2</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>2</dim>
					<dim>2</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>2</dim>
					<dim>2</dim>
				</port>
			</output>
		</layer>
		<layer id="3" name="Output_Result" type="Result" version="opset1">
			<input>
				<port id="0" precision="FP32">
					<dim>2</dim>
					<dim>2</dim>
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

    std::vector<unsigned char> binBuffer(16, 0);
    float* fBuf = reinterpret_cast<float*>(binBuffer.data());
    fBuf[0] = 1.0f;
    fBuf[1] = 2.0f;
    fBuf[2] = 3.0f;
    fBuf[3] = 4.0f;

    createTemporalModelFile(xmlModel, binBuffer);

    std::shared_ptr<ov::Model> model;
    EXPECT_NO_THROW(model = core.read_model(xmlFileName, binFileName));
    ASSERT_NE(model, nullptr);
    EXPECT_EQ(model->get_ops().size(), 4);
}
