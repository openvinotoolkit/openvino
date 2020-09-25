// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <fstream>

#include <gtest/gtest.h>

#include "common_test_utils/ngraph_test_utils.hpp"
#include "ngraph/attribute_visitor.hpp"
#include "pugixml.hpp"
#include <ie_core.hpp>
#include <ngraph/ngraph.hpp>

struct BasicModel {
  const std::string model = R"V0G0N(
<?xml version="1.0" ?>
<net name="add_abc" version="10">
	<layers>
		<layer id="0" name="A" type="Parameter" version="opset1">
			<data element_type="f32" shape="1"/>
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="1" name="B" type="Parameter" version="opset1">
			<data element_type="f32" shape="1"/>
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="2" name="add_node1" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
				</port>
				<port id="1">
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="3" name="C" type="Parameter" version="opset1">
			<data element_type="f32" shape="1"/>
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="4" name="Y" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
				</port>
				<port id="1">
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="5" name="Y/sink_port_0" type="Result" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
				</port>
			</input>
		</layer>
	</layers>
	<edges>
		<edge from-layer="0" from-port="0" to-layer="2" to-port="0"/>
		<edge from-layer="1" from-port="0" to-layer="2" to-port="1"/>
		<edge from-layer="2" from-port="2" to-layer="4" to-port="0"/>
		<edge from-layer="3" from-port="0" to-layer="4" to-port="1"/>
		<edge from-layer="4" from-port="2" to-layer="5" to-port="0"/>
	</edges>
</net>
)V0G0N";

  const InferenceEngine::Blob::Ptr weights{};
};

struct ConstModel {
  const std::string model = R"V0G0N(
<?xml version="1.0" ?>
<net name="add_abc_const" version="10">
	<layers>
		<layer id="0" name="add_node1/Output_0/Data__const" type="Const" version="opset1">
			<data element_type="f32" offset="0" shape="2,2" size="16"/>
			<output>
				<port id="1" precision="FP32">
					<dim>2</dim>
					<dim>2</dim>
				</port>
			</output>
		</layer>
		<layer id="1" name="C" type="Parameter" version="opset1">
			<data element_type="f32" shape="2,2"/>
			<output>
				<port id="0" precision="FP32">
					<dim>2</dim>
					<dim>2</dim>
				</port>
			</output>
		</layer>
		<layer id="2" name="Y" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>2</dim>
					<dim>2</dim>
				</port>
				<port id="1">
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
		<layer id="3" name="Y/sink_port_0" type="Result" version="opset1">
			<input>
				<port id="0">
					<dim>2</dim>
					<dim>2</dim>
				</port>
			</input>
		</layer>
	</layers>
	<edges>
		<edge from-layer="0" from-port="1" to-layer="2" to-port="0"/>
		<edge from-layer="1" from-port="0" to-layer="2" to-port="1"/>
		<edge from-layer="2" from-port="2" to-layer="3" to-port="0"/>
	</edges>
</net>
)V0G0N";

  std::vector<float> weights{2, 4, 6, 8};
  InferenceEngine::Blob::CPtr weights_blob =
      InferenceEngine::make_shared_blob<float>(
          InferenceEngine::TensorDesc(InferenceEngine::Precision::FP32, {2, 2},
                                      InferenceEngine::NC),
          &weights[0]);
};

class SerializationTest : public ::testing::Test {
protected:
  std::string m_out_xml_path = "tmp.xml";
  std::string m_out_bin_path = "tmp.bin";

  void TearDown() override {
#if 0 // TODO: remove debug code
    std::remove(m_out_xml_path.c_str());
    std::remove(m_out_bin_path.c_str());
#endif
  }
};

// TODO: compare_functions() is missing atribute comparison
TEST_F(SerializationTest, BasicModel) {
  InferenceEngine::Core ie;
  BasicModel m;
  auto expected = ie.ReadNetwork(m.model, m.weights);
  expected.serialize(m_out_xml_path, m_out_bin_path);
  auto result = ie.ReadNetwork(m_out_xml_path, m_out_bin_path);

  bool success;
  std::string message;
  std::tie(success, message) =
      compare_functions(result.getFunction(), expected.getFunction());

  ASSERT_TRUE(success) << message;
}

TEST_F(SerializationTest, DISABLED_ModelWithMultipleOutputs) {
  FAIL() << "not implemented";
}

TEST_F(SerializationTest, DISABLED_ModelWithMultipleLayers) {
  FAIL() << "not implemented";
}

TEST_F(SerializationTest, ModelWithConstants) {
  InferenceEngine::Core ie;

  ConstModel m;
  auto expected = ie.ReadNetwork(m.model, m.weights_blob);
  expected.serialize(m_out_xml_path, m_out_bin_path);
  auto result = ie.ReadNetwork(m_out_xml_path, m_out_bin_path);

  bool success;
  std::string message;
  std::tie(success, message) =
      compare_functions(result.getFunction(), expected.getFunction());

  ASSERT_TRUE(success) << message;
}
