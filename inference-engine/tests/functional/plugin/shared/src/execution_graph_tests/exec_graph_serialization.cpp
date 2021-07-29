// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <common_test_utils/file_utils.hpp>
#include "execution_graph_tests/exec_graph_serialization.hpp"

namespace ExecutionGraphTests {

const char serialize_test_model[] = R"V0G0N(<?xml version="1.0" ?>
<?xml version="1.0" ?>
<net name="addmul_abc" version="10">
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
		<layer id="3" name="add_node2" type="Multiply" version="opset1">
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
		<layer id="4" name="add_node3" type="Add" version="opset1">
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
		<layer id="5" name="C" type="Parameter" version="opset1">
			<data element_type="f32" shape="1"/>
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="6" name="add_node4" type="Add" version="opset1">
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
		<layer id="7" name="Y" type="Add" version="opset1">
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
		<layer id="8" name="Y/sink_port_0" type="Result" version="opset1">
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
		<edge from-layer="0" from-port="0" to-layer="3" to-port="0"/>
		<edge from-layer="1" from-port="0" to-layer="3" to-port="1"/>
		<edge from-layer="2" from-port="2" to-layer="4" to-port="0"/>
		<edge from-layer="3" from-port="2" to-layer="4" to-port="1"/>
		<edge from-layer="4" from-port="2" to-layer="6" to-port="0"/>
		<edge from-layer="5" from-port="0" to-layer="6" to-port="1"/>
		<edge from-layer="6" from-port="2" to-layer="7" to-port="0"/>
		<edge from-layer="5" from-port="0" to-layer="7" to-port="1"/>
		<edge from-layer="7" from-port="2" to-layer="8" to-port="0"/>
	</edges>
</net>
)V0G0N";

const char expected_serialized_model[] = R"V0G0N(
<?xml version="1.0"?>
<net name="addmul_abc" version="10">
	<layers>
		<layer id="0" name="C" type="Input">
			<data shape="1" element_type="f32" execOrder="3" execTimeMcs="not_executed" originalLayersNames="C" outputLayouts="x" outputPrecisions="FP32" primitiveType="unknown_FP32" runtimePrecision="FP32" />
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="1" name="B" type="Input">
			<data shape="1" element_type="f32" execOrder="1" execTimeMcs="not_executed" originalLayersNames="B" outputLayouts="x" outputPrecisions="FP32" primitiveType="unknown_FP32" runtimePrecision="FP32"/>
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="2" name="A" type="Input">
			<data shape="1" element_type="f32" execOrder="0" execTimeMcs="not_executed" originalLayersNames="A" outputLayouts="x" outputPrecisions="FP32" primitiveType="unknown_FP32" runtimePrecision="FP32"/>
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="3" name="add_node2" type="Eltwise">
			<data execOrder="2" execTimeMcs="not_executed" originalLayersNames="add_node2" outputLayouts="x" outputPrecisions="FP32" primitiveType="jit_avx512_FP32" runtimePrecision="FP32"/>
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
		<layer id="4" name="add_node1" type="Eltwise">
			<data execOrder="4" execTimeMcs="not_executed" originalLayersNames="add_node1,add_node3,add_node4" outputLayouts="x" outputPrecisions="FP32" primitiveType="jit_avx512_FP32" runtimePrecision="FP32"/>
			<input>
				<port id="0">
					<dim>1</dim>
				</port>
				<port id="1">
					<dim>1</dim>
				</port>
				<port id="2">
					<dim>1</dim>
				</port>
				<port id="3">
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="4" precision="FP32">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="5" name="Y" type="Eltwise">
			<data execOrder="5" execTimeMcs="not_executed" originalLayersNames="Y" outputLayouts="x" outputPrecisions="FP32" primitiveType="jit_avx512_FP32" runtimePrecision="FP32"/>
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
		<layer id="6" name="out_Y" type="Output">
			<data execOrder="6" execTimeMcs="not_executed" originalLayersNames="" outputLayouts="undef" outputPrecisions="FP32" primitiveType="unknown_FP32" runtimePrecision="FP32"/>
			<input>
				<port id="0">
					<dim>1</dim>
				</port>
			</input>
		</layer>
	</layers>
	<edges>
		<edge from-layer="0" from-port="0" to-layer="4" to-port="3" />
		<edge from-layer="0" from-port="0" to-layer="5" to-port="1" />
		<edge from-layer="1" from-port="0" to-layer="3" to-port="1" />
		<edge from-layer="1" from-port="0" to-layer="4" to-port="1" />
		<edge from-layer="2" from-port="0" to-layer="3" to-port="0" />
		<edge from-layer="2" from-port="0" to-layer="4" to-port="0" />
		<edge from-layer="3" from-port="2" to-layer="4" to-port="2" />
		<edge from-layer="4" from-port="4" to-layer="5" to-port="0" />
		<edge from-layer="5" from-port="2" to-layer="6" to-port="0" />
	</edges>
</net>
)V0G0N";


std::string ExecGraphSerializationTest::getTestCaseName(testing::TestParamInfo<std::string> obj) {
    std::ostringstream result;
    std::string model_name, expected_model_name;
    std::string targetDevice = obj.param;
    result << "ModelName=" << model_name << "_";
    result << "ExpectedModelName=" << expected_model_name << "_";
    result << "TargetDevice=" << targetDevice;
    return result.str();
}

void ExecGraphSerializationTest::SetUp() {
    const std::string XML_EXT = ".xml";
    const std::string BIN_EXT = ".bin";

    std::string test_name = ::testing::UnitTest::GetInstance()->current_test_info()->name();
    test_name.erase(std::remove(test_name.begin(), test_name.end(), '/'), test_name.end());
    test_name += getTimestamp();

    m_out_xml_path = test_name + XML_EXT;
    m_out_bin_path = test_name + BIN_EXT;

    deviceName = this->GetParam();
}

void ExecGraphSerializationTest::TearDown() {
    CommonTestUtils::removeIRFiles(m_out_xml_path, m_out_bin_path);
}

std::string ExecGraphSerializationTest::getTimestamp() {
    auto now = std::chrono::system_clock::now();
    auto epoch = now.time_since_epoch();
    auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(epoch);
    return std::to_string(ns.count());
}

bool ExecGraphSerializationTest::exec_graph_walker::for_each(pugi::xml_node &node) {
    std::string node_name{node.name()};
    if (node_name == "layer" || node_name == "data") {
        nodes.push_back(node);
    }
    return true;  // continue traversal
}

std::pair<bool, std::string> ExecGraphSerializationTest::compare_nodes(const pugi::xml_node &node1,
                                                                       const pugi::xml_node &node2) {
    // node names must be the same
    const std::string node1_name{node1.name()};
    const std::string node2_name{node2.name()};
    if (node1_name != node2_name) {
        return {false, "Node name differ: " + node1_name + " != " + node2_name};
    }

    // node attribute count must be the same
    const auto attr1 = node1.attributes();
    const auto attr2 = node2.attributes();
    const auto attr1_size = std::distance(attr1.begin(), attr1.end());
    const auto attr2_size = std::distance(attr2.begin(), attr2.end());
    if (attr1_size != attr2_size) {
        return {false, "Attribute count differ in <" + node1_name + "> :" +
                       std::to_string(attr1_size) + " != " +
                       std::to_string(attr2_size)};
    }

    // every node attribute name must be the same
    auto a1 = attr1.begin();
    auto a2 = attr2.begin();
    for (int j = 0; j < attr1_size; ++j, ++a1, ++a2) {
        const std::string a1_name{a1->name()};
        const std::string a2_name{a2->name()};
        const std::string a1_value{a1->value()};
        const std::string a2_value{a2->value()};
        if (a1_name != a2_name || (a1_name == "type" && a1_value != a2_value)) {
            return {false, "Attributes differ in <" + node1_name + "> : " +
                           a1_name + "=" + a1_value + " != " + a2_name +
                           "=" + a2_value};
        }
    }
    return {true, ""};
}

std::pair<bool, std::string> ExecGraphSerializationTest::compare_docs(const pugi::xml_document &doc1,
                                                                      const pugi::xml_document &doc2) {
    // traverse document and prepare vector of <layer> & <data> nodes to compare
    exec_graph_walker walker1, walker2;
    doc1.child("net").child("layers").traverse(walker1);
    doc2.child("net").child("layers").traverse(walker2);

    // nodes count must be the same
    const auto &nodes1 = walker1.nodes;
    const auto &nodes2 = walker2.nodes;
    if (nodes1.size() != nodes2.size()) {
        return {false, "Node count differ: " + std::to_string(nodes1.size()) +
                       " != " + std::to_string(nodes2.size())};
    }

    // every node must be equivalent
    for (int i = 0; i < nodes1.size(); i++) {
        const auto res = compare_nodes(nodes1[i], nodes2[i]);
        if (res.first == false) {
            return res;
        }
    }
    return {true, ""};
}

TEST_P(ExecGraphSerializationTest, ExecutionGraph) {
    InferenceEngine::Core ie;
    InferenceEngine::Blob::Ptr a;
    auto cnnNet = ie.ReadNetwork(serialize_test_model, a);
    auto execNet = ie.LoadNetwork(cnnNet, deviceName);
    auto execGraph = execNet.GetExecGraphInfo();
    InferenceEngine::InferRequest req = execNet.CreateInferRequest();
    execGraph.serialize(m_out_xml_path, m_out_bin_path);

    pugi::xml_document expected;
    pugi::xml_document result;
    ASSERT_TRUE(expected.load_string(expected_serialized_model));
    ASSERT_TRUE(result.load_file(m_out_xml_path.c_str()));

    bool status;
    std::string message;
    std::tie(status, message) = this->compare_docs(expected, result);

    ASSERT_TRUE(status) << message;
}
} // namespace ExecutionGraphTests
