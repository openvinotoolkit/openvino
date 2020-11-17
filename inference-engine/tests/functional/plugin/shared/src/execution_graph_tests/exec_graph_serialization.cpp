// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "execution_graph_tests/exec_graph_serialization.hpp"

namespace ExecutionGraphTests {

std::string ExecGraphSerializationTest::getTestCaseName(testing::TestParamInfo<ExecGraphSerializationParam> obj) {
    std::ostringstream result;
    std::string model_name, expected_model_name;
    std::tie(model_name, expected_model_name) = std::get<0>(obj.param);
    std::string targetDevice = std::get<1>(obj.param);
    result << "ModelName=" << model_name << "_";
    result << "ExpectedModelName=" << expected_model_name << "_";
    result << "TargetDevice=" << targetDevice;
    return result.str();
}

void ExecGraphSerializationTest::SetUp() {
    const std::string XML_EXT = ".xml";
    const std::string BIN_EXT = ".bin";

    std::string test_name = ::testing::UnitTest::GetInstance()->current_test_info()->name();
    std::remove(test_name.begin(), test_name.end(), '/');

    m_out_xml_path = test_name + XML_EXT;
    m_out_bin_path = test_name + BIN_EXT;

    deviceName = std::get<1>(this->GetParam());

    std::string model_name, expected_model_name;
    std::tie(model_name, expected_model_name) = std::get<0>(this->GetParam());
    model_name += XML_EXT;
    expected_model_name += XML_EXT;
    source_model = IR_SERIALIZATION_MODELS_PATH + model_name;
    expected_model = IR_SERIALIZATION_MODELS_PATH + expected_model_name;
}

void ExecGraphSerializationTest::TearDown() {
    std::remove(m_out_xml_path.c_str());
    std::remove(m_out_bin_path.c_str());
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
        if ((a1_name != a2_name)) {
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
    auto cnnNet = ie.ReadNetwork(source_model);
    auto execNet = ie.LoadNetwork(cnnNet, deviceName);
    auto execGraph = execNet.GetExecGraphInfo();
    InferenceEngine::InferRequest req = execNet.CreateInferRequest();
    execGraph.serialize(m_out_xml_path, m_out_bin_path);

    pugi::xml_document expected;
    pugi::xml_document result;
    ASSERT_TRUE(expected.load_file(expected_model.c_str()));
    ASSERT_TRUE(result.load_file(m_out_xml_path.c_str()));

    bool status;
    std::string message;
    std::tie(status, message) = this->compare_docs(expected, result);

    ASSERT_TRUE(status) << message;
}
} // namespace ExecutionGraphTests
