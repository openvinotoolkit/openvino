// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <fstream>

#include "common_test_utils/ngraph_test_utils.hpp"
#include "gtest/gtest.h"
#include "ie_core.hpp"
#include "pugixml.hpp"

#ifndef IR_SERIALIZATION_MODELS_PATH  // should be already defined by cmake
#define IR_SERIALIZATION_MODELS_PATH ""
#endif

// walker traverse (DFS) xml document and store layer & data nodes in
// vector which is later used for comparison
struct exec_graph_walker : pugi::xml_tree_walker {
    std::vector<pugi::xml_node> nodes;

    virtual bool for_each(pugi::xml_node& node) {
        std::string node_name{node.name()};
        if (node_name == "layer" || node_name == "data") {
            nodes.push_back(node);
        }
        return true;  // continue traversal
    }
};

// compare_docs() helper
std::pair<bool, std::string> compare_nodes(const pugi::xml_node& node1,
                                           const pugi::xml_node& node2) {
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

// checks if two exec graph xml's are equivalent:
// - the same count of <layer> and <data> nodes
// - the same count of attributes of each node
// - the same name of each attribute (value is not checked, since it can differ
// beetween different devices)
std::pair<bool, std::string> compare_docs(const pugi::xml_document& doc1,
                                          const pugi::xml_document& doc2) {
    // traverse document and prepare vector of <layer> & <data> nodes to compare
    exec_graph_walker walker1, walker2;
    doc1.child("net").child("layers").traverse(walker1);
    doc2.child("net").child("layers").traverse(walker2);

    // nodes count must be the same
    const auto& nodes1 = walker1.nodes;
    const auto& nodes2 = walker2.nodes;
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

class ExecGraphSerializationTest : public ::testing::Test {
protected:
    std::string test_name =
        ::testing::UnitTest::GetInstance()->current_test_info()->name();
    std::string m_out_xml_path = test_name + ".xml";
    std::string m_out_bin_path = test_name + ".bin";

    void TearDown() override {
        std::remove(m_out_xml_path.c_str());
        std::remove(m_out_bin_path.c_str());
    }
};

TEST_F(ExecGraphSerializationTest, DISABLED_ExecutionGraph_CPU) {
    const std::string source_model =
        IR_SERIALIZATION_MODELS_PATH "addmul_abc.xml";
    const std::string expected_model =
        IR_SERIALIZATION_MODELS_PATH "addmul_abc_execution.xml";

    InferenceEngine::Core ie;
    auto devices = ie.GetAvailableDevices();
    if (std::find(devices.begin(), devices.end(), "CPU") != devices.end()) {
        auto cnnNet = ie.ReadNetwork(source_model);
        auto execNet = ie.LoadNetwork(cnnNet, "CPU");
        auto execGraph = execNet.GetExecGraphInfo();
        InferenceEngine::InferRequest req = execNet.CreateInferRequest();
        execGraph.serialize(m_out_xml_path, m_out_bin_path);

        pugi::xml_document expected;
        pugi::xml_document result;
        ASSERT_TRUE(expected.load_file(expected_model.c_str()));
        ASSERT_TRUE(result.load_file(m_out_xml_path.c_str()));

        bool success;
        std::string message;
        std::tie(success, message) = compare_docs(expected, result);

        ASSERT_TRUE(success) << message;
    } else {
        // no CPU device available so we are ignoring this test
        GTEST_SKIP();
    }
}
