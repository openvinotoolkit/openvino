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

// checks if layer & data nodes of two exec graph xml are the same
std::pair<bool, std::string> compare_docs(pugi::xml_document& doc1,
                                          pugi::xml_document& doc2) {
    exec_graph_walker walker1, walker2;
    doc1.child("net").child("layers").traverse(walker1);
    doc2.child("net").child("layers").traverse(walker2);

    if (walker1.nodes.size() != walker2.nodes.size()) {
        return {false, "XML's contain different number of nodes"};
    }

    // for every node
    for (int i = 0; i < walker1.nodes.size(); i++) {
        auto attr1 = walker1.nodes[i].attributes();
        auto attr2 = walker2.nodes[i].attributes();
        auto attr1_size = std::distance(attr1.begin(), attr1.end());
        auto attr2_size = std::distance(attr2.begin(), attr2.end());
        std::string node_name{walker1.nodes[i].name()};

        if (attr1_size != attr2_size) {
            return {false,
                    node_name + " contain different number of attributes"};
        }
        // for every attribute in node
        if (attr1_size > 0) {
            auto a1 = attr1.begin();
            auto a2 = attr2.begin();
            for (int j = 0; j < attr1_size; j++) {
                std::string a1_name{a1->name()};
                std::string a2_name{a2->name()};
                if (a1_name != a2_name) {
                    return {false,
                            node_name + " contain different attribute names"};
                }
                if (std::string(a1->value()) != std::string(a2->value())) {
                    return {false,
                            node_name + " contain different attribute values"};
                }
                ++a1;
                ++a2;
            }
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

TEST_F(ExecGraphSerializationTest, ExecutionGraph) {
    const std::string source_model =
        IR_SERIALIZATION_MODELS_PATH "addmul_abc.xml";
    const std::string expected_model =
        IR_SERIALIZATION_MODELS_PATH "addmul_abc_execution.xml";

    InferenceEngine::Core ie;
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
}
