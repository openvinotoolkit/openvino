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

namespace {
// walker traverse (DFS) xml document and store layer type attriubte in
// vector which is later used for comparison
struct type_attribute_walker : pugi::xml_tree_walker {
    std::vector<pugi::xml_attribute> attributes;

    virtual bool for_each(pugi::xml_node& node) {
        const std::string node_name{node.name()};
        if (node_name == "layer") {
            for (const auto& attribute : node.attributes()) {
                const std::string attribute_name{attribute.name()};
                if (attribute_name == "type") {
                    attributes.push_back(attribute);
                }
            }
        }
        return true;  // continue traversal
    }
};

std::pair<bool, std::string> compare_layer_ordering(
    const pugi::xml_document& doc1, const pugi::xml_document& doc2) {
    // traverse document and prepare vector of <layer type="value">
    type_attribute_walker walker1, walker2;
    doc1.child("net").child("layers").traverse(walker1);
    doc2.child("net").child("layers").traverse(walker2);

    // attributes count must be the same
    const auto& attributes1 = walker1.attributes;
    const auto& attributes2 = walker2.attributes;
    if (attributes1.size() != attributes2.size()) {
        return {false, "Attribute count differ: " +
                           std::to_string(attributes1.size()) + " != " +
                           std::to_string(attributes2.size())};
    }

    // every attribute must be equal
    for (int i = 0; i < attributes1.size(); i++) {
        const std::string a1_name{attributes1[i].name()};
        const std::string a2_name{attributes2[i].name()};
        const std::string a1_value{attributes1[i].value()};
        const std::string a2_value{attributes2[i].value()};
        if ((a1_name != a2_name) || (a1_value != a2_value)) {
            return {false, "Attributes types differ in <layer> : " + a1_name +
                               "=" + a1_value + " != " + a2_name + "=" +
                               a2_value};
        }
    }
    return {true, ""};
}
}  // namespace

class OrderingSerializationTest : public ::testing::Test {
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

TEST_F(OrderingSerializationTest, BasicModel_MO) {
    const std::string model = IR_SERIALIZATION_MODELS_PATH "add_abc.xml";
    const std::string weights = IR_SERIALIZATION_MODELS_PATH "add_abc.bin";

    InferenceEngine::Core ie;
    auto m = ie.ReadNetwork(model, weights);
    m.serialize(m_out_xml_path, m_out_bin_path);

    pugi::xml_document expected_xml;
    pugi::xml_document result_xml;
    ASSERT_TRUE(expected_xml.load_file(model.c_str()));
    ASSERT_TRUE(result_xml.load_file(m_out_xml_path.c_str()));

    bool success;
    std::string message;
    std::tie(success, message) =
        compare_layer_ordering(expected_xml, result_xml);
    ASSERT_TRUE(success) << message;
}

TEST_F(OrderingSerializationTest, ModelWithMultipleOutputs_MO) {
    const std::string model =
        IR_SERIALIZATION_MODELS_PATH "split_equal_parts_2d.xml";
    const std::string weights =
        IR_SERIALIZATION_MODELS_PATH "split_equal_parts_2d.bin";

    InferenceEngine::Core ie;
    auto m = ie.ReadNetwork(model, weights);
    m.serialize(m_out_xml_path, m_out_bin_path);

    pugi::xml_document expected_xml;
    pugi::xml_document result_xml;
    ASSERT_TRUE(expected_xml.load_file(model.c_str()));
    ASSERT_TRUE(result_xml.load_file(m_out_xml_path.c_str()));

    bool success;
    std::string message;
    std::tie(success, message) =
        compare_layer_ordering(expected_xml, result_xml);
    ASSERT_TRUE(success) << message;
}

TEST_F(OrderingSerializationTest, ModelWithMultipleLayers_MO) {
    const std::string model = IR_SERIALIZATION_MODELS_PATH "addmul_abc.xml";
    const std::string weights = IR_SERIALIZATION_MODELS_PATH "addmul_abc.bin";

    InferenceEngine::Core ie;
    auto m = ie.ReadNetwork(model, weights);
    m.serialize(m_out_xml_path, m_out_bin_path);

    pugi::xml_document expected_xml;
    pugi::xml_document result_xml;
    ASSERT_TRUE(expected_xml.load_file(model.c_str()));
    ASSERT_TRUE(result_xml.load_file(m_out_xml_path.c_str()));

    bool success;
    std::string message;
    std::tie(success, message) =
        compare_layer_ordering(expected_xml, result_xml);
    ASSERT_TRUE(success) << message;
}

TEST_F(OrderingSerializationTest, ModelWithConstants_MO) {
    const std::string model =
        IR_SERIALIZATION_MODELS_PATH "add_abc_initializers.xml";
    const std::string weights =
        IR_SERIALIZATION_MODELS_PATH "add_abc_initializers.bin";

    InferenceEngine::Core ie;
    auto m = ie.ReadNetwork(model, weights);
    m.serialize(m_out_xml_path, m_out_bin_path);

    pugi::xml_document expected_xml;
    pugi::xml_document result_xml;
    ASSERT_TRUE(expected_xml.load_file(model.c_str()));
    ASSERT_TRUE(result_xml.load_file(m_out_xml_path.c_str()));

    bool success;
    std::string message;
    std::tie(success, message) =
        compare_layer_ordering(expected_xml, result_xml);
    ASSERT_TRUE(success) << message;
}
