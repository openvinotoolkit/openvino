// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <sstream>
#include <string>

#include "common_test_utils/thread_utils.hpp"
#include "frontend_test.hpp"

using testing::HasSubstr;

class IRFrontendTestsIf : public ::testing::Test, public IRFrontendTestsImpl {
protected:
    void SetUp() override {}

    void TearDown() override {
        RemoveTemporalFiles();
    }
};

namespace {

// One level: Const condition (constant-condition branch), Const data and an If whose then-body
// holds the next level. Levels are self-contained, so the XML is built without recursion.
struct Level {
    std::string layers;
    std::string edges;
    int if_id = 0;  // output port 2 carries the level's result
};

Level make_level(int& next_id, const Level* inner) {
    const int cond_id = next_id++;
    const int data_id = next_id++;
    const int if_id = next_id++;
    const int then_param_id = next_id++;
    const int then_result_id = next_id++;
    const int else_param_id = next_id++;
    const int else_result_id = next_id++;

    // then-body: Parameter -> [inner If] -> Result, else-body: Parameter -> Result
    std::ostringstream then_layers, then_edges;
    then_layers << "<layer id=\"" << then_param_id << "\" name=\"then_param_" << then_param_id
                << "\" type=\"Parameter\" version=\"opset1\">"
                << "<data element_type=\"f32\" shape=\"1\"/>"
                << "<output><port id=\"0\" precision=\"FP32\"><dim>1</dim></port></output></layer>";
    if (inner != nullptr) {
        then_layers << inner->layers;
        then_edges << inner->edges;
    }
    then_layers << "<layer id=\"" << then_result_id << "\" name=\"then_result_" << then_result_id
                << "\" type=\"Result\" version=\"opset1\">"
                << "<input><port id=\"0\"><dim>1</dim></port></input></layer>";
    then_edges << "<edge from-layer=\"" << (inner != nullptr ? inner->if_id : then_param_id) << "\" from-port=\""
               << (inner != nullptr ? 2 : 0) << "\" to-layer=\"" << then_result_id << "\" to-port=\"0\"/>";

    std::ostringstream else_layers, else_edges;
    else_layers << "<layer id=\"" << else_param_id << "\" name=\"else_param_" << else_param_id
                << "\" type=\"Parameter\" version=\"opset1\">"
                << "<data element_type=\"f32\" shape=\"1\"/>"
                << "<output><port id=\"0\" precision=\"FP32\"><dim>1</dim></port></output></layer>"
                << "<layer id=\"" << else_result_id << "\" name=\"else_result_" << else_result_id
                << "\" type=\"Result\" version=\"opset1\">"
                << "<input><port id=\"0\"><dim>1</dim></port></input></layer>";
    else_edges << "<edge from-layer=\"" << else_param_id << "\" from-port=\"0\" to-layer=\"" << else_result_id
               << "\" to-port=\"0\"/>";

    Level level;
    level.if_id = if_id;

    std::ostringstream layers;
    layers << "<layer id=\"" << cond_id << "\" name=\"cond_" << cond_id << "\" type=\"Const\" version=\"opset1\">"
           << "<data element_type=\"boolean\" shape=\"\" offset=\"0\" size=\"1\"/>"
           << "<output><port id=\"0\" precision=\"BOOL\"/></output></layer>"
           << "<layer id=\"" << data_id << "\" name=\"data_" << data_id << "\" type=\"Const\" version=\"opset1\">"
           << "<data element_type=\"f32\" shape=\"1\" offset=\"4\" size=\"4\"/>"
           << "<output><port id=\"0\" precision=\"FP32\"><dim>1</dim></port></output></layer>"
           << "<layer id=\"" << if_id << "\" name=\"if_" << if_id << "\" type=\"If\" version=\"opset8\">"
           << "<input><port id=\"0\"/><port id=\"1\"><dim>1</dim></port></input>"
           << "<output><port id=\"2\" precision=\"FP32\"><dim>1</dim></port></output>"
           << "<then_port_map>"
           << "<input external_port_id=\"1\" internal_layer_id=\"" << then_param_id << "\"/>"
           << "<output external_port_id=\"2\" internal_layer_id=\"" << then_result_id << "\"/>"
           << "</then_port_map>"
           << "<else_port_map>"
           << "<input external_port_id=\"1\" internal_layer_id=\"" << else_param_id << "\"/>"
           << "<output external_port_id=\"2\" internal_layer_id=\"" << else_result_id << "\"/>"
           << "</else_port_map>"
           << "<then_body><layers>" << then_layers.str() << "</layers><edges>" << then_edges.str()
           << "</edges></then_body>"
           << "<else_body><layers>" << else_layers.str() << "</layers><edges>" << else_edges.str()
           << "</edges></else_body>"
           << "</layer>";
    level.layers = layers.str();

    std::ostringstream edges;
    edges << "<edge from-layer=\"" << cond_id << "\" from-port=\"0\" to-layer=\"" << if_id << "\" to-port=\"0\"/>"
          << "<edge from-layer=\"" << data_id << "\" from-port=\"0\" to-layer=\"" << if_id << "\" to-port=\"1\"/>";
    level.edges = edges.str();

    return level;
}

std::string generate_nested_if_xml(size_t depth) {
    int next_id = 0;
    Level level;
    for (size_t i = 0; i < depth; ++i) {
        level = make_level(next_id, i == 0 ? nullptr : &level);
    }
    const int result_id = next_id++;

    std::ostringstream xml;
    xml << "<?xml version=\"1.0\" ?>\n<net name=\"NestedIf\" version=\"11\">\n<layers>" << level.layers
        << "<layer id=\"" << result_id << "\" name=\"output\" type=\"Result\" version=\"opset1\">"
        << "<input><port id=\"0\"><dim>1</dim></port></input></layer>"
        << "</layers>\n<edges>" << level.edges << "<edge from-layer=\"" << level.if_id
        << "\" from-port=\"2\" to-layer=\"" << result_id << "\" to-port=\"0\"/>"
        << "</edges>\n</net>\n";
    return xml.str();
}

// Bool condition at offset 0, f32 data at offset 4.
std::vector<unsigned char> nested_if_weights() {
    return {1, 0, 0, 0, 0, 0, 0, 0};
}

constexpr size_t small_stack_size = 512 * 1024;

}  // namespace

TEST_F(IRFrontendTestsIf, nested_if_shallow_loads) {
    createTemporalModelFile(generate_nested_if_xml(8), nested_if_weights());

    std::shared_ptr<ov::Model> model;
    OV_ASSERT_NO_THROW(model = core.read_model(xmlFileName, binFileName));
    ASSERT_NE(model, nullptr);
}

// Runs on a small stack; asserts the error, not the depth at which it fires.
TEST_F(IRFrontendTestsIf, nested_if_deep_ir_is_rejected) {
    createTemporalModelFile(generate_nested_if_xml(512), nested_if_weights());

    const auto error = ov::test::utils::run_with_stack_size(small_stack_size, [this] {
        std::ignore = core.read_model(xmlFileName, binFileName);
    });
    EXPECT_THAT(error, HasSubstr("deeper than the stack of the current thread"));
}
