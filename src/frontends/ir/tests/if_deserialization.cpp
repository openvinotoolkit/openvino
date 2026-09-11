// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <sstream>

#include "frontend_test.hpp"
using testing::HasSubstr;

class IRFrontendTestsIf : public ::testing::Test, public IRFrontendTestsImpl {
protected:
    void SetUp() override {}

    void TearDown() override {
        RemoveTemporalFiles();
    }
};

// Build an IR XML string containing an If op whose then-body itself contains
// another If, nested to the requested depth.  Every condition is a Const(true),
// which forces the constant-condition branch of If::validate_and_infer_types.
//
// The topology at each nesting level is:
//   Const(bool) ──┐
//   Parameter ────┤  If ── Result
//                 │
//   then_body: contains the next-level If (or just Parameter→Result at leaf)
//   else_body: Parameter → Result (pass-through)
static std::string generate_nested_if_xml(size_t depth) {
    // Each nesting level needs unique layer IDs.  We use a simple counter.
    int next_id = 0;
    auto alloc_id = [&]() {
        return next_id++;
    };

    // Recursive lambda: returns the <layers>...</layers><edges>...</edges>
    // fragment for a single If level and all its children.  Also returns
    // the layer-id of the top-level If (or the pass-through Parameter at
    // the leaf) so the parent can wire edges.
    struct Fragment {
        std::string layers;
        std::string edges;
        int output_layer_id;
        int output_port_id;
    };

    // Forward-declare for the recursive lambda.
    std::function<Fragment(size_t, int /*param_id of data fed into this level*/)> build_level;

    build_level = [&](size_t remaining_depth, int input_param_id) -> Fragment {
        if (remaining_depth == 0) {
            // Leaf: just return the input parameter info (the body is a
            // trivial Parameter→Result; the Parameter is already created by
            // the caller as part of the body).
            return {"", "", input_param_id, 0};
        }

        std::ostringstream layers, edges;

        // -- Const condition (true) --
        int cond_id = alloc_id();
        layers << R"(        <layer id=")" << cond_id
               << R"(" name="cond_)" << cond_id
               << R"(" type="Const" version="opset1">)"
               << R"(
            <data element_type="boolean" shape="" offset="0" size="1"/>
            <output>
                <port id="0" precision="BOOL"/>
            </output>
        </layer>
)";

        // -- The If node --
        int if_id = alloc_id();

        // then-body: Parameter(f32, dynamic) → [nested-If or pass-through] → Result
        int then_param_id = alloc_id();
        int then_result_id = alloc_id();

        // else-body: Parameter(f32, dynamic) → Result (trivial pass-through)
        int else_param_id = alloc_id();
        int else_result_id = alloc_id();

        // Build the nested content for then-body
        auto inner = build_level(remaining_depth - 1, then_param_id);

        // Construct then-body layers
        std::ostringstream then_layers;
        then_layers << R"(                    <layer id=")" << then_param_id
                    << R"(" name="then_param_)" << then_param_id
                    << R"(" type="Parameter" version="opset1">)"
                    << R"(
                        <data element_type="f32" shape="1"/>
                        <output>
                            <port id="0" precision="FP32">
                                <dim>1</dim>
                            </port>
                        </output>
                    </layer>
)";
        then_layers << inner.layers;

        then_layers << R"(                    <layer id=")" << then_result_id
                    << R"(" name="then_result_)" << then_result_id
                    << R"(" type="Result" version="opset1">)"
                    << R"(
                        <input>
                            <port id="0">
                                <dim>1</dim>
                            </port>
                        </input>
                    </layer>
)";

        // then-body edges
        std::ostringstream then_edges;
        then_edges << inner.edges;
        // Connect inner output to Result
        int src_layer = (remaining_depth - 1 > 0) ? inner.output_layer_id : then_param_id;
        int src_port = inner.output_port_id;
        then_edges << R"(                    <edge from-layer=")" << src_layer
                   << R"(" from-port=")" << src_port
                   << R"(" to-layer=")" << then_result_id
                   << R"(" to-port="0"/>
)";

        // Construct else-body layers (trivial pass-through)
        std::ostringstream else_layers;
        else_layers << R"(                    <layer id=")" << else_param_id
                    << R"(" name="else_param_)" << else_param_id
                    << R"(" type="Parameter" version="opset1">)"
                    << R"(
                        <data element_type="f32" shape="1"/>
                        <output>
                            <port id="0" precision="FP32">
                                <dim>1</dim>
                            </port>
                        </output>
                    </layer>
)";
        else_layers << R"(                    <layer id=")" << else_result_id
                    << R"(" name="else_result_)" << else_result_id
                    << R"(" type="Result" version="opset1">)"
                    << R"(
                        <input>
                            <port id="0">
                                <dim>1</dim>
                            </port>
                        </input>
                    </layer>
)";

        std::ostringstream else_edges;
        else_edges << R"(                    <edge from-layer=")" << else_param_id
                   << R"(" from-port="0" to-layer=")" << else_result_id
                   << R"(" to-port="0"/>
)";

        // -- If layer --
        // Input ports: 0=condition, 1=data
        // Output ports: 2=output
        layers << R"(        <layer id=")" << if_id
               << R"(" name="if_)" << if_id
               << R"(" type="If" version="opset8">)"
               << R"(
            <input>
                <port id="0">
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
            <then_port_map>
                <input external_port_id="1" internal_layer_id=")" << then_param_id << R"("/>
                <output external_port_id="2" internal_layer_id=")" << then_result_id << R"("/>
            </then_port_map>
            <else_port_map>
                <input external_port_id="1" internal_layer_id=")" << else_param_id << R"("/>
                <output external_port_id="2" internal_layer_id=")" << else_result_id << R"("/>
            </else_port_map>
            <then_body>
                <layers>
)" << then_layers.str() << R"(                </layers>
                <edges>
)" << then_edges.str() << R"(                </edges>
            </then_body>
            <else_body>
                <layers>
)" << else_layers.str() << R"(                </layers>
                <edges>
)" << else_edges.str() << R"(                </edges>
            </else_body>
        </layer>
)";

        // Edges: Const→If:0, input_param→If:1
        edges << R"(        <edge from-layer=")" << cond_id
              << R"(" from-port="0" to-layer=")" << if_id << R"(" to-port="0"/>
)";
        edges << R"(        <edge from-layer=")" << input_param_id
              << R"(" from-port="0" to-layer=")" << if_id << R"(" to-port="1"/>
)";

        return {layers.str(), edges.str(), if_id, /*output port*/ 2};
    };

    // Top-level model: Parameter → nested If chain → Result
    int top_param_id = alloc_id();
    auto frag = build_level(depth, top_param_id);
    int top_result_id = alloc_id();

    std::ostringstream xml;
    xml << R"(<?xml version="1.0" ?>
<net name="NestedIf" version="11">
    <layers>
        <layer id=")" << top_param_id
        << R"(" name="input" type="Parameter" version="opset1">
            <data element_type="f32" shape="1"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                </port>
            </output>
        </layer>
)" << frag.layers
        << R"(        <layer id=")" << top_result_id
        << R"(" name="output" type="Result" version="opset1">
            <input>
                <port id="0">
                    <dim>1</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
)" << frag.edges
        << R"(        <edge from-layer=")" << frag.output_layer_id
        << R"(" from-port=")" << frag.output_port_id
        << R"(" to-layer=")" << top_result_id << R"(" to-port="0"/>
    </edges>
</net>
)";

    return xml.str();
}

// Boundary: depth=64 equals kMaxIfValidationDepth and must load successfully.
TEST_F(IRFrontendTestsIf, nested_if_at_max_depth_loads) {
    std::string xmlModel = generate_nested_if_xml(64);
    // Single-byte bin for the Const(true) boolean values
    std::vector<unsigned char> buffer(1, 1);
    createTemporalModelFile(xmlModel, buffer);

    std::shared_ptr<ov::Model> model;
    OV_ASSERT_NO_THROW(model = core.read_model(xmlFileName, binFileName));
    ASSERT_NE(model, nullptr);
}

// CWE-674: depth=1024 far exceeds kMaxIfValidationDepth (64); without the fix this
// would overflow the stack, with the fix it throws ov::Exception.
TEST_F(IRFrontendTestsIf, nested_if_depth_limit_is_rejected) {
    std::string xmlModel = generate_nested_if_xml(1024);
    std::vector<unsigned char> buffer(1, 1);
    createTemporalModelFile(xmlModel, buffer);

    OV_EXPECT_THROW(core.read_model(xmlFileName, binFileName),
        ov::Exception,
        HasSubstr("nesting depth exceeds the maximum"));
}
