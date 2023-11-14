// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <gtest/gtest.h>

#include "dummy_node.hpp"

#include "ov_models/builders.hpp"
#include "ie_ngraph_utils.hpp"
#include "nodes/memory.hpp"
#include "nodes/softmax.h"

using namespace ov::intel_cpu;

TEST(MemStateGraphTest, smoke_Check_Memory_Modification_Guard) {
    const ov::element::Type_t test_prec = ov::element::Type_t::f32;
    const ov::Shape test_shape{1, 3, 8, 16};

    auto param = std::make_shared<ov::op::v0::Parameter>(test_prec, test_shape);

    // The ReadValue/Assign operations must be used in pairs in the model.
    // For each such a pair, its own variable object must be created.
    const std::string variable_name("variable0");
    auto variable = std::make_shared<ov::op::util::Variable>(
        ov::op::util::VariableInfo{test_shape, test_prec, variable_name});

    // creat ngraph ops to build CPU nodes
    auto read = std::make_shared<ov::op::v6::ReadValue>(param, variable);
    auto softmax = std::make_shared<ov::op::v1::Softmax>(read);
    auto assign = std::make_shared<ov::op::v6::Assign>(softmax, variable);
    auto res = std::make_shared<ov::op::v0::Result>(softmax);

    std::unordered_set<NodePtr> nodes_set;
    std::vector<EdgePtr> graph_edges;

    auto add_edge = [&](const NodePtr& parent, const NodePtr& child, size_t parentPort, size_t childPort) -> void {
        auto edge = std::make_shared<Edge>(parent, child, parentPort, childPort);
        child->addEdge(edge);
        graph_edges.push_back(edge);
        nodes_set.insert(parent);
        nodes_set.insert(child);
    };

    //create graph context

    Config conf;
    conf.rtCacheCapacity = 0;
    auto context = std::make_shared<GraphContext>(conf, nullptr, nullptr, false);

    auto input_node = std::make_shared<node::Input>(param, context);
    auto memory_input = std::make_shared<node::MemoryInput>(read, context);
    auto dummy_look_down = std::make_shared<cpu_unit_test::DummyNode>(
        test_shape, test_prec, "look_down_dummy", "DummyNode", context, LayoutType::ncsp, Edge::LOOK::LOOK_DOWN, true);
    auto memory_output = std::make_shared<node::MemoryOutput>(assign, context);
    auto dummy_look_up = std::make_shared<cpu_unit_test::DummyNode>(
        test_shape, test_prec, "look_up_dummy", "DummyNode", context, LayoutType::ncsp, Edge::LOOK::LOOK_UP, true);
    auto softmax_node = std::make_shared<node::SoftMax>(softmax, context);
    auto output_node = std::make_shared<node::Input>(res, context);

    add_edge(input_node, memory_input, 0, 0);
    add_edge(memory_input, dummy_look_down, 0, 0);
    add_edge(dummy_look_down, dummy_look_up, 0, 0);
    add_edge(dummy_look_down, memory_output, 0, 0);
    add_edge(dummy_look_up, softmax_node, 0, 0);
    add_edge(softmax_node, output_node, 0, 0);

    std::vector<NodePtr> graph_nodes(nodes_set.begin(), nodes_set.end());

    Graph graph;
    graph.CreateGraph(graph_nodes, graph_edges, context, "test_graph");

    auto state = memory_input->makeState();
    memory_input->assignState(state);

    auto dummy_look_up_out_mem = dummy_look_up->getChildEdgeAt(0)->getMemoryPtr()->getData();
    auto memory_output_inp_mem = memory_output->getParentEdgeAt(0)->getMemoryPtr()->getData();
    //otherwise the memory will be modified by the dummy_look_up
    ASSERT_NE(dummy_look_up_out_mem, memory_output_inp_mem);

    auto state_mem = state->get_state()->data();
    ASSERT_EQ(state_mem, memory_output_inp_mem);

    ov::pass::Serialize serializer("graph.xml", "graph.bin");
    serializer.run_on_model(graph.dump());
}