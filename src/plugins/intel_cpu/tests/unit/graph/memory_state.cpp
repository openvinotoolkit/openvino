// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <gtest/gtest.h>

#include "dummy_node.hpp"

#include "ov_models/builders.hpp"
#include "nodes/memory.hpp"
#include "nodes/softmax.h"

using namespace ov::intel_cpu;

/*
              ┌────────┐
              │ Param  │
              └───┬────┘
                  │
                  │
              ┌───┴────┐
              │MemInput│
              └───┬────┘
                  │
                  │
            ┌─────┴────────┐
            │1 DummyUp/Down│
            └──────────────┘
                /   \
               /     \
              /       \
    ┌────────┐        ┌──────────┐
    │ MemOut │        │2 DummyUp │
    └────────┘        └───┬──────┘
                          │
                          │
                      ┌───┴────┐
                      │Subgraph│
                      └───┬────┘
                          │
                          │
                      ┌───┴────┐
                      │ Output │
                      └────────┘
*/

TEST(MemStateGraphTest, smoke_Check_Memory_Modification_Guard) {
    auto build_graph = [](const Edge::LOOK first_dummy_inplace_direction) {
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
        auto first_dummy = std::make_shared<cpu_unit_test::DummyNode>(
            test_shape, test_prec,
            "first_dummy",
            "DummyNode",
            context,
            LayoutType::ncsp,
            first_dummy_inplace_direction,
            true);

        auto memory_output = std::make_shared<node::MemoryOutput>(assign, context);
        auto second_dummy = std::make_shared<cpu_unit_test::DummyNode>(
            test_shape, test_prec, "second_dummy", "DummyNode", context, LayoutType::ncsp, Edge::LOOK::LOOK_UP, true);
        auto softmax_node = std::make_shared<node::SoftMax>(softmax, context);
        auto output_node = std::make_shared<node::Input>(res, context);

        add_edge(input_node, memory_input, 0, 0);
        add_edge(memory_input, first_dummy, 0, 0);
        add_edge(first_dummy, second_dummy, 0, 0);
        add_edge(first_dummy, memory_output, 0, 0);
        add_edge(second_dummy, softmax_node, 0, 0);
        add_edge(softmax_node, output_node, 0, 0);

        std::vector<NodePtr> graph_nodes(nodes_set.begin(), nodes_set.end());

        Graph graph;
        graph.CreateGraph(graph_nodes, graph_edges, context, "test_graph");
        return graph;
    };

    auto find_node_str = [](const Graph& graph, const char* name) -> NodePtr {
        auto&& nodes = graph.GetNodes();
        auto itr =
            std::find_if(nodes.begin(), nodes.end(), [=](const NodePtr& node){ return name == node->getName(); });

        if (itr == nodes.end()) {
            return nullptr;
        }

        return (*itr);
    };

    auto find_node_type = [](const Graph& graph, Type type) -> NodePtr {
        auto&& nodes = graph.GetNodes();
        auto itr =
            std::find_if(nodes.begin(), nodes.end(), [=](const NodePtr& node){ return type == node->getType(); });

        if (itr == nodes.end()) {
            return nullptr;
        }

        return (*itr);
    };

    // Test with the first dummy has inplace::LOOK_DOWN config
    {
        auto graph = build_graph(Edge::LOOK::LOOK_DOWN);

        auto memory_input_node = find_node_type(graph, Type::MemoryInput);
        ASSERT_NE(memory_input_node, nullptr);

        auto memory_input = std::dynamic_pointer_cast<node::MemoryInput>(memory_input_node);
        ASSERT_NE(memory_input, nullptr);

        auto state = memory_input->makeState();
        memory_input->assignState(state);

        auto second_dummy = find_node_str(graph, "second_dummy");
        ASSERT_NE(second_dummy, nullptr);

        auto memory_output = find_node_type(graph, Type::MemoryOutput);

        auto second_dummy_out_mem = second_dummy->getChildEdgeAt(0)->getMemoryPtr()->getData();
        auto memory_output_inp_mem = memory_output->getParentEdgeAt(0)->getMemoryPtr()->getData();
        //otherwise the memory will be modified by the dummy_look_up
        ASSERT_NE(second_dummy_out_mem, memory_output_inp_mem);

        // due to double buffer usage by default
        auto state_inp_mem = state->input_mem()->getData();
        ASSERT_NE(state_inp_mem, memory_output_inp_mem);

        // but the ouptut mem in the state is expected to be the same with the edge mem
        auto state_out_mem = state->output_mem()->getData();
        ASSERT_EQ(state_out_mem, memory_output_inp_mem);
    }

    // Very the same check but for the case when the first dummy has inplace::LOOK_UP config
    {
        auto graph = build_graph(Edge::LOOK::LOOK_UP);

        auto memory_input_node = find_node_type(graph, Type::MemoryInput);
        ASSERT_NE(memory_input_node, nullptr);

        auto memory_input = std::dynamic_pointer_cast<node::MemoryInput>(memory_input_node);
        ASSERT_NE(memory_input, nullptr);

        auto state = memory_input->makeState();
        memory_input->assignState(state);

        auto second_dummy = find_node_str(graph, "second_dummy");
        ASSERT_NE(second_dummy, nullptr);

        auto memory_output = find_node_type(graph, Type::MemoryOutput);

        auto second_dummy_out_mem = second_dummy->getChildEdgeAt(0)->getMemoryPtr()->getData();
        auto memory_output_inp_mem = memory_output->getParentEdgeAt(0)->getMemoryPtr()->getData();
        //otherwise the memory will be modified by the dummy_look_up
        ASSERT_NE(second_dummy_out_mem, memory_output_inp_mem);

        // as the input memory bypassed through a cascade of look_up inplace nodes, it's set directly to the output
        // input edge
        auto state_inp_mem = state->input_mem()->getData();
        ASSERT_EQ(state_inp_mem, memory_output_inp_mem);

        // in this configuration MemOutput is expected to perform a regular copy, so the memory on the edge and in the
        // internal state output buffer are expected to be different
        auto state_out_mem = state->output_mem()->getData();
        ASSERT_NE(state_out_mem, memory_output_inp_mem);
    }
}