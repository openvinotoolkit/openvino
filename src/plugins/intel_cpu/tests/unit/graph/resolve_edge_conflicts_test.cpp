// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <gtest/gtest.h>

#include "dummy_node.hpp"
#include "nodes/input.h"
#include "nodes/concat.h"

#include "ov_models/builders.hpp"

using namespace ov::intel_cpu;

/*
 * Test the CPU plugin-in edge method ResolveEdgeConflicts().
 * This case is to check the capability of graph to resolve complex inplace conflicts
 */

TEST(ResolveEdgeConflictsCPUTest, smoke_Run_ResolveEdgeConflicts) {
    /*  create graph:
                 Input
                /    \
            Dummy1   Dummy2     <*NOTE: unexcutable fake node with inplace from upstream*>
              |        |
              |      Dummy3     <*NOTE: excutable fake node with inplace from upstream*>
              |        |
              |      Dummy4     <*NOTE: fake node can not be inplace*>
                \    /
                Concat
                  |
                Output

        Dummy1, Dummy2 and Dummy3 can be inplace. In ResolveEdgeConflicts(), detect Dummy3 is 
        a modifying node. Collect consumers of edge Input->Dummy1 and find consumer execution
        order is after Dummy3. Then insert Reorder in edge Input->Dummy2.
    */
    Config conf;
    conf.rtCacheCapacity = 100;
    auto context = std::make_shared<GraphContext>(conf, nullptr, false);
    const dnnl::engine cpuEngine = context->getEngine();

    std::unique_ptr<Graph> graph = std::unique_ptr<Graph>(new Graph());

    const ov::element::Type_t testPrec = ov::element::Type_t::f32;
    const ov::Shape testShape{2, 1};

    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(testPrec, testShape)};
    auto concat = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{params[0], params[0]}, 1);
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(concat)};
    auto inputNode = std::make_shared<node::Input>(params[0], context);
    auto outputNode = std::make_shared<node::Input>(results[0], context);
    auto concatNode = std::make_shared<node::Concat>(concat, context);
    auto dummyNode1 = std::make_shared<cpu_unit_test::DummyNode>(
        testShape, testPrec, "Dummy1", "DummyNode", context);
    auto dummyNode2 = std::make_shared<cpu_unit_test::DummyNode>(
        testShape, testPrec, "Dummy2", "DummyNode", context);
    auto dummyNode3 = std::make_shared<cpu_unit_test::DummyNode>(
        testShape, testPrec, "Dummy3", "DummyNode", context, LayoutType::ncsp, Edge::LOOK::LOOK_UP, true);
    auto dummyNode4 = std::make_shared<cpu_unit_test::DummyNode>(
        testShape, testPrec, "Dummy4", "DummyNode", context, LayoutType::ncsp, 0, true);

    std::vector<NodePtr> graphNodes;
    std::vector<EdgePtr> graphEdges;

    std::unordered_set<NodePtr> nodesSet;
    auto addEdge = [&](const NodePtr& parent, const NodePtr& child, size_t parentPort, size_t childPort) -> void {
        auto edge = std::make_shared<Edge>(parent, child, parentPort, childPort);
        child->addEdge(edge);
        graphEdges.push_back(edge);
        nodesSet.insert(parent);
        nodesSet.insert(child);
    };
    addEdge(inputNode, dummyNode2, 0, 0);
    addEdge(dummyNode2, dummyNode3, 0, 0);
    addEdge(dummyNode3, dummyNode4, 0, 0);
    addEdge(dummyNode4, concatNode, 0, 1);
    addEdge(inputNode, dummyNode1, 0, 0);
    addEdge(dummyNode1, concatNode, 0, 0);
    addEdge(concatNode, outputNode, 0, 0);
    for (auto &node : nodesSet) graphNodes.emplace_back(node);
    graph->CreateGraph(graphNodes, graphEdges, context, "test_graph");

    // Check whether reorder is inserted
    NodePtr expected_reorder = dummyNode2->getParentEdgeAt(0)->getParent();
    ASSERT_EQ(expected_reorder->getType(), Type::Reorder);
}
