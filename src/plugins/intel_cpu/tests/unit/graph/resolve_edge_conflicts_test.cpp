// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <gtest/gtest.h>

#include "dummy_node.hpp"
#include "graph.h"
#include "memory_control.hpp"
#include "nodes/concat.h"
#include "nodes/input.h"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reduce_prod.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/scatter_nd_update.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/opsets/opset.hpp"

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
    auto dummyNode1 = std::make_shared<cpu_unit_test::DummyNode>(testShape, testPrec, "Dummy1", "DummyNode", context);
    auto dummyNode2 = std::make_shared<cpu_unit_test::DummyNode>(testShape, testPrec, "Dummy2", "DummyNode", context);
    auto dummyNode3 = std::make_shared<cpu_unit_test::DummyNode>(testShape,
                                                                 testPrec,
                                                                 "Dummy3",
                                                                 "DummyNode",
                                                                 context,
                                                                 LayoutType::ncsp,
                                                                 Edge::LOOK::LOOK_UP,
                                                                 true);
    auto dummyNode4 = std::make_shared<cpu_unit_test::DummyNode>(testShape,
                                                                 testPrec,
                                                                 "Dummy4",
                                                                 "DummyNode",
                                                                 context,
                                                                 LayoutType::ncsp,
                                                                 0,
                                                                 true);

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
    for (auto& node : nodesSet)
        graphNodes.emplace_back(node);
    graph->CreateGraph(graphNodes, graphEdges, context, "test_graph");

    // Check whether reorder is inserted
    NodePtr expected_reorder = dummyNode2->getParentEdgeAt(0)->getParent();
    ASSERT_EQ(expected_reorder->getType(), Type::Reorder);
}

TEST(ResolveEdgeConflictsCPUTest2, smoke_Run_ResolveEdgeConflicts2) {
    /*  create graph:
                                                    Parameter
                                                       |
                                                    ShapeOf      Parameter
        <*NOTE: Should insert reorder here>         /      \        /
        <*NOTE: inplaced>               ScatterNDUpdate     ReduceProd
                                           |                   |
                                         Result              Result
    */
    Config conf;
    conf.rtCacheCapacity = 100;
    auto context = std::make_shared<GraphContext>(conf, nullptr, false);

    std::unique_ptr<Graph> graph = std::unique_ptr<Graph>(new Graph());

    const ov::element::Type_t testPrec = ov::element::Type_t::f32;
    const ov::Shape testShape{1, 384};
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(testPrec, testShape),
                               std::make_shared<ov::op::v0::Parameter>(ov::element::Type_t::i32, ov::Shape{})};
    auto org_ShapeOf_386 = std::make_shared<ov::op::v3::ShapeOf>(params[0], ov::element::Type_t::i32);

    auto org_ReduceProd_423 = std::make_shared<ov::op::v1::ReduceProd>(org_ShapeOf_386, params[1]);

    auto org_Constant_387 =
        std::make_shared<ov::op::v0::Constant>(ov::element::Type_t::i32, ov::Shape{1, 1}, std::vector<int64_t>{1});
    auto org_Constant_1 =
        std::make_shared<ov::op::v0::Constant>(ov::element::Type_t::i32, ov::Shape{1}, std::vector<int64_t>{1});
    auto org_ScatterNDUpdate_411 =
        std::make_shared<ov::op::v3::ScatterNDUpdate>(org_ShapeOf_386, org_Constant_387, org_Constant_1);

    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(org_ScatterNDUpdate_411),
                             std::make_shared<ov::op::v0::Result>(org_ReduceProd_423)};

    const auto model = std::make_shared<const ov::Model>(results, params, "test_graph");
    graph->CreateGraph(model, context);
    for (auto node : graph->GetNodes()) {
        if (node->getType() == Type::ScatterNDUpdate) {
            NodePtr expected_reorder = node->getParentEdgeAt(0)->getParent();
            ASSERT_EQ(expected_reorder->getType(), Type::Reorder);
            break;
        }
    }
}
