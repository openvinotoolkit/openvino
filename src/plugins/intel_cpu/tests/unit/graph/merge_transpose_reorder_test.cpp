// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <gtest/gtest.h>

#include "dummy_node.hpp"
#include "graph.h"
#include "nodes/reorder.h"
#include "nodes/input.h"
#include "nodes/transpose.h"

#include "openvino/op/transpose.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/parameter.hpp"

#include "common_test_utils/node_builders/constant.hpp"

using namespace ov::intel_cpu;

class MergeTransposeReordersCPUTest : public ::testing::Test {
protected:
    /*  graph typology
                --------- 
                |Input  |
                ---------
                    |
                ----------
                |  Dummy |           <*NOTE: fake node with laytout NCSP, and inplace from upstream*>
                ----------
                    |
             |---------------|
             |   ----------  |
             |   |Transpose| |
             |   ---------   |
             |       |       |
             |   ---------   |
             |   |Reorder |  |          <*NOTE: Reorder is inheristically inserted since Multiply is asking NSPC input.*>
             |   ---------   |
             |---------------|
                    |
                -----------
                |  Dummy  |         <*NOTE: fake node with laytout NSPC, and inplace from downstream*>
                -----------
                    |
                ---------
                |Output |
                ---------
    */
    void CreateGraph(int num_consumers, int consumer_in_place_direction) {
        //
        Config conf;
        conf.rtCacheCapacity = 100;
        auto context = std::make_shared<GraphContext>(conf, nullptr, false);
        const dnnl::engine cpuEngine = context->getEngine();

        m_graph = std::unique_ptr<Graph>(new Graph());

        // ov::Model with only a transpose node
        ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(testPrec, ov::Shape(testShape))};
        auto order = std::vector<int32_t>{0, 3, 1, 2};
        auto constOrder = ov::test::utils::deprecated::make_constant(ov::element::i32, {order.size()}, order);
        auto transpose = std::make_shared<ov::op::v1::Transpose>(params[0], constOrder);
        ov::ResultVector results;
        for (int i = 0; i < num_consumers; i++)
            results.push_back(std::make_shared<ov::op::v0::Result>(transpose));

        // Replicate
        auto replicate = [&](std::vector<NodePtr> &nodes, std::vector<EdgePtr> &edges) -> void {
            std::unordered_set<NodePtr> nodesSet;

            auto addEdge = [&](const NodePtr& parent, const NodePtr& child, size_t parentPort, size_t childPort) -> void {
                auto edge = std::make_shared<Edge>(parent, child, parentPort, childPort);
                Node::addEdge(edge);
                edges.push_back(edge);
                nodesSet.insert(parent);
                nodesSet.insert(child);
            };

            auto inputNode = std::make_shared<node::Input>(params[0], context);

            // dummy ncsp + inPlace LOOK_UP
            auto dummyNode1 = std::make_shared<cpu_unit_test::DummyNode>(
                testShape, testPrec, "reshape", "DummyNode", context, LayoutType::ncsp, Edge::LOOK::LOOK_UP);

            auto orderNode = std::make_shared<node::Input>(constOrder, context); // const order
            auto transposeNode = std::make_shared<node::Transpose>(transpose, context);
            transposeNode->filterSupportedPrimitiveDescriptors();


            addEdge(inputNode, dummyNode1, 0, 0);
            addEdge(dummyNode1, transposeNode, 0, 0);
            addEdge(orderNode, transposeNode, 0, 1);

            // dummy nspc + inPlace LOOK_DOWN
            const ov::Shape shape_tranpose{testShape[0],
                                           testShape[3],
                                           testShape[1],
                                           testShape[2]};  // shape after transpose
            for (int i = 0; i < num_consumers; i++) {
                auto dummyConsumer = std::make_shared<cpu_unit_test::DummyNode>(shape_tranpose,
                                                                                testPrec,
                                                                                "multiply",
                                                                                "DummyNode",
                                                                                context,
                                                                                LayoutType::nspc,
                                                                                consumer_in_place_direction);
                auto outputNode = std::make_shared<node::Input>(results[i], context);
                addEdge(transposeNode, dummyConsumer, 0, 0);
                addEdge(dummyConsumer, outputNode, 0, 0);
            }

            for (auto &node : nodesSet) nodes.emplace_back(node);
        };

        std::vector<NodePtr> graphNodes;
        std::vector<EdgePtr> graphEdges;
        replicate(graphNodes, graphEdges);

        m_graph->CreateGraph(graphNodes, graphEdges, context, "fused_graph");
    }

    // helper to check if Transpose node is fused.
    void CheckTransposeCount(const size_t expectedTransposeCount) const {
        const std::vector<NodePtr>& graph_nodes = m_graph->GetNodes();
        size_t actualTransposeCount = 0;
        for (auto &node : graph_nodes) {
            if (node->getType() == Type::Transpose) {
                actualTransposeCount++;
            }
        }

        ASSERT_EQ(expectedTransposeCount, actualTransposeCount);
    }

    // helper to check isOptimized of Reorder node with a part of its name
    void CheckReorderOptimized(const std::string &patial_name, const bool expectedOptimized) const {
        const std::vector<NodePtr>& graph_nodes = m_graph->GetNodes();
        size_t actualCount = 0;
        for (auto &node : graph_nodes) {
            auto reorder_node = std::dynamic_pointer_cast<node::Reorder>(node);
            if (reorder_node && node->getName().find(patial_name) != std::string::npos) {
                ASSERT_EQ(expectedOptimized, reorder_node->getOptimized());
                actualCount++;
            }
        }

        ASSERT_EQ(1, actualCount);
    }

protected:
    const ov::element::Type_t testPrec = ov::element::Type_t::f32;
    const ov::Shape testShape{1, 3, 8, 16};

    std::unique_ptr<Graph> m_graph;
}; // class MergeTransposeReordersCPUTest

// upstream node or downstream node is inPlaced thereby the inserted Reorder cannot be optimized.
TEST_F(MergeTransposeReordersCPUTest, smoke_Run_MergeTransposeReorders_isOptimized) {
    CreateGraph(1, Edge::LOOK::LOOK_DOWN);
    CheckTransposeCount(0);
    CheckReorderOptimized(std::string("_fake"), false);  // the fused node is of name "reshape_abcd_acdb_fake"
}

// 3 non-inplace consumers share a single optimized reorder fused with Transpose
TEST_F(MergeTransposeReordersCPUTest, smoke_Run_MergeTransposeReorders_shared) {
    CreateGraph(3, 0);
    CheckTransposeCount(0);
    CheckReorderOptimized(std::string("_fake"), true);
}

// 3 inplace consumers cannot share reorders thus transpose is not fused with reorders
// there will be also 3 reorders between 3 dummyNode-consumers and 3 Result nodes
TEST_F(MergeTransposeReordersCPUTest, smoke_Run_MergeTransposeReorders_notFused) {
    CreateGraph(3, Edge::LOOK::LOOK_DOWN);
    CheckTransposeCount(1);
    size_t reorderCount = 0;
    for (auto& node : m_graph->GetNodes()) {
        auto reorder_node = std::dynamic_pointer_cast<node::Reorder>(node);
        if (reorder_node) {
            // there should be no "_fake" reorders generated by merging transpose + reorder
            ASSERT_EQ(node->getName().find("_fake"), std::string::npos);
            reorderCount++;
        }
    }

    // 3 for layout conflist between [transpose => dummyConsumer]
    // 3 for layout conflist between [dummyConsumer => result]
    ASSERT_EQ(6, reorderCount);
}
