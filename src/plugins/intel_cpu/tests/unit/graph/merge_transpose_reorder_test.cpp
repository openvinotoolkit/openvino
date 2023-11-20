// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <gtest/gtest.h>

#include "dummy_node.hpp"
#include "nodes/reorder.h"
#include "nodes/input.h"
#include "nodes/transpose.h"

#include "ov_models/builders.hpp"

using namespace ov::intel_cpu;

/*
 * MergeTransposeReorderIsOptimizedCPUTest to test the CPU plugin-in MergeTransposeReorder graph optimizer
 * under the circumstance that the upstream node or downstream node is inPlaced thereby the inserted Reorder
 * cannot be optimized.
 */
class MergeTransposeReorderIsOptimizedCPUTest : public ::testing::Test {
public:
    void Validate() const {
        CheckTransposeCount(0);
        CheckReorderOptimized(std::string("_fake"), false);  // the fused node is of name "reshape_abcd_acdb_fake"
    }

    void SetUp() override {
        CreateGraph();
    }

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
    void CreateGraph() {
        //
        Config conf;
        conf.rtCacheCapacity = 100;
        auto context = std::make_shared<GraphContext>(conf, nullptr, nullptr, false);
        const dnnl::engine cpuEngine = context->getEngine();

        m_graph = std::unique_ptr<Graph>(new Graph());

        // ov::Model with only a transpose node
        ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(testPrec, ov::Shape(testShape))};
        auto order = std::vector<int32_t>{0, 3, 1, 2};
        auto constOrder = ngraph::builder::makeConstant(ngraph::element::i32, {order.size()}, order);
        auto transpose = std::make_shared<ngraph::opset5::Transpose>(params[0], constOrder);
        ov::ResultVector results{std::make_shared<ngraph::opset5::Result>(transpose)};

        // Replicate
        auto replicate = [&](std::vector<NodePtr> &nodes, std::vector<EdgePtr> &edges) -> void {
            std::unordered_set<NodePtr> nodesSet;

            auto addEdge = [&](const NodePtr& parent, const NodePtr& child, size_t parentPort, size_t childPort) -> void {
                auto edge = std::make_shared<Edge>(parent, child, parentPort, childPort);
                child->addEdge(edge);
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

            // dummy nspc + inPlace LOOK_DOWN
            const ov::Shape shape_tranpose{testShape[0], testShape[3], testShape[1], testShape[2]};  // shape after transpose
            auto dummyNode2 = std::make_shared<cpu_unit_test::DummyNode>(
                shape_tranpose, testPrec, "multiply", "DummyNode", context, LayoutType::nspc, Edge::LOOK::LOOK_DOWN);

            auto outputNode = std::make_shared<node::Input>(results[0], context);

            addEdge(inputNode, dummyNode1, 0, 0);
            addEdge(dummyNode1, transposeNode, 0, 0);
            addEdge(orderNode, transposeNode, 0, 1);
            addEdge(transposeNode, dummyNode2, 0, 0);
            addEdge(dummyNode2, outputNode, 0, 0);

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

private:
    const ov::element::Type_t testPrec = ov::element::Type_t::f32;
    const ov::Shape testShape{1, 3, 8, 16};

    std::unique_ptr<Graph> m_graph;
}; // class MergeTransposeReorderIsOptimizedCPUTest

TEST_F(MergeTransposeReorderIsOptimizedCPUTest, smoke_Run_MergeTransposeReorder_isOptimized) {
    Validate();
}