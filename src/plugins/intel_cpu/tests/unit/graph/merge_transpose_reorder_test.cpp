// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <gtest/gtest.h>
#include <memory_desc/cpu_blocked_memory_desc.h>

#include <common_test_utils/test_common.hpp>

#include "common_test_utils/node_builders/constant.hpp"
#include "dummy_node.hpp"
#include "graph.h"
#include "memory_control.hpp"
#include "nodes/input.h"
#include "nodes/reorder.h"
#include "nodes/reshape.h"
#include "nodes/transpose.h"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/transpose.hpp"

using namespace ov::intel_cpu;
using LOOK = Edge::LOOK;

struct Result {
    size_t transpose_count;
    size_t optimized_reorder_count;
    size_t not_optimized_reorder_count;
};

struct MergeTransposeReorderTestParam {
    LayoutType firstNodeLayout;
    LOOK firstNodeInplaceDirection;
    LayoutType lastNodeLayout;
    LOOK lastNodeInplaceDirection;
    size_t num_consumers;
    Result test_result;
};

using MergeTransposeReorderTestParams = std::tuple<ov::Shape, MergeTransposeReorderTestParam>;

/* graph topology
    ┌───────┐
    │ Input │
    └───┬───┘
        │
    ┌───┴───┐
    │ Dummy │      <*NOTE: fake node with firstNodeLayout, and firstNodeInplaceDirection*>
    └───┬───┘
        │
   ┌────┴────┐
   │Transpose│     <*NOTE: Reorder is inserted before/after Transpose depending on first/second node layouts.*>
   └────┬────┘
        │
    ┌───┴───┐
    │ Dummy │      <*NOTE: fake node with lastNodeLayout, and lastNodeInplaceDirection*>
    └───┬───┘
        │
   ┌────┴───┐
   │ Output │
   └────────┘
*/
class MergeTransposeReorderCPUTest : public testing::WithParamInterface<MergeTransposeReorderTestParams>,
                                     public ov::test::TestsCommon {
public:
    void Validate() const {
        const auto& result = std::get<1>(GetParam()).test_result;
        CheckTransposeCount(result.transpose_count);
        CheckReorderCount(result.optimized_reorder_count, result.not_optimized_reorder_count);
    }

protected:
    void SetUp() override {
        const auto& shape = std::get<0>(GetParam());
        const auto& params = std::get<1>(GetParam());
        OPENVINO_ASSERT(shape.size() == 4 || shape.size() == 3,
                        "MergeTransposeReorderCPUTest doesn't support shape",
                        shape,
                        ". Only 4D and 3D shapes are supported");
        Config conf;
        m_context =
            std::make_shared<GraphContext>(conf, nullptr, false);
        const auto replication_result = CreateModelAndReplicate(shape,
                                                                params.firstNodeLayout,
                                                                params.firstNodeInplaceDirection,
                                                                params.lastNodeLayout,
                                                                params.lastNodeInplaceDirection,
                                                                params.num_consumers);
        m_graph = std::unique_ptr<Graph>(new Graph());
        m_graph->CreateGraph(replication_result.first, replication_result.second, m_context, "fused_graph");
    }

    virtual std::pair<std::vector<NodePtr>, std::vector<EdgePtr>> CreateModelAndReplicate(
        const ov::Shape& testShape,
        LayoutType firstNodeLayout,
        LOOK firstNodeInplaceDirection,
        LayoutType lastNodeLayout,
        LOOK lastNodeInplaceDirection,
        size_t num_consumers) {
        const auto precision = ov::element::f32;
        // ov::Model with only a transpose node
        ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(precision, testShape)};
        auto order = testShape.size() == 4 ? std::vector<int32_t>{0, 3, 1, 2} : std::vector<int32_t>{0, 2, 1};
        auto constOrder = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{order.size()}, order);
        auto transpose = std::make_shared<ov::op::v1::Transpose>(params[0], constOrder);
        ov::ResultVector results;
        for (size_t i = 0; i < num_consumers; i++)
            results.push_back(std::make_shared<ov::op::v0::Result>(transpose));

        // Replicate
        std::vector<NodePtr> nodes;
        std::vector<EdgePtr> edges;
        std::unordered_set<NodePtr> nodesSet;

        auto addEdge = [&](const NodePtr& parent, const NodePtr& child, size_t parentPort, size_t childPort) -> void {
            auto edge = std::make_shared<Edge>(parent, child, parentPort, childPort);
            Node::addEdge(edge);
            edges.push_back(edge);
            nodesSet.insert(parent);
            nodesSet.insert(child);
        };

        auto inputNode = std::make_shared<node::Input>(params[0], m_context);

        auto dummyNode1 = std::make_shared<cpu_unit_test::DummyNode>(testShape,
                                                                     precision,
                                                                     "reshape",
                                                                     "DummyNode",
                                                                     m_context,
                                                                     firstNodeLayout,
                                                                     firstNodeInplaceDirection);

        auto orderNode = std::make_shared<node::Input>(constOrder, m_context);
        auto transposeNode = std::make_shared<node::Transpose>(transpose, m_context);
        transposeNode->filterSupportedPrimitiveDescriptors();

        addEdge(inputNode, dummyNode1, 0, 0);
        addEdge(dummyNode1, transposeNode, 0, 0);
        addEdge(orderNode, transposeNode, 0, 1);

        const auto& transpose_shape = transpose->get_output_shape(0);
        for (size_t i = 0; i < num_consumers; i++) {
            auto dummyConsumer = std::make_shared<cpu_unit_test::DummyNode>(transpose_shape,
                                                                            precision,
                                                                            "multiply",
                                                                            "DummyNode",
                                                                            m_context,
                                                                            lastNodeLayout,
                                                                            lastNodeInplaceDirection);
            auto outputNode = std::make_shared<node::Input>(results[i], m_context);
            addEdge(transposeNode, dummyConsumer, 0, 0);
            addEdge(dummyConsumer, outputNode, 0, 0);
        }
        for (auto& node : nodesSet)
            nodes.emplace_back(node);
        return {nodes, edges};
    }

    void CheckTransposeCount(size_t ref_transpose_count) const {
        size_t transpose_count = 0;
        for (auto& node : m_graph->GetNodes()) {
            if (node->getType() == Type::Transpose) {
                transpose_count++;
            }
        }
        ASSERT_EQ(ref_transpose_count, transpose_count);
    }

    void CheckReorderCount(size_t ref_optimized_reorder_count, size_t ref_not_optimized_reorder_count) const {
        size_t optimized_count = 0;
        size_t not_optimized_count = 0;
        for (auto& node : m_graph->GetNodes()) {
            if (auto reorder_node = std::dynamic_pointer_cast<node::Reorder>(node)) {
                if (reorder_node->getOptimized())
                    optimized_count++;
                else
                    not_optimized_count++;
            }
        }
        ASSERT_EQ(ref_optimized_reorder_count, optimized_count);
        ASSERT_EQ(ref_not_optimized_reorder_count, not_optimized_count);
    }

    std::shared_ptr<GraphContext> m_context;
    std::unique_ptr<Graph> m_graph;
};  // class MergeTransposeReorderCPUTest

/*
 ┌───────┐
 │ Input │
 └───┬───┘
     │
 ┌───┴───┐
 │ Dummy │
 └───┬───┘
     │
 ┌───┴───┐
 │Reshape│
 └───┬───┘
     │
┌────┴────┐
│Transpose│
└────┬────┘
     │
 ┌───┴───┐
 │ Dummy │
 └───┬───┘
     │
┌────┴───┐
│ Output │
└────────┘
 */
class MergeTransposeReorderWithReshapeCPUTest : public MergeTransposeReorderCPUTest {
    std::pair<std::vector<NodePtr>, std::vector<EdgePtr>> CreateModelAndReplicate(const ov::Shape& testShape,
                                                                                  LayoutType firstNodeLayout,
                                                                                  LOOK firstNodeInplaceDirection,
                                                                                  LayoutType lastNodeLayout,
                                                                                  LOOK lastNodeInplaceDirection,
                                                                                  size_t num_consumers) override {
        const auto precision = ov::element::f32;
        const auto param = std::make_shared<ov::op::v0::Parameter>(precision, testShape);
        auto reshape_const =
            std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{3}, std::vector<int>{0, 0, -1});
        auto reshape = std::make_shared<ov::op::v1::Reshape>(param, reshape_const, true);
        auto order = std::vector<int32_t>{0, 2, 1};
        auto transpose_order = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{order.size()}, order);
        auto transpose = std::make_shared<ov::op::v1::Transpose>(reshape, transpose_order);
        ov::ResultVector results;
        for (size_t i = 0; i < num_consumers; i++)
            results.push_back(std::make_shared<ov::op::v0::Result>(transpose));

        // Replicate
        std::vector<NodePtr> nodes;
        std::vector<EdgePtr> edges;
        std::unordered_set<NodePtr> nodesSet;

        auto addEdge = [&](const NodePtr& parent, const NodePtr& child, size_t parentPort, size_t childPort) -> void {
            auto edge = std::make_shared<Edge>(parent, child, parentPort, childPort);
            Node::addEdge(edge);
            edges.push_back(edge);
            nodesSet.insert(parent);
            nodesSet.insert(child);
        };

        auto inputNode = std::make_shared<node::Input>(param, m_context);
        auto dummyNode1 = std::make_shared<cpu_unit_test::DummyNode>(testShape,
                                                                     precision,
                                                                     "before_reshape",
                                                                     "DummyNode",
                                                                     m_context,
                                                                     LayoutType::nspc,
                                                                     LOOK::LOOK_UP);

        auto reshapeConstNode = std::make_shared<node::Input>(reshape_const, m_context);
        auto reshapeNode = std::make_shared<node::Reshape>(reshape, m_context);

        auto orderNode = std::make_shared<node::Input>(transpose_order, m_context);
        auto transposeNode = std::make_shared<node::Transpose>(transpose, m_context);
        transposeNode->filterSupportedPrimitiveDescriptors();

        addEdge(inputNode, dummyNode1, 0, 0);
        addEdge(dummyNode1, reshapeNode, 0, 0);
        addEdge(reshapeNode, transposeNode, 0, 0);
        addEdge(reshapeConstNode, reshapeNode, 0, 1);
        addEdge(orderNode, transposeNode, 0, 1);

        const auto& transpose_shape = transpose->get_output_shape(0);
        for (size_t i = 0; i < num_consumers; i++) {
            auto dummyConsumer = std::make_shared<cpu_unit_test::DummyNode>(transpose_shape,
                                                                            precision,
                                                                            "multiply",
                                                                            "DummyNode",
                                                                            m_context,
                                                                            LayoutType::ncsp,
                                                                            LOOK::LOOK_DOWN);
            auto outputNode = std::make_shared<node::Input>(results[i], m_context);
            addEdge(transposeNode, dummyConsumer, 0, 0);
            addEdge(dummyConsumer, outputNode, 0, 0);
        }
        for (auto& node : nodesSet)
            nodes.emplace_back(node);
        return {nodes, edges};
    }
};

TEST_P(MergeTransposeReorderCPUTest, smoke_Run_MergeTransposeReorder) {
    Validate();
}

TEST_P(MergeTransposeReorderWithReshapeCPUTest, smoke_Run_MergeTransposeReorderWithReshape) {
    Validate();
}

namespace {
const std::vector<ov::Shape> input_shapes{{1, 3, 8, 16}, {3, 8, 16}};

const std::vector<MergeTransposeReorderTestParam> test_params = {
    // upstream node or downstream node is inPlaced thereby the inserted Reorder cannot be optimized.
    {LayoutType::ncsp, LOOK::LOOK_UP, LayoutType::nspc, LOOK::LOOK_DOWN, 1, Result{0, 0, 2}},
    // no inplace conflict: a single optimized reorder fused with Transpose
    {LayoutType::ncsp, LOOK::LOOK_DOWN, LayoutType::nspc, LOOK::LOOK_UP, 1, Result{0, 1, 1}},
    // 3 non-inplace consumers share a single optimized reorder fused with Transpose
    {LayoutType::ncsp, LOOK::LOOK_UP, LayoutType::nspc, LOOK::LOOK_UP, 3, Result{0, 1, 3}},
    // 3 inplace consumers cannot share reorders thus transpose is not fused with reorders
    // there will be also 3 reorders between 3 dummyNode-consumers and 3 Result nodes
    {LayoutType::ncsp, LOOK::LOOK_UP, LayoutType::nspc, LOOK::LOOK_DOWN, 3, Result{1, 0, 6}},
};

INSTANTIATE_TEST_SUITE_P(smoke_Run_MergeTransposeReorder,
                         MergeTransposeReorderCPUTest,
                         ::testing::Combine(::testing::ValuesIn(input_shapes), ::testing::ValuesIn(test_params)));

const std::vector<ov::Shape> input_shapes_with_reshape{{1, 64, 128, 128}};

const std::vector<MergeTransposeReorderTestParam> test_params_with_reshape = {
    // In case of non optimized reorder OneDNN primitive is used,
    // which doesn't support reordering in case of different ranks on input and output.
    // So the fusion is skipped for such case.
    {LayoutType::nspc, LOOK::LOOK_UP, LayoutType::ncsp, LOOK::LOOK_DOWN, 1, Result{1, 0, 2}},
};

INSTANTIATE_TEST_SUITE_P(smoke_Run_MergeTransposeReorderWithReshape,
                         MergeTransposeReorderWithReshapeCPUTest,
                         ::testing::Combine(::testing::ValuesIn(input_shapes_with_reshape),
                                            ::testing::ValuesIn(test_params_with_reshape)));

}  // namespace

TEST(MergeTransposeReorder, smoke_InplaceConflict) {
    /*  Initial graph:
                  Parameter(Layout: nspc)
                   /                 \
                Reshape             Dummy1(Layout: nspc, LOOK_UP INPLACE)
                   |                   |
               Transpose             Result1
                   |
               Dummy0(LOOK_UP INPLACE)
                   |
                Result0

        In CreateGraph():
            1. Reorder will be inserted between Parameter -> Reshape to convert layout
            2. Reorder -> Reshape -> Transpose will be merged to Reorder
            3. Reorder will be inserted between Parameter -> Dummy1 by ResolveComplexInplaceConflicts()
                   Parameter
                   /       \
                Reorder   Reorder
                   |         |
                Dummy0    Dummy1
                   |         |
                Result0   Result1
    */
    Config conf;
    conf.rtCacheCapacity = 100;
    auto context = std::make_shared<GraphContext>(conf, nullptr, false);

    std::unique_ptr<Graph> graph = std::unique_ptr<Graph>(new Graph());

    const ov::Shape testShape{1, 8, 8, 8};
    ov::ParameterVector params{
        std::make_shared<ov::op::v0::Parameter>(ov::element::Type_t::f32, ov::Shape{1, 8, 8, 8})};

    auto shape_constant =
        std::make_shared<ov::op::v0::Constant>(ov::element::Type_t::i32, ov::Shape{3}, std::vector<int64_t>{1, 8, 64});
    auto reshape_node = std::make_shared<ov::op::v1::Reshape>(params[0], shape_constant, true);
    auto order_constant =
        std::make_shared<ov::op::v0::Constant>(ov::element::Type_t::i32, ov::Shape{3}, std::vector<int64_t>{0, 2, 1});
    auto transpose_node = std::make_shared<ov::op::v1::Transpose>(reshape_node, order_constant);

    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(transpose_node),
                             std::make_shared<ov::op::v0::Result>(params[0])};

    auto nspcCreator = BlockedDescCreator::getCommonCreators().at(LayoutType::nspc);
    auto inDesc = nspcCreator->createSharedDesc(ov::element::Type_t::f32,
                                                ov::intel_cpu::Shape(ov::intel_cpu::VectorDims{1, 8, 8, 8}));
    auto inputNode = std::make_shared<node::Input>(inDesc->clone(), "Input0", "Parameter", context);

    auto shapeConst = std::make_shared<node::Input>(shape_constant, context);
    auto reshapeNode = std::make_shared<node::Reshape>(reshape_node, context);
    auto orderConst = std::make_shared<node::Input>(order_constant, context);
    auto transposeNode = std::make_shared<node::Transpose>(transpose_node, context);
    auto dummyNode0 = std::make_shared<cpu_unit_test::DummyNode>(ov::Shape{1, 64, 8},
                                                                 ov::element::Type_t::f32,
                                                                 "Dummy0",
                                                                 "DummyNode",
                                                                 context,
                                                                 LayoutType::ncsp,
                                                                 Edge::LOOK::LOOK_UP,
                                                                 true);
    auto outputNode0 = std::make_shared<node::Input>(results[0], context);

    auto dummyNode1 = std::make_shared<cpu_unit_test::DummyNode>(ov::Shape{1, 8, 8, 8},
                                                                 ov::element::Type_t::f32,
                                                                 "Dummy1",
                                                                 "DummyNode",
                                                                 context,
                                                                 LayoutType::nspc,
                                                                 Edge::LOOK::LOOK_UP,
                                                                 true);
    auto outputNode1 = std::make_shared<node::Input>(results[1], context);

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

    addEdge(inputNode, dummyNode1, 0, 0);
    addEdge(dummyNode1, outputNode1, 0, 0);

    addEdge(inputNode, reshapeNode, 0, 0);
    addEdge(shapeConst, reshapeNode, 0, 1);
    addEdge(reshapeNode, transposeNode, 0, 0);
    addEdge(orderConst, transposeNode, 0, 1);
    addEdge(transposeNode, dummyNode0, 0, 0);
    addEdge(dummyNode0, outputNode0, 0, 0);

    for (auto& node : nodesSet)
        graphNodes.emplace_back(node);

    graph->CreateGraph(graphNodes, graphEdges, context, "test_graph");
    auto expected_reorder_node0 = dummyNode0->getParentEdgeAt(0)->getParent();
    auto expected_reorder_node1 = dummyNode1->getParentEdgeAt(0)->getParent();

    ASSERT_EQ(expected_reorder_node0->getType(), Type::Reorder);
    ASSERT_EQ(expected_reorder_node1->getType(), Type::Reorder);
    auto merged_reorder = std::dynamic_pointer_cast<node::Reorder>(expected_reorder_node0);
    ASSERT_TRUE(merged_reorder != nullptr);
    ASSERT_TRUE(merged_reorder->getOptimized());
}
