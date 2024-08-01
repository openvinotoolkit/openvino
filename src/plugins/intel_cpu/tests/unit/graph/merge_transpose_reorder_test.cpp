// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <gtest/gtest.h>

#include <common_test_utils/test_common.hpp>

#include "common_test_utils/node_builders/constant.hpp"
#include "dummy_node.hpp"
#include "graph.h"
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
                        "MergeTransposeReorderCPUTest doesn't support shape", shape,
                        ". Only 4D and 3D shapes are supported");
        m_context = std::make_shared<GraphContext>(Config(), nullptr, false);
        const auto replication_result = CreateModelAndReplicate(shape,
                                                                params.firstNodeLayout,
                                                                params.firstNodeInplaceDirection,
                                                                params.lastNodeLayout,
                                                                params.lastNodeInplaceDirection,
                                                                params.num_consumers);
        m_graph = std::unique_ptr<Graph>(new Graph());
        m_graph->CreateGraph(replication_result.first, replication_result.second, m_context, "fused_graph");
    }

    virtual std::pair<std::vector<NodePtr>, std::vector<EdgePtr>> CreateModelAndReplicate(const ov::Shape& testShape,
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

        auto dummyNode1 = std::make_shared<cpu_unit_test::DummyNode>(
            testShape, precision, "reshape", "DummyNode", m_context, firstNodeLayout, firstNodeInplaceDirection);

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
        for (auto &node : nodesSet) nodes.emplace_back(node);
        return {nodes, edges};
    }

    void CheckTransposeCount(size_t ref_transpose_count) const {
        size_t transpose_count = 0;
        for (auto &node : m_graph->GetNodes()) {
            if (node->getType() == Type::Transpose) {
                transpose_count++;
            }
        }
        ASSERT_EQ(ref_transpose_count, transpose_count);
    }

    void CheckReorderCount(size_t ref_optimized_reorder_count, size_t ref_not_optimized_reorder_count) const {
        size_t optimized_count = 0;
        size_t not_optimized_count = 0;
        for (auto &node : m_graph->GetNodes()) {
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
        auto reshape_const = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{3}, std::vector<int>{0, 0, -1});
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
        auto dummyNode1 = std::make_shared<cpu_unit_test::DummyNode>(
            testShape, precision, "before_reshape", "DummyNode", m_context, LayoutType::nspc, LOOK::LOOK_UP);

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
        for (auto &node : nodesSet) nodes.emplace_back(node);
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