// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <gtest/gtest.h>

#include "dummy_node.hpp"
#include "graph.h"
#include "memory_control.hpp"
#include "nodes/concat.h"
#include "nodes/input.h"
#include "nodes/rnn.h"
#include "openvino/op/concat.hpp"
#include "openvino/op/gru_sequence.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"

using namespace ov::intel_cpu;
using namespace ov::op;

// helper to check the inplace direction of a node with a part of its name
static void CheckInplaceDirection(const std::shared_ptr<Graph> graph,
                                  const std::string& partial_name,
                                  size_t inport,
                                  const Edge::LOOK undesiredDirection) {
    const std::vector<NodePtr>& graph_nodes = graph->GetNodes();
    size_t actualCount = 0;
    for (auto& node : graph_nodes) {
        if (node->getName().find(partial_name) != std::string::npos) {
            auto parentEdge = node->getParentEdgeAt(inport);
            if (undesiredDirection == 0)
                ASSERT_TRUE(parentEdge->inPlace(Edge::LOOK_BOTH));
            else
                ASSERT_FALSE(parentEdge->inPlace(undesiredDirection));
            actualCount++;
        }
    }

    ASSERT_EQ(1, actualCount);
}

class InplaceResolveIOCPUTestBase : public ::testing::Test {
public:
    std::shared_ptr<Graph> create_graph(const std::vector<ov::PartialShape>& input_shapes,
                                        const size_t num_consumers = 1) {
        Config conf;
        conf.rtCacheCapacity = 100;
        const auto context =
            std::make_shared<const GraphContext>(conf, nullptr, false);

        std::shared_ptr<Graph> graph = std::shared_ptr<Graph>(new Graph());

        ov::ParameterVector params;
        ov::ResultVector results;
        for (auto& input_shape : input_shapes)
            params.push_back(std::make_shared<v0::Parameter>(testPrec, input_shape));
        for (size_t i = 0; i < num_consumers; i++) {
            auto res = std::make_shared<v0::Result>(params.front());  // default, will be changed by impl
            res->set_friendly_name("_result" + std::to_string(i));
            results.push_back(res);
        }

        testShape = params.front()->get_output_partial_shape(0);

        // Replicate
        Replicate(params, results, context);
        graph->CreateGraph(nodes, edges, context, "inplace_resolve_testgraph");

        return graph;
    }

protected:
    virtual void replicate_impl(ov::ParameterVector params, ov::ResultVector results, GraphContext::CPtr context) = 0;

    void addEdge(const NodePtr& parent, const NodePtr& child, size_t parentPort, size_t childPort) {
        auto edge = std::make_shared<Edge>(parent, child, parentPort, childPort);
        Node::addEdge(edge);
        edges.push_back(edge);
        nodesSet.insert(parent);
        nodesSet.insert(child);
    }

    const ov::element::Type_t testPrec = ov::element::Type_t::f32;
    ov::PartialShape testShape;

private:
    void Replicate(ov::ParameterVector params, ov::ResultVector results, GraphContext::CPtr context) {
        replicate_impl(params, results, context);
        for (auto& node : nodesSet)
            nodes.emplace_back(node);
    }

    std::vector<NodePtr> nodes;
    std::vector<EdgePtr> edges;
    std::unordered_set<NodePtr> nodesSet;
};

class RNNConcatCPUTest : public InplaceResolveIOCPUTestBase {
    /*This test runs the following subgraph:
                        param0  param1 param2
                          H_t      X  seq_lens
                         /   \     |     /
                        /     \    |    /
                     Softmax0  RNNSequence
                        \       /(Ho)   \(Y)
                         \     /         \
                         Concat       Reshape1
                           |              |
                           |              |
                         Result0       Result1

    Edge Concat -> Result0 can share memory of inference output; Reshape1 -> Result1 can share memory of inference
    output;
    */
protected:
    void replicate_impl(ov::ParameterVector params, ov::ResultVector results, GraphContext::CPtr context) override {
        std::vector<std::shared_ptr<node::Input>> inputNodes;
        for (size_t i = 0; i < params.size(); i++) {
            inputNodes.push_back(std::make_shared<node::Input>(params[i], context));
        }

        auto dummy_softmax = std::make_shared<cpu_unit_test::DummyNode>(params[0]->get_output_partial_shape(0),
                                                                        testPrec,
                                                                        "Softmax0" /*name*/,
                                                                        "DummyNode" /*type*/,
                                                                        context,
                                                                        LayoutType::ncsp,
                                                                        0 /*look*/);

        auto concat = std::make_shared<v0::Concat>(ov::OutputVector{params[0], params[0]},
                                                   0);  // default, the connection will be reset by addEdge
        auto concatNode = std::make_shared<node::Concat>(concat, context);

        constexpr size_t input_size = 8;
        constexpr size_t num_directions = 1;
        constexpr size_t hidden_size = 3;
        auto W = std::make_shared<v0::Constant>(testPrec, ov::Shape{num_directions, hidden_size * 3, input_size});
        auto R = std::make_shared<v0::Constant>(testPrec, ov::Shape{num_directions, hidden_size * 3, hidden_size});
        auto B = std::make_shared<v0::Constant>(testPrec, ov::Shape{num_directions, hidden_size * 3});
        auto wNode = std::make_shared<node::Input>(W, context);
        auto rNode = std::make_shared<node::Input>(R, context);
        auto bNode = std::make_shared<node::Input>(B, context);
        auto rnnseq = std::make_shared<v5::GRUSequence>(params[1],  // X
                                                        params[0],  // H_t
                                                        params[2],  // sequence_lengths
                                                        W,
                                                        R,
                                                        B,
                                                        hidden_size,
                                                        RecurrentSequenceDirection::FORWARD);
        auto rnnseqNode = std::make_shared<node::RNN>(rnnseq, context);

        auto dummy_reshape = std::make_shared<cpu_unit_test::DummyNode>(rnnseq->get_output_partial_shape(0),
                                                                        testPrec,
                                                                        "Reshape1" /*name*/,
                                                                        "DummyNode" /*type*/,
                                                                        context,
                                                                        LayoutType::ncsp,
                                                                        Edge::LOOK::LOOK_BOTH);

        auto outputNode0 = std::make_shared<node::Input>(results.front(), context);
        auto outputNode1 = std::make_shared<node::Input>(dummy_reshape->getOutputShapeAtPort(0),
                                                         testPrec,
                                                         "_result1",
                                                         "Result",
                                                         context);

        addEdge(inputNodes[0], dummy_softmax, 0, 0);
        addEdge(inputNodes[0], rnnseqNode, 0, 0);

        addEdge(dummy_softmax, concatNode, 0, 0);
        addEdge(concatNode, outputNode0, 0, 0);

        addEdge(inputNodes[1], rnnseqNode, 0, 1);
        addEdge(inputNodes[2], rnnseqNode, 0, 2);

        addEdge(wNode, rnnseqNode, 0, 3);
        addEdge(rNode, rnnseqNode, 0, 4);
        addEdge(bNode, rnnseqNode, 0, 5);

        addEdge(rnnseqNode, dummy_reshape, 0, 0);  // Y
        addEdge(rnnseqNode, concatNode, 1, 1);     // Ho

        addEdge(dummy_reshape, outputNode1, 0, 0);
    }
};

TEST_F(RNNConcatCPUTest, smoke_resolve_inplace_io) {
    auto graph = create_graph({ov::PartialShape{-1, 1, 3}, ov::PartialShape{-1, 10, 8}, ov::PartialShape{-1}}, 2);
    CheckInplaceDirection(graph,
                          std::string("Concat"),
                          0 /*inport*/,
                          Edge::LOOK::LOOK_UP /*undesired edge look direction*/);
    CheckInplaceDirection(graph,
                          std::string("_result1"),
                          0 /*inport*/,
                          Edge::LOOK::LOOK_UP /*undesired edge look direction*/);
}

class SoftmaxAddReshapeOutputCPUTest : public InplaceResolveIOCPUTestBase {
    /*This test runs the following subgraph:

                          param
                            |
                            |
                          Softmax
                         /     \
                        /       \
                       Add     Reshape0
                        |         |
                        |         |
                     Result0   Result1

    expect edge Reshape0->Result1 to be referenced by its upstreams, instead of referencing to its upstreams.
    */
protected:
    void replicate_impl(ov::ParameterVector params, ov::ResultVector results, GraphContext::CPtr context) override {
        std::vector<std::shared_ptr<node::Input>> inputNodes;
        for (size_t i = 0; i < params.size(); i++) {
            inputNodes.push_back(std::make_shared<node::Input>(params[i], context));
        }

        std::vector<std::shared_ptr<node::Input>> outputNodes;
        for (size_t i = 0; i < results.size(); i++) {
            outputNodes.push_back(std::make_shared<node::Input>(results[i], context));
        }

        auto dummy_softmax = std::make_shared<cpu_unit_test::DummyNode>(testShape,
                                                                        testPrec,
                                                                        "softmax" /*name*/,
                                                                        "DummyNode" /*type*/,
                                                                        context,
                                                                        LayoutType::ncsp,
                                                                        0 /*look*/);

        auto dummy_add = std::make_shared<cpu_unit_test::DummyNode>(testShape,
                                                                    testPrec,
                                                                    "add" /*name*/,
                                                                    "DummyNode" /*type*/,
                                                                    context,
                                                                    LayoutType::ncsp,
                                                                    0 /*look*/);

        auto dummy_reshape = std::make_shared<cpu_unit_test::DummyNode>(testShape,
                                                                        testPrec,
                                                                        "reshape" /*name*/,
                                                                        "DummyNode" /*type*/,
                                                                        context,
                                                                        LayoutType::ncsp,
                                                                        Edge::LOOK::LOOK_BOTH);

        addEdge(inputNodes.front(), dummy_softmax, 0, 0);

        addEdge(dummy_softmax, dummy_add, 0, 0);
        addEdge(dummy_add, outputNodes.front(), 0, 0);

        addEdge(dummy_softmax, dummy_reshape, 0, 0);
        addEdge(dummy_reshape, outputNodes.back(), 0, 0);
    }
};

TEST_F(SoftmaxAddReshapeOutputCPUTest, smoke_resolve_inplace_io) {
    auto graph = create_graph({ov::PartialShape{2, -1}}, 2);
    CheckInplaceDirection(graph,
                          std::string("_result1"),
                          0 /*inport*/,
                          Edge::LOOK::LOOK_UP /*undesired edge look direction*/);
}

class SoftmaxAddReshapeTwoOutputsCPUTest : public InplaceResolveIOCPUTestBase {
    /*This test runs the following subgraph:

                          param
                            |
                            |
                          Softmax
                         /       \
                        /         \
                       Add       Reshape0
                        |         |      \
                        |         |       \
                     Result0   Reshape1  Result2
                                  |
                                  |
                                Result1

    Hope Reshape0 could resolve downstream, so either edge Reshape1 -> Result1 or Reshape0 -> Result2
    could get a chance to be referenced by infer request.
    */
protected:
    void replicate_impl(ov::ParameterVector params, ov::ResultVector results, GraphContext::CPtr context) override {
        std::vector<std::shared_ptr<node::Input>> inputNodes;
        for (size_t i = 0; i < params.size(); i++) {
            inputNodes.push_back(std::make_shared<node::Input>(params[i], context));
        }

        std::vector<std::shared_ptr<node::Input>> outputNodes;
        for (size_t i = 0; i < results.size(); i++) {
            outputNodes.push_back(std::make_shared<node::Input>(results[i], context));
        }

        auto dummy_softmax = std::make_shared<cpu_unit_test::DummyNode>(testShape,
                                                                        testPrec,
                                                                        "softmax" /*name*/,
                                                                        "DummyNode" /*type*/,
                                                                        context,
                                                                        LayoutType::ncsp,
                                                                        0 /*look*/);

        auto dummy_add = std::make_shared<cpu_unit_test::DummyNode>(testShape,
                                                                    testPrec,
                                                                    "add" /*name*/,
                                                                    "DummyNode" /*type*/,
                                                                    context,
                                                                    LayoutType::ncsp,
                                                                    0 /*look*/);

        auto dummy_reshape0 = std::make_shared<cpu_unit_test::DummyNode>(testShape,
                                                                         testPrec,
                                                                         "reshape0" /*name*/,
                                                                         "DummyNode" /*type*/,
                                                                         context,
                                                                         LayoutType::ncsp,
                                                                         Edge::LOOK::LOOK_BOTH);

        auto dummy_reshape1 = std::make_shared<cpu_unit_test::DummyNode>(testShape,
                                                                         testPrec,
                                                                         "reshape1" /*name*/,
                                                                         "DummyNode" /*type*/,
                                                                         context,
                                                                         LayoutType::ncsp,
                                                                         Edge::LOOK::LOOK_BOTH);

        addEdge(inputNodes.front(), dummy_softmax, 0, 0);

        addEdge(dummy_softmax, dummy_add, 0, 0);
        addEdge(dummy_add, outputNodes[0], 0, 0);

        addEdge(dummy_softmax, dummy_reshape0, 0, 0);
        addEdge(dummy_reshape0, dummy_reshape1, 0, 0);
        addEdge(dummy_reshape1, outputNodes[1], 0, 0);

        addEdge(dummy_reshape0, outputNodes[2], 0, 0);
    }
};

TEST_F(SoftmaxAddReshapeTwoOutputsCPUTest, smoke_resolve_inplace_io) {
    auto graph = create_graph({ov::PartialShape{2, -1}}, 3);
    CheckInplaceDirection(graph,
                          std::string("reshape0"),
                          0 /*inport*/,
                          Edge::LOOK::LOOK_UP /*undesired edge look direction*/);
}

class InputReshapeOutputCPUTest : public InplaceResolveIOCPUTestBase {
    /*This test runs the following subgraph:

                          param
                            |
                            |
                        Reshape0
                            |
                            |
                         Result0

    Edge Reshape0 -> Result0 cannot be referenced by its upstreams as its upstream is an input.
    */
protected:
    void replicate_impl(ov::ParameterVector params, ov::ResultVector results, GraphContext::CPtr context) override {
        std::vector<std::shared_ptr<node::Input>> inputNodes;
        for (size_t i = 0; i < params.size(); i++) {
            inputNodes.push_back(std::make_shared<node::Input>(params[i], context));
        }

        std::vector<std::shared_ptr<node::Input>> outputNodes;
        for (size_t i = 0; i < results.size(); i++) {
            outputNodes.push_back(std::make_shared<node::Input>(results[i], context));
        }

        auto dummy_reshape = std::make_shared<cpu_unit_test::DummyNode>(testShape,
                                                                        testPrec,
                                                                        "reshape0" /*name*/,
                                                                        "DummyNode" /*type*/,
                                                                        context,
                                                                        LayoutType::ncsp,
                                                                        Edge::LOOK::LOOK_BOTH);

        addEdge(inputNodes.front(), dummy_reshape, 0, 0);
        addEdge(dummy_reshape, outputNodes.front(), 0, 0);
    }
};

TEST_F(InputReshapeOutputCPUTest, smoke_resolve_inplace_io) {
    auto graph = create_graph({ov::PartialShape{2, -1}});
    CheckInplaceDirection(graph,
                          std::string("reshape0"),
                          0 /*inport*/,
                          Edge::LOOK::LOOK_DOWN /*undesired edge look direction*/);
}
