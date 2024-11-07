//
// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>

#include "graph.hpp"
#include "scenario/accuracy_metrics.hpp"
#include "scenario/inference.hpp"
#include "utils/data_providers.hpp"

struct Source {};
struct Data {};

class DataNode {
public:
    DataNode(Graph* graph, NodeHandle nh);

private:
    friend class ScenarioGraph;
    NodeHandle m_nh;
};

class OpNode;
template <>
struct std::hash<OpNode>;

class OpNode {
public:
    OpNode(NodeHandle nh, DataNode out_data);
    DataNode out();

private:
    friend class ScenarioGraph;
    friend struct std::hash<OpNode>;
    NodeHandle m_nh;
    DataNode m_out_data;
};

namespace std {
template <>
struct hash<OpNode> {
    uint64_t operator()(const OpNode& op_node) const {
        return std::hash<NodeHandle>()(op_node.m_nh);
    }
};
}  // namespace std

class ScenarioGraph {
public:
    DataNode makeSource();
    OpNode makeInfer(const std::string& tag);
    OpNode makeDelay(uint64_t time_in_us);
    OpNode makeCompound(uint64_t repeat_count, ScenarioGraph subgraph, InferenceParamsMap infer_params,
                        const std::string& tag);

    void link(DataNode data, OpNode op);

    template <typename F>
    void pass(F&& f) {
        f(m_graph);
    }

private:
    template <typename Kind>
    OpNode makeOp(Kind&& kind);

private:
    Graph m_graph;
};

struct Infer {
    std::string tag;
};

struct Delay {
    uint64_t time_in_us;
};

struct Compound {
    uint64_t repeat_count;
    ScenarioGraph subgraph;
    InferenceParamsMap infer_params;
    std::string tag;
};

struct Op {
    using Kind = std::variant<Infer, Delay, Compound>;
    Kind kind;
};

template <typename Kind>
OpNode ScenarioGraph::makeOp(Kind&& kind) {
    auto op_nh = m_graph.create();
    auto out_nh = m_graph.create();
    m_graph.meta(op_nh).set(Op{std::forward<Kind>(kind)});
    m_graph.link(op_nh, out_nh);
    return OpNode(op_nh, DataNode(&m_graph, out_nh));
}
