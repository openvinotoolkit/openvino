//
// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "scenario/scenario_graph.hpp"

DataNode::DataNode(Graph* graph, NodeHandle nh): m_nh(nh) {
    graph->meta(nh).set(Data{});
};

OpNode::OpNode(NodeHandle nh, DataNode out_data): m_nh(nh), m_out_data(out_data) {
}

DataNode OpNode::out() {
    return m_out_data;
}

DataNode ScenarioGraph::makeSource() {
    NodeHandle nh = m_graph.create();
    m_graph.meta(nh).set(Source{});
    return DataNode(&m_graph, nh);
}

void ScenarioGraph::link(DataNode data, OpNode op) {
    m_graph.link(data.m_nh, op.m_nh);
}

OpNode ScenarioGraph::makeInfer(const std::string& tag) {
    return makeOp(Infer{tag});
}

OpNode ScenarioGraph::makeDelay(uint64_t time_in_us) {
    return makeOp(Delay{time_in_us});
}

OpNode ScenarioGraph::makeCompound(uint64_t repeat_count, ScenarioGraph subgraph, InferenceParamsMap infer_params,
                                   const std::string& tag) {
    return makeOp(Compound{repeat_count, subgraph, infer_params, tag});
}
