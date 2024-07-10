// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>

#include "typedefs.hpp"


namespace ov {
namespace mlir {

struct Subgraph {
    ov::NodeVector nodes;
    ov::OutputVector inputs;
    ov::OutputVector outputs;
    std::vector<InputVector> output_consumers;

    // Consumes other subgraph
    void merge (Subgraph& other);
};


using SubgraphPtr = std::shared_ptr<Subgraph>;
using SubgraphID = SymbolPtr;


class SubgraphTracker {
public:

    // callback to finalize the subgraph when it is terminated
    using Finalizer = std::function<void(SubgraphPtr)>;

    SubgraphTracker(Finalizer finalizer);
    void add_node (NodePtr node, bool belongs);
    void finalize();

private:

    std::unordered_map<SubgraphID, SubgraphPtr> m_subgraphs;
    using Dependencies = std::unordered_set<SubgraphID>;
    Finalizer m_finalizer;

    SubgraphID new_subgraph();
    void add_node_to_subgraph(NodePtr node, SubgraphID id);
    void merge_subgraphs(SubgraphID id1, SubgraphID id2);
    SubgraphPtr get_subgraph(SubgraphID id);

    // set/get all subgraph ids that contribute to a given node
    const Dependencies& get_dependencies(NodePtr node);
    void set_dependencies(NodePtr node, const Dependencies& dependencies);

    // set/get subgraph id that a give node belongs to
    SubgraphID get_subgraph_id(NodePtr node);
    void set_subgraph_id(NodePtr node, SubgraphID id);

    bool intersected(const Dependencies& a, const Dependencies& b);
    void terminate_subgraph(SubgraphID id);
    void try_terminate_subgraphs(const Dependencies& subgraphs, NodePtr terminator);
};


} // namespace mlir
} // namespace ov
