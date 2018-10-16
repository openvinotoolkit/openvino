// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#ifndef NODE_HPP
#define NODE_HPP

#include <vector>
#include <utility>
#include <memory>

#include "util/range.hpp"
#include "util/map_range.hpp"

#include "edge.hpp"
#include "handle.hpp"
#include "metadata.hpp"

namespace ade
{

class Graph;
class Edge;
class Node;
using NodeHandle = Handle<Node>;

class Node final : public std::enable_shared_from_this<Node>
{
public:
    struct HandleMapper final
    {
        EdgeHandle operator()(Edge* obj) const;
    };
    struct InEdgeMapper final
    {
        NodeHandle operator()(const EdgeHandle& handle) const;
    };
    struct OutEdgeMapper final
    {
        NodeHandle operator()(const EdgeHandle& handle) const;
    };

    using EdgeSet = std::vector<Edge*>;
    using EdgeSetRange  = util::MapRange<util::IterRange<EdgeSet::iterator>, HandleMapper>;
    using EdgeSetCRange = util::MapRange<util::IterRange<EdgeSet::const_iterator>, HandleMapper>;
    using InNodeSetRange  = util::MapRange<EdgeSetRange,  InEdgeMapper>;
    using InNodeSetCRange = util::MapRange<EdgeSetCRange, InEdgeMapper>;
    using OutNodeSetRange  = util::MapRange<EdgeSetRange,  OutEdgeMapper>;
    using OutNodeSetCRange = util::MapRange<EdgeSetCRange, OutEdgeMapper>;

    EdgeSetRange  inEdges();
    EdgeSetCRange inEdges() const;

    EdgeSetRange  outEdges();
    EdgeSetCRange outEdges() const;

    InNodeSetRange  inNodes();
    InNodeSetCRange inNodes() const;

    OutNodeSetRange  outNodes();
    OutNodeSetCRange outNodes() const;
private:
    friend class Graph;
    friend class Edge;

    Node(Graph* parent);
    ~Node();
    Node(const Node&) = delete;
    Node& operator=(const Node&) = delete;

    Graph* getParent() const;

    void unlink();

    void addInEdge(Edge* edge);
    void removeInEdge(Edge* edge);
    void addOutEdge(Edge* edge);
    void removeOutEdge(Edge* edge);

    Graph* m_parent = nullptr;
    EdgeSet m_inEdges;
    EdgeSet m_outEdges;
};

}

#endif // NODE_HPP
