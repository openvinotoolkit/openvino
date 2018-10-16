// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#ifndef GRAPH_LISTENER_HPP
#define GRAPH_LISTENER_HPP

#include "handle.hpp"

namespace ade
{

class Graph;
class Node;
class Edge;
using EdgeHandle = Handle<Edge>;
using NodeHandle = Handle<Node>;

class IGraphListener
{
public:
    virtual ~IGraphListener() = default;

    virtual void nodeCreated(const Graph& graph, const NodeHandle& node) = 0;
    virtual void nodeAboutToBeDestroyed(const Graph& graph, const NodeHandle& node) = 0;

    virtual void edgeCreated(const Graph&, const EdgeHandle& edge) = 0;
    virtual void edgeAboutToBeDestroyed(const Graph& graph, const EdgeHandle& edge) = 0;
    virtual void edgeAboutToBeRelinked(const Graph& graph,
                                       const EdgeHandle& edge,
                                       const NodeHandle& newSrcNode,
                                       const NodeHandle& newDstNode) = 0;
};

}

#endif // GRAPH_LISTENER_HPP
