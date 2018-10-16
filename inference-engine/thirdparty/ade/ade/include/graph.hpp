// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#ifndef GRAPH_HPP
#define GRAPH_HPP

#include <memory>
#include <vector>
#include <unordered_map>

#include "util/assert.hpp"
#include "util/range.hpp"
#include "util/map_range.hpp"

#include "handle.hpp"
#include "edge.hpp"
#include "node.hpp"
#include "metadata.hpp"

namespace ade
{

class Node;
class Edge;
class IGraphListener;

class Graph final
{
public:
    struct HandleMapper
    {
        template<typename T>
        Handle<T> operator()(T* obj) const
        {
            ASSERT(nullptr != obj);
            return Handle<T>(obj->shared_from_this());
        }

        template<typename T>
        Handle<T> operator()(const std::shared_ptr<T>& obj) const
        {
            ASSERT(nullptr != obj);
            return Handle<T>(obj);
        }
    };

    using EdgePtr = std::shared_ptr<Edge>;
    using EdgeList = std::vector<EdgePtr>;

    using NodePtr = std::shared_ptr<Node>;
    using NodesList = std::vector<NodePtr>;
    using NodesListRange  = util::MapRange<util::IterRange<NodesList::iterator>, HandleMapper>;
    using NodesListCRange = util::MapRange<util::IterRange<NodesList::const_iterator>, HandleMapper>;

    Graph();
    ~Graph();

    /// Create new node
    NodeHandle createNode();

    /// Delete node and all connected edges from graph and null all handles pointing to it
    void erase(const NodeHandle& node);

    /// Delete all edges, connected to this node
    void unlink(const NodeHandle& node);

    /// Delete edge from graph and null all handles pointing to it
    void erase(const EdgeHandle& edge);


    /// Create new edge between src_node and dst_node
    EdgeHandle link(const NodeHandle& src_node, const NodeHandle& dst_node);

    /// Change src_edge destination node to dst_node
    /// noop if src_edge destination node is already dst_node
    /// returns src_edge
    EdgeHandle link(const EdgeHandle& src_edge, const NodeHandle& dst_node);

    /// Change dst_edge source node to src_node
    /// noop if dst_edge source node is already src_node
    /// returns dst_edge
    EdgeHandle link(const NodeHandle& src_node, const EdgeHandle& dst_edge);

    NodesListRange nodes();
    NodesListCRange nodes() const;

    MetadataId getMetadataId(const std::string& name) const;

    Metadata&       metadata();
    const Metadata& metadata() const;

    Metadata&       metadata(const NodeHandle handle);
    const Metadata& metadata(const NodeHandle handle) const;

    Metadata&       metadata(const EdgeHandle handle);
    const Metadata& metadata(const EdgeHandle handle) const;

    void setListener(IGraphListener* listener);
    IGraphListener* getListener() const;
private:
    friend class Node;
    friend class Edge;

    struct ElemDeleter final
    {
        void operator()(Node* node) const;
        void operator()(Edge* edge) const;
    };

    Graph(const Graph&) = delete;
    Graph& operator=(const Graph&) = delete;

    EdgeHandle createEdge(Node* src_node, Node* dst_node);

    void removeNode(Node* node);
    void removeEdge(Edge* edge);

    Metadata& geMetadataImpl(void* handle) const;

    NodesList m_nodes;
    EdgeList m_edges;

    struct Id
    {
        std::unique_ptr<char> p;

        Id():p(new char) {} // enforce identity
        Id(const Id&) = delete;
        Id& operator=(const Id&) = delete;
        Id(Id&&) = default;
        Id& operator=(Id&&) = default;
    };

    // To make getMetadataId() const
    mutable std::unordered_map<std::string, Id> m_ids;

    // We will create metadata lazily
    // Stores node/edge metadata per object ptr
    // Stores graph metadata per nullptr
    mutable std::unordered_map<void*, std::unique_ptr<Metadata>> m_metadata;

    IGraphListener* m_listener = nullptr;
};

}

#endif // GRAPH_HPP
