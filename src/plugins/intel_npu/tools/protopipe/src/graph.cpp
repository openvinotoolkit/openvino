//
// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <stack>

#include "graph.hpp"

Nodes Node::srcNodes() const {
    Nodes src_nodes;
    src_nodes.reserve(m_src_edges.size());
    std::transform(m_src_edges.begin(), m_src_edges.end(), std::back_inserter(src_nodes), [](EdgeHandle edge) {
        return edge->srcNode();
    });
    return src_nodes;
}

Nodes Node::dstNodes() const {
    Nodes dst_nodes;
    dst_nodes.reserve(m_dst_edges.size());
    std::transform(m_dst_edges.begin(), m_dst_edges.end(), std::back_inserter(dst_nodes), [](EdgeHandle edge) {
        return edge->dstNode();
    });
    return dst_nodes;
}

Edges Node::srcEdges() const {
    return {m_src_edges.begin(), m_src_edges.end()};
}

Edges Node::dstEdges() const {
    return {m_dst_edges.begin(), m_dst_edges.end()};
}

NodeHandle Graph::create() {
    auto node = std::make_shared<Node>();
    NodeHandle nh(node);
    m_nodes.emplace(node.get(), MetaPtr<Node>{node, Meta{}});
    return nh;
}

void Graph::remove(NodeHandle nh) {
    auto src_edges = nh->srcEdges();
    for (size_t i = 0; i < src_edges.size(); ++i) {
        remove(src_edges[i]);
    }
    auto dst_edges = nh->dstEdges();
    for (size_t i = 0; i < dst_edges.size(); ++i) {
        remove(dst_edges[i]);
    }
    m_nodes.erase(nh.get());
}

void Graph::remove(EdgeHandle eh) {
    auto src = eh->srcNode();
    auto dst = eh->dstNode();
    src->m_dst_edges.erase(eh);
    dst->m_src_edges.erase(eh);
    m_edges.erase(eh.get());
};

EdgeHandle Graph::link(NodeHandle src, NodeHandle dst) {
    auto edge = std::make_shared<Edge>(src, dst);
    EdgeHandle eh{edge};
    m_edges.emplace(edge.get(), MetaPtr<Edge>{edge, Meta{}});
    src->m_dst_edges.insert(eh);
    dst->m_src_edges.insert(eh);
    return eh;
}

Meta& Graph::meta(NodeHandle handle) {
    const auto it = m_nodes.find(handle.get());
    ASSERT(it != m_nodes.end());
    return it->second.meta;
}

const Meta& Graph::meta(NodeHandle handle) const {
    const auto it = m_nodes.find(handle.get());
    ASSERT(it != m_nodes.end());
    return it->second.meta;
}

Meta& Graph::meta(EdgeHandle handle) {
    const auto it = m_edges.find(handle.get());
    ASSERT(it != m_edges.end());
    return it->second.meta;
}

const Meta& Graph::meta(EdgeHandle handle) const {
    const auto it = m_edges.find(handle.get());
    ASSERT(it != m_edges.end());
    return it->second.meta;
}

std::vector<NodeHandle> Graph::nodes() const {
    std::vector<NodeHandle> ret;
    std::transform(m_nodes.begin(), m_nodes.end(), std::back_inserter(ret), [](const auto& p) {
        return NodeHandle{p.second.ptr};
    });
    return ret;
}

static void dfs(NodeHandle& nh, std::unordered_set<NodeHandle>& visited, std::stack<NodeHandle>& stack) {
    visited.insert(nh);
    auto dst_nodes = nh->dstNodes();
    for (auto dst_nh : dst_nodes) {
        auto it = visited.find(dst_nh);
        if (it == visited.end()) {
            dfs(dst_nh, visited, stack);
        }
    }
    stack.push(nh);
};

std::vector<NodeHandle> Graph::sorted() const {
    std::unordered_set<NodeHandle> visited;
    std::stack<NodeHandle> stack;
    const auto nodes = this->nodes();
    for (auto nh : nodes) {
        auto it = visited.find(nh);
        if (it == visited.end()) {
            dfs(nh, visited, stack);
        }
    }
    std::vector<NodeHandle> sorted;
    while (!stack.empty()) {
        sorted.push_back(stack.top());
        stack.pop();
    }
    return sorted;
}

Meta& Meta::operator+=(const Meta& other) {
    for (const auto& p : other.store) {
        ASSERT(store.emplace(p.first, p.second).second);
    }
    return *this;
}
