//
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "graph.hpp"

#include <algorithm>
#include <stack>

own::ade::Nodes own::ade::Node::srcNodes() const {
    own::ade::Nodes src_nodes;
    src_nodes.reserve(m_src_edges.size());
    std::transform(m_src_edges.begin(),
                   m_src_edges.end(),
                   std::back_inserter(src_nodes),
                   [](own::ade::EdgeHandle edge) {
                       return edge->srcNode();
                   });
    // FIXME: this was introduced to make the graph
    // the same every run when created the same way.
    // FIXME: cache this information
    std::sort(src_nodes.begin(), src_nodes.end(), [&](const own::ade::NodeHandle& a, const own::ade::NodeHandle& b) {
        auto locked_graph = m_graph.lock();
        return locked_graph->meta(a).get<detail::CreateIdx>().m_idx <
               locked_graph->meta(b).get<detail::CreateIdx>().m_idx;
    });
    return src_nodes;
}

own::ade::Nodes own::ade::Node::dstNodes() const {
    own::ade::Nodes dst_nodes;
    dst_nodes.reserve(m_dst_edges.size());
    std::transform(m_dst_edges.begin(),
                   m_dst_edges.end(),
                   std::back_inserter(dst_nodes),
                   [](own::ade::EdgeHandle edge) {
                       return edge->dstNode();
                   });
    // FIXME: this was introduced to make the graph
    // the same every run when created the same way.
    // FIXME: cache this information
    std::sort(dst_nodes.begin(), dst_nodes.end(), [&](const own::ade::NodeHandle& a, const own::ade::NodeHandle& b) {
        auto locked_graph = m_graph.lock();
        return locked_graph->meta(a).get<detail::CreateIdx>().m_idx <
               locked_graph->meta(b).get<detail::CreateIdx>().m_idx;
    });
    return dst_nodes;
}

own::ade::Edges own::ade::Node::srcEdges() const {
    return {m_src_edges.begin(), m_src_edges.end()};
}

own::ade::Edges own::ade::Node::dstEdges() const {
    return {m_dst_edges.begin(), m_dst_edges.end()};
}

own::ade::NodeHandle own::ade::Graph::create() {
    auto node = std::make_shared<own::ade::Node>(shared_from_this());
    own::ade::NodeHandle nh(node);
    m_nodes.emplace(node.get(), MetaPtr<own::ade::Node>{node, own::ade::Meta{}});
    this->meta(nh).set(detail::CreateIdx{m_create_idx++});
    return nh;
}

void own::ade::Graph::remove(own::ade::NodeHandle nh) {
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

void own::ade::Graph::remove(own::ade::EdgeHandle eh) {
    auto src = eh->srcNode();
    auto dst = eh->dstNode();
    src->m_dst_edges.erase(eh);
    dst->m_src_edges.erase(eh);
    m_edges.erase(eh.get());
}

own::ade::EdgeHandle own::ade::Graph::link(own::ade::NodeHandle src, own::ade::NodeHandle dst) {
    auto edge = std::make_shared<own::ade::Edge>(src, dst);
    own::ade::EdgeHandle eh{edge};
    m_edges.emplace(edge.get(), MetaPtr<own::ade::Edge>{edge, own::ade::Meta{}});
    src->m_dst_edges.insert(eh);
    dst->m_src_edges.insert(eh);
    return eh;
}

own::ade::Meta& own::ade::Graph::meta(own::ade::NodeHandle handle) {
    const auto it = m_nodes.find(handle.get());
    ASSERT(it != m_nodes.end());
    return it->second.meta;
}

const own::ade::Meta& own::ade::Graph::meta(own::ade::NodeHandle handle) const {
    const auto it = m_nodes.find(handle.get());
    ASSERT(it != m_nodes.end());
    return it->second.meta;
}

own::ade::Meta& own::ade::Graph::meta(own::ade::EdgeHandle handle) {
    const auto it = m_edges.find(handle.get());
    ASSERT(it != m_edges.end());
    return it->second.meta;
}

const own::ade::Meta& own::ade::Graph::meta(own::ade::EdgeHandle handle) const {
    const auto it = m_edges.find(handle.get());
    ASSERT(it != m_edges.end());
    return it->second.meta;
}

bool own::ade::Graph::contains(own::ade::NodeHandle handle) const {
    return m_nodes.find(handle.get()) != m_nodes.end();
}

bool own::ade::Graph::linked(own::ade::NodeHandle src, own::ade::NodeHandle dst) {
    for (const auto& edge : src->m_dst_edges) {
        if (edge->dstNode() == dst) {
            return true;
        }
    }
    return false;
}

std::vector<own::ade::NodeHandle> own::ade::Graph::nodes() const {
    std::vector<own::ade::NodeHandle> ret;
    std::transform(m_nodes.begin(), m_nodes.end(), std::back_inserter(ret), [](const auto& p) {
        return own::ade::NodeHandle{p.second.ptr};
    });
    return ret;
}

void own::ade::Graph::dfs(own::ade::NodeHandle& nh,
                          std::unordered_set<own::ade::NodeHandle>& visited,
                          std::stack<own::ade::NodeHandle>& stack) const {
    visited.insert(nh);
    auto dst_nodes = nh->dstNodes();

    // FIXME: this was introduced to make the graph
    // the same every run when created the same way.
    std::sort(dst_nodes.begin(), dst_nodes.end(), [&](const own::ade::NodeHandle& a, const own::ade::NodeHandle& b) {
        return this->meta(a).get<detail::CreateIdx>().m_idx < this->meta(b).get<detail::CreateIdx>().m_idx;
    });

    for (auto dst_nh : dst_nodes) {
        auto it = visited.find(dst_nh);
        if (it == visited.end()) {
            dfs(dst_nh, visited, stack);
        }
    }
    stack.push(nh);
}

std::vector<own::ade::NodeHandle> own::ade::Graph::sorted() const {
    std::unordered_set<own::ade::NodeHandle> visited;
    std::stack<own::ade::NodeHandle> stack;
    auto nodes = this->nodes();

    // FIXME: this was introduced to make the graph
    // the same every run when created the same way.
    std::sort(nodes.begin(), nodes.end(), [&](const own::ade::NodeHandle& a, const own::ade::NodeHandle& b) {
        return this->meta(a).get<detail::CreateIdx>().m_idx < this->meta(b).get<detail::CreateIdx>().m_idx;
    });

    for (auto nh : nodes) {
        auto it = visited.find(nh);
        if (it == visited.end()) {
            dfs(nh, visited, stack);
        }
    }
    std::vector<own::ade::NodeHandle> sorted;
    while (!stack.empty()) {
        sorted.push_back(stack.top());
        stack.pop();
    }
    return sorted;
}

own::ade::Meta& own::ade::Meta::operator+=(const own::ade::Meta& other) {
    for (const auto& p : other.store) {
        ASSERT(store.emplace(p.first, p.second).second);
    }
    return *this;
}
